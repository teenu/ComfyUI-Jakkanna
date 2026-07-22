#!/usr/bin/env python3
"""Export the posed Jakkanna mannequin as a mesh file.

Reads a ComfyUI workflow JSON (or a raw Pose Studio pose_data JSON), rebuilds
the morphed character through the same CharacterData pipeline the node uses,
applies the stored per-bone rotations, and linear-blend-skins the vertices
with the skeleton's canonical weights. Only the face groups the viewer shows
are exported (body, eyes, teeth, tongue; genitals follow the gender rule).

The pose math is imported from nodes/openpose_export.py rather than copied,
so the exported mesh always matches the node's own OpenPose projection.

Usage (run with ComfyUI's venv python — torch must be importable):
    .venv/bin/python custom_nodes/ComfyUI-Jakkanna/tools/export_posed_mesh.py \
        user/default/workflows/Krea_2_Community.json -o mannequin.obj

Notes:
  - UVs come from base.obj collapsed to one UV per vertex (seam vertices keep
    the last UV encountered), matching obj_loader's behaviour.
  - Root translation from dragged hips (hipBonePosition) is not applied,
    matching the behaviour of the node's OpenPose export.
"""

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = "jakkanna_repo"

# Face groups the Pose Studio viewer displays (see _jakkanna_build_update_preview_payload)
VISIBLE_GROUPS = {
    "body",
    "helper-r-eye",
    "helper-l-eye",
    "helper-upper-teeth",
    "helper-lower-teeth",
    "helper-tongue",
    "helper-genital",
}


def _import_repo_modules():
    """Register the repo as an importable package and load the node modules."""
    if PKG not in sys.modules:
        spec = importlib.machinery.ModuleSpec(PKG, None, is_package=True)
        spec.submodule_search_locations = [str(REPO_ROOT)]
        sys.modules[PKG] = importlib.util.module_from_spec(spec)
    openpose = importlib.import_module(f"{PKG}.nodes.openpose_export")
    matrix = importlib.import_module(f"{PKG}.CharacterData.matrix")
    return openpose, matrix


def _extract_pose_data(doc):
    """Return the pose_data dict from a workflow JSON or a raw pose_data JSON."""
    if isinstance(doc, dict) and "poses" in doc and "mesh" in doc:
        return doc

    found = []
    if isinstance(doc, dict) and isinstance(doc.get("nodes"), list):
        # UI workflow format
        for node in doc["nodes"]:
            if "PoseStudio" not in str(node.get("type", "")):
                continue
            for value in node.get("widgets_values") or []:
                if not (isinstance(value, str) and value.lstrip().startswith("{")):
                    continue
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "poses" in parsed and "mesh" in parsed:
                    found.append((node.get("id"), parsed))
    elif isinstance(doc, dict):
        # API (prompt) workflow format
        for node_id, node in doc.items():
            if not isinstance(node, dict) or "PoseStudio" not in str(node.get("class_type", "")):
                continue
            value = (node.get("inputs") or {}).get("pose_data")
            if isinstance(value, str) and value.lstrip().startswith("{"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "poses" in parsed and "mesh" in parsed:
                    found.append((node_id, parsed))

    if not found:
        raise SystemExit("No Pose Studio pose_data found in the input JSON.")
    if len(found) > 1:
        ids = ", ".join(str(node_id) for node_id, _ in found)
        print(f"Multiple Pose Studio nodes found (ids: {ids}); using the first.")
    return found[0][1]


def _visible_faces(base_mesh, gender):
    faces = []
    for face, group in zip(base_mesh.faces, base_mesh.face_groups or []):
        name = group.strip()
        if name not in VISIBLE_GROUPS:
            continue
        if name == "helper-genital" and gender < 0.99:
            continue
        faces.append([int(v[0] if isinstance(v, (list, tuple)) else v) for v in face])
    if not faces:
        raise SystemExit("No visible faces after group filtering; base mesh groups missing?")
    return faces


def _compact(faces, arrays):
    """Drop unreferenced vertices and remap face indices."""
    used = sorted({index for face in faces for index in face})
    remap = np.full(max(used) + 1, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_faces = [[int(remap[index]) for index in face] for face in faces]
    return new_faces, [array[used] for array in arrays], np.asarray(used)


def _skin_vertices(vertices, skeleton, world_matrices, matrix_mod):
    positions = np.zeros((len(vertices), 3), dtype=np.float64)
    total = np.zeros(len(vertices), dtype=np.float64)
    homogeneous = np.hstack([vertices.astype(np.float64), np.ones((len(vertices), 1))])

    for bone_name, (indices, weights) in skeleton.vertexWeights.data.items():
        bone = skeleton.getBone(bone_name)
        world = world_matrices.get(bone_name)
        if bone is None or world is None:
            continue
        # Rest local axes are world-aligned, so bind inverse is translate(-head)
        skin_matrix = np.asarray(world @ np.asarray(matrix_mod.translate(-bone.headPos)))
        transformed = homogeneous[indices] @ skin_matrix.T
        positions[indices] += transformed[:, :3] * weights[:, None].astype(np.float64)
        total[indices] += weights

    unweighted = np.flatnonzero(total < 1e-6)
    if unweighted.size:
        print(f"Warning: {unweighted.size} vertices have no skin weights; left at rest position.")
        positions[unweighted] = vertices[unweighted]
        total[unweighted] = 1.0
    return positions / total[:, None]


def _joint_dump(skeleton, world_matrices):
    joints = {}
    for bone in skeleton.boneslist:
        world = np.asarray(world_matrices[bone.name])
        tail_offset = np.append(bone.tailPos - bone.headPos, 1.0)
        joints[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "head": [round(float(v), 6) for v in world[:3, 3]],
            "tail": [round(float(v), 6) for v in (world @ tail_offset)[:3]],
        }
    return joints


def _write_obj(path, vertices, uvs, faces):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Jakkanna posed mannequin export\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            handle.write(f"vt {u:.6f} {v:.6f}\n")
        handle.write("g body\n")
        for face in faces:
            handle.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in face) + "\n")


def _write_glb(path, vertices, uvs, faces):
    try:
        import trimesh
    except ImportError:
        raise SystemExit("GLB export needs the 'trimesh' package (pip install trimesh).")
    triangles = []
    for face in faces:
        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i + 1]])
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(triangles, dtype=np.int64),
        visual=trimesh.visual.TextureVisuals(uv=uvs),
        process=False,
    )
    mesh.export(path)


def _output_path(template, pose_index, multiple):
    if not multiple:
        return template
    return template.with_name(f"{template.stem}_pose{pose_index}{template.suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="Workflow JSON or raw pose_data JSON")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output mesh path (.obj or .glb); default <input>_pose<N>.obj")
    parser.add_argument("--pose-index", type=int, default=0, help="Pose to export (default 0)")
    parser.add_argument("--all", action="store_true", help="Export every pose in the pose_data")
    parser.add_argument("--apply-model-rotation", action="store_true",
                        help="Bake the pose's modelRotation into the vertices")
    parser.add_argument("--triangulate", action="store_true", help="Split quads into triangles")
    parser.add_argument("--joints", type=Path, default=None,
                        help="Also write posed skeleton joint positions to this JSON path")
    args = parser.parse_args()

    openpose, matrix_mod = _import_repo_modules()

    doc = json.loads(args.input.read_text(encoding="utf-8"))
    data = _extract_pose_data(doc)
    openpose._validate_pose_data(data)
    mesh_params = data.get("mesh", {})
    poses = data.get("poses") or [{}]

    if args.all:
        selected = list(enumerate(poses))
    else:
        if not 0 <= args.pose_index < len(poses):
            raise SystemExit(f"pose-index {args.pose_index} out of range (found {len(poses)} poses)")
        selected = [(args.pose_index, poses[args.pose_index])]

    openpose._ensure_data_loaded()
    from importlib import import_module
    pose_studio = import_module(f"{PKG}.nodes.pose_studio")
    base_mesh = pose_studio.POSE_STUDIO_CACHE["base_mesh"]

    rest_vertices = openpose._solve_vertices(mesh_params)
    gender = float(mesh_params.get("gender", 0.5)) if isinstance(mesh_params.get("gender", 0.5), (int, float)) else 0.5
    faces = _visible_faces(base_mesh, gender)
    if args.triangulate:
        faces = [tri for face in faces for tri in
                 ([[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)])]
    faces, (uvs,), used = _compact(faces, [base_mesh.vertex_uvs])

    default_output = args.output or args.input.with_suffix("").with_name(args.input.stem + ".obj")
    multiple = len(selected) > 1

    for pose_index, pose in selected:
        skeleton, world_matrices = openpose._pose_bones(rest_vertices, mesh_params, pose)
        skinned = _skin_vertices(rest_vertices, skeleton, world_matrices, matrix_mod)[used]
        if args.apply_model_rotation:
            skinned = skinned @ openpose._model_rotation(pose).T

        out_path = _output_path(default_output, pose_index, multiple)
        if out_path.suffix.lower() == ".glb":
            _write_glb(out_path, skinned, uvs, faces)
        else:
            _write_obj(out_path, skinned, uvs, faces)

        lower = skinned.min(axis=0)
        upper = skinned.max(axis=0)
        print(f"Pose {pose_index}: {len(skinned)} vertices, {len(faces)} faces -> {out_path}")
        print(f"  bounds min [{lower[0]:.2f} {lower[1]:.2f} {lower[2]:.2f}] "
              f"max [{upper[0]:.2f} {upper[1]:.2f} {upper[2]:.2f}] (MakeHuman units, Y-up)")

        if args.joints is not None:
            joints_path = _output_path(args.joints, pose_index, multiple)
            joints_path.write_text(
                json.dumps({"units": "makehuman-decimeters", "up": "Y",
                            "joints": _joint_dump(skeleton, world_matrices)}, indent=2),
                encoding="utf-8",
            )
            print(f"  joints -> {joints_path}")


if __name__ == "__main__":
    main()
