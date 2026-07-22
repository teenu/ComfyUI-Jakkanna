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
  - UVs are read per face corner from base.obj, so texture seams are exported
    correctly (OBJ uses split v/vt indices; GLB duplicates seam vertices).
  - GLB coordinates and their optional joints sidecar are exported in metres;
    OBJ coordinates retain MakeHuman's native decimetre scale.
  - A dragged hips root (hipBonePosition) is applied as a translation of the
    pelvis subtree, mirroring the viewer's root-effector translate mode.
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

GLB_METERS_PER_MAKEHUMAN_UNIT = 0.1


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


def _load_uv_topology():
    """Parse base.obj for texcoords and per-face-corner vt indices."""
    for rel in ("makehuman/makehuman/data/3dobjs/base.obj", "makehuman/data/3dobjs/base.obj"):
        path = REPO_ROOT / "CharacterData" / rel
        if path.exists():
            break
    else:
        raise SystemExit("base.obj not found under CharacterData.")
    texcoords = []
    face_corner_vts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("vt "):
                parts = line.split()
                texcoords.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                corners = []
                for token in line.split()[1:]:
                    components = token.split("/")
                    if not components[0]:
                        continue
                    if len(components) > 1 and components[1]:
                        corners.append(int(components[1]) - 1)
                    else:
                        corners.append(-1)
                face_corner_vts.append(corners)
    return np.asarray(texcoords, dtype=np.float64), face_corner_vts


def _visible_face_indices(base_mesh, gender):
    indices = []
    for position, group in enumerate(base_mesh.face_groups or []):
        name = group.strip()
        if name not in VISIBLE_GROUPS:
            continue
        if name == "helper-genital" and gender < 0.99:
            continue
        indices.append(position)
    if not indices:
        raise SystemExit("No visible faces after group filtering; base mesh groups missing?")
    return indices


def _triangulate(faces):
    return [[face[0], face[i], face[i + 1]] for face in faces for i in range(1, len(face) - 1)]


def _compact_indices(faces):
    """Return (remapped_faces, used_original_indices)."""
    used = sorted({index for face in faces for index in face})
    remap = np.full(max(used) + 1, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return [[int(remap[index]) for index in face] for face in faces], np.asarray(used)


def _hip_translation(pose, skeleton, world_matrices, matrix_mod):
    """Apply a dragged-hips root translation to the pelvis subtree, if any.

    The viewer stores the hips effector bone's local position, which in the
    payload's convention equals pelvis.headPos - Root.headPos at rest; any
    difference is a user drag in skeleton units.
    """
    stored = (pose.get("hipBonePosition") or {}).get("hips")
    pelvis = skeleton.getBone("pelvis")
    if stored is None or pelvis is None or pelvis.parent is None:
        return world_matrices
    rest = np.asarray(pelvis.headPos, dtype=np.float64) - np.asarray(pelvis.parent.headPos, dtype=np.float64)
    delta = np.asarray(stored, dtype=np.float64) - rest
    if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) < 1e-4:
        return world_matrices
    print(f"Applying hips drag translation delta {delta.round(4).tolist()}")
    subtree = {"pelvis"}
    for bone in skeleton.boneslist:
        if bone.parent is not None and bone.parent.name in subtree:
            subtree.add(bone.name)
    parent_world = np.asarray(world_matrices[pelvis.parent.name])
    world_delta = parent_world[:3, :3] @ delta
    shift = np.asarray(matrix_mod.translate(world_delta))
    return {name: (shift @ np.asarray(world) if name in subtree else world)
            for name, world in world_matrices.items()}


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


def _joint_dump(skeleton, world_matrices, model_rotation=None, scale=1.0):
    rotation = np.asarray(model_rotation) if model_rotation is not None else np.identity(3)
    joints = {}
    for bone in skeleton.boneslist:
        world = np.asarray(world_matrices[bone.name])
        tail_offset = np.append(bone.tailPos - bone.headPos, 1.0)
        joints[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "head": [round(float(v), 6) for v in scale * (rotation @ world[:3, 3])],
            "tail": [round(float(v), 6) for v in scale * (rotation @ (world @ tail_offset)[:3])],
        }
    return joints


def _write_obj(path, vertices, texcoords, v_faces, vt_faces):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Jakkanna posed mannequin export\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in texcoords:
            handle.write(f"vt {u:.6f} {v:.6f}\n")
        handle.write("g body\n")
        for v_face, vt_face in zip(v_faces, vt_faces):
            handle.write("f " + " ".join(f"{v + 1}/{vt + 1}" for v, vt in zip(v_face, vt_face)) + "\n")


def _write_glb(path, vertices, texcoords, v_faces, vt_faces):
    try:
        import trimesh
    except ImportError:
        raise SystemExit("GLB export needs the 'trimesh' package (pip install trimesh).")
    # Split vertices per unique (position, uv) pair so texture seams survive.
    corner_lookup = {}
    split_vertices = []
    split_uvs = []
    triangles = []
    for v_face, vt_face in zip(v_faces, vt_faces):
        corners = []
        for v_index, vt_index in zip(v_face, vt_face):
            key = (v_index, vt_index)
            corner = corner_lookup.get(key)
            if corner is None:
                corner = len(split_vertices)
                corner_lookup[key] = corner
                split_vertices.append(vertices[v_index])
                split_uvs.append(texcoords[vt_index])
            corners.append(corner)
        for i in range(1, len(corners) - 1):
            triangles.append([corners[0], corners[i], corners[i + 1]])
    mesh = trimesh.Trimesh(
        vertices=np.asarray(split_vertices) * GLB_METERS_PER_MAKEHUMAN_UNIT,
        faces=np.asarray(triangles, dtype=np.int64),
        visual=trimesh.visual.TextureVisuals(uv=np.asarray(split_uvs)),
        process=False,
    )
    mesh.export(path)


def _output_path(template, pose_index, multiple):
    if not multiple:
        return template
    return template.with_name(f"{template.stem}_pose{pose_index}{template.suffix}")


def _same_path(first, second):
    return first.resolve() == second.resolve() or (first.exists() and second.exists() and first.samefile(second))


def _validate_output_paths(input_path, mesh_paths, joints_paths):
    destinations = []
    for label, paths, suffixes in (
        ("mesh output", mesh_paths, {".obj", ".glb"}),
        ("joints output", joints_paths, {".json"}),
    ):
        for path in paths:
            if path.suffix.lower() not in suffixes:
                expected = " or ".join(sorted(suffixes))
                raise SystemExit(f"{label} must use a {expected} suffix: {path}")
            if _same_path(path, input_path):
                raise SystemExit(f"{label} must not overwrite the input file: {path}")
            for previous_label, previous_path in destinations:
                if _same_path(path, previous_path):
                    raise SystemExit(f"{label} collides with {previous_label}: {path}")
            destinations.append((label, path))


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

    texcoords, face_corner_vts = _load_uv_topology()
    if len(face_corner_vts) != len(base_mesh.faces):
        raise SystemExit("base.obj face count mismatch between obj_loader and UV parser.")
    kept = _visible_face_indices(base_mesh, gender)
    v_faces = [[int(v[0] if isinstance(v, (list, tuple)) else v) for v in base_mesh.faces[i]] for i in kept]
    vt_faces = [face_corner_vts[i] for i in kept]
    if any(vt < 0 for face in vt_faces for vt in face):
        raise SystemExit("base.obj is missing texture coordinates on visible faces.")
    if args.triangulate:
        v_faces = _triangulate(v_faces)
        vt_faces = _triangulate(vt_faces)
    v_faces, used = _compact_indices(v_faces)
    vt_faces, used_vt = _compact_indices(vt_faces)
    texcoords = texcoords[used_vt]

    default_output = args.output or args.input.with_suffix("").with_name(args.input.stem + ".obj")
    multiple = len(selected) > 1
    mesh_paths = {pose_index: _output_path(default_output, pose_index, multiple) for pose_index, _ in selected}
    joints_paths = ({pose_index: _output_path(args.joints, pose_index, multiple) for pose_index, _ in selected}
                    if args.joints is not None else {})
    _validate_output_paths(args.input, mesh_paths.values(), joints_paths.values())

    for pose_index, pose in selected:
        skeleton, world_matrices = openpose._pose_bones(rest_vertices, mesh_params, pose)
        world_matrices = _hip_translation(pose, skeleton, world_matrices, matrix_mod)
        skinned = _skin_vertices(rest_vertices, skeleton, world_matrices, matrix_mod)[used]
        model_rotation = None
        if args.apply_model_rotation:
            model_rotation = openpose._model_rotation(pose)
            skinned = skinned @ model_rotation.T

        out_path = mesh_paths[pose_index]
        if out_path.suffix.lower() == ".glb":
            _write_glb(out_path, skinned, texcoords, v_faces, vt_faces)
        else:
            _write_obj(out_path, skinned, texcoords, v_faces, vt_faces)

        lower = skinned.min(axis=0)
        upper = skinned.max(axis=0)
        print(f"Pose {pose_index}: {len(skinned)} vertices, {len(v_faces)} faces -> {out_path}")
        print(f"  bounds min [{lower[0]:.2f} {lower[1]:.2f} {lower[2]:.2f}] "
              f"max [{upper[0]:.2f} {upper[1]:.2f} {upper[2]:.2f}] (MakeHuman units, Y-up)")

        if args.joints is not None:
            joints_path = joints_paths[pose_index]
            glb = out_path.suffix.lower() == ".glb"
            joints_path.write_text(
                json.dumps({"units": "meters" if glb else "makehuman-decimeters", "up": "Y",
                            "joints": _joint_dump(skeleton, world_matrices, model_rotation,
                                                  GLB_METERS_PER_MAKEHUMAN_UNIT if glb else 1.0)}, indent=2),
                encoding="utf-8",
            )
            print(f"  joints -> {joints_path}")


if __name__ == "__main__":
    main()
