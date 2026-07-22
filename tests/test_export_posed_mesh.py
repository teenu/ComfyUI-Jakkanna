import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPEC = importlib.util.spec_from_file_location(
    "jakkanna_export_posed_mesh",
    os.path.join(ROOT, "tools", "export_posed_mesh.py"),
)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class Bone:
    def __init__(self, name, head, tail, parent=None):
        self.name = name
        self.headPos = np.asarray(head, dtype=np.float64)
        self.tailPos = np.asarray(tail, dtype=np.float64)
        self.parent = parent


class Skeleton:
    def __init__(self, bones):
        self.boneslist = bones
        self.bones = {bone.name: bone for bone in bones}

    def getBone(self, name):
        return self.bones.get(name)


class Matrix:
    @staticmethod
    def translate(offset):
        result = np.identity(4)
        result[:3, 3] = offset
        return result


class ExportTransformTests(unittest.TestCase):
    def test_hip_translation_uses_parent_coordinate_space(self):
        root = Bone("Root", [0, 0, 0], [0, 1, 0])
        pelvis = Bone("pelvis", [1, 0, 0], [1, 1, 0], root)
        child = Bone("spine", [1, 1, 0], [1, 2, 0], pelvis)
        skeleton = Skeleton([root, pelvis, child])

        root_world = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        pelvis_world = root_world @ Matrix.translate([1, 0, 0])
        spine_world = pelvis_world @ Matrix.translate([0, 1, 0])
        world = {"Root": root_world, "pelvis": pelvis_world, "spine": spine_world}

        translated = EXPORTER._hip_translation(
            {"hipBonePosition": {"hips": [2, 0, 0]}},
            skeleton,
            world,
            Matrix,
        )

        np.testing.assert_allclose(translated["Root"], root_world)
        np.testing.assert_allclose(translated["pelvis"][:3, 3], pelvis_world[:3, 3] + [0, 1, 0])
        np.testing.assert_allclose(translated["spine"][:3, 3], spine_world[:3, 3] + [0, 1, 0])

    def test_joint_dump_applies_model_rotation(self):
        root = Bone("Root", [1, 0, 0], [1, 1, 0])
        skeleton = Skeleton([root])
        world = {"Root": Matrix.translate([1, 0, 0])}
        rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        joints = EXPORTER._joint_dump(skeleton, world, rotation)

        self.assertEqual(joints["Root"]["head"], [0.0, 1.0, 0.0])
        self.assertEqual(joints["Root"]["tail"], [-1.0, 1.0, 0.0])

    def test_joint_dump_scales_glb_coordinates_to_meters(self):
        root = Bone("Root", [10, 0, 0], [10, 20, 0])
        skeleton = Skeleton([root])
        world = {"Root": Matrix.translate([10, 0, 0])}

        joints = EXPORTER._joint_dump(
            skeleton,
            world,
            scale=EXPORTER.GLB_METERS_PER_MAKEHUMAN_UNIT,
        )

        self.assertEqual(joints["Root"]["head"], [1.0, 0.0, 0.0])
        self.assertEqual(joints["Root"]["tail"], [1.0, 2.0, 0.0])

    def test_glb_vertices_are_exported_in_meters(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.glb"
            EXPORTER._write_glb(
                path,
                np.asarray([[0, 0, 0], [10, 0, 0], [0, 20, 0]], dtype=np.float64),
                np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
                [[0, 1, 2]],
                [[0, 1, 2]],
            )

            mesh = trimesh.load(path, force="mesh", process=False)
            np.testing.assert_allclose(mesh.bounds, [[0, 0, 0], [1, 2, 0]])

    def test_output_paths_reject_unsupported_suffixes_and_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "workflow.json"
            source.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "mesh output must use"):
                EXPORTER._validate_output_paths(source, [Path(directory) / "mesh.json"], [])
            with self.assertRaisesRegex(SystemExit, "must not overwrite the input"):
                EXPORTER._validate_output_paths(source, [], [source])

            hardlink = Path(directory) / "mesh.obj"
            os.link(source, hardlink)
            with self.assertRaisesRegex(SystemExit, "must not overwrite the input"):
                EXPORTER._validate_output_paths(source, [hardlink], [])

    def test_output_paths_accept_distinct_mesh_and_joint_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            EXPORTER._validate_output_paths(
                root / "workflow.json",
                [root / "mannequin.glb"],
                [root / "joints.json"],
            )


if __name__ == "__main__":
    unittest.main()
