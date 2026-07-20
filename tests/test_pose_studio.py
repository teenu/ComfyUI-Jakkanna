import base64
import importlib
import io
import json
import os
import sys
import types
import unittest

import numpy as np
from PIL import Image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from CharacterData.mh_parser import HumanSolver, TargetParser
from CharacterData.mh_skeleton import Skeleton
from CharacterData.obj_loader import load_obj


def load_pose_studio_module():
    package_name = "ComfyUI_VNCCS_Utils"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [ROOT]
        sys.modules[package_name] = package

    nodes_name = f"{package_name}.nodes"
    nodes = sys.modules.get(nodes_name)
    if nodes is None:
        nodes = types.ModuleType(nodes_name)
        nodes.__path__ = [os.path.join(ROOT, "nodes")]
        sys.modules[nodes_name] = nodes

    return importlib.import_module(f"{nodes_name}.pose_studio")


class SkeletonWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        makehuman = os.path.join(ROOT, "CharacterData", "makehuman", "makehuman")
        cls.mesh = load_obj(os.path.join(makehuman, "data", "3dobjs", "base.obj"))
        cls.skeleton = Skeleton()
        cls.skeleton.fromFile(os.path.join(makehuman, "data", "rigs", "game_engine.mhskel"), cls.mesh)

    def test_weights_are_unique_normalized_and_limited(self):
        vertex_count = len(self.mesh.vertices)
        weight_sum = np.zeros(vertex_count, dtype=np.float64)
        influence_count = np.zeros(vertex_count, dtype=np.uint32)

        for indices, weights in self.skeleton.vertexWeights.data.values():
            self.assertEqual(len(indices), len(np.unique(indices)))
            weight_sum[indices] += weights
            influence_count[indices] += 1

        np.testing.assert_allclose(weight_sum, 1.0, rtol=0.0, atol=1e-6)
        self.assertGreaterEqual(int(influence_count.min()), 1)
        self.assertLessEqual(int(influence_count.max()), 4)
        np.testing.assert_array_equal(influence_count, self.skeleton.vertexWeights._wCounts)

    def test_weight_canonicalization_is_idempotent(self):
        before = [
            (bone_name, indices.copy(), weights.copy())
            for bone_name, (indices, weights) in self.skeleton.vertexWeights.data.items()
        ]

        self.skeleton._canonicalize_weights(self.mesh)

        after = list(self.skeleton.vertexWeights.data.items())
        self.assertEqual(len(before), len(after))
        for (before_name, before_indices, before_weights), (after_name, (after_indices, after_weights)) in zip(before, after):
            self.assertEqual(before_name, after_name)
            np.testing.assert_array_equal(before_indices, after_indices)
            np.testing.assert_array_equal(before_weights, after_weights)


class CapturedImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pose_studio = load_pose_studio_module()
        image = Image.new("RGB", (2, 2), (12, 34, 56))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        cls.data_url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

    def test_decodes_eighty_one_captures(self):
        images = self.pose_studio._decode_captured_images([self.data_url] * 81)

        self.assertEqual(len(images), 81)
        self.assertTrue(all(image.size == (2, 2) for image in images))

    def test_sparse_capture_slots_remain_supported(self):
        images = self.pose_studio._decode_captured_images([None, self.data_url, "", self.data_url])

        self.assertEqual(len(images), 2)

    def test_capture_count_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "limit is 256"):
            self.pose_studio._decode_captured_images([None] * 257)

    def test_invalid_base64_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid base64"):
            self.pose_studio._decode_captured_images(["data:image/png;base64,not-valid!"])

    def test_generate_returns_all_eighty_one_captures(self):
        data = {
            "poses": [{} for _ in range(81)],
            "captured_images": [self.data_url] * 81,
            "lighting_prompts": [f"prompt {index}" for index in range(81)],
            "export": {"output_mode": "LIST"},
        }

        images, prompts = self.pose_studio.VNCCS_PoseStudio().generate(json.dumps(data))

        self.assertEqual(len(images), 81)
        self.assertEqual(prompts, data["lighting_prompts"])
        self.assertTrue(all(tuple(image.shape) == (1, 2, 2, 3) for image in images))
        self.assertAlmostEqual(float(images[0][0, 0, 0, 0]), 12.0 / 255.0)

    def test_generate_rejects_incomplete_capture_set(self):
        data = {
            "poses": [{}, {}],
            "captured_images": [self.data_url, None],
        }

        with self.assertRaisesRegex(RuntimeError, "1 of 2 required frontend captures"):
            self.pose_studio.VNCCS_PoseStudio().generate(json.dumps(data))


class PoseDataValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pose_studio = load_pose_studio_module()

    def test_rejects_malformed_bone_rotation(self):
        data = {"poses": [{"bones": {"upperarm_l": [10, float("nan"), 20]}}]}

        with self.assertRaisesRegex(ValueError, "finite number"):
            self.pose_studio._validate_pose_data(data)

    def test_rejects_invalid_export_dimensions(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 4096"):
            self.pose_studio._validate_pose_data({
                "poses": [{}],
                "export": {"view_width": 0},
            })

    def test_pose_image_hash_covers_every_element(self):
        first = self.pose_studio.torch.zeros((1, 1, 2000, 1), dtype=self.pose_studio.torch.float32)
        second = first.clone()
        second.reshape(-1)[1] = 1.0

        first_hash = self.pose_studio.VNCCS_PoseStudio.IS_CHANGED("{}", first)
        second_hash = self.pose_studio.VNCCS_PoseStudio.IS_CHANGED("{}", second)

        self.assertNotEqual(first_hash, second_hash)


class MorphFactorTests(unittest.TestCase):
    def test_proportion_targets_are_tagged(self):
        makehuman = os.path.join(ROOT, "CharacterData", "makehuman")
        targets = TargetParser(makehuman).scan_targets()
        proportions = [
            target["tags"].get("proportions")
            for target in targets
            if "proportions" in target["tags"]
        ]

        self.assertEqual(proportions.count("idealproportions"), 54)
        self.assertEqual(proportions.count("uncommonproportions"), 54)

    def test_proportion_and_race_factors_are_normalized(self):
        factors = HumanSolver().calculate_factors(
            0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            proportions=0.8, african=2.0, asian=-1.0, caucasian=1.0,
        )

        self.assertAlmostEqual(factors["idealproportions"], 0.8)
        self.assertAlmostEqual(factors["uncommonproportions"], 0.2)
        self.assertAlmostEqual(factors["african"], 2.0 / 3.0)
        self.assertEqual(factors["asian"], 0.0)
        self.assertAlmostEqual(factors["caucasian"], 1.0 / 3.0)

    def test_live_worker_defines_python_morph_factors(self):
        worker_path = os.path.join(ROOT, "web", "vnccs_pose_morph_worker.js")
        with open(worker_path, "r", encoding="utf-8") as handle:
            worker = handle.read()

        for factor in ("idealproportions", "uncommonproportions", "african", "asian", "caucasian"):
            self.assertIn(f"{factor}:", worker)


if __name__ == "__main__":
    unittest.main()
