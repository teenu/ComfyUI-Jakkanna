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
    package_name = "ComfyUI_Jakkanna"
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


def load_openpose_module():
    load_pose_studio_module()
    return importlib.import_module("ComfyUI_Jakkanna.nodes.openpose_export")


def image_data_url(color=(12, 34, 56), size=(2, 2)):
    image = Image.new("RGB", size, color)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def openpose_frame(marker=1.0, width=2, height=2):
    fields = {
        "pose_keypoints_2d": [0.0] * 54,
        "foot_keypoints_2d": [0.0] * 18,
        "face_keypoints_2d": [0.0] * 210,
        "hand_right_keypoints_2d": [0.0] * 63,
        "hand_left_keypoints_2d": [0.0] * 63,
    }
    fields["pose_keypoints_2d"][0] = marker
    return {
        "canvas_width": width,
        "canvas_height": height,
        "people": [fields],
    }


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
        cls.data_url = image_data_url()

    def test_decodes_eighty_one_captures(self):
        images = self.pose_studio._decode_captured_images([self.data_url] * 81)

        self.assertEqual(len(images), 81)
        self.assertTrue(all(image.size == (2, 2) for image in images))

    def test_sparse_capture_slots_remain_supported(self):
        images = self.pose_studio._decode_captured_images([None, self.data_url, "", self.data_url])

        self.assertEqual(len(images), 2)

    def test_capture_count_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "limit is 128"):
            self.pose_studio._decode_captured_images([None] * 129)

    def test_default_resolution_fits_the_structural_pose_limit(self):
        self.assertEqual(self.pose_studio._CAPTURED_IMAGE_MAX_COUNT, 128)
        self.assertLessEqual(
            self.pose_studio._CAPTURED_IMAGE_MAX_COUNT * 1024 * 1024,
            self.pose_studio._CAPTURED_IMAGE_MAX_TOTAL_PIXELS,
        )

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

        images, prompts = self.pose_studio.JakkannaPoseStudio().generate(json.dumps(data))

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
            self.pose_studio.JakkannaPoseStudio().generate(json.dumps(data))


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

        with self.assertRaisesRegex(ValueError, "integer"):
            self.pose_studio._validate_pose_data({
                "poses": [{}],
                "export": {"view_width": 1024.5},
            })

    def test_rejects_frontend_state_beyond_light_and_prompt_limits(self):
        with self.assertRaisesRegex(ValueError, "too many entries"):
            self.pose_studio._validate_pose_data({
                "poses": [{}],
                "lights": [{} for _ in range(33)],
            })

        with self.assertRaisesRegex(ValueError, "4096 characters"):
            self.pose_studio._validate_pose_data({
                "poses": [{}],
                "lighting_prompts": ["x" * 4097],
            })

    def test_rejects_invalid_capture_version(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            self.pose_studio._validate_pose_data({
                "poses": [{}],
                "capture_version": -1,
            })

    def test_rejects_animation_provenance_that_does_not_match_poses(self):
        with self.assertRaisesRegex(ValueError, "must match the poses list"):
            self.pose_studio._validate_pose_data({
                "poses": [{}, {}],
                "animation": {
                    "sampled_frames": 81,
                    "sample_times_seconds": [0.0, 1.0],
                },
            })

        with self.assertRaisesRegex(ValueError, "non-negative and ordered"):
            self.pose_studio._validate_pose_data({
                "poses": [{}, {}],
                "animation": {
                    "sampled_frames": 2,
                    "sample_times_seconds": [1.0, 0.0],
                },
            })

    def test_validates_realtime_animation_settings(self):
        self.pose_studio._validate_pose_data({
            "poses": [{}],
            "export": {
                "animation_frames": 81,
                "animation_fps": 16,
                "animation_start_seconds": 0,
                "animation_timing": "REALTIME",
            },
        })

        with self.assertRaisesRegex(ValueError, "REALTIME or FIT_CLIP"):
            self.pose_studio._validate_pose_data({
                "poses": [{}],
                "export": {"animation_timing": "STRETCH"},
            })

    def test_prompt_from_list_selects_one_scalar_prompt(self):
        node = self.pose_studio.JakkannaPromptFromList()
        self.assertEqual(node.select(["appearance", "motion"], [1]), ("motion",))

        with self.assertRaisesRegex(ValueError, "outside"):
            node.select(["appearance"], [1])

    def test_complete_payload_does_not_request_frontend_sync(self):
        class NoSyncPoseStudio(self.pose_studio.JakkannaPoseStudio):
            def _discard_frontend_sync(self, unique_id):
                raise AssertionError("complete captures must not request frontend sync")

        data = {
            "poses": [{}],
            "captured_images": [image_data_url()],
        }

        images, _prompts = NoSyncPoseStudio().generate(json.dumps(data), unique_id="42")

        self.assertEqual(len(images), 1)

    def test_pose_image_failure_is_not_silently_ignored(self):
        class FailedPoseImage(self.pose_studio.JakkannaPoseStudio):
            def _apply_pose_image_via_frontend(self, pose_image, unique_id):
                raise RuntimeError("synchronization failed")

        data = {
            "poses": [{}],
            "captured_images": [image_data_url()],
        }

        with self.assertRaisesRegex(RuntimeError, "synchronization failed"):
            FailedPoseImage().generate(json.dumps(data), pose_image=object(), unique_id="42")

    def test_pose_image_hash_covers_every_element(self):
        first = self.pose_studio.torch.zeros((1, 1, 2000, 1), dtype=self.pose_studio.torch.float32)
        second = first.clone()
        second.reshape(-1)[1] = 1.0

        first_hash = self.pose_studio.JakkannaPoseStudio.IS_CHANGED("{}", first)
        second_hash = self.pose_studio.JakkannaPoseStudio.IS_CHANGED("{}", second)

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
        worker_path = os.path.join(ROOT, "web", "jakkanna_pose_morph_worker.js")
        with open(worker_path, "r", encoding="utf-8") as handle:
            worker = handle.read()

        for factor in ("idealproportions", "uncommonproportions", "african", "asian", "caucasian"):
            self.assertIn(f"{factor}:", worker)


class OpenPoseExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.openpose = load_openpose_module()

    def test_pose_image_uses_one_effective_state_for_images_and_keypoints(self):
        original = {
            "poses": [{}],
            "captured_images": [image_data_url((255, 0, 0))],
            "openpose_keypoints": [openpose_frame(1.0)],
        }
        synchronized = {
            "poses": [{}],
            "captured_images": [image_data_url((0, 255, 0))],
            "openpose_keypoints": [openpose_frame(2.0)],
        }

        class SynchronizedOpenPose(self.openpose.JakkannaPoseStudioOpenPose):
            def _apply_pose_image_via_frontend(self, pose_image, unique_id):
                return synchronized

        images, _prompts, keypoints = SynchronizedOpenPose().generate_with_openpose(
            json.dumps(original),
            pose_image=object(),
            unique_id="42",
        )

        np.testing.assert_allclose(images[0][0, 0, 0].numpy(), [0.0, 1.0, 0.0])
        self.assertEqual(keypoints[0]["people"][0]["pose_keypoints_2d"][0], 2.0)

    def test_complete_openpose_payload_does_not_request_frontend_sync(self):
        class NoSyncOpenPose(self.openpose.JakkannaPoseStudioOpenPose):
            def _discard_frontend_sync(self, unique_id):
                raise AssertionError("complete captures must not request frontend sync")

        data = {
            "poses": [{}],
            "captured_images": [image_data_url()],
            "openpose_keypoints": [openpose_frame()],
        }

        images, _prompts, keypoints = NoSyncOpenPose().generate_with_openpose(
            json.dumps(data),
            unique_id="42",
        )

        self.assertEqual(len(images), 1)
        self.assertEqual(len(keypoints), 1)

    def test_fallback_openpose_uses_captured_image_dimensions(self):
        data = {
            "poses": [{}],
            "captured_images": [image_data_url(size=(8, 6))],
        }

        images, _prompts, keypoints = self.openpose.JakkannaPoseStudioOpenPose().generate_with_openpose(
            json.dumps(data)
        )

        self.assertEqual(tuple(images[0].shape), (1, 6, 8, 3))
        self.assertEqual((keypoints[0]["canvas_width"], keypoints[0]["canvas_height"]), (8, 6))

    def test_malformed_openpose_keypoints_are_rejected(self):
        data = {
            "poses": [{}],
            "captured_images": [image_data_url()],
            "openpose_keypoints": ["garbage"],
        }

        with self.assertRaisesRegex(ValueError, "must be an object"):
            self.openpose.JakkannaPoseStudioOpenPose().generate_with_openpose(json.dumps(data))

    def test_openpose_frame_count_must_match_pose_count(self):
        data = {
            "poses": [{}, {}],
            "captured_images": [image_data_url(), image_data_url()],
            "openpose_keypoints": [openpose_frame()],
        }

        with self.assertRaisesRegex(ValueError, "exactly 2 frames"):
            self.openpose.JakkannaPoseStudioOpenPose().generate_with_openpose(json.dumps(data))

    def test_openpose_canvas_must_match_captured_image(self):
        data = {
            "poses": [{}],
            "captured_images": [image_data_url()],
            "openpose_keypoints": [openpose_frame(width=4, height=4)],
        }

        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            self.openpose.JakkannaPoseStudioOpenPose().generate_with_openpose(json.dumps(data))

    def test_grid_openpose_combines_pose_frames_into_grid_coordinates(self):
        data = {
            "poses": [{}, {}],
            "export": {"output_mode": "GRID", "grid_columns": 2},
            "captured_images": [image_data_url(), image_data_url()],
            "openpose_keypoints": [openpose_frame(), openpose_frame()],
        }
        data["openpose_keypoints"][0]["people"][0]["pose_keypoints_2d"][:3] = [1.0, 1.0, 1.0]
        data["openpose_keypoints"][1]["people"][0]["pose_keypoints_2d"][:3] = [1.0, 1.0, 1.0]

        images, prompts, keypoints = self.openpose.JakkannaPoseStudioOpenPose().generate_with_openpose(
            json.dumps(data)
        )

        self.assertEqual((len(images), len(prompts), len(keypoints)), (1, 1, 1))
        self.assertEqual((keypoints[0]["canvas_width"], keypoints[0]["canvas_height"]), (4, 2))
        self.assertEqual(len(keypoints[0]["people"]), 2)
        self.assertEqual(keypoints[0]["people"][0]["pose_keypoints_2d"][:3], [1.0, 1.0, 1.0])
        self.assertEqual(keypoints[0]["people"][1]["pose_keypoints_2d"][:3], [3.0, 1.0, 1.0])


class FrontendContractTests(unittest.TestCase):
    def test_automatic_repository_refresh_is_absent(self):
        with open(os.path.join(ROOT, "web", "jakkanna_pose_studio.js"), "r", encoding="utf-8") as handle:
            studio = handle.read()
        with open(os.path.join(ROOT, "api", "pose_library.py"), "r", encoding="utf-8") as handle:
            library = handle.read()

        self.assertNotIn("autoRefreshEnabledPoseRepositories", studio)
        self.assertNotIn("/vnccs/pose_library/repositories/auto_refresh", studio)
        self.assertNotIn("auto_refresh_enabled_pose_repositories", library)
        self.assertNotIn("cdn.buymeacoffee.com", studio)

    def test_frontend_limits_match_python_limits(self):
        with open(os.path.join(ROOT, "web", "jakkanna_pose_studio.js"), "r", encoding="utf-8") as handle:
            studio = handle.read()

        self.assertIn("const JAKKANNA_POSE_MAX_COUNT = 128;", studio)
        self.assertIn("const JAKKANNA_LIGHT_MAX_COUNT = 32;", studio)
        self.assertIn("const JAKKANNA_LIGHTING_PROMPT_MAX_LENGTH = 4096;", studio)
        self.assertIn("const JAKKANNA_CAPTURE_MAX_TOTAL_PIXELS = 128 * 1024 * 1024;", studio)
        self.assertIn("promptArea.maxLength = JAKKANNA_LIGHTING_PROMPT_MAX_LENGTH;", studio)
        self.assertIn("capture_version: this._captureVersion", studio)
        self.assertIn("if (!node.studioWidget.syncToNode(true)) return;", studio)

    def test_mixamo_sampler_is_hard_capped(self):
        with open(os.path.join(ROOT, "web", "jakkanna_mixamo_import.js"), "r", encoding="utf-8") as handle:
            mixamo = handle.read()

        self.assertIn("const frameCount = Math.min(frameLimit", mixamo)
        self.assertIn("{ length: frameCount }", mixamo)
        self.assertNotIn("times.push(safeDuration)", mixamo)
        self.assertIn("options.frameCount", mixamo)
        self.assertIn("exactFrames !== null && exactFrames !== undefined", mixamo)
        self.assertIn('timingMode === "REALTIME"', mixamo)
        self.assertIn("safeStart + index / safeFps", mixamo)
        self.assertIn('file_sha256: await crypto.subtle.digest("SHA-256"', mixamo)
        self.assertIn("action.setLoop(THREE.LoopOnce, 0)", mixamo)
        self.assertIn("action.clampWhenFinished = true", mixamo)
        self.assertIn("Could not retarget the sampled frame", mixamo)
        self.assertNotIn("if (!applied) continue", mixamo)
        self.assertIn("The FBX skeleton is not compatible with the Mixamo bone layout.", mixamo)
        self.assertIn("sourceBones.normalizedBones[`mixamorig${name}`]", mixamo)

    def test_head_retarget_options_stay_in_their_own_method(self):
        with open(os.path.join(ROOT, "web", "jakkanna_pose_studio_core.js"), "r", encoding="utf-8") as handle:
            core = handle.read()

        import_start = core.index("_applyImportPelvisAndTorso(")
        import_end = core.index("_buildWorldKeypointsFromSAM3D(", import_start)
        import_method = core[import_start:import_end]
        self.assertIn("const includeHead = options.includeHead !== false;", import_method)
        self.assertIn("if (includeHead)", import_method)

        hmr_start = core.index("fitMannequinToHMR2(")
        hmr_end = core.index("setMannequinVisible(", hmr_start)
        self.assertNotIn("includeHead", core[hmr_start:hmr_end])

    def test_pose_studio_tracks_imported_animation_contract(self):
        with open(os.path.join(ROOT, "web", "jakkanna_pose_studio.js"), "r", encoding="utf-8") as handle:
            studio = handle.read()

        self.assertIn("animation_frames: 81", studio)
        self.assertIn("animation_fps: 16", studio)
        self.assertIn('animation_timing: "REALTIME"', studio)
        self.assertIn("frameCount: this.exportParams.animation_frames", studio)
        self.assertIn("fps: this.exportParams.animation_fps", studio)
        self.assertIn("animation: this.animationMetadata", studio)
        self.assertIn("this.reconcileAnimationMetadata();", studio)
        self.assertIn("this.invalidateAnimationMetadata();", studio)
        self.assertIn("if (!this.animationMetadata)", studio)

    def test_frontend_openpose_uses_face_landmarks(self):
        with open(os.path.join(ROOT, "web", "jakkanna_pose_studio_core.js"), "r", encoding="utf-8") as handle:
            core = handle.read()

        self.assertIn("project(face?.nose || headCenter)", core)
        self.assertIn("project(face?.right)", core)
        self.assertIn("project(face?.left)", core)


if __name__ == "__main__":
    unittest.main()
