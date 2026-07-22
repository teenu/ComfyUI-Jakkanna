import ast
import json
import os
import tomllib
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTERED_NODE_MAPPINGS = {
    "VNCCS_PositionControl": "JakkannaPositionControl",
    "VNCCS_VisualPositionControl": "JakkannaVisualPositionControl",
    "VNCCS_QWEN_Detailer": "JakkannaQwenDetailer",
    "VNCCS_BBox_Extractor": "JakkannaBBoxExtractor",
    "VNCCS_ModelManager": "JakkannaModelManager",
    "VNCCS_ModelSelector": "JakkannaModelSelector",
    "VNCCS_PoseStudio": "JakkannaPoseStudio",
    "JakkannaPromptFromList": "JakkannaPromptFromList",
    "JakkannaSCAIL2FlowUniPC": "JakkannaSCAIL2FlowUniPC",
    "JakkannaSCAIL2ProductionInputValidate": "JakkannaSCAIL2ProductionInputValidate",
    "JakkannaSCAIL2ProductionManifest": "JakkannaSCAIL2ProductionManifest",
    "JakkannaSCAIL2ReferencePNG16Save": "JakkannaSCAIL2ReferencePNG16Save",
    "JakkannaSCAIL2SingleSubjectMask": "JakkannaSCAIL2SingleSubjectMask",
    "JakkannaSCAIL2TrackValidate": "JakkannaSCAIL2TrackValidate",
    "JakkannaSCAIL2UpstreamNoise": "JakkannaSCAIL2UpstreamNoise",
    "VNCCS_PoseStudioOpenPose": "JakkannaPoseStudioOpenPose",
    "VNCCSReplaceOpenPoseHands": "JakkannaReplaceOpenPoseHandsLegacy",
    "VNCCS_ReplaceOpenPoseHands": "JakkannaReplaceOpenPoseHands",
    "VNCCS_UniCanvas": "JakkannaCanvas",
}


class IdentityBoundaryTests(unittest.TestCase):
    def test_implementation_paths_use_jakkanna_names(self):
        expected_paths = [
            "jakkanna_sam3d",
            "nodes/camera_control.py",
            "nodes/model_manager.py",
            "nodes/openpose_export.py",
            "nodes/qwen_detailer.py",
            "web/jakkanna_camera_control.js",
            "web/jakkanna_model_manager.js",
            "web/jakkanna_pose_studio.js",
            "web/jakkanna_unicanvas.js",
        ]
        legacy_paths = [
            "vnccs_sam3d",
            "nodes/vnccs_nodes.py",
            "nodes/vnccs_model_manager.py",
            "nodes/vnccs_openpose_export.py",
            "nodes/vnccs_qwen_detailer.py",
            "web/vnccs_camera_control.js",
            "web/vnccs_model_manager.js",
            "web/vnccs_pose_studio.js",
            "web/vnccs_unicanvas.js",
            "images/VNCCS_Discord_Button.png",
            "images/VNCCS_Donate_Button.png",
            "images/VNCCS_GITHUB_CARD.png",
            "web/assets/VNCCS_Donate_Button.png",
        ]

        for path in expected_paths:
            self.assertTrue(os.path.exists(os.path.join(ROOT, path)), path)
        for path in legacy_paths:
            self.assertFalse(os.path.exists(os.path.join(ROOT, path)), path)

    def test_frontend_uses_jakkanna_registration_and_styles(self):
        for filename in os.listdir(os.path.join(ROOT, "web")):
            if not filename.endswith(".js"):
                continue
            with open(os.path.join(ROOT, "web", filename), "r", encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn('name: "VNCCS.', source, filename)
            self.assertNotIn('name: "vnccs.', source, filename)
            self.assertNotIn("VNCCS Settings", source, filename)
            self.assertNotIn('class="vnccs-', source, filename)
            self.assertNotIn("className = 'vnccs-", source, filename)
            self.assertNotIn('className = "vnccs-', source, filename)
            self.assertNotIn(".vnccs-", source, filename)

    def test_frontend_has_no_runtime_esm_sh_imports(self):
        for directory, _subdirectories, filenames in os.walk(os.path.join(ROOT, "web")):
            for filename in filenames:
                if not filename.endswith((".js", ".mjs")):
                    continue
                path = os.path.join(directory, filename)
                with open(path, "r", encoding="utf-8") as handle:
                    self.assertNotIn("https://esm.sh", handle.read(), os.path.relpath(path, ROOT))

    def test_registered_node_ids_map_to_jakkanna_classes(self):
        with open(os.path.join(ROOT, "__init__.py"), "r", encoding="utf-8") as handle:
            module = ast.parse(handle.read())

        mappings = next(
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS" for target in node.targets)
        )
        actual_mappings = {key.value: value.id for key, value in zip(mappings.keys, mappings.values)}

        self.assertEqual(actual_mappings, REGISTERED_NODE_MAPPINGS)

    def test_old_hands_identifier_is_a_deprecated_compatibility_wrapper(self):
        with open(os.path.join(ROOT, "__init__.py"), "r", encoding="utf-8") as handle:
            module = ast.parse(handle.read())

        wrapper = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "JakkannaReplaceOpenPoseHandsLegacy"
        )
        self.assertEqual([base.id for base in wrapper.bases], ["JakkannaReplaceOpenPoseHands"])
        deprecated = next(
            node.value
            for node in wrapper.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DEPRECATED" for target in node.targets)
        )
        self.assertIs(deprecated.value, True)

    def test_registry_ids_match_registered_nodes(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            metadata = tomllib.load(handle)

        self.assertEqual(set(metadata["tool"]["comfy"]["NodeIds"]), set(REGISTERED_NODE_MAPPINGS))

    def test_example_workflows_use_current_jakkanna_version(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            version = tomllib.load(handle)["project"]["version"]

        found = 0

        def check(value, filename):
            nonlocal found
            if isinstance(value, dict):
                if value.get("cnr_id") == "jakkanna":
                    found += 1
                    self.assertEqual(value.get("ver"), version, filename)
                for child in value.values():
                    check(child, filename)
            elif isinstance(value, list):
                for child in value:
                    check(child, filename)

        workflow_dir = os.path.join(ROOT, "workflows")
        for filename in os.listdir(workflow_dir):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(workflow_dir, filename), "r", encoding="utf-8") as handle:
                check(json.load(handle), filename)

        self.assertGreater(found, 0)

    def test_scail2_animation_workflows_are_self_contained(self):
        filenames = (
            "Jakkanna Krea 2 SCAIL 2 Animate Fast.json",
            "Jakkanna Krea 2 SCAIL 2 Animate Production.json",
        )
        legacy_types = {
            node_id.removeprefix("Jakkanna")
            for node_id in REGISTERED_NODE_MAPPINGS
            if node_id.startswith("JakkannaSCAIL2")
        }

        for filename in filenames:
            with open(os.path.join(ROOT, "workflows", filename), "r", encoding="utf-8") as handle:
                workflow = json.load(handle)

            node_types = {node["type"] for node in workflow["nodes"]}
            self.assertFalse(node_types.intersection(legacy_types), filename)
            self.assertTrue(any(node_type.startswith("JakkannaSCAIL2") for node_type in node_types), filename)

            pose_node = next(node for node in workflow["nodes"] if node["type"] == "VNCCS_PoseStudio")
            pose_data = json.loads(pose_node["widgets_values"][0])
            self.assertEqual(pose_data["export"]["animation_frames"], 81, filename)
            self.assertEqual(pose_data["export"]["animation_fps"], 16, filename)
            self.assertEqual(pose_data["export"]["animation_timing"], "FIT_CLIP", filename)

            video_node = next(node for node in workflow["nodes"] if node["type"] == "VHS_VideoCombine")
            self.assertEqual(video_node["properties"].get("cnr_id"), "comfyui-videohelpersuite", filename)
            self.assertGreaterEqual(video_node["properties"].get("ver", ""), "1.7.9", filename)

            serialized = json.dumps(workflow)
            self.assertNotIn("experiments/scail_jakkanna", serialized, filename)
            self.assertNotIn("/home/", serialized, filename)

    def test_scail2_release_dependencies_are_declared(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            dependencies = tomllib.load(handle)["project"]["dependencies"]

        self.assertIn("av>=16.0.0", dependencies)
        self.assertIn("diffusers>=0.31.0,<0.40", dependencies)
        self.assertIn("trimesh", dependencies)

        with open(os.path.join(ROOT, "requirements.txt"), "r", encoding="utf-8") as handle:
            requirements = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        self.assertEqual(dependencies, requirements)


if __name__ == "__main__":
    unittest.main()
