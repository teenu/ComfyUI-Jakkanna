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


if __name__ == "__main__":
    unittest.main()
