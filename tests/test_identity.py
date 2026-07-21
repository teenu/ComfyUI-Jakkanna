import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEGACY_NODE_IDS = {
    "VNCCS_PositionControl",
    "VNCCS_VisualPositionControl",
    "VNCCS_QWEN_Detailer",
    "VNCCS_BBox_Extractor",
    "VNCCS_ModelManager",
    "VNCCS_ModelSelector",
    "VNCCS_PoseStudio",
    "VNCCS_PoseStudioOpenPose",
    "VNCCS_ReplaceOpenPoseHands",
    "VNCCS_UniCanvas",
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

    def test_legacy_node_ids_map_to_jakkanna_classes(self):
        with open(os.path.join(ROOT, "__init__.py"), "r", encoding="utf-8") as handle:
            module = ast.parse(handle.read())

        mappings = next(
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS" for target in node.targets)
        )
        actual_ids = {key.value for key in mappings.keys}
        class_names = {value.id for value in mappings.values}

        self.assertEqual(actual_ids, LEGACY_NODE_IDS)
        self.assertTrue(all(name.startswith("Jakkanna") for name in class_names))


if __name__ == "__main__":
    unittest.main()
