"""Jakkanna Pose Studio - combined mesh editor and multi-pose generator.

Combines Character Studio mesh sliders with dynamic pose tabs.
Each pose stores bone rotations and global model rotation.
Outputs rendered mesh images with skin material.

This node is fully self-contained with all data loading logic.
"""

import json
import os
import base64
import binascii
import math
import threading
from io import BytesIO
import hashlib
import torch
import numpy as np
from PIL import Image

# Import from CharacterData module
from ..CharacterData.mh_parser import TargetParser
from ..CharacterData.obj_loader import load_obj
from ..CharacterData.mh_skeleton import Skeleton

_CACHE_LOCK = threading.Lock()
_CAPTURED_IMAGE_MAX_COUNT = 128
_CAPTURED_IMAGE_MAX_TOTAL_CHARS = 64 * 1024 * 1024
_CAPTURED_IMAGE_MAX_BYTES = 32 * 1024 * 1024
_CAPTURED_IMAGE_MAX_PIXELS = 4096 * 4096
_CAPTURED_IMAGE_MAX_TOTAL_PIXELS = 128 * 1024 * 1024


# === Data Cache and Loader (from Character Studio) ===

# Singleton storage for loaded MH data to avoid reloading every time
POSE_STUDIO_CACHE = {
    "base_mesh": None,
    "targets": None,
    "parser": None,
    "skeleton": None
}


def _finite_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return value


def _vector3(value, path):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{path} must contain three numbers")
    for index, component in enumerate(value):
        _finite_number(component, f"{path}[{index}]")


def _validate_pose_data(data):
    if not isinstance(data, dict):
        raise ValueError("pose_data must be a JSON object")

    mesh = data.get("mesh", {})
    if not isinstance(mesh, dict):
        raise ValueError("mesh must be an object")
    mesh_numbers = {
        "age", "gender", "weight", "muscle", "height", "breast_size", "firmness",
        "penis_len", "penis_circ", "penis_test", "genital_size", "proportions",
        "african", "asian", "caucasian", "head_size", "arm_size", "hand_size",
        "foot_size", "upper_arm_l_length", "upper_arm_r_length", "forearm_l_length",
        "forearm_r_length", "thigh_l_length", "thigh_r_length", "shin_l_length",
        "shin_r_length", "spine_length",
    }
    for name in mesh_numbers.intersection(mesh):
        if name == "caucasian" and mesh[name] is None:
            continue
        _finite_number(mesh[name], f"mesh.{name}")

    export = data.get("export", {})
    if not isinstance(export, dict):
        raise ValueError("export must be an object")
    for name in (
        "view_width", "view_height", "view_size", "cam_zoom", "cam_offset_x",
        "cam_offset_y", "cam_yaw_deg", "cam_pitch_deg", "grid_columns",
    ):
        if name in export:
            _finite_number(export[name], f"export.{name}")
    for name in ("view_width", "view_height", "view_size"):
        if name in export and (export[name] != int(export[name]) or not 1 <= export[name] <= 4096):
            raise ValueError(f"export.{name} must be an integer between 1 and 4096")
    if "grid_columns" in export:
        if export["grid_columns"] != int(export["grid_columns"]) or not 1 <= export["grid_columns"] <= _CAPTURED_IMAGE_MAX_COUNT:
            raise ValueError(f"export.grid_columns must be an integer between 1 and {_CAPTURED_IMAGE_MAX_COUNT}")
    output_mode = export.get("output_mode", "LIST")
    if output_mode not in ("LIST", "GRID"):
        raise ValueError("export.output_mode must be LIST or GRID")
    bg_color = export.get("bg_color", [40, 40, 40])
    _vector3(bg_color, "export.bg_color")
    if any(component != int(component) or component < 0 or component > 255 for component in bg_color):
        raise ValueError("export.bg_color values must be integers between 0 and 255")

    poses = data.get("poses", [{}])
    if not isinstance(poses, list):
        raise ValueError("poses must be a list")
    if not poses:
        raise ValueError("poses must contain at least one pose")
    if len(poses) > _CAPTURED_IMAGE_MAX_COUNT:
        raise ValueError(f"poses limit is {_CAPTURED_IMAGE_MAX_COUNT}")
    for pose_index, pose in enumerate(poses):
        path = f"poses[{pose_index}]"
        if not isinstance(pose, dict):
            raise ValueError(f"{path} must be an object")
        bones = pose.get("bones", {})
        if not isinstance(bones, dict):
            raise ValueError(f"{path}.bones must be an object")
        if len(bones) > 256:
            raise ValueError(f"{path}.bones contains too many entries")
        for bone_name, rotation in bones.items():
            if not isinstance(bone_name, str) or not bone_name or len(bone_name) > 128:
                raise ValueError(f"{path}.bones contains an invalid bone name")
            _vector3(rotation, f"{path}.bones.{bone_name}")
        if "modelRotation" in pose:
            _vector3(pose["modelRotation"], f"{path}.modelRotation")
        for map_name in ("ikEffectorPositions", "poleTargetPositions", "hipBonePosition"):
            positions = pose.get(map_name, {})
            if not isinstance(positions, dict):
                raise ValueError(f"{path}.{map_name} must be an object")
            if len(positions) > 256:
                raise ValueError(f"{path}.{map_name} contains too many entries")
            for position_name, position in positions.items():
                if not isinstance(position_name, str) or len(position_name) > 128:
                    raise ValueError(f"{path}.{map_name} contains an invalid name")
                _vector3(position, f"{path}.{map_name}.{position_name}")
        for camera_name in ("camera", "cameraParams"):
            camera = pose.get(camera_name)
            if camera is None:
                continue
            if not isinstance(camera, dict):
                raise ValueError(f"{path}.{camera_name} must be an object")
            for name, value in camera.items():
                _finite_number(value, f"{path}.{camera_name}.{name}")
        if pose.get("prompt") is not None and not isinstance(pose["prompt"], str):
            raise ValueError(f"{path}.prompt must be a string")

    lights = data.get("lights", [])
    if not isinstance(lights, list):
        raise ValueError("lights must be a list")
    if len(lights) > 32:
        raise ValueError("lights contains too many entries")
    for light_index, light in enumerate(lights):
        if not isinstance(light, dict):
            raise ValueError(f"lights[{light_index}] must be an object")
        for name in ("intensity", "x", "y", "z", "angle", "penumbra", "distance", "decay"):
            if name in light:
                _finite_number(light[name], f"lights[{light_index}].{name}")

    captured_images = data.get("captured_images", [])
    if not isinstance(captured_images, list):
        raise ValueError("captured_images must be a list")
    if len(captured_images) > _CAPTURED_IMAGE_MAX_COUNT:
        raise ValueError(f"captured_images limit is {_CAPTURED_IMAGE_MAX_COUNT}")
    for image in captured_images:
        if image is not None and not isinstance(image, str):
            raise ValueError("captured_images entries must be strings or null")
    if captured_images and len(captured_images) != len(poses):
        raise ValueError("captured_images must have one slot per pose")

    lighting_prompts = data.get("lighting_prompts", [])
    if not isinstance(lighting_prompts, list):
        raise ValueError("lighting_prompts must be a list")
    if len(lighting_prompts) > _CAPTURED_IMAGE_MAX_COUNT:
        raise ValueError(f"lighting_prompts limit is {_CAPTURED_IMAGE_MAX_COUNT}")
    if any(not isinstance(prompt, str) for prompt in lighting_prompts):
        raise ValueError("lighting_prompts entries must be strings")
    if any(len(prompt) > 4096 for prompt in lighting_prompts):
        raise ValueError("lighting_prompts entries must not exceed 4096 characters")
    if len(lighting_prompts) > len(poses):
        raise ValueError("lighting_prompts cannot contain more entries than poses")

    capture_id = data.get("capture_id")
    if capture_id is not None and (not isinstance(capture_id, str) or len(capture_id) > 256):
        raise ValueError("capture_id must be a string of at most 256 characters")

    capture_version = data.get("capture_version")
    if capture_version is not None:
        if isinstance(capture_version, bool) or not isinstance(capture_version, int) or capture_version < 0:
            raise ValueError("capture_version must be a non-negative integer")

    active_tab = data.get("activeTab")
    if active_tab is not None:
        if isinstance(active_tab, bool) or not isinstance(active_tab, int):
            raise ValueError("activeTab must be an integer")
        if not 0 <= active_tab < len(poses):
            raise ValueError("activeTab is outside the poses list")

    return data


def _captures_complete(data):
    poses = data.get("poses", [{}])
    captures = data.get("captured_images", [])
    return len(captures) >= len(poses) and all(
        isinstance(captures[index], str) and bool(captures[index])
        for index in range(len(poses))
    )


def _decode_captured_images(captured_images):
    if not isinstance(captured_images, list):
        raise ValueError("captured_images must be a list")
    if len(captured_images) > _CAPTURED_IMAGE_MAX_COUNT:
        raise ValueError(f"captured_images limit is {_CAPTURED_IMAGE_MAX_COUNT}")

    rendered_images = []
    total_chars = 0
    total_pixels = 0
    for b64 in captured_images:
        if b64 is None or b64 == "":
            continue
        if not isinstance(b64, str):
            raise ValueError("captured_images entries must be strings or null")
        total_chars += len(b64)
        if total_chars > _CAPTURED_IMAGE_MAX_TOTAL_CHARS:
            raise ValueError("captured_images payload is too large")
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        try:
            img_data = base64.b64decode(b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("captured image is not valid base64") from exc
        if len(img_data) > _CAPTURED_IMAGE_MAX_BYTES:
            raise ValueError("captured image is too large")
        try:
            with Image.open(BytesIO(img_data)) as img:
                if img.width * img.height > _CAPTURED_IMAGE_MAX_PIXELS:
                    raise ValueError("captured image dimensions are too large")
                total_pixels += img.width * img.height
                if total_pixels > _CAPTURED_IMAGE_MAX_TOTAL_PIXELS:
                    raise ValueError("captured images contain too many total pixels")
                rendered_images.append(img.convert('RGB'))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("captured image data is invalid") from exc
    return rendered_images


def _get_character_data_path():
    """Get the path to CharacterData folder."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CharacterData"))


def _ensure_data_loaded():
    """Load MakeHuman data if not already loaded."""
    if POSE_STUDIO_CACHE['base_mesh'] is not None:
        return  # fast path without lock
    with _CACHE_LOCK:
        if POSE_STUDIO_CACHE['base_mesh'] is not None:
            return  # double-check after acquiring lock

        char_data_path = _get_character_data_path()
        mh_path = os.path.join(char_data_path, "makehuman")

        if not os.path.exists(mh_path):
            raise Exception(f"MakeHuman data not found at: {mh_path}")

        print(f"[Jakkanna Pose Studio] Loading MakeHuman data from {mh_path}...")

        # 1. Load Base Mesh
        base_obj_paths = [
            os.path.join(mh_path, "makehuman", "data", "3dobjs", "base.obj"),
            os.path.join(mh_path, "data", "3dobjs", "base.obj"),
        ]

        base_path = next((p for p in base_obj_paths if os.path.exists(p)), None)
        if not base_path:
            raise Exception("Could not find base.obj inside makehuman data.")

        base_mesh = load_obj(base_path)

        # 2. Load Targets
        parser = TargetParser(mh_path)
        targets = parser.scan_targets()

        print(f"[Jakkanna Pose Studio] Loaded {len(targets)} targets.")

        # 3. Load Skeleton (Preference: game_engine > default)
        skeleton = None
        skel_path = os.path.join(mh_path, "makehuman", "data", "rigs", "game_engine.mhskel")
        if not os.path.exists(skel_path):
            skel_path = os.path.join(mh_path, "makehuman", "data", "rigs", "default.mhskel")

        if os.path.exists(skel_path):
            print(f"[Jakkanna Pose Studio] Loading skeleton from {skel_path}...")
            skeleton = Skeleton()
            skeleton.fromFile(skel_path, base_mesh)
        else:
            print(f"[Jakkanna Pose Studio] Warning: Default skeleton not found at {skel_path}")

        POSE_STUDIO_CACHE.update({
            "base_mesh": base_mesh,
            "targets": targets,
            "parser": parser,
            "skeleton": skeleton,
        })


# === Main Node Class ===

class JakkannaPoseStudio:
    """Pose Studio with mesh editing and multiple pose generation."""
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "lighting_prompt")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate"
    CATEGORY = "Jakkanna/pose"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ALL settings come from widget via pose_data
                "pose_data": ("STRING", {"multiline": True, "default": "{}"}),
            },
            "optional": {
                "pose_image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    @classmethod
    def IS_CHANGED(cls, pose_data: str = "{}", pose_image=None, unique_id: str = None):
        # Force re-execution if Debug Mode is enabled
        try:
            data = json.loads(pose_data)
            export = data.get("export", {})
            if export.get("debugMode", False):
                return float("NaN")
        except Exception:
            pass
        if pose_image is None:
            return pose_data
        try:
            tensor = pose_image.detach().cpu().contiguous()
            arr = tensor.numpy()
            hasher = hashlib.sha256()
            hasher.update(f"{arr.dtype}:{arr.shape}:".encode("ascii"))
            hasher.update(arr.tobytes())
            return f"{pose_data}|pose_image:{hasher.hexdigest()}"
        except Exception:
            return f"{pose_data}|pose_image:{id(pose_image)}"

    def _discard_frontend_sync(self, unique_id):
        import folder_paths

        filepath = os.path.join(folder_paths.get_temp_directory(), f"jakkanna_debug_{unique_id}.json")
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass

    def _wait_for_frontend_sync(self, unique_id, start_time, timeout=15.0):
        import time
        import folder_paths

        temp_dir = folder_paths.get_temp_directory()
        filepath = os.path.join(temp_dir, f"jakkanna_debug_{unique_id}.json")

        while time.time() - start_time < timeout:
            if os.path.exists(filepath) and os.path.getmtime(filepath) >= start_time:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        sync_data = json.load(f)
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    return sync_data
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(0.1)
        return None

    def _apply_pose_image_via_frontend(self, pose_image, unique_id):
        if not unique_id:
            raise RuntimeError("pose_image requires an active Pose Studio frontend node")
        try:
            from server import PromptServer
            import time
            from ..jakkanna_sam3d import process_image_to_pose_json, progress

            task_id = f"node-{unique_id}-pose-image"
            progress.start_task(task_id)
            with progress.task_context(task_id):
                progress.update("Step 1/6: Pose image input received. Preparing SAM 3D Body import...", 2)
                pose_json = process_image_to_pose_json(pose_image[:1])
            try:
                pose_payload = json.loads(pose_json)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError("pose_image SAM import returned invalid pose data") from exc
            if not isinstance(pose_payload, dict) or not pose_payload:
                raise RuntimeError("pose_image SAM import returned empty pose data")

            self._discard_frontend_sync(unique_id)
            start_time = time.time()
            PromptServer.instance.send_sync("vnccs_apply_sam3d_pose", {
                "node_id": unique_id,
                "pose_data": pose_payload,
            })
            synced = self._wait_for_frontend_sync(unique_id, start_time, timeout=30.0)
            if synced:
                print("[Jakkanna Pose Studio] Applied pose_image SAM pose through frontend sync.")
                return synced
            raise RuntimeError(
                "pose_image was analyzed, but the Pose Studio frontend did not return synchronized captures"
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"pose_image SAM import failed: {exc}") from exc

    def _resolve_execution_data(self, pose_data, pose_image, unique_id):
        try:
            data = json.loads(pose_data) if pose_data else {}
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("pose_data must contain valid JSON") from exc
        _validate_pose_data(data)

        if pose_image is not None:
            if data.get("export", {}).get("interface_mode") == "manager":
                raise RuntimeError("pose_image is unavailable in Pose Studio manager mode")
            data = self._apply_pose_image_via_frontend(pose_image, unique_id)
            if not isinstance(data, dict):
                raise RuntimeError("pose_image did not return synchronized Pose Studio data")
            _validate_pose_data(data)
        elif unique_id and not _captures_complete(data):
            synced = None
            try:
                from server import PromptServer
                import time

                self._discard_frontend_sync(unique_id)
                start_time = time.time()
                PromptServer.instance.send_sync("vnccs_req_pose_sync", {"node_id": unique_id})
                synced = self._wait_for_frontend_sync(unique_id, start_time, timeout=30.0)
            except Exception as exc:
                print(f"[Jakkanna Pose Studio] Frontend sync failed: {exc}")
            if synced is not None:
                _validate_pose_data(synced)
                data = synced

        if not _captures_complete(data):
            capture_id = data.get("capture_id")
            if capture_id:
                try:
                    from .. import _jakkanna_get_capture_cache

                    cached = _jakkanna_get_capture_cache(capture_id, data.get("capture_version", 0))
                    if cached:
                        data["captured_images"] = cached.get("captured_images", [])
                        data["lighting_prompts"] = cached.get("lighting_prompts", [])
                        print(f"[Jakkanna Pose Studio] Loaded {len(data['captured_images'])} captures from LRU cache (id={capture_id})")
                except (ImportError, AttributeError) as exc:
                    print(f"[Jakkanna Pose Studio] Capture cache unavailable: {exc}")

        _validate_pose_data(data)
        if not _captures_complete(data):
            poses = data.get("poses", [{}])
            captures = data.get("captured_images", [])
            captured_count = sum(
                isinstance(capture, str) and bool(capture)
                for capture in captures[:len(poses)]
            )
            raise RuntimeError(
                f"Pose Studio received {captured_count} of {len(poses)} required frontend captures. "
                "Open the node, wait for the 3D viewer to load, and run the workflow again."
            )
        return data

    def _generate_from_execution_data(self, data):
        export = data.get("export", {})
        output_mode = export.get("output_mode", "LIST")
        grid_columns = int(export.get("grid_columns", 2))
        bg_color = export.get("bg_color", [40, 40, 40])
        captured_images = data.get("captured_images", [])
        prompts = data.get("lighting_prompts", [])
        pose_count = len(data.get("poses", [{}]))
        lighting_prompts = [
            prompts[index] if index < len(prompts) else ""
            for index in range(pose_count)
        ]
        rendered_images = _decode_captured_images(captured_images[:pose_count])
        tensors = [
            torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            for image in rendered_images
        ]

        if output_mode == "LIST":
            return ([tensor.unsqueeze(0) for tensor in tensors], lighting_prompts)

        grid_img = self._make_grid(rendered_images, grid_columns, tuple(bg_color))
        grid = torch.from_numpy(np.asarray(grid_img, dtype=np.float32) / 255.0).unsqueeze(0)
        return ([grid], [lighting_prompts[0] if lighting_prompts else ""])
    
    def generate(
        self,
        pose_data: str = "{}",
        pose_image=None,
        unique_id: str = None
    ):
        """Generate rendered mesh images for all poses."""
        data = self._resolve_execution_data(pose_data, pose_image, unique_id)
        return self._generate_from_execution_data(data)

    def _make_grid(self, images, columns, bg_color=(40, 40, 40)):
        """Combine images into a grid."""
        if not images:
            return Image.new('RGB', (512, 512), bg_color)
        
        n = len(images)
        cols = max(1, min(columns, n))
        rows = (n + cols - 1) // cols
        
        w, h = images[0].size
        if any(image.size != (w, h) for image in images[1:]):
            raise ValueError("GRID mode requires captured images with identical dimensions")
        grid = Image.new('RGB', (w * cols, h * rows), bg_color)
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid.paste(img, (col * w, row * h))
        
        return grid


# Node mappings
NODE_CLASS_MAPPINGS = {
    "VNCCS_PoseStudio": JakkannaPoseStudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_PoseStudio": "Jakkanna Pose Studio"
}
