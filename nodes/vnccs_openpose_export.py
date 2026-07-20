import copy
import json
import math
import os
import types
from functools import lru_cache

import numpy as np

from ..CharacterData import matrix
from ..CharacterData.mh_parser import HumanSolver
from .pose_studio import POSE_STUDIO_CACHE, VNCCS_PoseStudio, _ensure_data_loaded, _validate_pose_data


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_KEYPOINT_FIELD_LENGTHS = {
    "pose_keypoints_2d": 18 * 3,
    "foot_keypoints_2d": 6 * 3,
    "face_keypoints_2d": 70 * 3,
    "hand_right_keypoints_2d": 21 * 3,
    "hand_left_keypoints_2d": 21 * 3,
}
_LENGTH_CONTROLS = {
    "lowerarm_l": "upper_arm_l_length",
    "lowerarm_r": "upper_arm_r_length",
    "hand_l": "forearm_l_length",
    "hand_r": "forearm_r_length",
    "calf_l": "thigh_l_length",
    "calf_r": "thigh_r_length",
    "foot_l": "shin_l_length",
    "foot_r": "shin_r_length",
    "spine_02": "spine_length",
    "spine_03": "spine_length",
}


def _number(values, name, default):
    value = values.get(name, default)
    return float(value) if isinstance(value, (int, float)) else float(default)


def _validate_openpose_keypoints(keypoints, frame_count, people_per_frame=1, max_canvas_size=4096):
    if not isinstance(keypoints, list) or len(keypoints) != frame_count:
        raise ValueError(f"openpose_keypoints must contain exactly {frame_count} frames")
    for frame_index, frame in enumerate(keypoints):
        if not isinstance(frame, dict):
            raise ValueError(f"openpose_keypoints[{frame_index}] must be an object")
        for name in ("canvas_width", "canvas_height"):
            value = frame.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value != int(value)
                or value < 1
                or (max_canvas_size is not None and value > max_canvas_size)
            ):
                limit = f" between 1 and {max_canvas_size}" if max_canvas_size is not None else " greater than zero"
                raise ValueError(f"openpose_keypoints[{frame_index}].{name} must be an integer{limit}")
        people = frame.get("people")
        if not isinstance(people, list) or len(people) != people_per_frame:
            raise ValueError(
                f"openpose_keypoints[{frame_index}].people must contain exactly {people_per_frame} "
                f"{'person' if people_per_frame == 1 else 'people'}"
            )
        for person_index, person in enumerate(people):
            if not isinstance(person, dict):
                raise ValueError(f"openpose_keypoints[{frame_index}].people[{person_index}] must be an object")
            for field, expected_length in _KEYPOINT_FIELD_LENGTHS.items():
                values = person.get(field)
                if not isinstance(values, list) or len(values) != expected_length:
                    raise ValueError(
                        f"openpose_keypoints[{frame_index}].people[{person_index}].{field} "
                        f"must contain {expected_length} numbers"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                ):
                    raise ValueError(
                        f"openpose_keypoints[{frame_index}].people[{person_index}].{field} "
                        "must contain finite numbers"
                    )
    return keypoints


def _solve_vertices(mesh):
    age = max(0.0, min(1.0, (_number(mesh, "age", 25.0) - 1.0) / 89.0))
    solver = HumanSolver()
    factors = solver.calculate_factors(
        age,
        _number(mesh, "gender", 0.5),
        _number(mesh, "weight", 0.5),
        _number(mesh, "muscle", 0.5),
        _number(mesh, "height", 0.5),
        _number(mesh, "breast_size", 0.5),
        _number(mesh, "firmness", 0.5),
        _number(mesh, "penis_len", 0.5),
        _number(mesh, "penis_circ", 0.5),
        _number(mesh, "penis_test", 0.5),
        proportions=_number(mesh, "proportions", 0.5),
        african=_number(mesh, "african", 1.0 / 3),
        asian=_number(mesh, "asian", 1.0 / 3),
        caucasian=_number(mesh, "caucasian", 1.0 / 3) if mesh.get("caucasian") is not None else None,
    )
    return solver.solve_mesh(POSE_STUDIO_CACHE["base_mesh"], POSE_STUDIO_CACHE["targets"], factors)


def _pose_bones(vertices, mesh, pose):
    skeleton = POSE_STUDIO_CACHE["skeleton"].copy()
    skeleton.updateJointPositions(types.SimpleNamespace(vertices=vertices))

    bone_scales = {
        "head": _number(mesh, "head_size", 1.0),
        "upperarm_l": _number(mesh, "arm_size", 1.0),
        "upperarm_r": _number(mesh, "arm_size", 1.0),
        "hand_l": _number(mesh, "hand_size", 1.0),
        "hand_r": _number(mesh, "hand_size", 1.0),
    }
    rotations = pose.get("bones", {})
    world_matrices = {}
    for bone in skeleton.boneslist:
        if bone.parent is None:
            offset = bone.headPos
        else:
            offset = bone.headPos - bone.parent.headPos
        control_name = _LENGTH_CONTROLS.get(bone.name)
        if control_name is not None:
            offset = offset * max(0.25, min(2.0, 0.5 + _number(mesh, control_name, 0.5)))

        rx, ry, rz = rotations.get(bone.name, (0.0, 0.0, 0.0))
        local_matrix = np.asarray(matrix.translate(offset) @ matrix.rotx(rx) @ matrix.roty(ry) @ matrix.rotz(rz))
        scale = bone_scales.get(bone.name, 1.0)
        if scale != 1.0:
            local_matrix = local_matrix @ np.diag((scale, scale, scale, 1.0))
        world_matrices[bone.name] = (
            world_matrices[bone.parent.name] @ local_matrix
            if bone.parent is not None
            else local_matrix
        )
    return skeleton, world_matrices


def _bone_point(skeleton, world_matrices, bone_name, tail=False):
    bone = skeleton.getBone(bone_name)
    if bone is None:
        return np.zeros(3, dtype=np.float32)
    offset = bone.tailPos - bone.headPos if tail else np.zeros(3, dtype=np.float64)
    return (world_matrices[bone_name] @ np.append(offset, 1.0))[:3]


@lru_cache(maxsize=1)
def _face_joint_indices():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "CharacterData",
        "makehuman",
        "makehuman",
        "data",
        "rigs",
        "default.mhskel",
    )
    with open(path, "r", encoding="utf-8") as handle:
        joints = json.load(handle)["joints"]
    return {
        "nose": tuple(joints["special01____tail"]),
        "left_eye": tuple(joints["eye.L____head"]),
        "right_eye": tuple(joints["eye.R____head"]),
    }


def _face_points(vertices, skeleton, world_matrices):
    head = skeleton.getBone("head")
    transform = world_matrices["head"] @ np.asarray(matrix.translate(-head.headPos))
    points = {}
    for name, indices in _face_joint_indices().items():
        rest = vertices[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        points[name] = (transform @ np.append(rest, 1.0))[:3]
    return points


def _model_rotation(pose):
    rx, ry, rz = pose.get("modelRotation", (0.0, 0.0, 0.0))
    return np.asarray(matrix.rotx(rx) @ matrix.roty(ry) @ matrix.rotz(rz))[:3, :3]


def _projector(center, width, height, camera):
    zoom = max(0.1, _number(camera, "zoom", 1.0))
    target = np.asarray(center, dtype=np.float64).copy()
    target[0] -= _number(camera, "offset_x", 0.0)
    target[1] -= _number(camera, "offset_y", 0.0)

    yaw = _number(camera, "yaw_deg", 0.0)
    pitch = _number(camera, "pitch_deg", 0.0)
    camera_rotation = np.asarray(matrix.roty(yaw) @ matrix.rotx(pitch))[:3, :3]
    position = target + camera_rotation @ np.array((0.0, 0.0, 45.0))

    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_half_fov = math.tan(math.radians(15.0)) / zoom
    aspect = width / height

    def project(point):
        relative = point - position
        depth = float(np.dot(relative, forward))
        if depth <= 0.0:
            return (0.0, 0.0, 0.0)
        ndc_x = float(np.dot(relative, right)) / (depth * tan_half_fov * aspect)
        ndc_y = float(np.dot(relative, up)) / (depth * tan_half_fov)
        return ((ndc_x + 1.0) * width * 0.5, (1.0 - ndc_y) * height * 0.5, 1.0)

    return project


def _flatten(points):
    return [component for point in points for component in point]


def _hand_points(skeleton, world_matrices, side, rotation, project):
    points = [project(rotation @ _bone_point(skeleton, world_matrices, f"hand_{side}"))]
    for finger in _FINGERS:
        for joint in range(1, 4):
            points.append(project(rotation @ _bone_point(skeleton, world_matrices, f"{finger}_{joint:02d}_{side}")))
        points.append(project(rotation @ _bone_point(skeleton, world_matrices, f"{finger}_03_{side}", tail=True)))
    return points


def _openpose_frame(vertices, skeleton, world_matrices, export, pose):
    width = int(_number(export, "view_width", _number(export, "view_size", 512)))
    height = int(_number(export, "view_height", _number(export, "view_size", 512)))
    rotation = _model_rotation(pose)
    camera = dict(export)
    camera.update(pose.get("cameraParams") or {})
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    project = _projector(center, width, height, camera)

    face = {name: project(rotation @ point) for name, point in _face_points(vertices, skeleton, world_matrices).items()}
    missing = (0.0, 0.0, 0.0)
    body = [
        face["nose"],
        project(rotation @ _bone_point(skeleton, world_matrices, "neck_01")),
        project(rotation @ _bone_point(skeleton, world_matrices, "upperarm_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "lowerarm_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "hand_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "upperarm_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "lowerarm_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "hand_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "thigh_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "calf_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "foot_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "thigh_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "calf_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "foot_l")),
        face["right_eye"],
        face["left_eye"],
        missing,
        missing,
    ]
    feet = [
        project(rotation @ _bone_point(skeleton, world_matrices, "ball_l", tail=True)),
        project(rotation @ _bone_point(skeleton, world_matrices, "ball_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "foot_l")),
        project(rotation @ _bone_point(skeleton, world_matrices, "ball_r", tail=True)),
        project(rotation @ _bone_point(skeleton, world_matrices, "ball_r")),
        project(rotation @ _bone_point(skeleton, world_matrices, "foot_r")),
    ]
    empty_face = [missing] * 70
    return {
        "canvas_width": width,
        "canvas_height": height,
        "people": [{
            "pose_keypoints_2d": _flatten(body),
            "foot_keypoints_2d": _flatten(feet),
            "face_keypoints_2d": _flatten(empty_face),
            "hand_right_keypoints_2d": _flatten(_hand_points(skeleton, world_matrices, "r", rotation, project)),
            "hand_left_keypoints_2d": _flatten(_hand_points(skeleton, world_matrices, "l", rotation, project)),
        }],
    }


def pose_data_to_openpose(pose_data, image_sizes=None):
    data = pose_data if isinstance(pose_data, dict) else json.loads(pose_data) if pose_data else {}
    _validate_pose_data(data)
    mesh = data.get("mesh", {})
    export = data.get("export", {})
    poses = data.get("poses") or [{}]
    if image_sizes is not None and (
        not isinstance(image_sizes, (list, tuple))
        or len(image_sizes) != len(poses)
    ):
        raise ValueError(f"image_sizes must contain exactly {len(poses)} entries")

    _ensure_data_loaded()
    vertices = _solve_vertices(mesh)
    frames = []
    for pose_index, pose in enumerate(poses):
        skeleton, world_matrices = _pose_bones(vertices, mesh, pose)
        frame_export = export
        if image_sizes is not None:
            width, height = image_sizes[pose_index]
            frame_export = {**export, "view_width": width, "view_height": height}
        frames.append(_openpose_frame(vertices, skeleton, world_matrices, frame_export, pose))
    return frames


def _combine_openpose_grid(keypoints, columns, cell_width, cell_height):
    people = []
    for frame_index, frame in enumerate(keypoints):
        person = copy.deepcopy(frame["people"][0])
        offset_x = (frame_index % columns) * cell_width
        offset_y = (frame_index // columns) * cell_height
        for field in _KEYPOINT_FIELD_LENGTHS:
            values = person[field]
            for point_index in range(0, len(values), 3):
                if values[point_index] != 0 or values[point_index + 1] != 0 or values[point_index + 2] != 0:
                    values[point_index] += offset_x
                    values[point_index + 1] += offset_y
        people.append(person)
    rows = (len(keypoints) + columns - 1) // columns
    return [{
        "canvas_width": columns * cell_width,
        "canvas_height": rows * cell_height,
        "people": people,
    }]


class VNCCS_PoseStudioOpenPose(VNCCS_PoseStudio):
    RETURN_TYPES = ("IMAGE", "STRING", "POSE_KEYPOINT")
    RETURN_NAMES = ("images", "lighting_prompt", "keypoints")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "generate_with_openpose"

    def generate_with_openpose(self, pose_data="{}", pose_image=None, unique_id=None):
        data = self._resolve_execution_data(pose_data, pose_image, unique_id)
        images, prompts = self._generate_from_execution_data(data)
        pose_count = len(data.get("poses", [{}]))
        output_mode = data.get("export", {}).get("output_mode", "LIST")
        if output_mode == "GRID":
            columns = min(int(data.get("export", {}).get("grid_columns", 2)), pose_count)
            rows = (pose_count + columns - 1) // columns
            grid_height, grid_width = images[0].shape[1:3]
            if grid_width % columns or grid_height % rows:
                raise ValueError("GRID output dimensions are not divisible into equal pose cells")
            cell_width = grid_width // columns
            cell_height = grid_height // rows
            image_sizes = [(cell_width, cell_height)] * pose_count
        else:
            image_sizes = [(int(image.shape[2]), int(image.shape[1])) for image in images]

        keypoints = data.get("openpose_keypoints")
        if keypoints is None or keypoints == []:
            keypoints = pose_data_to_openpose(data, image_sizes)
        _validate_openpose_keypoints(keypoints, pose_count)
        for frame_index, ((width, height), frame) in enumerate(zip(image_sizes, keypoints)):
            if frame["canvas_width"] != width or frame["canvas_height"] != height:
                raise ValueError(
                    f"openpose_keypoints[{frame_index}] canvas dimensions do not match its captured image"
                )
        if output_mode == "GRID":
            keypoints = _combine_openpose_grid(keypoints, columns, cell_width, cell_height)
            _validate_openpose_keypoints(keypoints, 1, people_per_frame=pose_count, max_canvas_size=None)
            if keypoints[0]["canvas_width"] != grid_width or keypoints[0]["canvas_height"] != grid_height:
                raise ValueError("OpenPose GRID dimensions do not match the rendered grid")
        return images, prompts, keypoints


class VNCCSReplaceOpenPoseHands:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_keypoints": ("POSE_KEYPOINT",),
                "vnccs_keypoints": ("POSE_KEYPOINT",),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("keypoints",)
    FUNCTION = "replace_hands"
    CATEGORY = "VNCCS/Pose"

    def replace_hands(self, base_keypoints, vnccs_keypoints):
        merged = copy.deepcopy(base_keypoints)
        if not isinstance(merged, list) or not isinstance(vnccs_keypoints, list):
            return (merged,)

        for index, frame in enumerate(merged):
            if index >= len(vnccs_keypoints):
                break
            base_people = frame.get("people") if isinstance(frame, dict) else None
            direct_frame = vnccs_keypoints[index]
            direct_people = direct_frame.get("people") if isinstance(direct_frame, dict) else None
            if not base_people or not direct_people:
                continue

            base_person = base_people[0]
            direct_person = direct_people[0]
            for field in ("hand_right_keypoints_2d", "hand_left_keypoints_2d"):
                hand = direct_person.get(field)
                if isinstance(hand, list) and len(hand) == 63:
                    base_person[field] = copy.deepcopy(hand)

            body = base_person.get("pose_keypoints_2d")
            direct_body = direct_person.get("pose_keypoints_2d")
            if (
                isinstance(body, list)
                and len(body) >= 24
                and isinstance(direct_body, list)
                and len(direct_body) >= 24
            ):
                body[12:15] = direct_body[12:15]
                body[21:24] = direct_body[21:24]

        return (merged,)
