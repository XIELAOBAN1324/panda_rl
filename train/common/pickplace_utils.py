from __future__ import annotations

import numpy as np


LEGACY_PHASE_NAMES = (
    "approach",
    "descend",
    "grasp",
    "lift",
    "move",
    "place",
    "hold",
)

WINDOW_PHASE_NAMES = (
    "approach",
    "descend",
    "grasp",
    "lift",
    "reorient",
    "move",
    "place",
    "hold",
)

WINDOW_GRASP_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float32)
WINDOW_REORIENT_ALIGNMENT = 0.95
WINDOW_LIFT_MARGIN = 0.16
WINDOW_REORIENT_OFFSET = 0.14
WINDOW_PREALIGN_OFFSET = 0.14
WINDOW_PREALIGN_HEIGHT = 0.12
WINDOW_PREINSERT_OFFSET = 0.10
WINDOW_PREINSERT_HEIGHT = 0.01


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    safe_norm = np.maximum(norm, eps)
    return vector / safe_norm


def goal_position(goal: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal, dtype=np.float32)
    return goal[..., :3]


def goal_direction(goal: np.ndarray) -> np.ndarray | None:
    goal = np.asarray(goal, dtype=np.float32)
    if goal.shape[-1] < 6:
        return None
    return normalize_vector(goal[..., 3:6])


def is_window_goal(goal: np.ndarray) -> bool:
    goal = np.asarray(goal)
    return goal.shape[-1] >= 6


def goal_alignment(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    desired_direction = goal_direction(desired_goal)
    if desired_direction is None:
        achieved_goal = np.asarray(achieved_goal)
        shape = achieved_goal.shape[:-1] if achieved_goal.ndim > 1 else ()
        return np.ones(shape, dtype=np.float32)
    achieved_direction = normalize_vector(np.asarray(achieved_goal, dtype=np.float32)[..., 3:6])
    return np.sum(achieved_direction * desired_direction, axis=-1)


def goal_orientation_error(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    alignment = np.asarray(goal_alignment(achieved_goal, desired_goal), dtype=np.float32)
    return 1.0 - np.clip(alignment, -1.0, 1.0)


def pickplace_phase_names(action_dim: int, desired_goal_dim: int) -> tuple[str, ...]:
    if action_dim >= 7 or desired_goal_dim >= 6:
        return WINDOW_PHASE_NAMES
    return LEGACY_PHASE_NAMES


def rotation_action_from_axes(
    current_axis: np.ndarray,
    target_axis: np.ndarray,
    rotation_scale: float = 0.20,
) -> np.ndarray:
    current_axis = normalize_vector(np.asarray(current_axis, dtype=np.float32))
    target_axis = normalize_vector(np.asarray(target_axis, dtype=np.float32))
    dot = float(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0))
    cross = np.cross(current_axis, target_axis)
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-6:
        if dot > 0.999:
            return np.zeros(3, dtype=np.float32)
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(fallback, current_axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = normalize_vector(np.cross(current_axis, fallback))
        rotvec = axis * np.pi
    else:
        axis = cross / cross_norm
        angle = float(np.arctan2(cross_norm, dot))
        rotvec = axis * angle

    return np.clip(rotvec / max(float(rotation_scale), 1e-6), -1.0, 1.0).astype(np.float32)


def window_stage_waypoints(
    object_position: np.ndarray,
    goal_pos: np.ndarray,
    goal_normal: np.ndarray,
    initial_object_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_position = np.asarray(object_position, dtype=np.float32)
    goal_pos = np.asarray(goal_pos, dtype=np.float32)
    goal_normal = normalize_vector(np.asarray(goal_normal, dtype=np.float32))

    safe_lift_z = float(
        max(
            initial_object_height + WINDOW_LIFT_MARGIN,
            float(object_position[2]) + 0.12,
            float(goal_pos[2]) + WINDOW_LIFT_MARGIN,
        )
    )
    retreat = goal_pos - goal_normal * WINDOW_REORIENT_OFFSET
    lift_target = np.array(
        [min(float(object_position[0]), float(retreat[0])), object_position[1], safe_lift_z],
        dtype=np.float32,
    )
    prealign = goal_pos - goal_normal * WINDOW_PREALIGN_OFFSET
    prealign = prealign + np.array([0.0, 0.0, WINDOW_PREALIGN_HEIGHT], dtype=np.float32)
    preinsert = goal_pos - goal_normal * WINDOW_PREINSERT_OFFSET
    preinsert = preinsert + np.array([0.0, 0.0, WINDOW_PREINSERT_HEIGHT], dtype=np.float32)
    return lift_target, prealign.astype(np.float32), preinsert.astype(np.float32)


def pickplace_expert_action(
    observation,
    phase: str,
    initial_object_height: float,
    action_dim: int,
    hint_style: str = "staged",
    ee_forward_axis: np.ndarray | None = None,
    position_scale: float = 0.05,
    rotation_scale: float = 0.20,
) -> np.ndarray:
    ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
    object_position = goal_position(observation["achieved_goal"])
    goal_pos = goal_position(observation["desired_goal"])
    desired_goal_dim = int(np.asarray(observation["desired_goal"]).shape[-1])

    if action_dim >= 7 or desired_goal_dim >= 6:
        current_axis = (
            normalize_vector(ee_forward_axis)
            if ee_forward_axis is not None
            else WINDOW_GRASP_AXIS.copy()
        )
        goal_normal = goal_direction(observation["desired_goal"])
        if goal_normal is None:
            goal_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        lift_target, prealign, preinsert = window_stage_waypoints(
            object_position,
            goal_pos,
            goal_normal,
            initial_object_height,
        )
        desired_axis = WINDOW_GRASP_AXIS if phase in {"approach", "descend", "grasp", "lift"} else goal_normal
        targets = {
            "approach": (object_position + np.array([0.0, 0.0, 0.10], dtype=np.float32), 1.0, False),
            "descend": (object_position + np.array([0.0, 0.0, 0.060], dtype=np.float32), 1.0, False),
            "grasp": (object_position + np.array([0.0, 0.0, 0.058], dtype=np.float32), -1.0, False),
            "lift": (lift_target, -1.0, False),
            "reorient": (lift_target, -1.0, False),
            "move": (prealign, -1.0, True),
            "place": (preinsert, -1.0, True),
            "hold": (goal_pos.astype(np.float32), -1.0, True),
        }
        target_position, gripper_action, object_space_target = targets[phase]
        delta = target_position - (object_position if object_space_target else ee_position)
        pos_action = np.clip(delta / max(float(position_scale), 1e-6), -1.0, 1.0)
        rot_action = rotation_action_from_axes(current_axis, desired_axis, rotation_scale=rotation_scale)
        return np.concatenate([pos_action, rot_action, np.array([gripper_action], dtype=np.float32)]).astype(
            np.float32
        )

    if hint_style == "direct":
        xy_distance = float(np.linalg.norm((goal_pos - object_position)[:2]))
        goal_height = max(float(goal_pos[2] - initial_object_height), 0.0)
        carry_height = float(np.clip(goal_height + 0.035 + 0.35 * xy_distance, 0.05, 0.11))
        carry_z = float(max(goal_pos[2] + 0.015, initial_object_height + carry_height))
        targets = {
            "approach": (object_position + np.array([0.0, 0.0, 0.075], dtype=np.float32), 1.0),
            "descend": (object_position + np.array([0.0, 0.0, 0.008], dtype=np.float32), 1.0),
            "grasp": (object_position + np.array([0.0, 0.0, 0.004], dtype=np.float32), -1.0),
            "lift": (np.array([object_position[0], object_position[1], carry_z], dtype=np.float32), -1.0),
            "move": (np.array([goal_pos[0], goal_pos[1], carry_z], dtype=np.float32), -1.0),
            "place": (goal_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32), -1.0),
            "hold": (goal_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32), -1.0),
        }
        target_position, gripper_action = targets[phase]
        delta = np.clip((target_position - ee_position) / max(float(position_scale), 1e-6), -1.0, 1.0)
        return np.concatenate([delta, np.array([gripper_action], dtype=np.float32)]).astype(np.float32)

    safe_z = max(float(goal_pos[2] + 0.10), float(object_position[2] + 0.10), 0.18)
    targets = {
        "approach": (object_position + np.array([0.0, 0.0, 0.10], dtype=np.float32), 1.0),
        "descend": (object_position + np.array([0.0, 0.0, 0.01], dtype=np.float32), 1.0),
        "grasp": (object_position + np.array([0.0, 0.0, 0.005], dtype=np.float32), -1.0),
        "lift": (np.array([object_position[0], object_position[1], safe_z], dtype=np.float32), -1.0),
        "move": (np.array([goal_pos[0], goal_pos[1], safe_z], dtype=np.float32), -1.0),
        "place": (goal_pos + np.array([0.0, 0.0, 0.02], dtype=np.float32), -1.0),
        "hold": (goal_pos + np.array([0.0, 0.0, 0.02], dtype=np.float32), -1.0),
    }
    target_position, gripper_action = targets[phase]
    delta = np.clip((target_position - ee_position) / max(float(position_scale), 1e-6), -1.0, 1.0)
    return np.concatenate([delta, np.array([gripper_action], dtype=np.float32)]).astype(np.float32)


def pickplace_next_phase(
    observation,
    phase_name: str,
    phase_steps: int,
    initial_object_height: float,
    hint_style: str = "staged",
    action_dim: int = 4,
    ee_forward_axis: np.ndarray | None = None,
) -> tuple[str, int]:
    ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
    object_position = goal_position(observation["achieved_goal"])
    goal_pos = goal_position(observation["desired_goal"])
    desired_goal_dim = int(np.asarray(observation["desired_goal"]).shape[-1])

    horizontal_distance = float(np.linalg.norm((ee_position - object_position)[:2]))
    ee_object_distance = float(np.linalg.norm(ee_position - object_position))
    object_goal_distance = float(np.linalg.norm(object_position - goal_pos))

    if action_dim >= 7 or desired_goal_dim >= 6:
        goal_normal = goal_direction(observation["desired_goal"])
        if goal_normal is None:
            goal_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        current_axis = (
            normalize_vector(ee_forward_axis)
            if ee_forward_axis is not None
            else WINDOW_GRASP_AXIS.copy()
        )
        alignment = float(np.dot(current_axis, goal_normal))
        lift_target, prealign, preinsert = window_stage_waypoints(
            object_position,
            goal_pos,
            goal_normal,
            initial_object_height,
        )
        lift_distance = float(np.linalg.norm(object_position - lift_target))
        prealign_distance = float(np.linalg.norm(object_position - prealign))
        preinsert_distance = float(np.linalg.norm(object_position - preinsert))

        if phase_name == "approach" and horizontal_distance < 0.02 and ee_position[2] > object_position[2] + 0.05:
            return "descend", 0
        if phase_name == "descend" and ee_object_distance < 0.065:
            return "grasp", 0
        if phase_name == "grasp" and phase_steps > 10:
            return "lift", 0
        if (
            phase_name == "lift"
            and object_position[2] > initial_object_height + 0.08
            and object_position[0] <= float(lift_target[0]) + 0.03
        ):
            return "reorient", 0
        if phase_name == "reorient" and alignment > WINDOW_REORIENT_ALIGNMENT:
            return "move", 0
        if phase_name == "move" and prealign_distance < 0.04:
            return "place", 0
        if phase_name == "place" and preinsert_distance < 0.035 and alignment > WINDOW_REORIENT_ALIGNMENT:
            return "hold", 0
        return phase_name, phase_steps + 1

    if hint_style == "direct":
        goal_height = max(float(goal_pos[2] - initial_object_height), 0.0)
        carry_height = float(np.clip(goal_height + 0.035 + 0.35 * object_goal_distance, 0.05, 0.11))
        required_lift = max(0.025, 0.55 * carry_height)
        if phase_name == "approach" and horizontal_distance < 0.015 and ee_position[2] > object_position[2] + 0.05:
            return "descend", 0
        if phase_name == "descend" and ee_object_distance < 0.02:
            return "grasp", 0
        if phase_name == "grasp" and phase_steps > 10:
            return "lift", 0
        if phase_name == "lift" and object_position[2] > initial_object_height + required_lift:
            return "move", 0
        if phase_name == "move" and float(np.linalg.norm((object_position - goal_pos)[:2])) < 0.03:
            return "place", 0
        if phase_name == "place" and object_goal_distance < 0.04:
            return "hold", 0
        return phase_name, phase_steps + 1

    if phase_name == "approach" and horizontal_distance < 0.015 and ee_position[2] > object_position[2] + 0.06:
        return "descend", 0
    if phase_name == "descend" and ee_object_distance < 0.02:
        return "grasp", 0
    if phase_name == "grasp" and phase_steps > 12:
        return "lift", 0
    if phase_name == "lift" and object_position[2] > goal_pos[2] + 0.03:
        return "move", 0
    if phase_name == "move" and float(np.linalg.norm((object_position - goal_pos)[:2])) < 0.03:
        return "place", 0
    if phase_name == "place" and object_goal_distance < 0.04:
        return "hold", 0
    return phase_name, phase_steps + 1
