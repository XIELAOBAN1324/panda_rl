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

WINDOW_INSERT_PHASE_NAMES = (
    "reorient",
    "transport",
    "prealign",
    "insert",
    "hold",
)

WINDOW_GRASP_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float32)
WINDOW_REORIENT_ALIGNMENT = 0.95
WINDOW_LIFT_MARGIN = 0.16
WINDOW_REORIENT_OFFSET = 0.20
WINDOW_PREALIGN_OFFSET = 0.14
WINDOW_PREALIGN_HEIGHT = 0.0
WINDOW_PREINSERT_OFFSET = 0.03
WINDOW_PREINSERT_HEIGHT = 0.0
WINDOW_FINAL_INSERT_OFFSET = 0.003
WINDOW_HOLD_LATERAL_MARGIN = 0.003
WINDOW_HOLD_LATERAL_GAIN = 0.5
WINDOW_PICKUP_APPROACH_HEIGHT = 0.12
WINDOW_PICKUP_DESCEND_HEIGHT = 0.01
WINDOW_INSERT_TRANSPORT_HEIGHT = 0.14
WINDOW_INSERT_REORIENT_ALIGNMENT = 0.98


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
    alignment = np.sum(achieved_direction * desired_direction, axis=-1)
    return alignment


def goal_orientation_error(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    alignment = np.asarray(goal_alignment(achieved_goal, desired_goal), dtype=np.float32)
    return 1.0 - np.clip(alignment, -1.0, 1.0)


def pickplace_phase_names(action_dim: int, desired_goal_dim: int) -> tuple[str, ...]:
    if action_dim >= 7 or desired_goal_dim >= 6:
        return WINDOW_PHASE_NAMES
    return LEGACY_PHASE_NAMES


def is_insertion_task(observation, initial_object_height: float | None = None) -> bool:
    if initial_object_height is not None:
        return float(initial_object_height) > 0.35
    observation_vec = np.asarray(observation["observation"], dtype=np.float32)
    attached_flag = float(observation_vec[-1]) if observation_vec.size > 0 else 0.0
    object_position = goal_position(observation["achieved_goal"])
    return attached_flag > 0.5 and float(object_position[2]) > 0.35


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


def rotation_action_from_matrices(
    current_rot: np.ndarray,
    target_rot: np.ndarray,
    rotation_scale: float = 0.20,
) -> np.ndarray:
    current_rot = np.asarray(current_rot, dtype=np.float32).reshape(3, 3)
    target_rot = np.asarray(target_rot, dtype=np.float32).reshape(3, 3)
    rel_rot = target_rot @ current_rot.T
    trace = float(np.trace(rel_rot))
    cos_angle = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)

    axis = np.array(
        [
            rel_rot[2, 1] - rel_rot[1, 2],
            rel_rot[0, 2] - rel_rot[2, 0],
            rel_rot[1, 0] - rel_rot[0, 1],
        ],
        dtype=np.float32,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rel_rot)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(np.real(eigenvalues) - 1.0)))])
        axis = normalize_vector(axis.astype(np.float32))
    else:
        axis = axis / axis_norm
    rotvec = axis * angle
    return np.clip(rotvec / max(float(rotation_scale), 1e-6), -1.0, 1.0).astype(np.float32)


def rotation_alignment_from_matrices(
    current_rot: np.ndarray,
    target_rot: np.ndarray,
) -> float:
    current_rot = np.asarray(current_rot, dtype=np.float32).reshape(3, 3)
    target_rot = np.asarray(target_rot, dtype=np.float32).reshape(3, 3)
    rel_rot = target_rot.T @ current_rot
    cos_angle = float(np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0))
    return cos_angle


def window_stage_waypoints(
    object_position: np.ndarray,
    goal_pos: np.ndarray,
    goal_normal: np.ndarray,
    initial_object_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    final_insert = goal_pos - goal_normal * WINDOW_FINAL_INSERT_OFFSET
    return (
        lift_target,
        prealign.astype(np.float32),
        preinsert.astype(np.float32),
        final_insert.astype(np.float32),
    )


def window_insert_waypoints(
    object_position: np.ndarray,
    goal_pos: np.ndarray,
    goal_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_position = np.asarray(object_position, dtype=np.float32)
    goal_pos = np.asarray(goal_pos, dtype=np.float32)
    goal_normal = normalize_vector(np.asarray(goal_normal, dtype=np.float32))

    retreat = goal_pos - goal_normal * WINDOW_REORIENT_OFFSET
    transport = np.array(
        [retreat[0], goal_pos[1], max(float(object_position[2]), float(goal_pos[2]) + WINDOW_INSERT_TRANSPORT_HEIGHT)],
        dtype=np.float32,
    )
    prealign = goal_pos - goal_normal * WINDOW_PREALIGN_OFFSET
    prealign = prealign + np.array([0.0, 0.0, WINDOW_PREALIGN_HEIGHT], dtype=np.float32)
    preinsert = goal_pos - goal_normal * WINDOW_PREINSERT_OFFSET
    final_insert = goal_pos - goal_normal * WINDOW_FINAL_INSERT_OFFSET
    return (
        transport.astype(np.float32),
        prealign.astype(np.float32),
        preinsert.astype(np.float32),
        final_insert.astype(np.float32),
    )


def window_hold_waypoint(
    object_position: np.ndarray,
    goal_pos: np.ndarray,
    goal_normal: np.ndarray,
) -> np.ndarray:
    object_position = np.asarray(object_position, dtype=np.float32)
    goal_pos = np.asarray(goal_pos, dtype=np.float32)
    goal_normal = normalize_vector(np.asarray(goal_normal, dtype=np.float32))
    lateral_error = float(np.linalg.norm((goal_pos - object_position)[1:3]))
    guarded_offset = max(
        WINDOW_FINAL_INSERT_OFFSET,
        WINDOW_HOLD_LATERAL_GAIN * lateral_error + WINDOW_HOLD_LATERAL_MARGIN,
    )
    return (goal_pos - goal_normal * guarded_offset).astype(np.float32)


def window_target_suction_axis(
    object_position: np.ndarray,
    goal_pos: np.ndarray,
    goal_normal: np.ndarray,
) -> np.ndarray:
    del object_position, goal_pos
    goal_normal = normalize_vector(np.asarray(goal_normal, dtype=np.float32))
    # The suction tool should stay on the robot side of the glass, so its
    # forward axis must oppose the desired glass normal throughout transport.
    return (-goal_normal).astype(np.float32)


def window_target_rotation(goal_normal: np.ndarray) -> np.ndarray:
    # The goal normal is the desired glass normal. The suction tool itself
    # should point along the opposite direction, which is handled separately
    # when converting the object target pose into an EE target pose.
    normal_axis = normalize_vector(np.asarray(goal_normal, dtype=np.float32))
    world_vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(world_vertical, normal_axis))) > 0.95:
        world_vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    local_x_world = world_vertical - float(np.dot(world_vertical, normal_axis)) * normal_axis
    local_x_world = normalize_vector(local_x_world)
    local_y_world = normalize_vector(np.cross(normal_axis, local_x_world))
    return np.stack([local_x_world, local_y_world, normal_axis], axis=1).astype(np.float32)


def window_target_rotation_variants(goal_normal: np.ndarray) -> tuple[np.ndarray, ...]:
    base_rotation = window_target_rotation(goal_normal).astype(np.float32)
    symmetry_transforms = (
        np.diag([1.0, 1.0, 1.0]).astype(np.float32),
        np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
        np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        np.diag([-1.0, 1.0, -1.0]).astype(np.float32),
    )
    return tuple((base_rotation @ transform).astype(np.float32) for transform in symmetry_transforms)


def select_window_target_rotation(
    current_object_rot: np.ndarray | None,
    goal_normal: np.ndarray,
) -> np.ndarray:
    candidates = window_target_rotation_variants(goal_normal)
    if current_object_rot is None:
        return candidates[0]

    current_object_rot = np.asarray(current_object_rot, dtype=np.float32).reshape(3, 3)
    best_candidate = candidates[0]
    best_alignment = float("-inf")
    for candidate in candidates:
        alignment = rotation_alignment_from_matrices(current_object_rot, candidate)
        if alignment > best_alignment:
            best_alignment = alignment
            best_candidate = candidate
    return best_candidate.astype(np.float32)


def estimate_grasp_relative_rotation(
    ee_rotation_matrix: np.ndarray,
    object_rotation_matrix: np.ndarray,
) -> np.ndarray:
    ee_rotation_matrix = np.asarray(ee_rotation_matrix, dtype=np.float32).reshape(3, 3)
    object_rotation_matrix = np.asarray(object_rotation_matrix, dtype=np.float32).reshape(3, 3)
    return (ee_rotation_matrix.T @ object_rotation_matrix).astype(np.float32)


def window_rotation_action_and_alignment(
    goal_normal: np.ndarray,
    current_axis: np.ndarray,
    ee_rotation_matrix: np.ndarray | None,
    object_rotation_matrix: np.ndarray | None,
    rotation_scale: float,
) -> tuple[np.ndarray, float]:
    desired_axis = window_target_suction_axis(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), goal_normal)
    axis_alignment = float(np.dot(current_axis, desired_axis))
    if ee_rotation_matrix is None or object_rotation_matrix is None:
        return (
            rotation_action_from_axes(current_axis, desired_axis, rotation_scale=rotation_scale),
            axis_alignment,
        )

    target_object_rotation = select_window_target_rotation(object_rotation_matrix, goal_normal)
    grasp_relative_rotation = estimate_grasp_relative_rotation(ee_rotation_matrix, object_rotation_matrix)
    target_ee_rotation = target_object_rotation @ grasp_relative_rotation.T
    rotation_alignment = rotation_alignment_from_matrices(object_rotation_matrix, target_object_rotation)
    return (
        rotation_action_from_matrices(ee_rotation_matrix, target_ee_rotation, rotation_scale=rotation_scale),
        float(rotation_alignment),
    )


def translation_scale_from_alignment(alignment: float) -> float:
    alignment = float(alignment)
    return float(np.clip((alignment - 0.70) / 0.28, 0.0, 1.0))


def pickplace_expert_action(
    observation,
    phase: str,
    initial_object_height: float,
    action_dim: int,
    hint_style: str = "staged",
    ee_forward_axis: np.ndarray | None = None,
    ee_rotation_matrix: np.ndarray | None = None,
    object_rotation_matrix: np.ndarray | None = None,
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

        if is_insertion_task(observation, initial_object_height):
            transport, prealign, preinsert, final_insert = window_insert_waypoints(
                object_position,
                goal_pos,
                goal_normal,
            )
            hold_target = window_hold_waypoint(object_position, goal_pos, goal_normal)
            targets = {
                "reorient": (object_position.copy(), -1.0, True),
                "transport": (transport, -1.0, True),
                "prealign": (prealign, -1.0, True),
                "insert": (preinsert, -1.0, True),
                "hold": (hold_target, -1.0, True),
            }
            target_position, gripper_action, object_space_target = targets[phase]
            delta = target_position - (object_position if object_space_target else ee_position)
            pos_action = np.clip(delta / max(float(position_scale), 1e-6), -1.0, 1.0)
            if phase == "hold":
                lateral_error = float(np.linalg.norm((goal_pos - object_position)[1:3]))
                if lateral_error > 0.003:
                    pos_action[0] *= 0.25
                    pos_action[1:] = np.clip(pos_action[1:] * 2.5, -1.0, 1.0)
            if phase in {"reorient", "transport"}:
                desired_axis = window_target_suction_axis(object_position, goal_pos, goal_normal)
                rot_action = rotation_action_from_axes(current_axis, desired_axis, rotation_scale=rotation_scale)
                alignment_for_translation = float(np.dot(current_axis, desired_axis))
            else:
                rot_action, alignment_for_translation = window_rotation_action_and_alignment(
                    goal_normal,
                    current_axis,
                    ee_rotation_matrix,
                    object_rotation_matrix,
                    rotation_scale,
                )
            if phase != "reorient":
                pos_action *= translation_scale_from_alignment(alignment_for_translation)
            components = [pos_action, rot_action]
            if action_dim > 6:
                components.append(np.array([gripper_action], dtype=np.float32))
            return np.concatenate(components).astype(np.float32)

        lift_target, prealign, preinsert, final_insert = window_stage_waypoints(
            object_position,
            goal_pos,
            goal_normal,
            initial_object_height,
        )
        desired_axis = (
            WINDOW_GRASP_AXIS
            if phase in {"approach", "descend", "grasp", "lift"}
            else window_target_suction_axis(object_position, goal_pos, goal_normal)
        )
        if phase in {"approach", "descend", "grasp", "lift"}:
            rot_action = rotation_action_from_axes(current_axis, desired_axis, rotation_scale=rotation_scale)
            alignment_for_translation = float(np.dot(current_axis, desired_axis))
        else:
            rot_action, alignment_for_translation = window_rotation_action_and_alignment(
                goal_normal,
                current_axis,
                ee_rotation_matrix,
                object_rotation_matrix,
                rotation_scale,
            )
        targets = {
            "approach": (object_position + np.array([0.0, 0.0, WINDOW_PICKUP_APPROACH_HEIGHT], dtype=np.float32), 1.0, False),
            "descend": (object_position + np.array([0.0, 0.0, WINDOW_PICKUP_DESCEND_HEIGHT], dtype=np.float32), 1.0, False),
            "grasp": (object_position + np.array([0.0, 0.0, WINDOW_PICKUP_DESCEND_HEIGHT], dtype=np.float32), -1.0, False),
            "lift": (lift_target, -1.0, False),
            "reorient": (lift_target, -1.0, False),
            "move": (prealign, -1.0, True),
            "place": (preinsert, -1.0, True),
            "hold": (final_insert, -1.0, True),
        }
        target_position, gripper_action, object_space_target = targets[phase]
        delta = target_position - (object_position if object_space_target else ee_position)
        pos_action = np.clip(delta / max(float(position_scale), 1e-6), -1.0, 1.0)
        if phase in {"move", "place", "hold"}:
            pos_action *= translation_scale_from_alignment(alignment_for_translation)
        components = [pos_action, rot_action]
        if action_dim > 6:
            components.append(np.array([gripper_action], dtype=np.float32))
        return np.concatenate(components).astype(np.float32)

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
    ee_rotation_matrix: np.ndarray | None = None,
    object_rotation_matrix: np.ndarray | None = None,
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

        if is_insertion_task(observation, initial_object_height):
            transport, prealign, preinsert, final_insert = window_insert_waypoints(
                object_position,
                goal_pos,
                goal_normal,
            )
            transport_distance = float(np.linalg.norm(object_position - transport))
            prealign_distance = float(np.linalg.norm(object_position - prealign))
            preinsert_distance = float(np.linalg.norm(object_position - preinsert))
            final_insert_distance = float(np.linalg.norm(object_position - final_insert))
            target_axis = window_target_suction_axis(object_position, goal_pos, goal_normal)
            axis_alignment = float(np.dot(current_axis, target_axis))
            _, pose_alignment = window_rotation_action_and_alignment(
                goal_normal,
                current_axis,
                ee_rotation_matrix,
                object_rotation_matrix,
                rotation_scale=0.20,
            )

            if phase_name == "reorient" and axis_alignment > WINDOW_INSERT_REORIENT_ALIGNMENT:
                return "transport", 0
            if phase_name == "transport" and transport_distance < 0.05 and axis_alignment > 0.94:
                return "prealign", 0
            if phase_name == "prealign" and prealign_distance < 0.04 and pose_alignment > 0.95:
                return "insert", 0
            if phase_name == "insert" and preinsert_distance < 0.02 and pose_alignment > 0.97:
                return "hold", 0
            if phase_name == "hold" and final_insert_distance < 0.006 and pose_alignment > WINDOW_INSERT_REORIENT_ALIGNMENT:
                return "hold", phase_steps + 1
            return phase_name, phase_steps + 1

        if phase_name in {"approach", "descend", "grasp", "lift"}:
            target_axis = window_target_suction_axis(object_position, goal_pos, goal_normal)
            alignment = float(np.dot(current_axis, target_axis))
        else:
            _, alignment = window_rotation_action_and_alignment(
                goal_normal,
                current_axis,
                ee_rotation_matrix,
                object_rotation_matrix,
                rotation_scale=0.20,
            )
        lift_target, prealign, preinsert, final_insert = window_stage_waypoints(
            object_position,
            goal_pos,
            goal_normal,
            initial_object_height,
        )
        lift_distance = float(np.linalg.norm(object_position - lift_target))
        prealign_distance = float(np.linalg.norm(object_position - prealign))
        preinsert_distance = float(np.linalg.norm(object_position - preinsert))
        final_insert_distance = float(np.linalg.norm(object_position - final_insert))

        if phase_name == "approach" and horizontal_distance < 0.03 and ee_position[2] > object_position[2] + 0.06:
            return "descend", 0
        if phase_name == "descend" and ee_object_distance < 0.028:
            return "grasp", 0
        if phase_name == "grasp" and phase_steps > 6:
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
        if phase_name == "place" and preinsert_distance < 0.03 and alignment > WINDOW_REORIENT_ALIGNMENT:
            return "hold", 0
        if phase_name == "hold" and final_insert_distance < 0.01 and alignment > WINDOW_REORIENT_ALIGNMENT:
            return "hold", phase_steps + 1
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
