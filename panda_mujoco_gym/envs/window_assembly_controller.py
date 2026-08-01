"""Scripted Cartesian motion controller for window assembly."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from panda_mujoco_gym.envs.window_assembly_geometry import (
    project_to_rotation_matrix,
    so3_exp,
    so3_log,
)
from panda_mujoco_gym.envs.window_assembly_pipeline import WindowAssemblyStage


class WindowAssemblyController:
    """Execute and track non-policy window-assembly motion."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def run_stage(
        self,
        stage: WindowAssemblyStage,
        operation: Callable[[], None],
    ) -> None:
        env = self.env
        env.stage = stage
        env.stage_tracker.begin(stage, env._simulation_step_count)
        try:
            operation()
        except Exception as exc:
            env.stage_tracker.end(
                env._simulation_step_count,
                success=False,
                failure_reason=str(exc),
            )
            raise
        env.stage_tracker.end(env._simulation_step_count, success=True)

    def move_mocap_pose(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        steps: int,
    ) -> None:
        env = self.env
        start_position = env.get_ee_position().copy()
        start_quaternion = env.get_mocap_quaternion().copy()
        target_position = np.asarray(target_position, dtype=np.float64)
        target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
        for step_index in range(max(int(steps), 1)):
            alpha = float(step_index + 1) / float(max(int(steps), 1))
            position = (1.0 - alpha) * start_position + alpha * target_position
            quaternion = env._slerp(
                start_quaternion, target_quaternion, alpha
            )
            env.set_mocap_pose(position, quaternion)
            env._mujoco_step()
            env._simulation_step_count += 1
            env._mujoco.mj_forward(env.model, env.data)
            env._emit_pipeline_frame()
        env._last_ee_target_error = float(
            np.linalg.norm(env.get_ee_position() - target_position)
        )

    def move_attached_object_pose(
        self,
        target_object_center: np.ndarray,
        target_object_rotation: np.ndarray,
        steps: int,
        *,
        center_feedback_gain: float = 0.0,
    ) -> None:
        env = self.env
        if not env._grasp_weld_is_active():
            raise RuntimeError("Cannot move an object that is not suction-attached")
        target_center = np.asarray(target_object_center, dtype=np.float64)
        target_object_rotation = project_to_rotation_matrix(
            target_object_rotation
        )
        start_center = env.get_object_position().copy()
        start_object_rotation = env.get_object_rotation_matrix().copy()
        start_ee_rotation = np.asarray(
            env.data.xmat[env.ee_center_body_id], dtype=np.float64
        ).reshape(3, 3).copy()
        start_ee_quaternion = env.get_mocap_quaternion().copy()
        world_rotation_delta = target_object_rotation @ start_object_rotation.T
        world_rotation_vector = so3_log(world_rotation_delta)
        last_ee_target = env.get_ee_position().copy()
        for step_index in range(max(int(steps), 1)):
            alpha = float(step_index + 1) / float(max(int(steps), 1))
            desired_center = (1.0 - alpha) * start_center + alpha * target_center
            interpolated_delta = so3_exp(alpha * world_rotation_vector)
            delta_quaternion = env._matrix_to_quaternion(interpolated_delta)
            ee_quaternion = env._normalize_vector(
                env._quat_multiply(delta_quaternion, start_ee_quaternion)
            )
            desired_ee_rotation = interpolated_delta @ start_ee_rotation
            ee_position = (
                desired_center
                - desired_ee_rotation @ env._grasp_relative_position
            )
            if center_feedback_gain != 0.0:
                center_error = desired_center - env.get_object_position().copy()
                ee_position += float(center_feedback_gain) * center_error
            last_ee_target = ee_position.copy()
            env.set_mocap_pose(ee_position, ee_quaternion)
            env._mujoco_step()
            env._simulation_step_count += 1
            env._mujoco.mj_forward(env.model, env.data)
            env._emit_pipeline_frame()
        env._last_ee_target_error = float(
            np.linalg.norm(env.get_ee_position() - last_ee_target)
        )
