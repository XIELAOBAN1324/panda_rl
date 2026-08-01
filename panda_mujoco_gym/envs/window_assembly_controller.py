"""Scripted Cartesian motion controller for window assembly."""

from __future__ import annotations

from typing import Any, Callable, Mapping

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
        self._last_insert_diagnostics: dict[str, Any] = {}

    def reset_episode(self) -> None:
        """Clear controller-owned diagnostics for a newly sampled scene."""

        self._last_insert_diagnostics = self._empty_insert_diagnostics()

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

    def run_approach(self, frame_rotation: np.ndarray) -> None:
        """Move the suction cup to the scripted pickup approach pose."""

        env = self.env
        object_position = env.get_object_position().copy()
        approach_position = object_position + frame_rotation[:, 0] * 0.12
        self.run_stage(
            WindowAssemblyStage.APPROACH,
            lambda: self.move_mocap_pose(
                approach_position,
                env.grasp_site_pose,
                env.post_grasp_motion_steps,
            ),
        )

    def run_descend(self, frame_rotation: np.ndarray) -> None:
        """Descend from the approach pose to the suction pickup pose."""

        env = self.env
        object_position = env.get_object_position().copy()
        descend_position = object_position + frame_rotation[:, 0] * 0.01
        self.run_stage(
            WindowAssemblyStage.DESCEND,
            lambda: self.move_mocap_pose(
                descend_position,
                env.grasp_site_pose,
                env.post_grasp_motion_steps,
            ),
        )

    def run_grasp(self) -> None:
        """Activate and verify the scripted suction attachment."""

        env = self.env

        def grasp() -> None:
            env._activate_grasp_weld()
            env._mujoco_step()
            env._simulation_step_count += 1
            env._mujoco.mj_forward(env.model, env.data)
            env._emit_pipeline_frame()
            if not env._grasp_weld_is_active():
                raise RuntimeError("Scripted suction weld did not activate")

        self.run_stage(WindowAssemblyStage.GRASP, grasp)

    def run_lift(self) -> np.ndarray:
        """Lift the attached glass to the existing clearance height."""

        env = self.env
        lift_height = float(
            max(
                env.window_center[2] + env.post_grasp_canonical_goal_margin,
                env.initial_object_height + env.post_grasp_min_lift_above_object,
            )
        )
        lift_position = env.get_object_position().copy()
        lift_position[2] = lift_height
        self.run_stage(
            WindowAssemblyStage.LIFT,
            lambda: self.move_attached_object_pose(
                lift_position,
                env.get_object_rotation_matrix(),
                2 * env.post_grasp_motion_steps,
            ),
        )
        return lift_position

    def run_reorient(
        self,
        frame_center: np.ndarray,
        frame_rotation: np.ndarray,
        orientation_frame: np.ndarray,
        lift_position: np.ndarray,
    ) -> None:
        """Reorient at the post-lift center, then transport toward the frame."""

        env = self.env
        frame_normal = frame_rotation[:, 2]
        transport_position = frame_center + frame_normal * max(
            env.preinsert_normal_offset + 0.12, 0.28
        )
        transport_position[2] = max(
            float(lift_position[2]), float(frame_center[2] + 0.12)
        )
        target_object_rotation = project_to_rotation_matrix(
            orientation_frame @ env.GLASS_ASSEMBLY_FRAME
        )

        def reorient() -> None:
            flip_center = env.get_object_position().copy()
            self.move_attached_object_pose(
                flip_center,
                target_object_rotation,
                3 * env.post_grasp_motion_steps,
                center_feedback_gain=1.0,
            )
            self.move_attached_object_pose(
                transport_position,
                target_object_rotation,
                3 * env.post_grasp_motion_steps,
                center_feedback_gain=1.0,
            )

        self.run_stage(WindowAssemblyStage.REORIENT, reorient)

    def run_coarse_align(self, frame_rotation: np.ndarray) -> None:
        """Sample and execute the recoverable scripted preinsert pose."""

        self.run_stage(
            WindowAssemblyStage.COARSE_ALIGN,
            lambda: self._sample_and_apply_coarse_alignment(frame_rotation),
        )

    def _sample_and_apply_coarse_alignment(self, frame_rotation: np.ndarray) -> None:
        env = self.env
        canonical_state = env._capture_sim_state()
        frame_center, frame_rotation = env._current_fine_frame_pose()
        nominal_preinsert_center = (
            frame_center + env.preinsert_normal_offset * frame_rotation[:, 2]
        )
        last_reason = "no_candidate"
        last_errors: Mapping[str, Any] | None = None
        last_target_center: np.ndarray | None = None
        last_object_target_error = 0.0
        last_ee_target_error = 0.0
        for _ in range(env.coarse_sampling_attempts):
            env._restore_sim_state(canonical_state)
            sample = np.array(
                [
                    env.np_random.uniform(
                        -env.coarse_translation_range,
                        env.coarse_translation_range,
                    ),
                    env.np_random.uniform(
                        -env.coarse_translation_range,
                        env.coarse_translation_range,
                    ),
                    env.np_random.uniform(
                        -env.coarse_tilt_range_rad,
                        env.coarse_tilt_range_rad,
                    ),
                    env.np_random.uniform(
                        -env.coarse_tilt_range_rad,
                        env.coarse_tilt_range_rad,
                    ),
                    env.np_random.uniform(
                        -env.coarse_yaw_range_rad,
                        env.coarse_yaw_range_rad,
                    ),
                ],
                dtype=np.float64,
            )
            target_center = (
                nominal_preinsert_center
                + sample[0] * frame_rotation[:, 0]
                + sample[1] * frame_rotation[:, 1]
            )
            commanded_gap = float(
                np.dot(frame_rotation[:, 2], target_center - frame_center)
            )
            if not np.isclose(
                commanded_gap, env.preinsert_normal_offset, atol=1e-8
            ):
                raise RuntimeError("Coarse preinsert target left the fixed normal plane")
            rotation_vector_world = (
                sample[2] * frame_rotation[:, 0]
                + sample[3] * frame_rotation[:, 1]
                + sample[4] * frame_rotation[:, 2]
            )
            target_assembly_rotation = (
                so3_exp(rotation_vector_world) @ env._current_orientation_frame()
            )
            target_rotation = (
                target_assembly_rotation @ env.GLASS_ASSEMBLY_FRAME
            )
            self.move_attached_object_pose(
                target_center,
                target_rotation,
                5 * env.post_grasp_motion_steps,
            )
            env._stabilize_reset_state()
            errors = env.get_fine_alignment_errors()
            last_target_center = target_center.copy()
            last_object_target_error = float(
                np.linalg.norm(env.get_object_position().copy() - target_center)
            )
            last_ee_target_error = float(env._last_ee_target_error)
            last_errors = errors
            last_reason = self._coarse_candidate_failure(
                errors,
                target_center=target_center,
                object_target_error=last_object_target_error,
            )
            if not last_reason:
                env.coarse_alignment_sample = sample
                return
        env._restore_sim_state(canonical_state)
        diagnostics = ""
        if last_errors is not None:
            diagnostics = (
                f" (u={float(last_errors['error_u']):.4f}, "
                f"v={float(last_errors['error_v']):.4f}, "
                f"gap={float(last_errors['current_normal_gap']):.4f}, "
                f"gap_error={float(last_errors['normal_gap_error']):.4f}, "
                f"tilt=({float(last_errors['tilt_error_t1']):.4f},"
                f"{float(last_errors['tilt_error_t2']):.4f}), "
                f"yaw={float(last_errors['yaw_error']):.4f}, "
                f"ee_target_error={last_ee_target_error:.4f}, "
                f"object_target_error={last_object_target_error:.4f})"
            )
        if last_target_center is not None:
            diagnostics += f" target_center={last_target_center.tolist()}"
        raise RuntimeError(
            "Unable to sample a recoverable coarse-alignment state: "
            f"{last_reason}{diagnostics}"
        )

    def _coarse_candidate_failure(
        self,
        errors: Mapping[str, Any],
        *,
        target_center: np.ndarray,
        object_target_error: float,
    ) -> str:
        env = self.env
        if not env._grasp_weld_is_active():
            return "suction_weld_broken"
        if (
            env._last_ee_target_error > env.unreachable_position_threshold
            or object_target_error > env.unreachable_position_threshold
            or not np.all(np.isfinite(np.asarray(target_center, dtype=np.float64)))
        ):
            return "coarse_align_preinsert_pose_unreachable"
        if env._has_window_collision():
            return "illegal_window_collision"
        if env._max_object_penetration() > env.plane_constraint_eps:
            return "glass_entered_insertion_region"
        if float(errors["current_normal_gap"]) < 0.02:
            return "normal_gap_not_safe"
        if (
            abs(
                float(errors["current_normal_gap"])
                - env.preinsert_normal_offset
            )
            > 0.015
        ):
            return "coarse_preinsert_gap_not_reached"
        if abs(float(errors["error_u"])) > env.coarse_translation_range:
            return "coarse_translation_target_not_reached"
        if abs(float(errors["error_v"])) > env.coarse_translation_range:
            return "coarse_translation_target_not_reached"
        if abs(float(errors["tilt_error_t1"])) > env.coarse_tilt_range_rad:
            return "coarse_rotation_target_not_reached"
        if abs(float(errors["tilt_error_t2"])) > env.coarse_tilt_range_rad:
            return "coarse_rotation_target_not_reached"
        if abs(float(errors["yaw_error"])) > env.coarse_yaw_range_rad:
            return "coarse_rotation_target_not_reached"
        if abs(float(errors["error_u"])) > env.max_translation_error:
            return "translation_not_recoverable"
        if abs(float(errors["error_v"])) > env.max_translation_error:
            return "translation_not_recoverable"
        if abs(float(errors["tilt_error_t1"])) > env.max_tilt_error_rad:
            return "tilt_not_recoverable"
        if abs(float(errors["tilt_error_t2"])) > env.max_tilt_error_rad:
            return "tilt_not_recoverable"
        if abs(float(errors["yaw_error"])) > env.max_yaw_error_rad:
            return "yaw_not_recoverable"
        if env._within_alignment_tolerance(errors):
            return "already_fine_aligned"
        return ""

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
            quaternion = self._slerp(
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
            delta_quaternion = self._matrix_to_quaternion(interpolated_delta)
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

    def _matrix_to_quaternion(self, rotation: np.ndarray) -> np.ndarray:
        env = self.env
        quaternion = np.empty(4, dtype=np.float64)
        env._mujoco.mju_mat2Quat(
            quaternion,
            project_to_rotation_matrix(rotation).reshape(9),
        )
        return env._normalize_vector(quaternion)

    def _slerp(
        self,
        first: np.ndarray,
        second: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        env = self.env
        first = env._normalize_vector(np.asarray(first, dtype=np.float64))
        second = env._normalize_vector(np.asarray(second, dtype=np.float64))
        dot = float(np.dot(first, second))
        if dot < 0.0:
            second = -second
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return env._normalize_vector(
                (1.0 - alpha) * first + alpha * second
            )
        angle = float(np.arccos(dot))
        sin_angle = float(np.sin(angle))
        return env._normalize_vector(
            np.sin((1.0 - alpha) * angle) / sin_angle * first
            + np.sin(alpha * angle) / sin_angle * second
        )

    def _empty_insert_diagnostics(self) -> dict[str, Any]:
        env = self.env
        return {
            "insert_start_error_u": float("nan"),
            "insert_start_error_v": float("nan"),
            "insert_start_tilt_t1": float("nan"),
            "insert_start_tilt_t2": float("nan"),
            "insert_start_yaw": float("nan"),
            "insert_start_normal_gap": float("nan"),
            "insert_max_abs_error_u": 0.0,
            "insert_max_abs_error_v": 0.0,
            "insert_max_abs_tilt_t1": 0.0,
            "insert_max_abs_tilt_t2": 0.0,
            "insert_max_abs_yaw": 0.0,
            "insert_alignment_violation_count": 0,
            "insert_alignment_violation_hold_steps": int(
                env.insert_alignment_violation_hold_steps
            ),
            "insert_success_hold_steps": int(env.insert_success_hold_steps),
            "insert_final_alignment_success_count": 0,
            "insert_final_verification_steps": 0,
            "insert_final_verification_max_steps": int(
                env.insert_final_verification_max_steps
            ),
            "insert_depth_tolerance": float(env.insert_depth_tolerance),
            "insert_final_depth_reached": False,
            "insert_final_command_finished": False,
            "insert_final_alignment_ok": False,
            "insert_action_sim_steps": int(env.insert_action_sim_steps),
            "insert_translation_abort_tolerance": float(
                env.insert_translation_abort_tolerance
            ),
            "insert_tilt_abort_tolerance": float(
                env.insert_tilt_abort_tolerance_rad
            ),
            "insert_yaw_abort_tolerance": float(
                env.insert_yaw_abort_tolerance_rad
            ),
            "insert_step_size": float(env.insert_step_size),
            "insert_steps": 0,
            "insert_failure_error_u": float("nan"),
            "insert_failure_error_v": float("nan"),
            "insert_failure_tilt_t1": float("nan"),
            "insert_failure_tilt_t2": float("nan"),
            "insert_failure_yaw": float("nan"),
            "insert_failure_normal_gap": float("nan"),
            "insert_failure_step": -1,
            "insert_failure_depth": float("nan"),
            "insert_ee_target_error": float("nan"),
            "insert_depth": 0.0,
            "insert_final_error_u": float("nan"),
            "insert_final_error_v": float("nan"),
            "insert_final_tilt_t1": float("nan"),
            "insert_final_tilt_t2": float("nan"),
            "insert_final_yaw": float("nan"),
            "insert_final_normal_gap": float("nan"),
            "hold_max_abs_error_u": 0.0,
            "hold_max_abs_error_v": 0.0,
            "hold_max_abs_tilt_t1": 0.0,
            "hold_max_abs_tilt_t2": 0.0,
            "hold_max_abs_yaw": 0.0,
            "hold_min_normal_gap": float("nan"),
            "hold_max_normal_gap": float("nan"),
            "assembly_verified": False,
            "assembly_alignment_ok": False,
            "assembly_depth_ok": False,
            "assembly_attachment_ok": False,
            "assembly_collision_ok": False,
            "final_error_u": float("nan"),
            "final_error_v": float("nan"),
            "final_tilt_error_t1": float("nan"),
            "final_tilt_error_t2": float("nan"),
            "final_yaw_error": float("nan"),
            "final_normal_gap": float("nan"),
            "failure_reason": "",
            "failure_category": "",
        }

    def get_last_insert_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_insert_diagnostics)

    def _initialize_insert_diagnostics(
        self,
        start_errors: Mapping[str, Any],
    ) -> None:
        self._last_insert_diagnostics = self._empty_insert_diagnostics()
        self._last_insert_diagnostics.update(
            {
                "insert_start_error_u": float(start_errors["error_u"]),
                "insert_start_error_v": float(start_errors["error_v"]),
                "insert_start_tilt_t1": float(start_errors["tilt_error_t1"]),
                "insert_start_tilt_t2": float(start_errors["tilt_error_t2"]),
                "insert_start_yaw": float(start_errors["yaw_error"]),
                "insert_start_normal_gap": float(
                    start_errors["current_normal_gap"]
                ),
            }
        )
        self._update_insert_error_diagnostics(start_errors)

    def _update_insert_error_diagnostics(
        self,
        errors: Mapping[str, Any],
    ) -> None:
        diagnostics = self._last_insert_diagnostics
        for diagnostic_key, error_key in (
            ("insert_max_abs_error_u", "error_u"),
            ("insert_max_abs_error_v", "error_v"),
            ("insert_max_abs_tilt_t1", "tilt_error_t1"),
            ("insert_max_abs_tilt_t2", "tilt_error_t2"),
            ("insert_max_abs_yaw", "yaw_error"),
        ):
            diagnostics[diagnostic_key] = max(
                float(diagnostics[diagnostic_key]), abs(float(errors[error_key]))
            )
        diagnostics["insert_ee_target_error"] = float(
            self.env._last_ee_target_error
        )

    def _record_insert_failure(
        self,
        errors: Mapping[str, Any],
        *,
        failure_reason: str,
        step_index: int,
        depth: float,
        failure_category: str = "",
    ) -> None:
        env = self.env
        self._last_insert_diagnostics.update(
            {
                "insert_failure_error_u": float(errors["error_u"]),
                "insert_failure_error_v": float(errors["error_v"]),
                "insert_failure_tilt_t1": float(errors["tilt_error_t1"]),
                "insert_failure_tilt_t2": float(errors["tilt_error_t2"]),
                "insert_failure_yaw": float(errors["yaw_error"]),
                "insert_failure_normal_gap": float(
                    errors["current_normal_gap"]
                ),
                "insert_failure_step": int(step_index),
                "insert_failure_depth": float(depth),
                "insert_ee_target_error": float(env._last_ee_target_error),
                "failure_reason": str(failure_reason),
                "failure_category": str(failure_category),
            }
        )

    def run_insert(self, *, max_steps: int = 300) -> bool:
        """Insert only after fine-align success, strictly along frame normal."""

        env = self.env
        if not env.ready_for_insert:
            return False
        if env.stage is not WindowAssemblyStage.FINE_ALIGN or not env._episode_done:
            return False
        env.stage = WindowAssemblyStage.INSERT
        env.stage_tracker.begin(env.stage, env._simulation_step_count)
        failure_reason = "insert_step_limit"
        failure_category = ""
        success = False
        start_errors = env.get_fine_alignment_errors()
        insert_start_center = env.get_object_position().copy()
        insert_start_rotation = env.get_object_rotation_matrix().copy()
        insert_start_gap = float(start_errors["current_normal_gap"])
        frame_normal = np.asarray(
            start_errors["frame_normal_world"], dtype=np.float64
        )
        target_gap = -abs(env.insert_final_gap)
        insert_direction = (
            -frame_normal if insert_start_gap >= target_gap else frame_normal
        )
        total_depth = abs(float(insert_start_gap - target_gap))
        commanded_depth = 0.0
        inserted_steps = 0
        alignment_violation_count = 0
        final_alignment_success_count = 0
        final_verification_steps = 0
        final_verification_started = False
        self._initialize_insert_diagnostics(start_errors)

        last_errors = start_errors
        for step_index in range(max(int(max_steps), 1)):
            at_final_command_depth = commanded_depth >= total_depth - 1e-12
            if not at_final_command_depth:
                current_gap = float(last_errors["current_normal_gap"])
                next_gap = max(
                    target_gap,
                    min(insert_start_gap, current_gap) - env.insert_step_size,
                )
                next_depth = float(
                    np.clip(insert_start_gap - next_gap, 0.0, total_depth)
                )
                if next_depth <= commanded_depth:
                    next_depth = float(
                        min(commanded_depth + env.insert_step_size, total_depth)
                    )
                commanded_depth = next_depth

            target_center = (
                insert_start_center + commanded_depth * insert_direction
            )
            self.move_attached_object_pose(
                target_center,
                insert_start_rotation,
                env.insert_action_sim_steps,
                center_feedback_gain=1.0,
            )
            inserted_steps += 1
            errors = env.get_fine_alignment_errors()
            last_errors = errors
            self._update_insert_error_diagnostics(errors)

            immediate_failure = self._insert_immediate_failure_reason(
                errors,
                target_center=target_center,
            )
            alignment_reason = self.get_insert_alignment_failure_reason(errors)
            alignment_violation_count, alignment_failure = (
                self._update_insert_alignment_violation_count(
                    errors,
                    alignment_violation_count,
                )
            )
            depth_reached = bool(
                float(errors["current_normal_gap"])
                <= target_gap + env.insert_depth_tolerance
            )
            command_finished = bool(commanded_depth >= total_depth - 1e-12)
            at_final_depth = bool(depth_reached and command_finished)
            if at_final_depth:
                final_verification_started = True
            if final_verification_started and command_finished:
                final_verification_steps += 1

            final_alignment_ok = self.insert_alignment_within_abort_tolerance(
                errors
            )
            if at_final_depth and final_alignment_ok:
                final_alignment_success_count += 1
            else:
                final_alignment_success_count = 0

            self._last_insert_diagnostics.update(
                {
                    "insert_steps": int(inserted_steps),
                    "insert_depth": float(commanded_depth),
                    "insert_alignment_violation_count": int(
                        alignment_violation_count
                    ),
                    "insert_final_alignment_success_count": int(
                        final_alignment_success_count
                    ),
                    "insert_final_verification_steps": int(
                        final_verification_steps
                    ),
                    "insert_final_depth_reached": bool(depth_reached),
                    "insert_final_command_finished": bool(command_finished),
                    "insert_final_alignment_ok": bool(final_alignment_ok),
                    "insert_final_error_u": float(errors["error_u"]),
                    "insert_final_error_v": float(errors["error_v"]),
                    "insert_final_tilt_t1": float(errors["tilt_error_t1"]),
                    "insert_final_tilt_t2": float(errors["tilt_error_t2"]),
                    "insert_final_yaw": float(errors["yaw_error"]),
                    "insert_final_normal_gap": float(
                        errors["current_normal_gap"]
                    ),
                }
            )

            if immediate_failure:
                failure_reason = immediate_failure
                self._record_insert_failure(
                    errors,
                    failure_reason=failure_reason,
                    step_index=step_index,
                    depth=commanded_depth,
                )
                break

            if alignment_failure:
                failure_reason = (
                    self._final_insert_failure_reason(alignment_failure)
                    if at_final_depth
                    else alignment_failure
                )
                failure_category = "alignment_degraded_during_insert"
                self._record_insert_failure(
                    errors,
                    failure_reason=failure_reason,
                    step_index=step_index,
                    depth=commanded_depth,
                    failure_category=failure_category,
                )
                break

            if final_alignment_success_count >= env.insert_success_hold_steps:
                success = True
                failure_reason = ""
                break

            if (
                final_verification_steps
                >= env.insert_final_verification_max_steps
            ):
                failure_reason = (
                    self._final_insert_failure_reason(alignment_reason)
                    if alignment_reason
                    else "insert_final_alignment_not_stable"
                )
                failure_category = "alignment_degraded_during_insert"
                self._record_insert_failure(
                    errors,
                    failure_reason=failure_reason,
                    step_index=step_index,
                    depth=commanded_depth,
                    failure_category=failure_category,
                )
                break
        else:
            self._record_insert_failure(
                last_errors,
                failure_reason=failure_reason,
                step_index=max(int(max_steps), 1),
                depth=commanded_depth,
                failure_category=failure_category,
            )
        env.insert_success = success
        if success:
            self._last_insert_diagnostics.update(
                {
                    "failure_reason": "",
                    "failure_category": "",
                    "insert_alignment_violation_count": int(
                        alignment_violation_count
                    ),
                    "insert_depth": float(commanded_depth),
                    "insert_ee_target_error": float(env._last_ee_target_error),
                }
            )
        env.stage_tracker.end(
            env._simulation_step_count,
            success=success,
            failure_reason=failure_reason or None,
        )
        return success

    def run_hold(self, *, hold_steps: int | None = None) -> bool:
        """Hold the inserted pose and verify its relative and absolute state."""

        env = self.env
        if not env.insert_success or env.stage is not WindowAssemblyStage.INSERT:
            return False
        env.stage = WindowAssemblyStage.HOLD
        env.stage_tracker.begin(env.stage, env._simulation_step_count)
        start_center = env.get_object_position().copy()
        start_rotation = env.get_object_rotation_matrix().copy()
        failure_reason = ""
        min_normal_gap = float("inf")
        max_normal_gap = float("-inf")
        steps = env.hold_steps if hold_steps is None else max(int(hold_steps), 1)
        for _ in range(steps):
            self.move_attached_object_pose(start_center, start_rotation, 1)
            errors = env.get_fine_alignment_errors()
            min_normal_gap = min(
                min_normal_gap, float(errors["current_normal_gap"])
            )
            max_normal_gap = max(
                max_normal_gap, float(errors["current_normal_gap"])
            )
            for diagnostic_key, error_key in (
                ("hold_max_abs_error_u", "error_u"),
                ("hold_max_abs_error_v", "error_v"),
                ("hold_max_abs_tilt_t1", "tilt_error_t1"),
                ("hold_max_abs_tilt_t2", "tilt_error_t2"),
                ("hold_max_abs_yaw", "yaw_error"),
            ):
                self._last_insert_diagnostics[diagnostic_key] = max(
                    float(self._last_insert_diagnostics[diagnostic_key]),
                    abs(float(errors[error_key])),
                )
            self._last_insert_diagnostics["hold_min_normal_gap"] = float(
                min_normal_gap
            )
            self._last_insert_diagnostics["hold_max_normal_gap"] = float(
                max_normal_gap
            )

            immediate_failure = self._insert_immediate_failure_reason(
                errors,
                target_center=start_center,
            )
            if immediate_failure:
                failure_reason = immediate_failure
                break
            absolute_failure = self.get_hold_assembly_failure_reason(errors)
            if absolute_failure:
                failure_reason = absolute_failure
                break
            if (
                np.linalg.norm(env.get_object_position() - start_center)
                > env.hold_position_tolerance
            ):
                failure_reason = "glass_slipped_during_hold"
                break
            rotation_drift = so3_log(
                start_rotation.T @ env.get_object_rotation_matrix()
            )
            if np.linalg.norm(rotation_drift) > env.hold_angle_tolerance_rad:
                failure_reason = "glass_rotated_during_hold"
                break
        env.hold_success = not failure_reason
        if failure_reason:
            self._last_insert_diagnostics.update(
                {
                    "failure_reason": failure_reason,
                    "failure_category": "hold_verification_failed",
                }
            )
        env.stage_tracker.end(
            env._simulation_step_count,
            success=env.hold_success,
            failure_reason=failure_reason or None,
        )
        return env.hold_success

    def verify_final_assembly(self) -> tuple[bool, dict[str, Any]]:
        """Re-read actual geometry and verify the final assembled state."""

        env = self.env
        errors = env.get_fine_alignment_errors()
        alignment_ok = self.insert_alignment_within_abort_tolerance(errors)
        depth_ok = bool(
            float(errors["current_normal_gap"])
            <= -abs(env.insert_final_gap) + env.insert_depth_tolerance
        )
        attachment_ok = bool(env._grasp_weld_is_active())
        collision_ok = not bool(env._window_collision_reason())
        verified = bool(
            alignment_ok and depth_ok and attachment_ok and collision_ok
        )
        diagnostics = {
            "assembly_verified": verified,
            "assembly_alignment_ok": bool(alignment_ok),
            "assembly_depth_ok": bool(depth_ok),
            "assembly_attachment_ok": bool(attachment_ok),
            "assembly_collision_ok": bool(collision_ok),
            "final_error_u": float(errors["error_u"]),
            "final_error_v": float(errors["error_v"]),
            "final_tilt_error_t1": float(errors["tilt_error_t1"]),
            "final_tilt_error_t2": float(errors["tilt_error_t2"]),
            "final_yaw_error": float(errors["yaw_error"]),
            "final_normal_gap": float(errors["current_normal_gap"]),
        }
        diagnostics["failure_reason"] = self._final_assembly_failure_reason(
            errors,
            alignment_ok=alignment_ok,
            depth_ok=depth_ok,
            attachment_ok=attachment_ok,
            collision_ok=collision_ok,
        )
        self._last_insert_diagnostics.update(diagnostics)
        return verified, diagnostics

    def insert_alignment_within_abort_tolerance(
        self,
        errors: Mapping[str, Any],
    ) -> bool:
        return not bool(self.get_insert_alignment_failure_reason(errors))

    def get_insert_alignment_failure_reason(
        self,
        errors: Mapping[str, Any],
    ) -> str:
        env = self.env
        ratios = {
            "insert_error_u_exceeded": abs(float(errors["error_u"]))
            / env.insert_translation_abort_tolerance,
            "insert_error_v_exceeded": abs(float(errors["error_v"]))
            / env.insert_translation_abort_tolerance,
            "insert_tilt_t1_exceeded": abs(float(errors["tilt_error_t1"]))
            / env.insert_tilt_abort_tolerance_rad,
            "insert_tilt_t2_exceeded": abs(float(errors["tilt_error_t2"]))
            / env.insert_tilt_abort_tolerance_rad,
            "insert_yaw_exceeded": abs(float(errors["yaw_error"]))
            / env.insert_yaw_abort_tolerance_rad,
        }
        exceeded = {
            reason: ratio
            for reason, ratio in ratios.items()
            if np.isfinite(ratio) and ratio >= 1.0
        }
        if not exceeded:
            return ""
        return max(exceeded.items(), key=lambda item: item[1])[0]

    def get_hold_assembly_failure_reason(
        self,
        errors: Mapping[str, Any],
    ) -> str:
        env = self.env
        insert_reason = self.get_insert_alignment_failure_reason(errors)
        if insert_reason:
            return {
                "insert_error_u_exceeded": "hold_error_u_exceeded",
                "insert_error_v_exceeded": "hold_error_v_exceeded",
                "insert_tilt_t1_exceeded": "hold_tilt_t1_exceeded",
                "insert_tilt_t2_exceeded": "hold_tilt_t2_exceeded",
                "insert_yaw_exceeded": "hold_yaw_exceeded",
            }[insert_reason]
        if (
            float(errors["current_normal_gap"])
            > -abs(env.insert_final_gap) + env.insert_depth_tolerance
        ):
            return "hold_insert_depth_lost"
        return ""

    @staticmethod
    def _final_insert_failure_reason(failure_reason: str) -> str:
        return failure_reason.replace("insert_", "insert_final_", 1)

    def _final_assembly_failure_reason(
        self,
        errors: Mapping[str, Any],
        *,
        alignment_ok: bool,
        depth_ok: bool,
        attachment_ok: bool,
        collision_ok: bool,
    ) -> str:
        if not alignment_ok:
            insert_reason = self.get_insert_alignment_failure_reason(errors)
            if insert_reason:
                return insert_reason.replace("insert_", "final_assembly_", 1)
            return "final_assembly_alignment_invalid"
        if not depth_ok:
            return "final_assembly_depth_invalid"
        if not attachment_ok:
            return "final_assembly_attachment_invalid"
        if not collision_ok:
            return "final_assembly_collision_invalid"
        return ""

    def _update_insert_alignment_violation_count(
        self,
        errors: Mapping[str, Any],
        current_count: int,
    ) -> tuple[int, str]:
        failure_reason = self.get_insert_alignment_failure_reason(errors)
        if not failure_reason:
            return 0, ""
        next_count = int(current_count) + 1
        if next_count >= self.env.insert_alignment_violation_hold_steps:
            return next_count, failure_reason
        return next_count, ""

    def _insert_immediate_failure_reason(
        self,
        errors: Mapping[str, Any],
        *,
        target_center: np.ndarray,
    ) -> str:
        env = self.env
        scalar_keys = (
            "error_u",
            "error_v",
            "tilt_error_t1",
            "tilt_error_t2",
            "yaw_error",
            "normal_gap_error",
            "current_normal_gap",
        )
        if (
            not all(np.isfinite(float(errors[key])) for key in scalar_keys)
            or not np.all(
                np.isfinite(np.asarray(target_center, dtype=np.float64))
            )
        ):
            return "nan_or_inf"
        if not env._grasp_weld_is_active():
            return "suction_weld_broken"
        if float(np.asarray(errors["glass_center_world"])[2]) < 0.10:
            return "glass_dropped"
        collision_reason = env._window_collision_reason()
        if collision_reason:
            return collision_reason
        if env._joint_limit_violated():
            return "robot_joint_limit"
        if env._last_ee_target_error > env.unreachable_position_threshold:
            return "end_effector_unreachable"
        return ""
