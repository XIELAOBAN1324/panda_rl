"""Vision-free reinforcement-learning environment for window fine alignment.

Only the five in-plane/rotational alignment degrees of freedom are exposed to
the policy. Pickup, transport, coarse alignment, insertion, and holding are
scripted and are represented explicitly by :class:`WindowAssemblyStage`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from panda_mujoco_gym.envs.pick_and_place_window import FrankaPickAndPlaceWindowEnv
from panda_mujoco_gym.envs.window_assembly_geometry import (
    alignment_errors_from_corners,
    fine_alignment_costs,
    ordered_rectangle_corners,
    project_to_rotation_matrix,
    so3_exp,
    so3_log,
)
from panda_mujoco_gym.envs.window_assembly_pipeline import (
    WindowAssemblyStage,
    WindowAssemblyStageTracker,
)


class FrankaWindowFineAlignEnv(FrankaPickAndPlaceWindowEnv):
    """Five-DoF dense-reward environment for the fine-alignment stage only."""

    # The suction cup grasps the XML glass from its local +normal side.  The
    # assembly-facing directed rectangle therefore uses local axes
    # ``(+x, -y, -z)``.  This is a fixed physical convention, not dynamic
    # nearest-corner matching.
    GLASS_ASSEMBLY_FRAME = np.diag([1.0, -1.0, -1.0])

    def __init__(self, reward_type: str = "dense", **kwargs: Any) -> None:
        if reward_type != "dense":
            raise ValueError("FrankaWindowFineAlignEnv supports reward_type='dense' only")

        internal_max_steps = kwargs.pop("fine_align_max_episode_steps", None)
        registered_max_steps = kwargs.pop("max_episode_steps", 100)
        self.max_episode_steps = int(
            registered_max_steps if internal_max_steps is None else internal_max_steps
        )
        self.coarse_translation_range = float(kwargs.pop("coarse_translation_range", 0.015))
        self.coarse_tilt_range_rad = np.deg2rad(float(kwargs.pop("coarse_tilt_range_deg", 2.0)))
        self.coarse_yaw_range_rad = np.deg2rad(float(kwargs.pop("coarse_yaw_range_deg", 4.0)))
        self.coarse_normal_gap = float(kwargs.pop("coarse_normal_gap", 0.08))
        self.coarse_sampling_attempts = int(kwargs.pop("coarse_sampling_attempts", 24))

        self.fine_position_action_scale = float(kwargs.pop("fine_position_action_scale", 0.0015))
        self.fine_tilt_action_scale_rad = np.deg2rad(
            float(kwargs.pop("fine_tilt_action_scale_deg", 0.25))
        )
        self.fine_yaw_action_scale_rad = np.deg2rad(
            float(kwargs.pop("fine_yaw_action_scale_deg", 0.4))
        )

        self.translation_tolerance = float(kwargs.pop("translation_tolerance", 0.002))
        self.tilt_tolerance_rad = np.deg2rad(float(kwargs.pop("tilt_tolerance_deg", 0.5)))
        self.yaw_tolerance_rad = np.deg2rad(float(kwargs.pop("yaw_tolerance_deg", 0.5)))
        self.normal_gap_tolerance = float(kwargs.pop("normal_gap_tolerance", 0.001))
        self.success_hold_steps = int(kwargs.pop("success_hold_steps", 5))

        self.max_translation_error = float(kwargs.pop("max_translation_error", 0.05))
        self.max_tilt_error_rad = np.deg2rad(float(kwargs.pop("max_tilt_error_deg", 8.0)))
        self.max_yaw_error_rad = np.deg2rad(float(kwargs.pop("max_yaw_error_deg", 15.0)))
        self.max_normal_gap_drift = float(kwargs.pop("max_normal_gap_drift", 0.005))
        self.unreachable_position_threshold = float(
            kwargs.pop("unreachable_position_threshold", 0.03)
        )

        self.measurement_noise_translation_std = float(
            kwargs.pop("measurement_noise_translation_std", 0.0005)
        )
        self.measurement_noise_angle_std_rad = np.deg2rad(
            float(kwargs.pop("measurement_noise_angle_std_deg", 0.1))
        )
        self.reward_uses_ground_truth = bool(kwargs.pop("reward_uses_ground_truth", True))
        self.success_uses_ground_truth = bool(kwargs.pop("success_uses_ground_truth", True))
        self.observation_clip = float(kwargs.pop("observation_clip", 10.0))

        self.insert_step_size = float(kwargs.pop("insert_step_size", 0.001))
        self.insert_final_gap = float(kwargs.pop("insert_final_gap", 0.005))
        self.hold_steps = int(kwargs.pop("hold_steps", 40))
        self.hold_position_tolerance = float(kwargs.pop("hold_position_tolerance", 0.003))
        self.hold_angle_tolerance_rad = np.deg2rad(
            float(kwargs.pop("hold_angle_tolerance_deg", 1.0))
        )

        for name in (
            "max_episode_steps",
            "coarse_sampling_attempts",
            "success_hold_steps",
            "hold_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "coarse_translation_range",
            "coarse_tilt_range_rad",
            "coarse_yaw_range_rad",
            "coarse_normal_gap",
            "fine_position_action_scale",
            "fine_tilt_action_scale_rad",
            "fine_yaw_action_scale_rad",
            "translation_tolerance",
            "tilt_tolerance_rad",
            "yaw_tolerance_rad",
            "normal_gap_tolerance",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

        self.previous_action = np.zeros(5, dtype=np.float32)
        self.initial_normal_gap = 0.0
        self._previous_state_cost = 0.0
        self._success_counter = 0
        self._elapsed_fine_steps = 0
        self._simulation_step_count = 0
        self._last_ee_target_error = 0.0
        self._episode_done = False
        self.ready_for_insert = False
        self.insert_success = False
        self.hold_success = False
        self.stage = WindowAssemblyStage.APPROACH
        self.stage_tracker = WindowAssemblyStageTracker()
        self.coarse_alignment_sample = np.zeros(5, dtype=np.float64)
        self.pipeline_frame_callback: Any | None = None

        kwargs.pop("reset_mode", None)
        kwargs.pop("scripted_pickup", None)
        kwargs.pop("constrain_robot_to_window_plane", None)
        super().__init__(
            reward_type=reward_type,
            reset_mode="full_task",
            scripted_pickup=True,
            constrain_robot_to_window_plane=False,
            terminate_on_success=False,
            **kwargs,
        )

        # GoalEnv validates its internal Dict observation during ``super().reset``.
        # Keep that private space for the reset call, while exposing only the
        # policy's flat fine-alignment state to users and Stable-Baselines3.
        self._robot_observation_space = self.observation_space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self._fine_observation_space = spaces.Box(
            low=-self.observation_clip,
            high=self.observation_clip,
            shape=(11,),
            dtype=np.float32,
        )
        self.observation_space = self._fine_observation_space

    # ------------------------------------------------------------------
    # Ordered physical geometry
    # ------------------------------------------------------------------
    def get_glass_corners_world(self) -> np.ndarray:
        """Return ordered glass corners with shape ``(4, 3)``."""

        center = self.get_object_position().copy()
        rotation = project_to_rotation_matrix(
            self.get_object_rotation_matrix() @ self.GLASS_ASSEMBLY_FRAME
        )
        return ordered_rectangle_corners(
            center,
            rotation,
            float(self.glass_half_extents[0]),
            float(self.glass_half_extents[1]),
        )

    def get_frame_corners_world(self) -> np.ndarray:
        """Return ordered opening corners with shape ``(4, 3)``.

        The XML frame's local axes are ``normal=+x``, ``width=-y``, and
        ``height=+z``. They are transformed by the actual body orientation;
        no world-axis alignment is assumed.
        """

        center = np.asarray(self.data.xpos[self.window_body_id], dtype=np.float64).copy()
        body_rotation = project_to_rotation_matrix(
            np.asarray(self.data.xmat[self.window_body_id], dtype=np.float64).reshape(3, 3)
        )
        frame_rotation = np.column_stack(
            [body_rotation[:, 2], -body_rotation[:, 1], body_rotation[:, 0]]
        )
        return ordered_rectangle_corners(
            center,
            frame_rotation,
            float(self.window_opening_size[0] * 0.5),
            float(self.window_opening_size[1] * 0.5),
        )

    def get_fine_alignment_errors(self) -> dict[str, np.ndarray | float]:
        """Return ground-truth five-DoF errors plus normal-gap diagnostics."""

        errors = alignment_errors_from_corners(
            self.get_glass_corners_world(), self.get_frame_corners_world()
        )
        center_delta = np.asarray(errors["glass_center_world"]) - np.asarray(
            errors["frame_center_world"]
        )
        current_gap = float(np.dot(np.asarray(errors["frame_normal_world"]), center_delta))
        errors["initial_normal_gap"] = float(self.initial_normal_gap)
        errors["current_normal_gap"] = current_gap
        errors["normal_gap_drift"] = current_gap - float(self.initial_normal_gap)
        return errors

    # ------------------------------------------------------------------
    # Reset: scripted approach -> coarse_align, then expose fine_align
    # ------------------------------------------------------------------
    def _reset_sim(self) -> bool:
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self.data.qacc[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        if self.model.na != 0:
            self.data.act[:] = 0.0
        self.set_joint_neutral()
        self.data.ctrl[:] = 0.0
        self.data.ctrl[:7] = self.neutral_joint_values[:7]
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._set_grasp_weld_active(False)
        self._last_collision = False
        self._last_collision_with_window = False
        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0
        self._simulation_step_count = 0
        self.stage_tracker = WindowAssemblyStageTracker()

        self.window_center = self._sample_window_center()
        self.model.body_pos[self.window_body_id] = self.window_center
        self._set_target_visibility()
        self._sample_object()
        self._mujoco.mj_forward(self.model, self.data)
        self._settle_object()
        self._mujoco.mj_forward(self.model, self.data)

        frame_errors = alignment_errors_from_corners(
            self.get_glass_corners_world(), self.get_frame_corners_world()
        )
        frame_rotation = np.asarray(frame_errors["frame_rotation_world"], dtype=np.float64)
        self.window_normal = frame_rotation[:, 2].copy()
        self.target_glass_normal = frame_rotation[:, 2].copy()
        self.window_center = np.asarray(frame_errors["frame_center_world"], dtype=np.float64).copy()
        self.window_safe_goal = self.window_center - self.window_normal * self.coarse_normal_gap

        object_position = self.get_object_position().copy()
        approach_position = object_position + frame_rotation[:, 0] * 0.12
        self._run_scripted_stage(
            WindowAssemblyStage.APPROACH,
            lambda: self._move_mocap_pose(approach_position, self.grasp_site_pose, self.post_grasp_motion_steps),
        )
        descend_position = object_position + frame_rotation[:, 0] * 0.01
        self._run_scripted_stage(
            WindowAssemblyStage.DESCEND,
            lambda: self._move_mocap_pose(descend_position, self.grasp_site_pose, self.post_grasp_motion_steps),
        )

        def suction_grasp() -> None:
            self._activate_grasp_weld()
            self._mujoco_step()
            self._simulation_step_count += 1
            self._mujoco.mj_forward(self.model, self.data)
            self._emit_pipeline_frame()
            if not self._grasp_weld_is_active():
                raise RuntimeError("Scripted suction weld did not activate")

        self._run_scripted_stage(WindowAssemblyStage.SUCTION_GRASP, suction_grasp)

        lift_height = float(
            max(
                self.window_center[2] + self.post_grasp_canonical_goal_margin,
                self.initial_object_height + self.post_grasp_min_lift_above_object,
            )
        )
        lift_position = self.get_object_position().copy()
        lift_position[2] = lift_height
        self._run_scripted_stage(
            WindowAssemblyStage.LIFT,
            lambda: self._move_attached_object_pose(
                lift_position, self.get_object_rotation_matrix(), 2 * self.post_grasp_motion_steps
            ),
        )

        transport_position = self.window_center - self.window_normal * max(
            self.coarse_normal_gap + 0.12, 0.16
        )
        transport_position[2] = max(
            float(lift_position[2]), float(self.window_center[2] + 0.12)
        )

        def reorient() -> None:
            self._move_attached_object_pose(
                transport_position,
                self.get_object_rotation_matrix(),
                3 * self.post_grasp_motion_steps,
            )
            self._move_attached_object_pose(
                transport_position,
                frame_rotation @ self.GLASS_ASSEMBLY_FRAME,
                3 * self.post_grasp_motion_steps,
            )

        self._run_scripted_stage(WindowAssemblyStage.REORIENT, reorient)
        self._run_scripted_stage(
            WindowAssemblyStage.COARSE_ALIGN,
            lambda: self._sample_and_apply_coarse_alignment(frame_rotation),
        )
        self._stabilize_reset_state()
        return True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.observation_space = self._robot_observation_space
        try:
            super().reset(seed=seed, options=options)
        finally:
            self.observation_space = self._fine_observation_space
        self.previous_action = np.zeros(5, dtype=np.float32)
        self._success_counter = 0
        self._elapsed_fine_steps = 0
        self._episode_done = False
        self.ready_for_insert = False
        self.insert_success = False
        self.hold_success = False

        raw_errors = alignment_errors_from_corners(
            self.get_glass_corners_world(), self.get_frame_corners_world()
        )
        center_delta = np.asarray(raw_errors["glass_center_world"]) - np.asarray(
            raw_errors["frame_center_world"]
        )
        self.initial_normal_gap = float(
            np.dot(np.asarray(raw_errors["frame_normal_world"]), center_delta)
        )
        errors = self.get_fine_alignment_errors()
        measured_errors = self._noisy_errors(errors)
        reward_errors = errors if self.reward_uses_ground_truth else measured_errors
        self._previous_state_cost = float(self._costs(reward_errors)["state_cost"])

        self.stage = WindowAssemblyStage.FINE_ALIGN
        self.stage_tracker.begin(self.stage, self._simulation_step_count)
        observation = self._policy_observation(measured_errors, self.previous_action)
        info = self._diagnostic_info(errors)
        info.update(
            {
                "is_success": False,
                "fine_align_success": False,
                "ready_for_insert": False,
                "success_hold_count": 0,
                "coarse_alignment_sample": self.coarse_alignment_sample.copy(),
                "stage_records": self.stage_tracker.as_dicts(),
            }
        )
        return observation, info

    def _run_scripted_stage(self, stage: WindowAssemblyStage, operation: Any) -> None:
        self.stage = stage
        self.stage_tracker.begin(stage, self._simulation_step_count)
        try:
            operation()
        except Exception as exc:
            self.stage_tracker.end(
                self._simulation_step_count, success=False, failure_reason=str(exc)
            )
            raise
        self.stage_tracker.end(self._simulation_step_count, success=True)

    def _sample_and_apply_coarse_alignment(self, frame_rotation: np.ndarray) -> None:
        canonical_state = self._capture_sim_state()
        frame_center = np.mean(self.get_frame_corners_world(), axis=0)
        last_reason = "no_candidate"
        last_errors: Mapping[str, Any] | None = None
        for _ in range(self.coarse_sampling_attempts):
            self._restore_sim_state(canonical_state)
            sample = np.array(
                [
                    self.np_random.uniform(-self.coarse_translation_range, self.coarse_translation_range),
                    self.np_random.uniform(-self.coarse_translation_range, self.coarse_translation_range),
                    self.np_random.uniform(-self.coarse_tilt_range_rad, self.coarse_tilt_range_rad),
                    self.np_random.uniform(-self.coarse_tilt_range_rad, self.coarse_tilt_range_rad),
                    self.np_random.uniform(-self.coarse_yaw_range_rad, self.coarse_yaw_range_rad),
                ],
                dtype=np.float64,
            )
            target_center = (
                frame_center
                - self.coarse_normal_gap * frame_rotation[:, 2]
                + sample[0] * frame_rotation[:, 0]
                + sample[1] * frame_rotation[:, 1]
            )
            target_assembly_rotation = frame_rotation @ so3_exp(sample[2:5])
            target_rotation = target_assembly_rotation @ self.GLASS_ASSEMBLY_FRAME
            self._move_attached_object_pose(
                target_center, target_rotation, 5 * self.post_grasp_motion_steps
            )
            self._stabilize_reset_state()
            errors = self.get_fine_alignment_errors()
            errors["normal_gap_drift"] = 0.0
            last_errors = errors
            last_reason = self._coarse_candidate_failure(errors)
            if not last_reason:
                self.coarse_alignment_sample = sample
                return
        self._restore_sim_state(canonical_state)
        diagnostics = ""
        if last_errors is not None:
            diagnostics = (
                f" (u={float(last_errors['error_u']):.4f}, "
                f"v={float(last_errors['error_v']):.4f}, "
                f"gap={float(last_errors['current_normal_gap']):.4f}, "
                f"tilt=({float(last_errors['tilt_error_t1']):.4f},"
                f"{float(last_errors['tilt_error_t2']):.4f}), "
                f"yaw={float(last_errors['yaw_error']):.4f})"
            )
        raise RuntimeError(
            f"Unable to sample a recoverable coarse-alignment state: {last_reason}{diagnostics}"
        )

    def _coarse_candidate_failure(self, errors: Mapping[str, Any]) -> str:
        if not self._grasp_weld_is_active():
            return "suction_weld_broken"
        if self._has_window_collision():
            return "illegal_window_collision"
        if self._max_object_penetration() > self.plane_constraint_eps:
            return "glass_entered_insertion_region"
        if abs(float(errors["current_normal_gap"])) < 0.02:
            return "normal_gap_not_safe"
        if abs(float(errors["current_normal_gap"]) + self.coarse_normal_gap) > 0.015:
            return "coarse_normal_gap_not_reached"
        if abs(float(errors["error_u"])) > self.coarse_translation_range:
            return "coarse_translation_target_not_reached"
        if abs(float(errors["error_v"])) > self.coarse_translation_range:
            return "coarse_translation_target_not_reached"
        if abs(float(errors["tilt_error_t1"])) > self.coarse_tilt_range_rad:
            return "coarse_rotation_target_not_reached"
        if abs(float(errors["tilt_error_t2"])) > self.coarse_tilt_range_rad:
            return "coarse_rotation_target_not_reached"
        if abs(float(errors["yaw_error"])) > self.coarse_yaw_range_rad:
            return "coarse_rotation_target_not_reached"
        if abs(float(errors["error_u"])) > self.max_translation_error:
            return "translation_not_recoverable"
        if abs(float(errors["error_v"])) > self.max_translation_error:
            return "translation_not_recoverable"
        if abs(float(errors["tilt_error_t1"])) > self.max_tilt_error_rad:
            return "tilt_not_recoverable"
        if abs(float(errors["tilt_error_t2"])) > self.max_tilt_error_rad:
            return "tilt_not_recoverable"
        if abs(float(errors["yaw_error"])) > self.max_yaw_error_rad:
            return "yaw_not_recoverable"
        if self._within_alignment_tolerance(errors):
            return "already_fine_aligned"
        return ""

    # ------------------------------------------------------------------
    # RL step: five fine-alignment controls, never insertion
    # ------------------------------------------------------------------
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (5,):
            raise ValueError(f"Expected action shape (5,), got {action_array.shape}")
        if self._episode_done or self.stage is not WindowAssemblyStage.FINE_ALIGN:
            raise RuntimeError("step() is only available during an active fine_align episode")

        old_action = self.previous_action.copy()
        finite_action = bool(np.all(np.isfinite(action_array)))
        clipped_action = np.clip(action_array, -1.0, 1.0) if finite_action else np.zeros(5, dtype=np.float32)
        target_center = self.get_object_position().copy()
        if finite_action and self._grasp_weld_is_active():
            errors_before = self.get_fine_alignment_errors()
            frame_rotation = np.asarray(errors_before["frame_rotation_world"], dtype=np.float64)
            translation_delta = (
                clipped_action[0] * self.fine_position_action_scale * frame_rotation[:, 0]
                + clipped_action[1] * self.fine_position_action_scale * frame_rotation[:, 1]
            )
            normal_component = abs(float(np.dot(translation_delta, frame_rotation[:, 2])))
            if normal_component > 1e-10:
                raise RuntimeError("Fine-alignment translation contains a normal component")
            rotation_vector_world = (
                clipped_action[2] * self.fine_tilt_action_scale_rad * frame_rotation[:, 0]
                + clipped_action[3] * self.fine_tilt_action_scale_rad * frame_rotation[:, 1]
                + clipped_action[4] * self.fine_yaw_action_scale_rad * frame_rotation[:, 2]
            )
            target_center = target_center + translation_delta
            target_rotation = so3_exp(rotation_vector_world) @ self.get_object_rotation_matrix()
            self._move_attached_object_pose(target_center, target_rotation, 1)

        self._elapsed_fine_steps += 1
        ground_truth_errors = self.get_fine_alignment_errors()
        measured_errors = self._noisy_errors(ground_truth_errors)
        reward_errors = ground_truth_errors if self.reward_uses_ground_truth else measured_errors
        success_errors = ground_truth_errors if self.success_uses_ground_truth else measured_errors
        costs = self._costs(reward_errors)
        current_state_cost = float(costs["state_cost"])
        progress_reward = 2.0 * float(
            np.clip(self._previous_state_cost - current_state_cost, -0.5, 0.5)
        )
        state_penalty = -0.1 * current_state_cost
        action_penalty = -0.005 * float(np.sum(np.square(clipped_action)))
        smoothness_penalty = -0.01 * float(np.sum(np.square(clipped_action - old_action)))

        within_tolerance = self._within_alignment_tolerance(success_errors)
        self._success_counter = self._success_counter + 1 if within_tolerance else 0
        fine_align_success = self._success_counter >= self.success_hold_steps
        failure_reason = self._safety_failure_reason(
            ground_truth_errors,
            action_was_finite=finite_action,
            target_center=target_center,
        )
        safety_failure = bool(failure_reason)
        terminal_reward = 10.0 if fine_align_success else (-10.0 if safety_failure else 0.0)
        reward = (
            progress_reward
            + state_penalty
            + action_penalty
            + smoothness_penalty
            + terminal_reward
        )

        terminated = bool(fine_align_success or safety_failure)
        truncated = bool(
            not terminated and self._elapsed_fine_steps >= self.max_episode_steps
        )
        self.ready_for_insert = bool(fine_align_success)
        self.previous_action = clipped_action.copy()
        self._previous_state_cost = current_state_cost
        self._episode_done = bool(terminated or truncated)

        if self._episode_done:
            self.stage_tracker.end(
                self._simulation_step_count,
                success=fine_align_success,
                failure_reason=failure_reason or ("time_limit" if truncated else None),
            )

        observation = self._policy_observation(measured_errors, self.previous_action)
        info = self._diagnostic_info(ground_truth_errors)
        info.update(
            {
                "is_success": bool(fine_align_success),
                "fine_align_success": bool(fine_align_success),
                "ready_for_insert": bool(self.ready_for_insert),
                "failure_reason": failure_reason or ("time_limit" if truncated else ""),
                "success_hold_count": int(self._success_counter),
                "within_alignment_tolerance": bool(within_tolerance),
                "reward_progress": float(progress_reward),
                "reward_state": float(state_penalty),
                "reward_action": float(action_penalty),
                "reward_smoothness": float(smoothness_penalty),
                "reward_terminal": float(terminal_reward),
                "mean_alignment_cost": float(costs["mean_alignment_cost"]),
                "worst_alignment_cost": float(costs["worst_alignment_cost"]),
                "alignment_cost": float(costs["alignment_cost"]),
                "gap_cost": float(costs["gap_cost"]),
                "state_cost": current_state_cost,
                "stage_records": self.stage_tracker.as_dicts(),
            }
        )
        return observation, float(reward), terminated, truncated, info

    def _costs(self, errors: Mapping[str, Any]) -> dict[str, float | np.ndarray]:
        return fine_alignment_costs(
            errors,
            translation_tolerance=self.translation_tolerance,
            tilt_tolerance_rad=self.tilt_tolerance_rad,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
            normal_gap_tolerance=self.normal_gap_tolerance,
        )

    def _within_alignment_tolerance(self, errors: Mapping[str, Any]) -> bool:
        return bool(
            abs(float(errors["error_u"])) < self.translation_tolerance
            and abs(float(errors["error_v"])) < self.translation_tolerance
            and abs(float(errors["tilt_error_t1"])) < self.tilt_tolerance_rad
            and abs(float(errors["tilt_error_t2"])) < self.tilt_tolerance_rad
            and abs(float(errors["yaw_error"])) < self.yaw_tolerance_rad
            and abs(float(errors["normal_gap_drift"])) < self.normal_gap_tolerance
        )

    def _safety_failure_reason(
        self,
        errors: Mapping[str, Any],
        *,
        action_was_finite: bool,
        target_center: np.ndarray,
    ) -> str:
        scalar_keys = (
            "error_u",
            "error_v",
            "tilt_error_t1",
            "tilt_error_t2",
            "yaw_error",
            "normal_gap_drift",
            "current_normal_gap",
        )
        if not action_was_finite or not all(np.isfinite(float(errors[key])) for key in scalar_keys):
            return "nan_or_inf"
        if not self._grasp_weld_is_active():
            return "suction_weld_broken"
        if float(np.asarray(errors["glass_center_world"])[2]) < 0.10:
            return "glass_dropped"
        collision_reason = self._window_collision_reason()
        if collision_reason:
            return collision_reason
        if abs(float(errors["normal_gap_drift"])) > self.max_normal_gap_drift:
            return "normal_gap_drift_exceeded"
        if max(abs(float(errors["error_u"])), abs(float(errors["error_v"]))) > self.max_translation_error:
            return "translation_error_exceeded"
        if max(
            abs(float(errors["tilt_error_t1"])), abs(float(errors["tilt_error_t2"]))
        ) > self.max_tilt_error_rad:
            return "tilt_error_exceeded"
        if abs(float(errors["yaw_error"])) > self.max_yaw_error_rad:
            return "yaw_error_exceeded"
        if self._joint_limit_violated():
            return "robot_joint_limit"
        if self._last_ee_target_error > self.unreachable_position_threshold:
            return "end_effector_unreachable"
        if self._max_object_penetration() > self.plane_constraint_eps:
            return "glass_entered_insertion_region"
        if float(errors["current_normal_gap"]) > -0.01:
            return "glass_entered_insertion_region"
        if not np.all(np.isfinite(np.asarray(target_center))):
            return "end_effector_unreachable"
        return ""

    # ------------------------------------------------------------------
    # Script-only insertion and hold. Neither method accepts a policy action.
    # ------------------------------------------------------------------
    def run_scripted_insert(self, *, max_steps: int = 200) -> bool:
        """Insert only after fine-align success, moving strictly along frame normal."""

        if not self.ready_for_insert:
            return False
        if self.stage is not WindowAssemblyStage.FINE_ALIGN or not self._episode_done:
            return False
        self.stage = WindowAssemblyStage.INSERT
        self.stage_tracker.begin(self.stage, self._simulation_step_count)
        failure_reason = "insert_step_limit"
        success = False
        start_errors = self.get_fine_alignment_errors()
        insert_start_center = self.get_object_position().copy()
        insert_start_rotation = self.get_object_rotation_matrix().copy()
        insert_start_gap = float(start_errors["current_normal_gap"])
        frame_normal = np.asarray(start_errors["frame_normal_world"], dtype=np.float64)
        commanded_progress = 0.0
        for _ in range(max(int(max_steps), 1)):
            errors = self.get_fine_alignment_errors()
            if not self._grasp_weld_is_active():
                failure_reason = "suction_weld_broken"
                break
            if not self._alignment_only_within_tolerance(errors):
                failure_reason = "alignment_degraded_during_insert"
                break
            collision_reason = self._window_collision_reason()
            if collision_reason:
                failure_reason = collision_reason
                break
            current_gap = float(errors["current_normal_gap"])
            target_gap = -abs(self.insert_final_gap)
            if current_gap >= target_gap - 5e-4:
                success = True
                failure_reason = ""
                break
            total_distance = target_gap - insert_start_gap
            commanded_progress = min(commanded_progress + self.insert_step_size, total_distance)
            # The commanded path has constant t1/t2 coordinates and constant
            # orientation. Feedback merely holds those values against tracking
            # drift; the script introduces no new in-plane/rotation command.
            target_center = insert_start_center + commanded_progress * frame_normal
            self._move_attached_object_pose(
                target_center, insert_start_rotation, 1
            )
        self.insert_success = success
        self.stage_tracker.end(
            self._simulation_step_count,
            success=success,
            failure_reason=failure_reason or None,
        )
        return success

    def run_scripted_hold(self, *, hold_steps: int | None = None) -> bool:
        """Hold the inserted pose and check attachment, slip, and collision."""

        if not self.insert_success or self.stage is not WindowAssemblyStage.INSERT:
            return False
        self.stage = WindowAssemblyStage.HOLD
        self.stage_tracker.begin(self.stage, self._simulation_step_count)
        start_center = self.get_object_position().copy()
        start_rotation = self.get_object_rotation_matrix().copy()
        failure_reason = ""
        for _ in range(self.hold_steps if hold_steps is None else max(int(hold_steps), 1)):
            self._move_attached_object_pose(start_center, start_rotation, 1)
            if not self._grasp_weld_is_active():
                failure_reason = "suction_weld_broken"
                break
            if self._window_collision_reason():
                failure_reason = "persistent_window_collision"
                break
            if np.linalg.norm(self.get_object_position() - start_center) > self.hold_position_tolerance:
                failure_reason = "glass_slipped_during_hold"
                break
            rotation_drift = so3_log(start_rotation.T @ self.get_object_rotation_matrix())
            if np.linalg.norm(rotation_drift) > self.hold_angle_tolerance_rad:
                failure_reason = "glass_rotated_during_hold"
                break
        self.hold_success = not failure_reason
        self.stage_tracker.end(
            self._simulation_step_count,
            success=self.hold_success,
            failure_reason=failure_reason or None,
        )
        return self.hold_success

    def _alignment_only_within_tolerance(self, errors: Mapping[str, Any]) -> bool:
        return bool(
            abs(float(errors["error_u"])) < self.translation_tolerance
            and abs(float(errors["error_v"])) < self.translation_tolerance
            and abs(float(errors["tilt_error_t1"])) < self.tilt_tolerance_rad
            and abs(float(errors["tilt_error_t2"])) < self.tilt_tolerance_rad
            and abs(float(errors["yaw_error"])) < self.yaw_tolerance_rad
        )

    # ------------------------------------------------------------------
    # Measurements, diagnostics, and low-level rigid motion
    # ------------------------------------------------------------------
    def _noisy_errors(self, errors: Mapping[str, Any]) -> dict[str, Any]:
        measured = dict(errors)
        if self.measurement_noise_translation_std > 0.0:
            for key in ("error_u", "error_v", "normal_gap_drift"):
                measured[key] = float(errors[key]) + float(
                    self.np_random.normal(0.0, self.measurement_noise_translation_std)
                )
        if self.measurement_noise_angle_std_rad > 0.0:
            for key in ("tilt_error_t1", "tilt_error_t2", "yaw_error"):
                measured[key] = float(errors[key]) + float(
                    self.np_random.normal(0.0, self.measurement_noise_angle_std_rad)
                )
        return measured

    def _policy_observation(
        self, errors: Mapping[str, Any], previous_action: np.ndarray
    ) -> np.ndarray:
        normalized = np.array(
            [
                float(errors["error_u"]) / self.translation_tolerance,
                float(errors["error_v"]) / self.translation_tolerance,
                float(errors["tilt_error_t1"]) / self.tilt_tolerance_rad,
                float(errors["tilt_error_t2"]) / self.tilt_tolerance_rad,
                float(errors["yaw_error"]) / self.yaw_tolerance_rad,
                float(errors["normal_gap_drift"]) / self.normal_gap_tolerance,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            [np.clip(normalized, -self.observation_clip, self.observation_clip), previous_action]
        ).astype(np.float32)
        return np.clip(
            observation, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)

    def _diagnostic_info(self, errors: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "error_u": float(errors["error_u"]),
            "error_v": float(errors["error_v"]),
            "tilt_error_t1": float(errors["tilt_error_t1"]),
            "tilt_error_t2": float(errors["tilt_error_t2"]),
            "yaw_error": float(errors["yaw_error"]),
            "initial_normal_gap": float(errors["initial_normal_gap"]),
            "current_normal_gap": float(errors["current_normal_gap"]),
            "normal_gap_drift": float(errors["normal_gap_drift"]),
            "corner_distances": np.asarray(errors["corner_distances"], dtype=np.float64).copy(),
            "glass_center_world": np.asarray(errors["glass_center_world"], dtype=np.float64).copy(),
            "frame_center_world": np.asarray(errors["frame_center_world"], dtype=np.float64).copy(),
            "glass_normal_world": np.asarray(errors["glass_normal_world"], dtype=np.float64).copy(),
            "frame_normal_world": np.asarray(errors["frame_normal_world"], dtype=np.float64).copy(),
            "stage": self.stage.value,
            "is_attached": bool(self._grasp_weld_is_active()),
        }

    def _move_mocap_pose(
        self, target_position: np.ndarray, target_quaternion: np.ndarray, n_steps: int
    ) -> None:
        start_position = self.get_ee_position().copy()
        start_quaternion = self.get_mocap_quaternion().copy()
        target_position = np.asarray(target_position, dtype=np.float64)
        target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
        for step_index in range(max(int(n_steps), 1)):
            alpha = float(step_index + 1) / float(max(int(n_steps), 1))
            position = (1.0 - alpha) * start_position + alpha * target_position
            quaternion = self._slerp(start_quaternion, target_quaternion, alpha)
            self.set_mocap_pose(position, quaternion)
            self._mujoco_step()
            self._simulation_step_count += 1
            self._mujoco.mj_forward(self.model, self.data)
            self._emit_pipeline_frame()
        self._last_ee_target_error = float(
            np.linalg.norm(self.get_ee_position() - target_position)
        )

    def _move_attached_object_pose(
        self,
        target_object_center: np.ndarray,
        target_object_rotation: np.ndarray,
        n_steps: int,
    ) -> None:
        if not self._grasp_weld_is_active():
            raise RuntimeError("Cannot move an object that is not suction-attached")
        target_center = np.asarray(target_object_center, dtype=np.float64)
        target_object_rotation = project_to_rotation_matrix(target_object_rotation)
        start_center = self.get_object_position().copy()
        start_object_rotation = self.get_object_rotation_matrix().copy()
        start_ee_rotation = np.asarray(
            self.data.xmat[self.ee_center_body_id], dtype=np.float64
        ).reshape(3, 3).copy()
        start_ee_quaternion = self.get_mocap_quaternion().copy()
        world_rotation_delta = target_object_rotation @ start_object_rotation.T
        world_rotation_vector = so3_log(world_rotation_delta)
        last_ee_target = self.get_ee_position().copy()
        for step_index in range(max(int(n_steps), 1)):
            alpha = float(step_index + 1) / float(max(int(n_steps), 1))
            desired_center = (1.0 - alpha) * start_center + alpha * target_center
            interpolated_delta = so3_exp(alpha * world_rotation_vector)
            delta_quaternion = self._matrix_to_quaternion(interpolated_delta)
            ee_quaternion = self._normalize_vector(
                self._quat_multiply(delta_quaternion, start_ee_quaternion)
            )
            desired_ee_rotation = interpolated_delta @ start_ee_rotation
            ee_position = desired_center - desired_ee_rotation @ self._grasp_relative_position
            last_ee_target = ee_position.copy()
            self.set_mocap_pose(ee_position, ee_quaternion)
            self._mujoco_step()
            self._simulation_step_count += 1
            self._mujoco.mj_forward(self.model, self.data)
            self._emit_pipeline_frame()
        self._last_ee_target_error = float(
            np.linalg.norm(self.get_ee_position() - last_ee_target)
        )

    def _matrix_to_quaternion(self, rotation: np.ndarray) -> np.ndarray:
        quaternion = np.empty(4, dtype=np.float64)
        self._mujoco.mju_mat2Quat(quaternion, project_to_rotation_matrix(rotation).reshape(9))
        return self._normalize_vector(quaternion)

    def _slerp(self, first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
        first = self._normalize_vector(np.asarray(first, dtype=np.float64))
        second = self._normalize_vector(np.asarray(second, dtype=np.float64))
        dot = float(np.dot(first, second))
        if dot < 0.0:
            second = -second
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._normalize_vector((1.0 - alpha) * first + alpha * second)
        angle = float(np.arccos(dot))
        sin_angle = float(np.sin(angle))
        return self._normalize_vector(
            np.sin((1.0 - alpha) * angle) / sin_angle * first
            + np.sin(alpha * angle) / sin_angle * second
        )

    def _grasp_weld_is_active(self) -> bool:
        return bool(
            self.is_attached
            and self.grasp_weld_eq_id >= 0
            and int(self.model.eq_active[self.grasp_weld_eq_id]) == 1
        )

    def _window_collision_reason(self) -> str:
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.window_geom_ids:
                other = geom2
            elif geom2 in self.window_geom_ids:
                other = geom1
            else:
                continue
            if other in self.object_collision_geom_ids:
                return "glass_window_illegal_collision"
            if other in self.eef_collision_geom_ids:
                return "robot_window_illegal_collision"
        return ""

    def _joint_limit_violated(self) -> bool:
        for joint_id in range(int(self.model.njnt)):
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            value = float(self.data.qpos[qpos_address])
            lower, upper = np.asarray(self.model.jnt_range[joint_id], dtype=np.float64)
            if value < lower - 1e-5 or value > upper + 1e-5:
                return True
        return False

    def _emit_pipeline_frame(self) -> None:
        if self.pipeline_frame_callback is not None:
            self.pipeline_frame_callback()
