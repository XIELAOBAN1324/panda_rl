"""Vision-free reinforcement-learning environment for window fine alignment.

Only the five in-plane/rotational alignment degrees of freedom are exposed to
the policy. Pickup, transport, coarse alignment, insertion, and holding are
scripted and are represented explicitly by :class:`WindowAssemblyStage`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from panda_mujoco_gym.envs.window_assembly_controller import (
    WindowAssemblyController,
)
from panda_mujoco_gym.envs.window_assembly_geometry import (
    alignment_errors_from_corners,
    fine_alignment_costs,
    normalize_vector,
    ordered_rectangle_corners,
    project_to_rotation_matrix,
    so3_exp,
    so3_log,
)
from panda_mujoco_gym.envs.window_assembly_pipeline import (
    WindowAssemblyStage,
    WindowAssemblyStageTracker,
)
from panda_mujoco_gym.envs.window_assembly_scene import WindowAssemblyScene


class FrankaWindowFineAlignEnv(WindowAssemblyScene):
    """Five-DoF dense-reward environment for the fine-alignment stage only."""

    # The suction cup grasps the XML glass from its local +normal side.  The
    # assembly-facing directed rectangle therefore uses local axes
    # ``(+x, -y, -z)``.  This is a fixed physical convention, not dynamic
    # nearest-corner matching.
    GLASS_ASSEMBLY_FRAME = np.diag([1.0, -1.0, -1.0])

    def __init__(self, **kwargs: Any) -> None:
        internal_max_steps = kwargs.pop("fine_align_max_episode_steps", None)
        registered_max_steps = kwargs.pop("max_episode_steps", 100)
        self.max_episode_steps = int(
            registered_max_steps if internal_max_steps is None else internal_max_steps
        )
        self.coarse_translation_range = float(kwargs.pop("coarse_translation_range", 0.015))
        self.coarse_tilt_range_rad = np.deg2rad(float(kwargs.pop("coarse_tilt_range_deg", 2.0)))
        self.coarse_yaw_range_rad = np.deg2rad(float(kwargs.pop("coarse_yaw_range_deg", 4.0)))
        self.preinsert_normal_offset = float(
            kwargs.pop("preinsert_normal_offset", 0.08)
        )
        self.coarse_normal_gap = self.preinsert_normal_offset
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
        self.max_normal_gap_error = float(kwargs.pop("max_normal_gap_error", 0.008))
        self.unreachable_position_threshold = float(
            kwargs.pop("unreachable_position_threshold", 0.03)
        )
        self.fine_action_sim_steps = int(kwargs.pop("fine_action_sim_steps", 4))
        self.normal_gap_correction_threshold = float(
            kwargs.pop("normal_gap_correction_threshold", 0.0005)
        )
        self.normal_gap_correction_sim_steps = int(
            kwargs.pop("normal_gap_correction_sim_steps", 2)
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
        self.insert_action_sim_steps = int(kwargs.pop("insert_action_sim_steps", 4))
        self.insert_translation_abort_tolerance = float(
            kwargs.pop("insert_translation_abort_tolerance", 0.0025)
        )
        self.insert_tilt_abort_tolerance_rad = np.deg2rad(
            float(kwargs.pop("insert_tilt_abort_tolerance_deg", 0.6))
        )
        self.insert_yaw_abort_tolerance_rad = np.deg2rad(
            float(kwargs.pop("insert_yaw_abort_tolerance_deg", 0.6))
        )
        self.insert_alignment_violation_hold_steps = int(
            kwargs.pop("insert_alignment_violation_hold_steps", 2)
        )
        self.insert_success_hold_steps = int(kwargs.pop("insert_success_hold_steps", 3))
        self.insert_final_verification_max_steps = int(
            kwargs.pop("insert_final_verification_max_steps", 20)
        )
        self.insert_depth_tolerance = float(kwargs.pop("insert_depth_tolerance", 0.0005))
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
            "fine_action_sim_steps",
            "normal_gap_correction_sim_steps",
            "insert_action_sim_steps",
            "insert_alignment_violation_hold_steps",
            "insert_success_hold_steps",
            "insert_final_verification_max_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "coarse_translation_range",
            "coarse_tilt_range_rad",
            "coarse_yaw_range_rad",
            "preinsert_normal_offset",
            "fine_position_action_scale",
            "fine_tilt_action_scale_rad",
            "fine_yaw_action_scale_rad",
            "translation_tolerance",
            "tilt_tolerance_rad",
            "yaw_tolerance_rad",
            "normal_gap_tolerance",
            "max_normal_gap_error",
            "insert_step_size",
            "insert_final_gap",
            "insert_translation_abort_tolerance",
            "insert_tilt_abort_tolerance_rad",
            "insert_yaw_abort_tolerance_rad",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.normal_gap_correction_threshold < 0.0:
            raise ValueError("normal_gap_correction_threshold must be non-negative")
        if self.insert_depth_tolerance < 0.0:
            raise ValueError("insert_depth_tolerance must be non-negative")
        if self.preinsert_normal_offset <= 0.0:
            raise ValueError("preinsert_normal_offset must be positive")
        if self.insert_translation_abort_tolerance < self.translation_tolerance:
            raise ValueError(
                "insert_translation_abort_tolerance must be at least translation_tolerance"
            )
        if self.insert_tilt_abort_tolerance_rad < self.tilt_tolerance_rad:
            raise ValueError(
                "insert_tilt_abort_tolerance_deg must be at least tilt_tolerance_deg"
            )
        if self.insert_yaw_abort_tolerance_rad < self.yaw_tolerance_rad:
            raise ValueError(
                "insert_yaw_abort_tolerance_deg must be at least yaw_tolerance_deg"
            )
        if self.insert_translation_abort_tolerance > self.max_translation_error:
            raise ValueError(
                "insert_translation_abort_tolerance must not exceed max_translation_error"
            )
        if self.insert_tilt_abort_tolerance_rad > self.max_tilt_error_rad:
            raise ValueError(
                "insert_tilt_abort_tolerance_deg must not exceed max_tilt_error_deg"
            )
        if self.insert_yaw_abort_tolerance_rad > self.max_yaw_error_rad:
            raise ValueError(
                "insert_yaw_abort_tolerance_deg must not exceed max_yaw_error_deg"
            )

        self.previous_action = np.zeros(5, dtype=np.float32)
        self.reference_normal_gap = float(self.preinsert_normal_offset)
        self.initial_normal_gap = float(self.reference_normal_gap)
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
        self.script_controller = WindowAssemblyController(self)
        self._fine_frame_center_world: np.ndarray | None = None
        self._fine_frame_rotation_world: np.ndarray | None = None
        self._fine_orientation_frame_world: np.ndarray | None = None
        self._last_step_gap_diagnostics = self._empty_gap_command_diagnostics()

        fine_action_space = spaces.Box(
            -1.0, 1.0, shape=(5,), dtype=np.float32
        )
        fine_observation_space = spaces.Box(
            low=-self.observation_clip,
            high=self.observation_clip,
            shape=(11,),
            dtype=np.float32,
        )
        super().__init__(
            policy_action_space=fine_action_space,
            policy_observation_space=fine_observation_space,
            **kwargs,
        )

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

    def _empty_gap_command_diagnostics(self) -> dict[str, float | bool]:
        return {
            "candidate_normal_gap": float("nan"),
            "commanded_normal_gap": float("nan"),
            "normal_gap_command_error": float("nan"),
            "post_action_normal_gap": float("nan"),
            "post_action_normal_gap_error": float("nan"),
            "normal_gap_correction_applied": False,
            "normal_gap_correction_magnitude": 0.0,
        }

    def _directed_frame_rotation(
        self,
        raw_frame_rotation: np.ndarray,
        frame_center: np.ndarray,
        glass_center: np.ndarray,
    ) -> np.ndarray:
        """Return orthonormal frame axes whose normal points to the glass side."""

        raw_rotation = project_to_rotation_matrix(raw_frame_rotation)
        raw_normal = normalize_vector(raw_rotation[:, 2])
        to_glass = np.asarray(glass_center, dtype=np.float64) - np.asarray(
            frame_center, dtype=np.float64
        )
        if float(np.dot(raw_normal, to_glass)) < 0.0:
            directed_normal = -raw_normal
        else:
            directed_normal = raw_normal
        return np.column_stack(
            [
                normalize_vector(raw_rotation[:, 0]),
                normalize_vector(raw_rotation[:, 1]),
                directed_normal,
            ]
        ).astype(np.float64)

    def _set_episode_frame_from_raw_errors(self, raw_errors: Mapping[str, Any]) -> None:
        frame_center = np.asarray(raw_errors["frame_center_world"], dtype=np.float64).copy()
        glass_center = np.asarray(raw_errors["glass_center_world"], dtype=np.float64).copy()
        raw_frame_rotation = np.asarray(raw_errors["frame_rotation_world"], dtype=np.float64)
        frame_rotation = self._directed_frame_rotation(
            raw_frame_rotation, frame_center, glass_center
        )
        self._fine_frame_center_world = frame_center
        self._fine_frame_rotation_world = frame_rotation
        self._fine_orientation_frame_world = project_to_rotation_matrix(raw_frame_rotation)
        self.reference_normal_gap = float(self.preinsert_normal_offset)

        # The inherited plane-penetration check treats the positive normal side as
        # the insertion region. The fine-alignment frame normal points the other
        # way, toward the glass preinsert side.
        frame_normal = frame_rotation[:, 2].copy()
        self.window_center = frame_center.copy()
        self.window_normal = -frame_normal
        self.target_glass_normal = self._fine_orientation_frame_world[:, 2].copy()
        self.window_safe_goal = frame_center + self.reference_normal_gap * frame_normal

    def _current_fine_frame_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._fine_frame_center_world is not None and self._fine_frame_rotation_world is not None:
            return (
                self._fine_frame_center_world.copy(),
                self._fine_frame_rotation_world.copy(),
            )
        raw_errors = alignment_errors_from_corners(
            self.get_glass_corners_world(), self.get_frame_corners_world()
        )
        frame_center = np.asarray(raw_errors["frame_center_world"], dtype=np.float64)
        frame_rotation = self._directed_frame_rotation(
            np.asarray(raw_errors["frame_rotation_world"], dtype=np.float64),
            frame_center,
            np.asarray(raw_errors["glass_center_world"], dtype=np.float64),
        )
        return frame_center.copy(), frame_rotation.copy()

    def _current_orientation_frame(self) -> np.ndarray:
        if self._fine_orientation_frame_world is not None:
            return self._fine_orientation_frame_world.copy()
        raw_errors = alignment_errors_from_corners(
            self.get_glass_corners_world(), self.get_frame_corners_world()
        )
        return project_to_rotation_matrix(
            np.asarray(raw_errors["frame_rotation_world"], dtype=np.float64)
        )

    def _normal_gap_for_center(
        self,
        center: np.ndarray,
        *,
        frame_center: np.ndarray | None = None,
        frame_rotation: np.ndarray | None = None,
    ) -> float:
        if frame_center is None or frame_rotation is None:
            frame_center, frame_rotation = self._current_fine_frame_pose()
        frame_normal = np.asarray(frame_rotation, dtype=np.float64)[:, 2]
        return float(np.dot(frame_normal, np.asarray(center, dtype=np.float64) - frame_center))

    def _project_center_to_reference_normal_gap(
        self,
        candidate_center: np.ndarray,
        *,
        frame_center: np.ndarray | None = None,
        frame_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Project only the normal component back to the fixed preinsert plane."""

        if frame_center is None or frame_rotation is None:
            frame_center, frame_rotation = self._current_fine_frame_pose()
        frame_center = np.asarray(frame_center, dtype=np.float64)
        frame_rotation = np.asarray(frame_rotation, dtype=np.float64)
        frame_t1 = frame_rotation[:, 0]
        frame_t2 = frame_rotation[:, 1]
        frame_normal = frame_rotation[:, 2]
        candidate_center = np.asarray(candidate_center, dtype=np.float64)
        candidate_gap = float(np.dot(frame_normal, candidate_center - frame_center))
        target_center = candidate_center + (
            self.reference_normal_gap - candidate_gap
        ) * frame_normal
        commanded_gap = float(np.dot(frame_normal, target_center - frame_center))
        command_error = float(commanded_gap - self.reference_normal_gap)
        candidate_u = float(np.dot(frame_t1, candidate_center - frame_center))
        target_u = float(np.dot(frame_t1, target_center - frame_center))
        candidate_v = float(np.dot(frame_t2, candidate_center - frame_center))
        target_v = float(np.dot(frame_t2, target_center - frame_center))
        if not np.isclose(commanded_gap, self.reference_normal_gap, atol=1e-8):
            raise RuntimeError("Projected center did not reach the reference normal gap")
        if not np.isclose(candidate_u, target_u, atol=1e-8):
            raise RuntimeError("Normal-gap projection changed the frame_t1 coordinate")
        if not np.isclose(candidate_v, target_v, atol=1e-8):
            raise RuntimeError("Normal-gap projection changed the frame_t2 coordinate")
        return target_center, {
            "candidate_normal_gap": candidate_gap,
            "commanded_normal_gap": commanded_gap,
            "normal_gap_command_error": command_error,
        }

    def get_fine_alignment_errors(self) -> dict[str, np.ndarray | float]:
        """Return five-DoF errors plus fixed preinsert-plane gap diagnostics."""

        errors = alignment_errors_from_corners(
            self.get_glass_corners_world(), self.get_frame_corners_world()
        )
        frame_center, frame_rotation = self._current_fine_frame_pose()
        orientation_frame = self._current_orientation_frame()
        glass_center = np.asarray(errors["glass_center_world"], dtype=np.float64)
        glass_rotation = np.asarray(errors["glass_rotation_world"], dtype=np.float64)
        center_delta = np.asarray(errors["glass_center_world"]) - np.asarray(
            frame_center
        )
        relative_rotation = orientation_frame.T @ glass_rotation
        rotation_error_world = orientation_frame @ so3_log(relative_rotation)
        current_gap = float(np.dot(frame_rotation[:, 2], center_delta))
        normal_gap_error = current_gap - float(self.reference_normal_gap)
        errors["error_u"] = float(np.dot(frame_rotation[:, 0], center_delta))
        errors["error_v"] = float(np.dot(frame_rotation[:, 1], center_delta))
        errors["tilt_error_t1"] = float(np.dot(frame_rotation[:, 0], rotation_error_world))
        errors["tilt_error_t2"] = float(np.dot(frame_rotation[:, 1], rotation_error_world))
        errors["yaw_error"] = float(np.dot(frame_rotation[:, 2], rotation_error_world))
        errors["frame_center_world"] = frame_center.copy()
        errors["frame_rotation_world"] = frame_rotation.copy()
        errors["frame_normal_world"] = frame_rotation[:, 2].copy()
        errors["initial_normal_gap"] = float(self.initial_normal_gap)
        errors["reference_normal_gap"] = float(self.reference_normal_gap)
        errors["current_normal_gap"] = current_gap
        errors["normal_gap_error"] = normal_gap_error
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
        self._fine_frame_center_world = None
        self._fine_frame_rotation_world = None
        self._fine_orientation_frame_world = None
        self._last_step_gap_diagnostics = self._empty_gap_command_diagnostics()
        self.script_controller.reset_episode()

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
        self._set_episode_frame_from_raw_errors(frame_errors)
        frame_center, frame_rotation = self._current_fine_frame_pose()
        orientation_frame = self._current_orientation_frame()
        self.script_controller.run_approach(frame_rotation)
        self.script_controller.run_descend(frame_rotation)
        self.script_controller.run_grasp()
        lift_position = self.script_controller.run_lift()
        self.script_controller.run_reorient(
            frame_center,
            frame_rotation,
            orientation_frame,
            lift_position,
        )
        self.script_controller.run_coarse_align(frame_rotation)
        self._stabilize_reset_state()
        return True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        # Reset the private RobotEnv state without entering GoalEnv's public
        # Dict-space validation; the public space remains the 11-D Box.
        self._reset_robot_simulation(seed=seed)
        self.previous_action = np.zeros(5, dtype=np.float32)
        self._success_counter = 0
        self._elapsed_fine_steps = 0
        self._episode_done = False
        self.ready_for_insert = False
        self.insert_success = False
        self.hold_success = False

        self.reference_normal_gap = float(self.preinsert_normal_offset)
        self.initial_normal_gap = float(self.reference_normal_gap)
        self._last_step_gap_diagnostics = self._empty_gap_command_diagnostics()
        self.script_controller.reset_episode()
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
        self._last_step_gap_diagnostics = self._empty_gap_command_diagnostics()
        if finite_action and self._grasp_weld_is_active():
            errors_before = self.get_fine_alignment_errors()
            frame_center = np.asarray(errors_before["frame_center_world"], dtype=np.float64)
            frame_rotation = np.asarray(errors_before["frame_rotation_world"], dtype=np.float64)
            current_center = self.get_object_position().copy()
            candidate_center = (
                current_center
                + clipped_action[0] * self.fine_position_action_scale * frame_rotation[:, 0]
                + clipped_action[1] * self.fine_position_action_scale * frame_rotation[:, 1]
            )
            translation_delta = candidate_center - current_center
            normal_component = abs(float(np.dot(translation_delta, frame_rotation[:, 2])))
            if normal_component > 1e-10:
                raise RuntimeError("Fine-alignment translation contains a normal component")
            target_center, command_diagnostics = self._project_center_to_reference_normal_gap(
                candidate_center,
                frame_center=frame_center,
                frame_rotation=frame_rotation,
            )
            rotation_vector_world = (
                clipped_action[2] * self.fine_tilt_action_scale_rad * frame_rotation[:, 0]
                + clipped_action[3] * self.fine_tilt_action_scale_rad * frame_rotation[:, 1]
                + clipped_action[4] * self.fine_yaw_action_scale_rad * frame_rotation[:, 2]
            )
            target_rotation = so3_exp(rotation_vector_world) @ self.get_object_rotation_matrix()
            self.script_controller.move_attached_object_pose(
                target_center,
                target_rotation,
                self.fine_action_sim_steps,
                center_feedback_gain=1.0,
            )
            post_center = self.get_object_position().copy()
            post_gap = float(np.dot(frame_rotation[:, 2], post_center - frame_center))
            post_gap_error = post_gap - float(self.reference_normal_gap)
            correction_applied = False
            correction_magnitude = 0.0
            if abs(post_gap_error) > self.normal_gap_correction_threshold:
                corrected_center = post_center - post_gap_error * frame_rotation[:, 2]
                current_rotation = self.get_object_rotation_matrix().copy()
                self.script_controller.move_attached_object_pose(
                    corrected_center,
                    current_rotation,
                    self.normal_gap_correction_sim_steps,
                    center_feedback_gain=1.0,
                )
                correction_applied = True
                correction_magnitude = abs(float(post_gap_error))
            self._last_step_gap_diagnostics = {
                **command_diagnostics,
                "post_action_normal_gap": post_gap,
                "post_action_normal_gap_error": post_gap_error,
                "normal_gap_correction_applied": bool(correction_applied),
                "normal_gap_correction_magnitude": float(correction_magnitude),
            }

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
            and abs(float(errors["normal_gap_error"])) < self.normal_gap_tolerance
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
            "normal_gap_error",
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
        if abs(float(errors["normal_gap_error"])) > self.max_normal_gap_error:
            return "normal_gap_error_exceeded"
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
        if float(errors["current_normal_gap"]) < 0.01:
            return "glass_entered_insertion_region"
        if not np.all(np.isfinite(np.asarray(target_center))):
            return "end_effector_unreachable"
        return ""

    # ------------------------------------------------------------------
    # Measurements, diagnostics, and low-level rigid motion
    # ------------------------------------------------------------------
    def _noisy_errors(self, errors: Mapping[str, Any]) -> dict[str, Any]:
        measured = dict(errors)
        if self.measurement_noise_translation_std > 0.0:
            for key in ("error_u", "error_v"):
                measured[key] = float(errors[key]) + float(
                    self.np_random.normal(0.0, self.measurement_noise_translation_std)
                )
            gap_noise = float(
                self.np_random.normal(0.0, self.measurement_noise_translation_std)
            )
            measured["normal_gap_error"] = float(errors["normal_gap_error"]) + gap_noise
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
                float(errors["normal_gap_error"]) / self.normal_gap_tolerance,
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
        info = {
            "error_u": float(errors["error_u"]),
            "error_v": float(errors["error_v"]),
            "tilt_error_t1": float(errors["tilt_error_t1"]),
            "tilt_error_t2": float(errors["tilt_error_t2"]),
            "yaw_error": float(errors["yaw_error"]),
            "preinsert_normal_offset": float(self.preinsert_normal_offset),
            "reference_normal_gap": float(errors["reference_normal_gap"]),
            "initial_normal_gap": float(errors["initial_normal_gap"]),
            "current_normal_gap": float(errors["current_normal_gap"]),
            "normal_gap_error": float(errors["normal_gap_error"]),
            "candidate_normal_gap": float(self._last_step_gap_diagnostics["candidate_normal_gap"]),
            "commanded_normal_gap": float(self._last_step_gap_diagnostics["commanded_normal_gap"]),
            "normal_gap_command_error": float(
                self._last_step_gap_diagnostics["normal_gap_command_error"]
            ),
            "post_action_normal_gap": float(
                self._last_step_gap_diagnostics["post_action_normal_gap"]
            ),
            "post_action_normal_gap_error": float(
                self._last_step_gap_diagnostics["post_action_normal_gap_error"]
            ),
            "normal_gap_correction_applied": bool(
                self._last_step_gap_diagnostics["normal_gap_correction_applied"]
            ),
            "normal_gap_correction_magnitude": float(
                self._last_step_gap_diagnostics["normal_gap_correction_magnitude"]
            ),
            "fine_action_sim_steps": int(self.fine_action_sim_steps),
            "insert_action_sim_steps": int(self.insert_action_sim_steps),
            "insert_translation_abort_tolerance": float(
                self.insert_translation_abort_tolerance
            ),
            "insert_tilt_abort_tolerance": float(self.insert_tilt_abort_tolerance_rad),
            "insert_tilt_abort_tolerance_deg": float(
                np.rad2deg(self.insert_tilt_abort_tolerance_rad)
            ),
            "insert_yaw_abort_tolerance": float(self.insert_yaw_abort_tolerance_rad),
            "insert_yaw_abort_tolerance_deg": float(
                np.rad2deg(self.insert_yaw_abort_tolerance_rad)
            ),
            "insert_alignment_violation_hold_steps": int(
                self.insert_alignment_violation_hold_steps
            ),
            "insert_success_hold_steps": int(self.insert_success_hold_steps),
            "insert_final_verification_max_steps": int(
                self.insert_final_verification_max_steps
            ),
            "insert_depth_tolerance": float(self.insert_depth_tolerance),
            "insert_step_size": float(self.insert_step_size),
            "ee_target_error": float(self._last_ee_target_error),
            "corner_distances": np.asarray(errors["corner_distances"], dtype=np.float64).copy(),
            "glass_center_world": np.asarray(errors["glass_center_world"], dtype=np.float64).copy(),
            "frame_center_world": np.asarray(errors["frame_center_world"], dtype=np.float64).copy(),
            "glass_normal_world": np.asarray(errors["glass_normal_world"], dtype=np.float64).copy(),
            "frame_normal_world": np.asarray(errors["frame_normal_world"], dtype=np.float64).copy(),
            "stage": self.stage.value,
            "is_attached": bool(self._grasp_weld_is_active()),
        }
        info.update(self.script_controller.get_last_insert_diagnostics())
        return info

    def _emit_pipeline_frame(self) -> None:
        if self.pipeline_frame_callback is not None:
            self.pipeline_frame_callback()
