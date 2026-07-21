"""Nine-stage state and timing helpers for window assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any

import numpy as np


class WindowAssemblyStage(str, Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    SUCTION_GRASP = "suction_grasp"
    LIFT = "lift"
    REORIENT = "reorient"
    COARSE_ALIGN = "coarse_align"
    FINE_ALIGN = "fine_align"
    INSERT = "insert"
    HOLD = "hold"


WINDOW_ASSEMBLY_STAGE_ORDER = tuple(WindowAssemblyStage)


@dataclass
class StageRecord:
    """One stage result, using wall-clock seconds and simulation step counts."""

    stage: str
    start_time: float
    end_time: float
    steps: int
    success: bool
    failure_reason: str | None = None


class WindowAssemblyStageTracker:
    """Track and validate the strict nine-stage execution order."""

    def __init__(self) -> None:
        self.records: list[StageRecord] = []
        self._active_stage: WindowAssemblyStage | None = None
        self._active_start_time = 0.0
        self._active_start_step = 0

    def begin(self, stage: WindowAssemblyStage, simulation_step: int) -> None:
        expected_index = len(self.records)
        if expected_index >= len(WINDOW_ASSEMBLY_STAGE_ORDER):
            raise RuntimeError("All window assembly stages have already completed")
        expected = WINDOW_ASSEMBLY_STAGE_ORDER[expected_index]
        if stage is not expected:
            raise RuntimeError(f"Expected stage {expected.value}, got {stage.value}")
        if self._active_stage is not None:
            raise RuntimeError(f"Stage {self._active_stage.value} is already active")
        self._active_stage = stage
        self._active_start_time = time.perf_counter()
        self._active_start_step = int(simulation_step)

    def end(
        self,
        simulation_step: int,
        *,
        success: bool,
        failure_reason: str | None = None,
    ) -> StageRecord:
        if self._active_stage is None:
            raise RuntimeError("No active window assembly stage")
        record = StageRecord(
            stage=self._active_stage.value,
            start_time=self._active_start_time,
            end_time=time.perf_counter(),
            steps=max(int(simulation_step) - self._active_start_step, 0),
            success=bool(success),
            failure_reason=failure_reason,
        )
        self.records.append(record)
        self._active_stage = None
        return record

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


class WindowAssemblyPipeline:
    """Run the complete task while invoking a policy only in ``fine_align``."""

    ERROR_KEYS = (
        "error_u",
        "error_v",
        "tilt_error_t1",
        "tilt_error_t2",
        "yaw_error",
        "normal_gap_error",
        "current_normal_gap",
    )
    INSERT_START_ERROR_KEYS = (
        "insert_start_error_u",
        "insert_start_error_v",
        "insert_start_tilt_t1",
        "insert_start_tilt_t2",
        "insert_start_yaw",
        "insert_start_normal_gap",
    )
    INSERT_MAX_ERROR_KEYS = (
        "insert_max_abs_error_u",
        "insert_max_abs_error_v",
        "insert_max_abs_tilt_t1",
        "insert_max_abs_tilt_t2",
        "insert_max_abs_yaw",
    )
    INSERT_FAILURE_ERROR_KEYS = (
        "insert_failure_error_u",
        "insert_failure_error_v",
        "insert_failure_tilt_t1",
        "insert_failure_tilt_t2",
        "insert_failure_yaw",
        "insert_failure_normal_gap",
    )

    def __init__(self, env: Any) -> None:
        self.env = env
        self.base = env.unwrapped

    def run_episode(
        self,
        policy: Any,
        *,
        seed: int,
        deterministic: bool = True,
        frame_callback: Any | None = None,
    ) -> dict[str, Any]:
        observation, reset_info = self.env.reset(seed=seed)
        if frame_callback is not None:
            frame_callback(self.env)
        initial_errors = {
            key: float(reset_info[key]) for key in self.ERROR_KEYS
        }
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        previous_action: np.ndarray | None = None
        smoothness: list[float] = []
        terminated = False
        truncated = False
        info = reset_info

        while not (terminated or truncated):
            prediction = policy.predict(observation, deterministic=deterministic)
            action = np.asarray(prediction[0], dtype=np.float32).reshape(5)
            observation, reward, terminated, truncated, info = self.env.step(action)
            actions.append(action.copy())
            rewards.append(float(reward))
            if previous_action is not None:
                smoothness.append(float(np.linalg.norm(action - previous_action)))
            previous_action = action
            if frame_callback is not None:
                frame_callback(self.env)

        fine_success = bool(info.get("fine_align_success", False))
        ready_for_insert = bool(info.get("ready_for_insert", False))
        fine_end_error_values = self.base.get_fine_alignment_errors()
        fine_end_errors = {
            key: float(fine_end_error_values[key]) for key in self.ERROR_KEYS
        }
        insert_success = False
        hold_success = False
        assembly_verified = False
        assembly_diagnostics: dict[str, Any] = {}
        failure_reason = str(info.get("failure_reason", ""))
        failure_category = ""
        if fine_success and ready_for_insert:
            insert_success = bool(self.base.run_scripted_insert())
            if frame_callback is not None:
                frame_callback(self.env)
            if insert_success:
                hold_success = bool(self.base.run_scripted_hold())
                if frame_callback is not None:
                    frame_callback(self.env)
            if hold_success:
                assembly_verified, assembly_diagnostics = self.base.verify_final_assembly()
                if not assembly_verified:
                    failure_reason = str(
                        assembly_diagnostics.get(
                            "failure_reason", "final_assembly_verification_failed"
                        )
                    )
                    failure_category = "final_assembly_verification_failed"
            if not insert_success:
                failure_reason = self._last_failure_reason("insert_failed")
            elif not hold_success:
                failure_reason = self._last_failure_reason("hold_failed")

        insert_diagnostics = self.base.get_last_insert_diagnostics()
        pipeline_end_error_values = self.base.get_fine_alignment_errors()
        pipeline_end_errors = {
            key: float(pipeline_end_error_values[key]) for key in self.ERROR_KEYS
        }
        stage_records = self.base.stage_tracker.as_dicts()
        stages = {
            stage.value: {
                "success": False,
                "start_time": None,
                "end_time": None,
                "steps": 0,
                "failure_reason": "not_run",
            }
            for stage in WINDOW_ASSEMBLY_STAGE_ORDER
        }
        for record in stage_records:
            stages[record["stage"]] = {
                "success": bool(record["success"]),
                "start_time": float(record["start_time"]),
                "end_time": float(record["end_time"]),
                "steps": int(record["steps"]),
                "failure_reason": record["failure_reason"] or "",
            }

        action_norms = [float(np.linalg.norm(action)) for action in actions]
        full_pipeline_success = bool(
            fine_success and insert_success and hold_success and assembly_verified
        )
        if not failure_category:
            failure_category = str(insert_diagnostics.get("failure_category", ""))
        return {
            "seed": int(seed),
            "stages": stages,
            "stage_records": stage_records,
            "initial_errors": initial_errors,
            "fine_end_errors": fine_end_errors,
            "final_errors": pipeline_end_errors,
            "pipeline_end_errors": pipeline_end_errors,
            "fine_align_reward": float(np.sum(rewards)) if rewards else 0.0,
            "fine_align_steps": len(actions),
            "fine_align_max_action_magnitude": max(action_norms, default=0.0),
            "fine_align_mean_action_magnitude": float(np.mean(action_norms)) if action_norms else 0.0,
            "fine_align_smoothness": float(np.mean(smoothness)) if smoothness else 0.0,
            "fine_align_success": fine_success,
            "ready_for_insert": ready_for_insert,
            "insert_success": insert_success,
            "hold_success": hold_success,
            "assembly_verified": bool(assembly_verified),
            "full_pipeline_success": full_pipeline_success,
            "insert_steps": int(insert_diagnostics.get("insert_steps", 0)),
            "insert_depth": float(insert_diagnostics.get("insert_depth", 0.0)),
            "insert_start_errors": {
                key: float(insert_diagnostics.get(key, float("nan")))
                for key in self.INSERT_START_ERROR_KEYS
            },
            "insert_max_errors": {
                key: float(insert_diagnostics.get(key, float("nan")))
                for key in self.INSERT_MAX_ERROR_KEYS
            },
            "insert_failure_errors": {
                key: float(insert_diagnostics.get(key, float("nan")))
                for key in self.INSERT_FAILURE_ERROR_KEYS
            },
            "insert_failure_step": int(insert_diagnostics.get("insert_failure_step", -1)),
            "insert_failure_depth": float(
                insert_diagnostics.get("insert_failure_depth", float("nan"))
            ),
            "insert_alignment_violation_count": int(
                insert_diagnostics.get("insert_alignment_violation_count", 0)
            ),
            "insert_alignment_violation_hold_steps": int(
                insert_diagnostics.get("insert_alignment_violation_hold_steps", 0)
            ),
            "insert_success_hold_steps": int(
                insert_diagnostics.get("insert_success_hold_steps", 0)
            ),
            "insert_final_alignment_success_count": int(
                insert_diagnostics.get("insert_final_alignment_success_count", 0)
            ),
            "insert_final_verification_steps": int(
                insert_diagnostics.get("insert_final_verification_steps", 0)
            ),
            "insert_final_verification_max_steps": int(
                insert_diagnostics.get("insert_final_verification_max_steps", 0)
            ),
            "insert_depth_tolerance": float(
                insert_diagnostics.get("insert_depth_tolerance", float("nan"))
            ),
            "insert_final_depth_reached": bool(
                insert_diagnostics.get("insert_final_depth_reached", False)
            ),
            "insert_final_command_finished": bool(
                insert_diagnostics.get("insert_final_command_finished", False)
            ),
            "insert_final_alignment_ok": bool(
                insert_diagnostics.get("insert_final_alignment_ok", False)
            ),
            "insert_final_errors": {
                key: float(insert_diagnostics.get(key, float("nan")))
                for key in (
                    "insert_final_error_u",
                    "insert_final_error_v",
                    "insert_final_tilt_t1",
                    "insert_final_tilt_t2",
                    "insert_final_yaw",
                    "insert_final_normal_gap",
                )
            },
            "hold_steps": int(stages[WindowAssemblyStage.HOLD.value]["steps"]),
            "hold_diagnostics": {
                key: float(insert_diagnostics.get(key, float("nan")))
                for key in (
                    "hold_max_abs_error_u",
                    "hold_max_abs_error_v",
                    "hold_max_abs_tilt_t1",
                    "hold_max_abs_tilt_t2",
                    "hold_max_abs_yaw",
                    "hold_min_normal_gap",
                    "hold_max_normal_gap",
                )
            },
            "assembly_alignment_ok": bool(
                insert_diagnostics.get("assembly_alignment_ok", False)
            ),
            "assembly_depth_ok": bool(insert_diagnostics.get("assembly_depth_ok", False)),
            "assembly_attachment_ok": bool(
                insert_diagnostics.get("assembly_attachment_ok", False)
            ),
            "assembly_collision_ok": bool(
                insert_diagnostics.get("assembly_collision_ok", False)
            ),
            "assembly_final_errors": {
                key: float(insert_diagnostics.get(key, float("nan")))
                for key in (
                    "final_error_u",
                    "final_error_v",
                    "final_tilt_error_t1",
                    "final_tilt_error_t2",
                    "final_yaw_error",
                    "final_normal_gap",
                )
            },
            "insert_action_sim_steps": int(
                insert_diagnostics.get("insert_action_sim_steps", 0)
            ),
            "insert_translation_abort_tolerance": float(
                insert_diagnostics.get("insert_translation_abort_tolerance", float("nan"))
            ),
            "insert_tilt_abort_tolerance": float(
                insert_diagnostics.get("insert_tilt_abort_tolerance", float("nan"))
            ),
            "insert_yaw_abort_tolerance": float(
                insert_diagnostics.get("insert_yaw_abort_tolerance", float("nan"))
            ),
            "insert_ee_target_error": float(
                insert_diagnostics.get("insert_ee_target_error", float("nan"))
            ),
            "failure_category": failure_category,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": "" if full_pipeline_success else (failure_reason or "pipeline_incomplete"),
            "stage": str(self.base.stage.value),
        }

    def _last_failure_reason(self, default: str) -> str:
        records = self.base.stage_tracker.records
        if records and records[-1].failure_reason:
            return str(records[-1].failure_reason)
        return default
