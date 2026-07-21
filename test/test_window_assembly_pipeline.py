"""Pipeline ordering and compatibility tests."""

import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np

import panda_mujoco_gym  # noqa: F401
from panda_mujoco_gym.envs.window_assembly_pipeline import (
    WINDOW_ASSEMBLY_STAGE_ORDER,
    WindowAssemblyPipeline,
)


class CorrectivePolicy:
    def __init__(self, base):
        self.base = base
        self.calls = 0

    def predict(self, observation, deterministic=True):
        del deterministic
        self.calls += 1
        observation = np.asarray(observation, dtype=np.float64)
        action = np.array(
            [
                -observation[0] * self.base.translation_tolerance / self.base.fine_position_action_scale,
                -observation[1] * self.base.translation_tolerance / self.base.fine_position_action_scale,
                -observation[2] * self.base.tilt_tolerance_rad / self.base.fine_tilt_action_scale_rad,
                -observation[3] * self.base.tilt_tolerance_rad / self.base.fine_tilt_action_scale_rad,
                -observation[4] * self.base.yaw_tolerance_rad / self.base.fine_yaw_action_scale_rad,
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0), None


def test_pipeline_runs_all_nine_stages_and_policy_only_runs_during_fine_align():
    env = gym.make(
        "FrankaWindowFineAlignDense-v0",
        measurement_noise_translation_std=0.0,
        measurement_noise_angle_std_deg=0.0,
        hold_steps=5,
    )
    try:
        policy = CorrectivePolicy(env.unwrapped)
        result = WindowAssemblyPipeline(env).run_episode(policy, seed=7)
        assert result["full_pipeline_success"]
        assert [record["stage"] for record in result["stage_records"]] == [
            stage.value for stage in WINDOW_ASSEMBLY_STAGE_ORDER
        ]
        assert all(record["success"] for record in result["stage_records"])
        assert policy.calls == result["fine_align_steps"]
        assert result["ready_for_insert"]
        assert result["insert_success"] and result["hold_success"]
    finally:
        env.close()


def test_old_environment_ids_and_training_entries_still_import():
    for env_id in (
        "FrankaPickAndPlaceWindowSparse-v0",
        "FrankaPickAndPlaceWindowDense-v0",
        "FrankaPickAndPlaceWindowInsertDense-v0",
        "FrankaPickAndPlaceWindowPrealignDense-v0",
    ):
        env = gym.make(env_id)
        env.close()
    importlib.import_module("train.train_sac")
    importlib.import_module("train.train_window_fine_align_sac")


def test_new_training_entry_has_no_legacy_expert_stack_calls():
    source = Path("train/train_window_fine_align_sac.py").read_text(encoding="utf-8")
    forbidden = (
        "collect_" + "pickplace_expert_dataset",
        "run_" + "behavior_cloning_pretrain",
        "prefill_" + "replay_buffer_from_dataset",
        "PickAndPlace" + "ResidualGuidanceWrapper",
        "Her" + "ReplayBuffer",
    )
    assert all(name not in source for name in forbidden)
    assert 'policy="MlpPolicy"' in source
