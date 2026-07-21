"""Pipeline ordering and compatibility tests."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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
    for flag in (
        "--preinsert-normal-offset",
        "--insert-action-sim-steps",
        "--insert-translation-abort-tolerance",
        "--insert-tilt-abort-tolerance-deg",
        "--insert-yaw-abort-tolerance-deg",
        "--insert-alignment-violation-hold-steps",
        "--insert-step-size",
    ):
        assert flag in source


def test_evaluation_restores_training_config_and_cli_override_wins(tmp_path, monkeypatch):
    eval_module = importlib.import_module("evaluate.evaluate_window_assembly_pipeline")
    run_dir = tmp_path / "run"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best_model.zip"
    model_path.write_bytes(b"placeholder")
    config = {
        "train_environment_kwargs": {
            "preinsert_normal_offset": 0.20,
            "insert_action_sim_steps": 1,
            "insert_translation_abort_tolerance": 0.003,
        }
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        model=str(model_path),
        measurement_noise_translation_std=0.0,
        measurement_noise_angle_std=0.0,
        preinsert_normal_offset=0.08,
        fine_action_sim_steps=4,
        normal_gap_correction_threshold=0.0005,
        normal_gap_correction_sim_steps=2,
        max_normal_gap_drift=0.008,
        insert_action_sim_steps=4,
        insert_translation_abort_tolerance=0.0025,
        insert_tilt_abort_tolerance_deg=0.6,
        insert_yaw_abort_tolerance_deg=0.6,
        insert_alignment_violation_hold_steps=2,
        insert_step_size=0.001,
    )
    monkeypatch.setattr(
        eval_module.sys,
        "argv",
        [
            "evaluate_window_assembly_pipeline.py",
            "--model",
            str(model_path),
            "--preinsert-normal-offset",
            "0.08",
        ],
    )

    env_kwargs, config_path, overrides = eval_module.environment_kwargs(args)

    assert config_path == run_dir / "training_config.json"
    assert env_kwargs["preinsert_normal_offset"] == 0.08
    assert env_kwargs["insert_action_sim_steps"] == 1
    assert env_kwargs["insert_translation_abort_tolerance"] == 0.003
    assert env_kwargs["insert_step_size"] == 0.001
    assert overrides == {"preinsert_normal_offset": (0.20, 0.08)}
