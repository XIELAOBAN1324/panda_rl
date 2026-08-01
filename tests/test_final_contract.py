"""Regression tests for the final window-assembly contract."""

import gymnasium as gym
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

import panda_mujoco_gym  # noqa: F401
from panda_mujoco_gym.envs.window_fine_align import FrankaWindowFineAlignEnv
from panda_mujoco_gym.envs.window_assembly_pipeline import (
    WINDOW_ASSEMBLY_STAGE_ORDER,
    WindowAssemblyPipeline,
)


EXPECTED_STAGES = (
    "approach",
    "descend",
    "grasp",
    "lift",
    "reorient",
    "coarse_align",
    "fine_align",
    "insert",
    "hold",
)


def test_only_final_environment_is_registered_and_removed_kwargs_are_rejected():
    project_ids = {
        env_id
        for env_id in gym.envs.registry
        if env_id.startswith("FrankaWindow")
    }
    assert project_ids == {"FrankaWindowFineAlignDense-v0"}
    for old_env_id in (
        "Franka" + "PickAndPlace-v0",
        "Franka" + "Insert-v0",
        "Franka" + "Prealign-v0",
    ):
        with pytest.raises(gym.error.Error):
            gym.spec(old_env_id)
    assert "reward_type" not in inspect.signature(
        FrankaWindowFineAlignEnv.__init__
    ).parameters
    for removed_kwargs in (
        {"reward_type": "dense"},
        {"reset_mode": "full_task"},
        {"full_task": True},
        {"post_grasp_lifted": True},
        {"post_grasp_prealign": True},
    ):
        with pytest.raises(TypeError):
            gym.make("FrankaWindowFineAlignDense-v0", **removed_kwargs)


def test_fine_align_spaces_and_actions_have_no_normal_insertion_control():
    env = gym.make(
        "FrankaWindowFineAlignDense-v0",
        measurement_noise_translation_std=0.0,
        measurement_noise_angle_std_deg=0.0,
    )
    try:
        observation, info = env.reset(seed=5)
        base = env.unwrapped

        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (5,)
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (11,)
        assert observation.shape == (11,)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert info["stage"] == "fine_align"

        frame_rotation = base.get_fine_alignment_errors()["frame_rotation_world"]
        for action in np.eye(5, dtype=np.float32):
            translation = (
                action[0] * base.fine_position_action_scale * frame_rotation[:, 0]
                + action[1] * base.fine_position_action_scale * frame_rotation[:, 1]
            )
            assert abs(float(np.dot(translation, frame_rotation[:, 2]))) < 1e-12

        step_observation, _, _, _, step_info = env.step(
            np.zeros(5, dtype=np.float32)
        )
        assert step_observation.shape == (11,)
        assert env.observation_space.contains(step_observation)
        assert step_info["stage"] == "fine_align"
        assert not base.insert_success
        assert not base.hold_success
    finally:
        env.close()


class _Tracker:
    def as_dicts(self):
        return [
            {
                "stage": stage,
                "start_time": float(index),
                "end_time": float(index + 1),
                "steps": 1,
                "success": True,
                "failure_reason": None,
            }
            for index, stage in enumerate(EXPECTED_STAGES)
        ]

    records = []


class _PipelineController:
    def __init__(self, env):
        self.env = env

    def run_insert(self):
        self.env.calls.append("insert")
        return True

    def run_hold(self):
        self.env.calls.append("hold")
        return True

    @staticmethod
    def verify_final_assembly():
        return True, {"assembly_verified": True}

    @staticmethod
    def get_last_insert_diagnostics():
        return {}


class _PipelineEnv:
    ERROR_VALUES = {
        "error_u": 0.0,
        "error_v": 0.0,
        "tilt_error_t1": 0.0,
        "tilt_error_t2": 0.0,
        "yaw_error": 0.0,
        "normal_gap_error": 0.0,
        "current_normal_gap": 0.08,
    }

    def __init__(self, calls):
        self.calls = calls
        self.unwrapped = self
        self.stage_tracker = _Tracker()
        self.stage = SimpleNamespace(value="hold")
        self.script_controller = _PipelineController(self)

    def reset(self, seed):
        self.calls.extend(EXPECTED_STAGES[:6])
        self.calls.append(("reset_seed", seed))
        return np.zeros(11, dtype=np.float32), dict(self.ERROR_VALUES)

    def step(self, action):
        assert np.asarray(action).shape == (5,)
        self.calls.append("fine_align_step")
        info = dict(self.ERROR_VALUES, fine_align_success=True, ready_for_insert=True)
        return np.zeros(11, dtype=np.float32), 1.0, True, False, info

    def get_fine_alignment_errors(self):
        return dict(self.ERROR_VALUES)


class _Policy:
    def __init__(self, calls):
        self.calls = calls
        self.call_count = 0

    def predict(self, observation, deterministic=True):
        assert np.asarray(observation).shape == (11,)
        assert deterministic
        self.calls.append("policy")
        self.call_count += 1
        return np.zeros(5, dtype=np.float32), None


def test_pipeline_order_and_control_ownership():
    assert tuple(stage.value for stage in WINDOW_ASSEMBLY_STAGE_ORDER) == EXPECTED_STAGES
    assert EXPECTED_STAGES.index("reorient") == EXPECTED_STAGES.index("lift") + 1

    calls = []
    policy = _Policy(calls)
    result = WindowAssemblyPipeline(_PipelineEnv(calls)).run_episode(policy, seed=17)

    assert calls == [
        "approach",
        "descend",
        "grasp",
        "lift",
        "reorient",
        "coarse_align",
        ("reset_seed", 17),
        "policy",
        "fine_align_step",
        "insert",
        "hold",
    ]
    assert policy.call_count == result["fine_align_steps"] == 1
    assert result["insert_success"]
    assert result["hold_success"]
    assert result["full_pipeline_success"]
