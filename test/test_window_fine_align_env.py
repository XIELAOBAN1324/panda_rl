"""MuJoCo integration tests for the fine-alignment environment."""

import gymnasium as gym
import numpy as np

import panda_mujoco_gym  # noqa: F401


ENV_ID = "FrankaWindowFineAlignDense-v0"
NO_NOISE = {
    "measurement_noise_translation_std": 0.0,
    "measurement_noise_angle_std_deg": 0.0,
}


def _corrective_action(base, info):
    return np.clip(
        np.array(
            [
                -info["error_u"] / base.fine_position_action_scale,
                -info["error_v"] / base.fine_position_action_scale,
                -info["tilt_error_t1"] / base.fine_tilt_action_scale_rad,
                -info["tilt_error_t2"] / base.fine_tilt_action_scale_rad,
                -info["yaw_error"] / base.fine_yaw_action_scale_rad,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )


def test_reset_spaces_seeded_coarse_state_and_no_normal_action():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        observation_a, info_a = env.reset(seed=19)
        sample_a = info_a["coarse_alignment_sample"].copy()
        observation_b, info_b = env.reset(seed=19)
        assert env.action_space.shape == (5,)
        assert observation_a.shape == (11,)
        assert observation_a.dtype == np.float32
        assert env.observation_space.contains(observation_a)
        assert np.allclose(observation_a, observation_b, atol=2e-4)
        assert np.allclose(sample_a, info_b["coarse_alignment_sample"])
        assert abs(info_a["error_u"]) <= env.unwrapped.coarse_translation_range
        assert abs(info_a["error_v"]) <= env.unwrapped.coarse_translation_range
        assert info_a["stage"] == "fine_align"

        frame_normal = info_a["frame_normal_world"]
        frame_rotation = info_a["frame_rotation_world"] if "frame_rotation_world" in info_a else None
        assert frame_rotation is None  # only measured diagnostics are exposed
        base = env.unwrapped
        frame_corners = base.get_frame_corners_world()
        glass_corners = base.get_glass_corners_world()
        assert np.isclose(np.linalg.norm(frame_corners[1] - frame_corners[0]), 0.54)
        assert np.isclose(np.linalg.norm(frame_corners[0] - frame_corners[3]), 0.36)
        assert np.isclose(np.linalg.norm(glass_corners[1] - glass_corners[0]), 0.53)
        assert np.isclose(np.linalg.norm(glass_corners[0] - glass_corners[3]), 0.35)
        for action in np.eye(5, dtype=np.float32):
            translation = (
                action[0] * base.fine_position_action_scale * base.get_fine_alignment_errors()["frame_rotation_world"][:, 0]
                + action[1] * base.fine_position_action_scale * base.get_fine_alignment_errors()["frame_rotation_world"][:, 1]
            )
            assert abs(float(np.dot(translation, frame_normal))) < 1e-12
    finally:
        env.close()


def test_step_keeps_suction_reports_reward_and_never_auto_inserts():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        env.reset(seed=7)
        base = env.unwrapped
        initial_gap = base.get_fine_alignment_errors()["current_normal_gap"]
        observation, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
        assert observation.shape == (11,)
        assert np.isfinite(reward)
        assert not terminated and not truncated
        assert info["is_attached"]
        assert base.stage.value == "fine_align"
        assert not base.insert_success
        assert abs(info["current_normal_gap"] - initial_gap) < base.max_normal_gap_drift
        for key in (
            "reward_progress",
            "reward_state",
            "reward_action",
            "reward_smoothness",
            "reward_terminal",
            "mean_alignment_cost",
            "worst_alignment_cost",
            "alignment_cost",
            "gap_cost",
            "state_cost",
        ):
            assert key in info and np.isfinite(info[key])
        assert info["reward_state"] < 0.0
    finally:
        env.close()


def test_cost_progress_and_round_trip_reward_properties():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        env.reset(seed=3)
        base = env.unwrapped
        good = {
            "error_u": 0.001,
            "error_v": 0.0,
            "tilt_error_t1": 0.0,
            "tilt_error_t2": 0.0,
            "yaw_error": 0.0,
            "normal_gap_drift": 0.0,
        }
        bad = dict(good, error_u=0.01)
        good_cost = float(base._costs(good)["state_cost"])
        bad_cost = float(base._costs(bad)["state_cost"])
        assert good_cost < bad_cost
        improving_progress = 2.0 * np.clip(bad_cost - good_cost, -0.5, 0.5)
        worsening_progress = 2.0 * np.clip(good_cost - bad_cost, -0.5, 0.5)
        assert improving_progress > worsening_progress
        round_trip_progress = improving_progress + worsening_progress
        action_penalties = -0.005 * 2.0 - 0.01 * 4.0
        assert round_trip_progress + action_penalties < 0.0
    finally:
        env.close()


def test_success_requires_consecutive_steps_then_allows_separate_insert_and_hold():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        observation, info = env.reset(seed=7)
        base = env.unwrapped
        terminated = truncated = False
        for _ in range(40):
            observation, _, terminated, truncated, info = env.step(_corrective_action(base, info))
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["fine_align_success"] and info["ready_for_insert"]
        assert info["success_hold_count"] == base.success_hold_steps
        assert not base.insert_success  # the successful RL step did not insert
        assert base.run_scripted_insert()
        assert base.run_scripted_hold(hold_steps=5)
    finally:
        env.close()


def test_time_limit_is_truncation_and_safety_failure_is_termination():
    timeout_env = gym.make(
        ENV_ID,
        max_episode_steps=2,
        fine_align_max_episode_steps=2,
        **NO_NOISE,
    )
    try:
        timeout_env.reset(seed=4)
        _, _, terminated, truncated, _ = timeout_env.step(np.zeros(5, dtype=np.float32))
        assert not terminated and not truncated
        _, _, terminated, truncated, info = timeout_env.step(np.zeros(5, dtype=np.float32))
        assert not terminated and truncated
        assert info["failure_reason"] == "time_limit"
    finally:
        timeout_env.close()

    failure_env = gym.make(ENV_ID, **NO_NOISE)
    try:
        failure_env.reset(seed=4)
        failure_env.unwrapped._set_grasp_weld_active(False)
        _, reward, terminated, truncated, info = failure_env.step(np.zeros(5, dtype=np.float32))
        assert terminated and not truncated
        assert reward < -9.0
        assert info["failure_reason"] == "suction_weld_broken"
    finally:
        failure_env.close()


def test_scripted_insert_is_rejected_before_ready():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        env.reset(seed=11)
        assert not env.unwrapped.run_scripted_insert()
        assert env.unwrapped.stage.value == "fine_align"
    finally:
        env.close()


def test_measurement_noise_changes_only_policy_observation():
    env = gym.make(
        ENV_ID,
        measurement_noise_translation_std=0.001,
        measurement_noise_angle_std_deg=0.25,
    )
    try:
        observation, info = env.reset(seed=29)
        base = env.unwrapped
        ground_truth_normalized = np.array(
            [
                info["error_u"] / base.translation_tolerance,
                info["error_v"] / base.translation_tolerance,
                info["tilt_error_t1"] / base.tilt_tolerance_rad,
                info["tilt_error_t2"] / base.tilt_tolerance_rad,
                info["yaw_error"] / base.yaw_tolerance_rad,
                info["normal_gap_drift"] / base.normal_gap_tolerance,
            ]
        )
        assert not np.allclose(observation[:6], ground_truth_normalized)
        qpos_before = base.data.qpos.copy()
        errors_before = base.get_fine_alignment_errors()
        base._noisy_errors(errors_before)
        assert np.array_equal(base.data.qpos, qpos_before)
        errors_after = base.get_fine_alignment_errors()
        for key in ("error_u", "error_v", "tilt_error_t1", "tilt_error_t2", "yaw_error"):
            assert np.isclose(errors_before[key], errors_after[key])
        assert base.reward_uses_ground_truth and base.success_uses_ground_truth
    finally:
        env.close()
