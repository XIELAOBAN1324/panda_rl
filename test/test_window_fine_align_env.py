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


def _inplane_error_norm(info):
    return float(np.hypot(info["error_u"], info["error_v"]))


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
        assert np.allclose(observation_a, observation_b, atol=2e-3)
        assert np.allclose(sample_a, info_b["coarse_alignment_sample"])
        assert abs(info_a["error_u"]) <= env.unwrapped.coarse_translation_range
        assert abs(info_a["error_v"]) <= env.unwrapped.coarse_translation_range
        assert info_a["stage"] == "fine_align"
        assert np.isclose(info_a["reference_normal_gap"], 0.08)
        assert abs(info_a["current_normal_gap"] - 0.08) <= 0.003
        assert np.isclose(info_a["normal_gap_drift"], info_a["normal_gap_error"])

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
        assert info["fine_action_sim_steps"] == 4
        assert np.isclose(info["commanded_normal_gap"], base.reference_normal_gap)
        assert abs(info["normal_gap_command_error"]) < 1e-8
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
            "normal_gap_error",
            "current_normal_gap",
            "ee_target_error",
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
        pre_insert_gap = float(base.get_fine_alignment_errors()["current_normal_gap"])
        assert pre_insert_gap > 0.075
        assert base.run_scripted_insert()
        post_insert_gap = float(base.get_fine_alignment_errors()["current_normal_gap"])
        assert post_insert_gap < 0.01
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


def test_reset_uses_fixed_preinsert_plane_and_keeps_random_inplane_error():
    env = gym.make(ENV_ID, **NO_NOISE)
    inplane_errors = []
    try:
        for seed in range(30, 40):
            _, info = env.reset(seed=seed)
            assert abs(info["current_normal_gap"] - 0.08) <= 0.003
            assert abs(info["measured_initial_normal_gap"] - 0.08) <= 0.003
            assert abs(info["error_u"]) <= env.unwrapped.coarse_translation_range
            assert abs(info["error_v"]) <= env.unwrapped.coarse_translation_range
            inplane_errors.append((round(float(info["error_u"]), 5), round(float(info["error_v"]), 5)))
        assert len(set(inplane_errors)) > 1
        assert any(abs(u) > 0.002 for u, _ in inplane_errors)
        assert any(abs(v) > 0.002 for _, v in inplane_errors)
    finally:
        env.close()


def test_normal_gap_projection_preserves_inplane_coordinates():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        frame_center, frame_rotation = base._current_fine_frame_pose()
        candidate_center = (
            frame_center
            + 0.010 * frame_rotation[:, 0]
            - 0.007 * frame_rotation[:, 1]
            + 0.123 * frame_rotation[:, 2]
        )
        target_center, diagnostics = base._project_center_to_reference_normal_gap(
            candidate_center,
            frame_center=frame_center,
            frame_rotation=frame_rotation,
        )
        for axis_index in (0, 1):
            candidate_coord = float(np.dot(frame_rotation[:, axis_index], candidate_center - frame_center))
            target_coord = float(np.dot(frame_rotation[:, axis_index], target_center - frame_center))
            assert np.isclose(candidate_coord, target_coord, atol=1e-8)
        target_gap = float(np.dot(frame_rotation[:, 2], target_center - frame_center))
        assert np.isclose(target_gap, base.reference_normal_gap, atol=1e-8)
        assert np.isclose(diagnostics["candidate_normal_gap"], 0.123)
        assert abs(diagnostics["normal_gap_command_error"]) < 1e-8
    finally:
        env.close()


def test_projection_does_not_auto_remove_inplane_error_but_policy_action_can_reduce_it():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        _, info = env.reset(seed=0)
        base = env.unwrapped
        before_norm = _inplane_error_norm(info)
        assert before_norm > 0.005

        _, _, terminated, truncated, zero_info = env.step(np.zeros(5, dtype=np.float32))
        assert not terminated and not truncated
        assert _inplane_error_norm(zero_info) > 0.5 * before_norm
        assert _inplane_error_norm(zero_info) > base.translation_tolerance

        action = np.array(
            [
                -zero_info["error_u"] / base.fine_position_action_scale,
                -zero_info["error_v"] / base.fine_position_action_scale,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        _, _, terminated, truncated, moved_info = env.step(np.clip(action, -1.0, 1.0))
        assert not terminated and not truncated
        assert _inplane_error_norm(moved_info) < _inplane_error_norm(zero_info)
    finally:
        env.close()


def test_pure_tilt_actions_do_not_accumulate_normal_gap_or_auto_center_inplane_error():
    actions = (
        np.array([0, 0, 1, 0, 0], dtype=np.float32),
        np.array([0, 0, -1, 0, 0], dtype=np.float32),
        np.array([0, 0, 0, 1, 0], dtype=np.float32),
        np.array([0, 0, 0, -1, 0], dtype=np.float32),
    )
    for index, action in enumerate(actions):
        env = gym.make(ENV_ID, **NO_NOISE)
        try:
            _, info = env.reset(seed=80 + index)
            initial_tilt = np.array([info["tilt_error_t1"], info["tilt_error_t2"]], dtype=np.float64)
            initial_inplane = _inplane_error_norm(info)
            gap_errors = [abs(float(info["normal_gap_error"]))]
            for _ in range(12):
                _, _, terminated, truncated, info = env.step(action)
                gap_errors.append(abs(float(info["normal_gap_error"])))
                assert info["failure_reason"] != "normal_gap_drift_exceeded"
                if terminated or truncated:
                    break
            final_tilt = np.array([info["tilt_error_t1"], info["tilt_error_t2"]], dtype=np.float64)
            assert float(np.linalg.norm(final_tilt - initial_tilt)) > 0.005
            assert _inplane_error_norm(info) > min(0.5 * initial_inplane, env.unwrapped.translation_tolerance)
            assert max(gap_errors) < 0.005
        finally:
            env.close()


def test_random_action_pressure_keeps_normal_gap_failure_near_zero():
    rng = np.random.default_rng(123)
    normal_gap_drift_failures = 0
    normal_gap_errors = []
    correction_count = 0
    for episode_index in range(20):
        env = gym.make(
            ENV_ID,
            normal_gap_correction_threshold=1e-7,
            **NO_NOISE,
        )
        try:
            _, info = env.reset(seed=1000 + episode_index)
            normal_gap_errors.append(abs(float(info["normal_gap_error"])))
            for _ in range(100):
                action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
                _, _, terminated, truncated, info = env.step(action)
                normal_gap_errors.append(abs(float(info["normal_gap_error"])))
                correction_count += int(bool(info["normal_gap_correction_applied"]))
                if terminated or truncated:
                    if info["failure_reason"] == "normal_gap_drift_exceeded":
                        normal_gap_drift_failures += 1
                    break
        finally:
            env.close()

    assert normal_gap_drift_failures == 0
    assert max(normal_gap_errors) < 0.005
    assert float(np.mean(normal_gap_errors)) < 0.002
    assert correction_count > 0


def test_fine_alignment_actions_do_not_move_glass_toward_insert_region():
    rng = np.random.default_rng(321)
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        _, info = env.reset(seed=44)
        min_gap = float(info["current_normal_gap"])
        for _ in range(50):
            action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)
            min_gap = min(min_gap, float(info["current_normal_gap"]))
            assert info["current_normal_gap"] > 0.075
            assert not env.unwrapped.insert_success
            if terminated or truncated:
                break
        assert min_gap > 0.075
        assert env.unwrapped.stage.value == "fine_align"
    finally:
        env.close()


def _insert_error_dict(**overrides):
    errors = {
        "error_u": 0.0,
        "error_v": 0.0,
        "tilt_error_t1": 0.0,
        "tilt_error_t2": 0.0,
        "yaw_error": 0.0,
        "normal_gap_error": 0.0,
        "current_normal_gap": 0.08,
        "glass_center_world": np.array([0.0, 0.0, 0.2], dtype=np.float64),
    }
    errors.update(overrides)
    return errors


def test_insert_abort_tolerance_adds_hysteresis_over_fine_success_threshold():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        base = env.unwrapped
        assert not base._within_alignment_tolerance(_insert_error_dict(error_u=0.0021))
        assert base.get_insert_alignment_failure_reason(
            _insert_error_dict(error_u=0.0021)
        ) == ""
    finally:
        env.close()


def test_insert_alignment_violation_requires_consecutive_steps():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        base = env.unwrapped
        count, reason = base._update_insert_alignment_violation_count(
            _insert_error_dict(error_u=0.0026),
            0,
        )
        assert count == 1
        assert reason == ""

        count, reason = base._update_insert_alignment_violation_count(
            _insert_error_dict(error_u=0.0),
            count,
        )
        assert count == 0
        assert reason == ""

        count, reason = base._update_insert_alignment_violation_count(
            _insert_error_dict(error_u=0.0026),
            count,
        )
        assert count == 1
        assert reason == ""
        count, reason = base._update_insert_alignment_violation_count(
            _insert_error_dict(error_u=0.0026),
            count,
        )
        assert count == base.insert_alignment_violation_hold_steps
        assert reason == "insert_error_u_exceeded"
    finally:
        env.close()


def test_insert_alignment_failure_reason_reports_largest_specific_excess():
    env = gym.make(ENV_ID, **NO_NOISE)
    try:
        base = env.unwrapped
        cases = (
            ("insert_error_u_exceeded", {"error_u": 0.0030}),
            ("insert_error_v_exceeded", {"error_v": -0.0030}),
            (
                "insert_tilt_t1_exceeded",
                {"tilt_error_t1": base.insert_tilt_abort_tolerance_rad * 1.2},
            ),
            (
                "insert_tilt_t2_exceeded",
                {"tilt_error_t2": -base.insert_tilt_abort_tolerance_rad * 1.2},
            ),
            (
                "insert_yaw_exceeded",
                {"yaw_error": base.insert_yaw_abort_tolerance_rad * 1.2},
            ),
            (
                "insert_yaw_exceeded",
                {
                    "error_u": base.insert_translation_abort_tolerance * 1.1,
                    "yaw_error": base.insert_yaw_abort_tolerance_rad * 1.5,
                },
            ),
        )
        for expected, overrides in cases:
            assert base.get_insert_alignment_failure_reason(
                _insert_error_dict(**overrides)
            ) == expected
    finally:
        env.close()


def test_scripted_insert_uses_configured_substeps_feedback_and_monotonic_gap(monkeypatch):
    env = gym.make(ENV_ID, insert_action_sim_steps=5, hold_steps=5, **NO_NOISE)
    target_gaps = []
    call_steps = []
    feedback_gains = []
    try:
        observation, info = env.reset(seed=7)
        base = env.unwrapped
        terminated = truncated = False
        for _ in range(40):
            observation, _, terminated, truncated, info = env.step(
                _corrective_action(base, info)
            )
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["fine_align_success"] and info["ready_for_insert"]
        pre_insert_gap = float(base.get_fine_alignment_errors()["current_normal_gap"])
        assert abs(pre_insert_gap - base.reference_normal_gap) <= 0.003

        original_move = base._move_attached_object_pose

        def wrapped_move(target_center, target_rotation, n_steps, *, center_feedback_gain=0.0):
            if base.stage.value == "insert":
                target_gaps.append(float(base._normal_gap_for_center(target_center)))
                call_steps.append(int(n_steps))
                feedback_gains.append(float(center_feedback_gain))
            return original_move(
                target_center,
                target_rotation,
                n_steps,
                center_feedback_gain=center_feedback_gain,
            )

        monkeypatch.setattr(base, "_move_attached_object_pose", wrapped_move)
        assert base.run_scripted_insert()
        post_insert_gap = float(base.get_fine_alignment_errors()["current_normal_gap"])
        assert post_insert_gap < 0.01
        assert target_gaps
        assert all(step == 5 for step in call_steps)
        assert all(np.isclose(gain, 1.0) for gain in feedback_gains)
        assert all(
            next_gap <= current_gap + 1e-9
            for current_gap, next_gap in zip(target_gaps, target_gaps[1:])
        )
        assert min(target_gaps) <= -base.insert_final_gap + 1e-9
        diagnostics = base.get_last_insert_diagnostics()
        assert diagnostics["insert_action_sim_steps"] == 5
        assert diagnostics["insert_steps"] == len(call_steps)
        assert diagnostics["failure_reason"] == ""
    finally:
        env.close()
