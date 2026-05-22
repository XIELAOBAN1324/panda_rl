import gymnasium as gym
import numpy as np

import panda_mujoco_gym  # noqa: F401
from train.common.expert_pickplace import collect_pickplace_expert_dataset
from train.common.config import SACConfig
from train.common.wrappers import PickAndPlaceDenseRewardWrapper
from train.train_sac import apply_pickplace_sparse_defaults


def _collect_goal_distances(env, num_resets: int = 64):
    distances = []
    for seed in range(num_resets):
        obs, _ = env.reset(seed=seed)
        distances.append(float(np.linalg.norm(obs["achieved_goal"][:3] - obs["desired_goal"][:3])))
    return distances


def test_pickplace_window_sparse_reset_avoids_immediate_success():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    min_distance = env.unwrapped.minimum_goal_object_distance()
    distances = _collect_goal_distances(env)
    env.close()

    assert min(distances) >= min_distance - 1e-6


def test_pickplace_window_reset_shapes_and_fixed_goal_height():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    try:
        for seed in range(8):
            obs, _ = env.reset(seed=seed)
            assert env.action_space.shape == (6,)
            assert obs["observation"].shape == (19,)
            assert obs["achieved_goal"].shape == (6,)
            assert obs["desired_goal"].shape == (6,)
            window_center = env.unwrapped.get_window_center()
            window_safe_goal = env.unwrapped.get_window_safe_goal()
            radius = float(np.linalg.norm(window_center[:2]))
            angle = float(np.rad2deg(np.arctan2(window_center[1], window_center[0])))

            assert abs(float(window_center[2]) - 0.65) < 1e-6
            assert abs(radius - 0.80) < 1e-6
            assert -60.0 <= angle <= 0.0
            assert np.allclose(obs["desired_goal"][:3], window_center, atol=1e-6)
            assert np.allclose(obs["desired_goal"][3:6], np.array([1.0, 0.0, 0.0]), atol=1e-6)
            assert np.allclose(window_safe_goal, window_center, atol=1e-6)
            assert env.unwrapped.is_attached
            assert obs["observation"][-1] > 0.5
    finally:
        env.close()


def test_pickplace_window_success_requires_position_and_alignment():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    base = env.unwrapped
    try:
        desired_goal = np.array([0.77, 0.00, 0.65, -1.0, 0.0, 0.0], dtype=np.float32)
        pos_only = np.array([0.77, 0.00, 0.65, 0.0, 0.0, 1.0], dtype=np.float32)
        align_only = np.array([0.65, 0.00, 0.65, -1.0, 0.0, 0.0], dtype=np.float32)
        both = np.array([0.77, 0.00, 0.65, -1.0, 0.0, 0.0], dtype=np.float32)
        flipped = np.array([0.77, 0.00, 0.65, 1.0, 0.0, 0.0], dtype=np.float32)

        assert float(base._is_success(pos_only, desired_goal)) == 0.0
        assert float(base._is_success(align_only, desired_goal)) == 0.0
        assert float(base._goal_alignment(both, desired_goal)) == 1.0
        assert float(base._goal_alignment(flipped, desired_goal)) == -1.0
    finally:
        env.close()


def test_pickplace_window_grasp_weld_attaches_and_releases():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        assert base.is_attached
        assert int(base.model.eq_active[base.grasp_weld_eq_id]) == 1

        base._set_grasp_weld_active(False)
        assert not base.is_attached
        assert int(base.model.eq_active[base.grasp_weld_eq_id]) == 0
    finally:
        env.close()


def test_pickplace_window_plane_violation_rolls_back():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        near_plane = base.get_window_center() - np.array([0.035, 0.0, 0.20], dtype=np.float64)
        base.set_mocap_pose(near_plane, base.grasp_site_pose)
        for _ in range(20):
            base._mujoco_step()
            base._mujoco.mj_forward(base.model, base.data)
        start_ee = base.get_ee_position().copy()
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        end_ee = base.get_ee_position().copy()

        assert bool(info["plane_violation"])
        assert float(info["max_plane_penetration"]) > 0.0
        assert float(end_ee[0]) <= float(base.get_window_center()[0]) + 1e-6
        assert float(end_ee[0]) <= float(start_ee[0]) + 1e-6
    finally:
        env.close()


def test_pickplace_window_insert_reset_starts_attached_and_lifted():
    env = gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")
    base = env.unwrapped
    try:
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            window_center = base.get_window_center()
            radius = float(np.linalg.norm(window_center[:2]))
            angle = float(np.rad2deg(np.arctan2(window_center[1], window_center[0])))

            assert base.is_attached
            assert obs["observation"][-1] > 0.5
            assert obs["achieved_goal"][2] > base.pickup_station_center[2] + 0.10
            assert abs(float(window_center[2]) - 0.65) < 1e-6
            assert abs(radius - 0.80) < 1e-6
            assert -60.0 <= angle <= 0.0
            assert base._max_object_penetration() <= base.plane_constraint_eps + 1e-9
            assert not base._has_window_collision()
    finally:
        env.close()


def test_pickplace_window_prealign_reset_starts_attached_near_window():
    env = gym.make("FrankaPickAndPlaceWindowPrealignSparse-v0")
    base = env.unwrapped
    try:
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            object_position = obs["achieved_goal"][:3]
            goal_position = obs["desired_goal"][:3]
            goal_normal = obs["desired_goal"][3:6]
            goal_distance = float(np.linalg.norm(object_position - goal_position))
            orientation_alignment = float(np.dot(obs["achieved_goal"][3:6], goal_normal))
            signed_plane_margin = float(np.dot(goal_position - object_position, goal_normal))

            assert base.is_attached
            assert obs["observation"][-1] > 0.5
            assert goal_distance < 0.35
            assert orientation_alignment > 0.90
            assert signed_plane_margin >= -1e-6
            assert base._max_object_penetration() <= base.plane_constraint_eps + 1e-9
            assert not base._has_window_collision()
    finally:
        env.close()


def test_pickplace_window_prealign_reset_is_closer_than_insert_reset():
    prealign_env = gym.make("FrankaPickAndPlaceWindowPrealignSparse-v0")
    insert_env = gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")
    try:
        prealign_distances = []
        insert_distances = []
        prealign_alignments = []
        insert_alignments = []
        for seed in range(8):
            prealign_obs, _ = prealign_env.reset(seed=seed)
            insert_obs, _ = insert_env.reset(seed=seed)
            prealign_distances.append(
                float(np.linalg.norm(prealign_obs["achieved_goal"][:3] - prealign_obs["desired_goal"][:3]))
            )
            insert_distances.append(
                float(np.linalg.norm(insert_obs["achieved_goal"][:3] - insert_obs["desired_goal"][:3]))
            )
            prealign_alignments.append(
                float(np.dot(prealign_obs["achieved_goal"][3:6], prealign_obs["desired_goal"][3:6]))
            )
            insert_alignments.append(
                float(np.dot(insert_obs["achieved_goal"][3:6], insert_obs["desired_goal"][3:6]))
            )

        assert float(np.mean(prealign_distances)) < float(np.mean(insert_distances))
        assert float(np.mean(prealign_alignments)) > float(np.mean(insert_alignments))
    finally:
        prealign_env.close()
        insert_env.close()


def test_pickplace_window_insert_success_thresholds():
    env = gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        desired_goal = base.goal.copy()

        within_pos = desired_goal[:3] + np.array([0.004, 0.0, 0.0], dtype=np.float32)
        outside_pos = desired_goal[:3] + np.array([0.006, 0.0, 0.0], dtype=np.float32)
        good_axis = desired_goal[3:6].copy()
        bad_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        assert float(base._is_success(np.concatenate([within_pos, good_axis]), desired_goal)) in {0.0, 1.0}
        assert float(base._is_success(np.concatenate([outside_pos, good_axis]), desired_goal)) == 0.0
        assert float(base._is_success(np.concatenate([within_pos, bad_axis]), desired_goal)) == 0.0
    finally:
        env.close()


def test_pickplace_window_insert_sparse_reward_matches_position_threshold():
    env = gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        desired_goal = base.goal.copy()
        outside_goal = desired_goal.copy()
        outside_goal[0] += 0.006
        info = {
            "collision": False,
            "plane_violation": False,
            "glass_fits_window": True,
        }

        assert float(base._is_success(outside_goal, desired_goal)) == 0.0
        assert float(base.compute_reward(outside_goal, desired_goal, info)) == -1.0
    finally:
        env.close()


def test_pickplace_window_prealign_thresholds_are_relaxed():
    prealign_env = gym.make("FrankaPickAndPlaceWindowPrealignSparse-v0")
    strict_env = gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")
    try:
        prealign_base = prealign_env.unwrapped
        strict_base = strict_env.unwrapped

        assert np.isclose(prealign_base.window_position_threshold, 0.01)
        assert np.isclose(strict_base.window_position_threshold, 0.005)
        assert np.isclose(prealign_base.orientation_threshold_cos, np.cos(np.deg2rad(8.0)))
        assert np.isclose(strict_base.orientation_threshold_cos, np.cos(np.deg2rad(5.0)))
        assert np.isclose(prealign_base.glass_alignment_threshold_cos, np.cos(np.deg2rad(8.0)))
        assert np.isclose(strict_base.glass_alignment_threshold_cos, np.cos(np.deg2rad(5.0)))
    finally:
        prealign_env.close()
        strict_env.close()


def test_pickplace_window_insert_plane_violation_terminates():
    env = gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        near_plane = base.get_window_center() - np.array([0.02, 0.0, 0.0], dtype=np.float64)
        base.set_mocap_pose(near_plane, base.get_mocap_quaternion())
        for _ in range(10):
            base._mujoco_step()
            base._mujoco.mj_forward(base.model, base.data)
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        assert terminated
        assert not truncated
        assert bool(info["collision"])
        assert bool(info["plane_violation"])
    finally:
        env.close()


def test_pickplace_window_prealign_plane_violation_terminates():
    env = gym.make("FrankaPickAndPlaceWindowPrealignSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        near_plane = base.get_window_center() - np.array([0.02, 0.0, 0.0], dtype=np.float64)
        base.set_mocap_pose(near_plane, base.get_mocap_quaternion())
        for _ in range(10):
            base._mujoco_step()
            base._mujoco.mj_forward(base.model, base.data)
        _, _, terminated, truncated, info = env.step(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        assert terminated
        assert not truncated
        assert bool(info["collision"])
        assert bool(info["plane_violation"])
    finally:
        env.close()


def test_apply_pickplace_sparse_defaults_for_window_prealign_and_insert():
    class Args:
        reward_scale = None
        her = None
        n_envs = None
        tau = None
        learning_rate = None
        batch_size = None
        learning_starts = None
        gradient_steps = None
        gamma = None
        use_sde = None
        action_noise_std = None
        ent_coef = None
        target_entropy = None
        min_ent_coef = None
        dense_reward_shaping = None
        dense_reward_style = None
        task_geometry_features = None
        task_stage_features = None
        task_progress_features = None
        residual_guidance = None
        normalize_env = None
        n_sampled_goal = None
        goal_selection_strategy = None
        expert_demo_episodes = 0
        expert_demo_style = None
        expert_hint_style = None
        demo_prefill_passes = 1
        safe_curriculum = None
        pickplace_profile = "auto"

    prealign_config = SACConfig(env_name="FrankaPickAndPlaceWindowPrealignSparse-v0", experiment_name="test_prealign")
    insert_config = SACConfig(env_name="FrankaPickAndPlaceWindowInsertSparse-v0", experiment_name="test_insert")
    args = Args()

    prealign_config = apply_pickplace_sparse_defaults(prealign_config, args)
    insert_config = apply_pickplace_sparse_defaults(insert_config, args)

    assert prealign_config.pickplace_profile == "prealign"
    assert prealign_config.her is False
    assert prealign_config.dense_reward_shaping is True
    assert prealign_config.dense_reward_style == "insert"
    assert prealign_config.task_geometry_features is True
    assert prealign_config.task_stage_features is False
    assert prealign_config.task_progress_features is False
    assert prealign_config.residual_guidance is False
    assert prealign_config.normalize_env is False
    assert prealign_config.n_envs == 1

    assert insert_config.pickplace_profile == "insert"
    assert insert_config.her is False
    assert insert_config.dense_reward_shaping is True
    assert insert_config.dense_reward_style == "insert"
    assert insert_config.task_geometry_features is True
    assert insert_config.residual_guidance is False
    assert insert_config.normalize_env is False
    assert insert_config.n_envs == 1


def test_pickplace_window_insert_dense_reward_penalizes_collision():
    env = gym.make("FrankaPickAndPlaceWindowInsertDense-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        desired_goal = base.goal.copy()
        good_info = {
            "collision": False,
            "plane_violation": False,
            "glass_fits_window": True,
        }
        bad_info = {
            "collision": True,
            "plane_violation": False,
            "glass_fits_window": False,
        }
        good_reward = float(base.compute_reward(desired_goal, desired_goal, good_info))
        bad_reward = float(base.compute_reward(desired_goal, desired_goal, bad_info))

        assert good_reward > bad_reward
    finally:
        env.close()


def test_pickplace_window_insert_dense_reward_penalizes_inplane_misalignment():
    env = gym.make("FrankaPickAndPlaceWindowInsertDense-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        desired_goal = base.goal.copy()
        good_info = {
            "collision": False,
            "plane_violation": False,
            "glass_fits_window": True,
            "inplane_alignment": 1.0,
        }
        bad_info = {
            "collision": False,
            "plane_violation": False,
            "glass_fits_window": True,
            "inplane_alignment": 0.0,
        }
        good_reward = float(base.compute_reward(desired_goal, desired_goal, good_info))
        bad_reward = float(base.compute_reward(desired_goal, desired_goal, bad_info))

        assert good_reward > bad_reward + 0.3
    finally:
        env.close()


def test_pickplace_insert_shaping_penalizes_inplane_misalignment():
    env = PickAndPlaceDenseRewardWrapper(gym.make("FrankaPickAndPlaceWindowInsertSparse-v0"), reward_style="insert")
    try:
        env.reset(seed=0)
        achieved_goal = env.unwrapped.goal.copy()
        desired_goal = env.unwrapped.goal.copy()
        good_reward = float(
            env.compute_reward(
                achieved_goal,
                desired_goal,
                {
                    "collision": False,
                    "plane_violation": False,
                    "inplane_alignment": 1.0,
                },
            )
        )
        bad_reward = float(
            env.compute_reward(
                achieved_goal,
                desired_goal,
                {
                    "collision": False,
                    "plane_violation": False,
                    "inplane_alignment": 0.0,
                },
            )
        )

        assert good_reward > bad_reward + 0.3
    finally:
        env.close()


def test_pickplace_insert_shaping_ignores_goal_specific_glass_fit_info():
    env = PickAndPlaceDenseRewardWrapper(gym.make("FrankaPickAndPlaceWindowInsertSparse-v0"), reward_style="insert")
    try:
        env.reset(seed=0)
        achieved_goal = env.unwrapped.goal.copy()
        desired_goal = env.unwrapped.goal.copy()
        false_fit_reward = float(
            env.compute_reward(
                achieved_goal,
                desired_goal,
                {
                    "collision": False,
                    "plane_violation": False,
                    "glass_fits_window": False,
                },
            )
        )
        true_fit_reward = float(
            env.compute_reward(
                achieved_goal,
                desired_goal,
                {
                    "collision": False,
                    "plane_violation": False,
                    "glass_fits_window": True,
                },
            )
        )

        assert np.isclose(false_fit_reward, true_fit_reward)
    finally:
        env.close()


def test_pickplace_expert_dataset_filters_failed_insert_seed():
    def make_env():
        return gym.make("FrankaPickAndPlaceWindowInsertSparse-v0")

    dataset = collect_pickplace_expert_dataset(
        make_env,
        num_episodes=9,
        seed_start=0,
        style="insert",
        keep_failed_episodes=False,
        stop_on_success=True,
    )

    kept_seeds = {
        int(info["expert_episode_seed"])
        for info in dataset.infos
    }
    assert np.isclose(dataset.success_rate, 8 / 9)
    assert 3 not in kept_seeds
    assert kept_seeds == {0, 1, 2, 4, 5, 6, 7, 8}
    assert dataset.num_transitions < 8 * 200
    assert all(bool(info["expert_episode_success"]) for info in dataset.infos)
