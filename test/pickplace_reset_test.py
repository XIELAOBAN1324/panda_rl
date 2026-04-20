import gymnasium as gym
import numpy as np

import panda_mujoco_gym  # noqa: F401
from train.common.curriculum import SafePickAndPlaceCurriculumWrapper


def _collect_goal_distances(env, num_resets: int = 64):
    distances = []
    for seed in range(num_resets):
        obs, _ = env.reset(seed=seed)
        distances.append(float(np.linalg.norm(obs["achieved_goal"][:3] - obs["desired_goal"][:3])))
    return distances


def test_pickplace_sparse_reset_avoids_immediate_success():
    env = gym.make("FrankaPickAndPlaceSparse-v0")
    min_distance = env.unwrapped.minimum_goal_object_distance()
    distances = _collect_goal_distances(env)
    env.close()

    assert min(distances) >= min_distance - 1e-6


def test_safe_curriculum_reset_starts_outside_success_threshold():
    base_env = gym.make("FrankaPickAndPlaceSparse-v0")
    env = SafePickAndPlaceCurriculumWrapper(base_env, total_env_steps_target=1_000_000)
    min_distance = env.unwrapped.minimum_goal_object_distance()
    distances = _collect_goal_distances(env)
    env.close()

    assert min(distances) >= min_distance - 1e-6


def test_pickplace_window_reset_shapes_and_fixed_goal_height():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    try:
        for seed in range(8):
            obs, _ = env.reset(seed=seed)
            assert env.action_space.shape == (7,)
            assert obs["achieved_goal"].shape == (6,)
            assert obs["desired_goal"].shape == (6,)
            window_center = env.unwrapped.get_window_center()
            window_safe_goal = env.unwrapped.get_window_safe_goal()
            radius = float(np.linalg.norm(window_center[:2]))
            angle = float(np.rad2deg(np.arctan2(window_center[1], window_center[0])))

            assert abs(float(window_center[2]) - 0.65) < 1e-6
            assert abs(radius - 0.80) < 1e-6
            assert -60.0 <= angle <= 0.0
            assert np.allclose(obs["desired_goal"][:3], window_safe_goal, atol=1e-6)
            assert np.allclose(obs["desired_goal"][3:6], np.array([1.0, 0.0, 0.0]), atol=1e-6)
            assert np.allclose(window_center - window_safe_goal, np.array([0.03, 0.0, 0.0]), atol=1e-6)
            assert float(obs["achieved_goal"][0]) <= float(window_center[0]) + 1e-6
            assert float(obs["achieved_goal"][2]) < 0.05
    finally:
        env.close()


def test_pickplace_window_success_requires_position_and_alignment():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    base = env.unwrapped
    try:
        desired_goal = np.array([0.77, 0.00, 0.65, 1.0, 0.0, 0.0], dtype=np.float32)
        pos_only = np.array([0.77, 0.00, 0.65, 0.0, 0.0, -1.0], dtype=np.float32)
        align_only = np.array([0.65, 0.00, 0.65, 1.0, 0.0, 0.0], dtype=np.float32)
        both = np.array([0.77, 0.00, 0.65, 1.0, 0.0, 0.0], dtype=np.float32)

        assert float(base._is_success(pos_only, desired_goal)) == 0.0
        assert float(base._is_success(align_only, desired_goal)) == 0.0
        assert float(base._is_success(both, desired_goal)) == 1.0
    finally:
        env.close()


def test_pickplace_window_grasp_weld_attaches_and_releases():
    env = gym.make("FrankaPickAndPlaceWindowSparse-v0")
    base = env.unwrapped
    try:
        env.reset(seed=0)
        ee_position = base.get_ee_position().copy()
        base._utils.set_joint_qpos(
            base.model,
            base.data,
            "obj_joint",
            np.concatenate([ee_position, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        base._utils.set_joint_qpos(base.model, base.data, "finger_joint1", 0.01)
        base._utils.set_joint_qpos(base.model, base.data, "finger_joint2", 0.01)
        base._mujoco.mj_forward(base.model, base.data)

        base._update_grasp_attachment(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        assert base.is_attached
        assert int(base.model.eq_active[base.grasp_weld_eq_id]) == 1

        base._utils.set_joint_qpos(base.model, base.data, "finger_joint1", 0.04)
        base._utils.set_joint_qpos(base.model, base.data, "finger_joint2", 0.04)
        base._mujoco.mj_forward(base.model, base.data)
        base._update_grasp_attachment(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
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
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        end_ee = base.get_ee_position().copy()

        assert bool(info["plane_violation"])
        assert float(info["max_plane_penetration"]) > 0.0
        assert float(end_ee[0]) <= float(base.get_window_center()[0]) + 1e-6
        assert np.allclose(start_ee, end_ee, atol=1e-5)
    finally:
        env.close()
