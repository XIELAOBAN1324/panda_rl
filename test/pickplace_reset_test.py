import gymnasium as gym
import numpy as np

import panda_mujoco_gym  # noqa: F401
from train.common.curriculum import SafePickAndPlaceCurriculumWrapper


def _collect_goal_distances(env, num_resets: int = 64):
    distances = []
    for seed in range(num_resets):
        obs, _ = env.reset(seed=seed)
        distances.append(float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
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
