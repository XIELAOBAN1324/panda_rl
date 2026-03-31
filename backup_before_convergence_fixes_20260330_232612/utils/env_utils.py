"""
환경 생성 유틸리티 함수들
"""

import os
import sys
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper


def create_env(env_name, render_mode=None, reward_scale=1.0, seed: Optional[int] = None):
    """환경 생성 (래퍼 적용)"""
    raw = gym.make(env_name, render_mode=render_mode)
    env = TimeLimit(raw, max_episode_steps=100)
    env = Monitor(env)

    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)

    env = SuccessTrackingWrapper(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def create_vec_env(
    env_name,
    n_envs=1,
    normalize=True,
    reward_scale=1.0,
    vec_normalize_path=None,
    training=True,
    render_mode=None,
    seed: Optional[int] = None,
    start_method: str = "forkserver",
):
    """벡터화된 환경 생성"""

    def make_env(rank: int):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_env(
                env_name,
                render_mode=render_mode,
                reward_scale=reward_scale,
                seed=env_seed,
            )
        return _init

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method=start_method)

    if normalize:
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        vec_env.training = training
        if not training:
            vec_env.norm_reward = False

    return vec_env
