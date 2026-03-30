"""
환경 생성 유틸리티 함수들
"""

import os
import sys
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper


def create_env(env_name, render_mode=None, reward_scale=1.0):
    """환경 생성 (래퍼 적용)"""
    raw = gym.make(env_name, render_mode=render_mode)
    env = TimeLimit(raw, max_episode_steps=100)
    env = Monitor(env)

    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)

    env = SuccessTrackingWrapper(env)
    return env


def create_vec_env(env_name, n_envs=1, normalize=True, reward_scale=1.0,
                   vec_normalize_path=None, training=True, render_mode=None):
    """벡터화된 환경 생성"""
    def make_env():
        return create_env(env_name, render_mode=render_mode, reward_scale=reward_scale)

    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    if normalize:
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        vec_env.training = training
        if not training:
            vec_env.norm_reward = False

    return vec_env
