"""
환경 생성 유틸리티 함수들
"""

import os
import sys
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from train.common.wrappers import (
    PickAndPlaceResidualGuidanceWrapper,
    PickAndPlaceTaskProgressWrapper,
    RewardScalingWrapper,
    SuccessTrackingWrapper,
)


def is_pick_and_place_sparse(env_name: str) -> bool:
    return ("PickAndPlace" in env_name) and ("Sparse" in env_name)


def create_env(
    env_name,
    render_mode=None,
    reward_scale=1.0,
    seed: Optional[int] = None,
    task_progress_features: bool = False,
    residual_guidance: bool = False,
    residual_action_scale: float = 0.1,
):
    """환경 생성 (래퍼 적용)"""
    env = gym.make(env_name, render_mode=render_mode)

    if task_progress_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceTaskProgressWrapper(env)
    if residual_guidance and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceResidualGuidanceWrapper(env, residual_scale=residual_action_scale)

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
    task_progress_features: bool = False,
    residual_guidance: bool = False,
    residual_action_scale: float = 0.1,
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
                task_progress_features=task_progress_features,
                residual_guidance=residual_guidance,
                residual_action_scale=residual_action_scale,
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
            norm_obs_keys = None
            obs_space = vec_env.observation_space
            if isinstance(obs_space, gym.spaces.Dict):
                goal_keys = {"observation", "achieved_goal", "desired_goal"}
                if goal_keys.issubset(set(obs_space.spaces.keys())):
                    norm_obs_keys = ["observation"]
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
                norm_obs_keys=norm_obs_keys,
            )

        vec_env.training = training
        if not training:
            vec_env.norm_reward = False

    return vec_env
