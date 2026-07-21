"""
环境创建工具函数
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
    PickAndPlaceDenseRewardWrapper,
    PickAndPlaceGeometryWrapper,
    PickAndPlaceResidualGuidanceWrapper,
    PickAndPlaceStageFeatureWrapper,
    PickAndPlaceTaskProgressWrapper,
    RewardScalingWrapper,
    SuccessTrackingWrapper,
)
from train.train_sac import validate_expert_runtime_config


def is_pick_and_place_sparse(env_name: str) -> bool:
    return ("PickAndPlace" in env_name) and ("Sparse" in env_name)


def create_env(
    env_name,
    render_mode=None,
    reward_scale=1.0,
    seed: Optional[int] = None,
    target_visible_in_rgb_array: bool = True,
    dense_reward_shaping: bool = False,
    dense_reward_style: str = "standard",
    task_geometry_features: bool = False,
    task_stage_features: bool = False,
    task_progress_features: bool = False,
    expert_hint_style: str = "staged",
    residual_guidance: bool = False,
    residual_action_scale: float = 0.1,
    expert_warmstart_only: bool = False,
):
    """创建环境（应用包装器）"""
    validate_expert_runtime_config(
        env_name=env_name,
        expert_warmstart_only=expert_warmstart_only,
        task_progress_features=task_progress_features,
        residual_guidance=residual_guidance,
    )
    env = gym.make(
        env_name,
        render_mode=render_mode,
        target_visible_in_rgb_array=target_visible_in_rgb_array,
    )

    if residual_guidance and is_pick_and_place_sparse(env_name):
        task_progress_features = True
        task_stage_features = False
        task_geometry_features = False

    if task_progress_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceTaskProgressWrapper(env, hint_style=expert_hint_style)
    elif task_stage_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceStageFeatureWrapper(env, hint_style=expert_hint_style)
    elif task_geometry_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceGeometryWrapper(env)
    if residual_guidance and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceResidualGuidanceWrapper(env, residual_scale=residual_action_scale)
    if dense_reward_shaping and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceDenseRewardWrapper(env, reward_style=dense_reward_style)

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
    target_visible_in_rgb_array: bool = True,
    start_method: str = "forkserver",
    dense_reward_shaping: bool = False,
    dense_reward_style: str = "standard",
    task_geometry_features: bool = False,
    task_stage_features: bool = False,
    task_progress_features: bool = False,
    expert_hint_style: str = "staged",
    residual_guidance: bool = False,
    residual_action_scale: float = 0.1,
    expert_warmstart_only: bool = False,
):
    """创建向量化环境"""

    def make_env(rank: int):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_env(
                env_name,
                render_mode=render_mode,
                reward_scale=reward_scale,
                seed=env_seed,
                target_visible_in_rgb_array=target_visible_in_rgb_array,
                dense_reward_shaping=dense_reward_shaping,
                dense_reward_style=dense_reward_style,
                task_geometry_features=task_geometry_features,
                task_stage_features=task_stage_features,
                task_progress_features=task_progress_features,
                expert_hint_style=expert_hint_style,
                residual_guidance=residual_guidance,
                residual_action_scale=residual_action_scale,
                expert_warmstart_only=expert_warmstart_only,
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
