#/home/dyros/panda_mujoco_gym/train/common/wrappers.py
"""
环境包装器集合
- RewardScalingWrapper: 奖励缩放
- SuccessTrackingWrapper: 成功率跟踪
"""

import gymnasium as gym
import numpy as np

from train.common.pickplace_utils import (
    goal_alignment,
    goal_orientation_error,
    goal_position,
    is_window_goal,
    pickplace_expert_action,
    pickplace_next_phase,
    pickplace_phase_names,
)


class RewardScalingWrapper(gym.Wrapper):
    """奖励缩放包装器"""
    
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale
        self.episode_reward = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        self.episode_reward += scaled_reward
        
        if terminated or truncated:
            info['original_reward'] = self.episode_reward / self.scale
            info['scaled_reward'] = self.episode_reward
            self.episode_reward = 0
            
        return obs, scaled_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.episode_reward = 0
        return self.env.reset(**kwargs)


class PickAndPlaceTaskProgressWrapper(gym.ObservationWrapper):
    """为 observation 添加 pick-and-place 任务进度的包装器。"""

    def __init__(self, env, hint_style: str = "staged"):
        super().__init__(env)
        observation_space = self.observation_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("PickAndPlaceTaskProgressWrapper expects a Dict observation space")
        desired_goal_dim = int(np.prod(observation_space["achieved_goal"].shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        self.phase_names = pickplace_phase_names(self.action_dim, desired_goal_dim)
        self.phase_name = self.phase_names[0]
        self.phase_steps = 0
        self.initial_object_height = 0.0
        self.last_expert_hint_action = np.zeros(self.action_dim, dtype=np.float32)
        self.hint_style = str(hint_style)

        base_observation_space = observation_space["observation"]
        extra_dim = 3 + 3 + 1 + 1 + 1 + len(self.phase_names) + self.action_dim
        low = np.concatenate([
            np.asarray(base_observation_space.low, dtype=np.float32),
            np.full(3 + 3 + 1 + 1 + 1 + self.action_dim, -np.inf, dtype=np.float32),
            np.zeros(len(self.phase_names), dtype=np.float32),
        ])
        high = np.concatenate([
            np.asarray(base_observation_space.high, dtype=np.float32),
            np.full(3 + 3 + 1 + 1 + 1 + self.action_dim, np.inf, dtype=np.float32),
            np.ones(len(self.phase_names), dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Dict({
            **observation_space.spaces,
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
        })

    def _phase_one_hot(self) -> np.ndarray:
        phase = np.zeros(len(self.phase_names), dtype=np.float32)
        phase[self.phase_names.index(self.phase_name)] = 1.0
        return phase

    def _expert_hint_action(self, observation) -> np.ndarray:
        ee_forward_axis = None
        if hasattr(self.unwrapped, "get_ee_forward_axis"):
            ee_forward_axis = self.unwrapped.get_ee_forward_axis()
        return pickplace_expert_action(
            observation,
            self.phase_name,
            self.initial_object_height,
            action_dim=self.action_dim,
            hint_style=self.hint_style,
            ee_forward_axis=ee_forward_axis,
        )

    def _augment_observation(self, observation):
        ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
        object_position = goal_position(observation["achieved_goal"])
        goal_pos = goal_position(observation["desired_goal"])
        ee_to_object = object_position - ee_position
        object_to_goal = goal_pos - object_position
        ee_object_distance = np.array([np.linalg.norm(ee_to_object)], dtype=np.float32)
        object_goal_distance = np.array([np.linalg.norm(object_to_goal)], dtype=np.float32)
        lift_height = np.array([object_position[2] - self.initial_object_height], dtype=np.float32)
        expert_hint_action = self._expert_hint_action(observation)
        self.last_expert_hint_action = expert_hint_action.copy()

        augmented = np.concatenate([
            np.asarray(observation["observation"], dtype=np.float32),
            ee_to_object.astype(np.float32),
            object_to_goal.astype(np.float32),
            ee_object_distance,
            object_goal_distance,
            lift_height,
            expert_hint_action,
            self._phase_one_hot(),
        ]).astype(np.float32)
        return {
            **observation,
            "observation": augmented,
        }

    def _update_phase(self, observation) -> None:
        ee_forward_axis = None
        if hasattr(self.unwrapped, "get_ee_forward_axis"):
            ee_forward_axis = self.unwrapped.get_ee_forward_axis()
        self.phase_name, self.phase_steps = pickplace_next_phase(
            observation,
            self.phase_name,
            self.phase_steps,
            self.initial_object_height,
            hint_style=self.hint_style,
            action_dim=self.action_dim,
            ee_forward_axis=ee_forward_axis,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phase_name = self.phase_names[0]
        self.phase_steps = 0
        self.initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])
        return self._augment_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._update_phase(observation)
        return self._augment_observation(observation), reward, terminated, truncated, info


class PickAndPlaceGeometryWrapper(gym.ObservationWrapper):
    """为 pick-and-place 添加纯几何特征的包装器。"""

    def __init__(self, env):
        super().__init__(env)
        self.initial_object_height = 0.0

        observation_space = self.observation_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("PickAndPlaceGeometryWrapper expects a Dict observation space")

        base_observation_space = observation_space["observation"]
        extra_dim = 3 + 3 + 1 + 1 + 1
        low = np.concatenate([
            np.asarray(base_observation_space.low, dtype=np.float32),
            np.full(extra_dim, -np.inf, dtype=np.float32),
        ])
        high = np.concatenate([
            np.asarray(base_observation_space.high, dtype=np.float32),
            np.full(extra_dim, np.inf, dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Dict({
            **observation_space.spaces,
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
        })

    def _augment_observation(self, observation):
        ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
        object_position = goal_position(observation["achieved_goal"])
        goal_pos = goal_position(observation["desired_goal"])
        ee_to_object = object_position - ee_position
        object_to_goal = goal_pos - object_position
        ee_object_distance = np.array([np.linalg.norm(ee_to_object)], dtype=np.float32)
        object_goal_distance = np.array([np.linalg.norm(object_to_goal)], dtype=np.float32)
        lift_height = np.array([object_position[2] - self.initial_object_height], dtype=np.float32)

        augmented = np.concatenate([
            np.asarray(observation["observation"], dtype=np.float32),
            ee_to_object.astype(np.float32),
            object_to_goal.astype(np.float32),
            ee_object_distance,
            object_goal_distance,
            lift_height,
        ]).astype(np.float32)
        return {
            **observation,
            "observation": augmented,
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])
        return self._augment_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_observation(observation), reward, terminated, truncated, info


class PickAndPlaceStageFeatureWrapper(gym.ObservationWrapper):
    """为 pick-and-place 添加 phase one-hot + 几何特征的包装器。"""

    def __init__(self, env, hint_style: str = "staged"):
        super().__init__(env)
        observation_space = self.observation_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("PickAndPlaceStageFeatureWrapper expects a Dict observation space")
        desired_goal_dim = int(np.prod(observation_space["achieved_goal"].shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        self.phase_names = pickplace_phase_names(self.action_dim, desired_goal_dim)
        self.phase_name = self.phase_names[0]
        self.phase_steps = 0
        self.initial_object_height = 0.0
        self.hint_style = str(hint_style)

        base_observation_space = observation_space["observation"]
        extra_dim = 3 + 3 + 1 + 1 + 1 + len(self.phase_names)
        low = np.concatenate([
            np.asarray(base_observation_space.low, dtype=np.float32),
            np.full(3 + 3 + 1 + 1 + 1, -np.inf, dtype=np.float32),
            np.zeros(len(self.phase_names), dtype=np.float32),
        ])
        high = np.concatenate([
            np.asarray(base_observation_space.high, dtype=np.float32),
            np.full(3 + 3 + 1 + 1 + 1, np.inf, dtype=np.float32),
            np.ones(len(self.phase_names), dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Dict({
            **observation_space.spaces,
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
        })

    def _phase_one_hot(self) -> np.ndarray:
        phase = np.zeros(len(self.phase_names), dtype=np.float32)
        phase[self.phase_names.index(self.phase_name)] = 1.0
        return phase

    def _augment_observation(self, observation):
        ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
        object_position = goal_position(observation["achieved_goal"])
        goal_pos = goal_position(observation["desired_goal"])
        ee_to_object = object_position - ee_position
        object_to_goal = goal_pos - object_position
        ee_object_distance = np.array([np.linalg.norm(ee_to_object)], dtype=np.float32)
        object_goal_distance = np.array([np.linalg.norm(object_to_goal)], dtype=np.float32)
        lift_height = np.array([object_position[2] - self.initial_object_height], dtype=np.float32)

        augmented = np.concatenate([
            np.asarray(observation["observation"], dtype=np.float32),
            ee_to_object.astype(np.float32),
            object_to_goal.astype(np.float32),
            ee_object_distance,
            object_goal_distance,
            lift_height,
            self._phase_one_hot(),
        ]).astype(np.float32)
        return {
            **observation,
            "observation": augmented,
        }

    def _update_phase(self, observation) -> None:
        ee_forward_axis = None
        if hasattr(self.unwrapped, "get_ee_forward_axis"):
            ee_forward_axis = self.unwrapped.get_ee_forward_axis()
        self.phase_name, self.phase_steps = pickplace_next_phase(
            observation,
            self.phase_name,
            self.phase_steps,
            self.initial_object_height,
            hint_style=self.hint_style,
            action_dim=self.action_dim,
            ee_forward_axis=ee_forward_axis,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phase_name = self.phase_names[0]
        self.phase_steps = 0
        self.initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])
        return self._augment_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._update_phase(observation)
        return self._augment_observation(observation), reward, terminated, truncated, info


class PickAndPlaceResidualGuidanceWrapper(gym.Wrapper):
    """让模型只学习叠加在 expert hint 之上的 residual action 的包装器。"""

    def __init__(self, env, residual_scale: float = 0.1):
        super().__init__(env)
        self.residual_scale = float(residual_scale)

    def step(self, action):
        expert_hint = np.asarray(getattr(self.env, "last_expert_hint_action", np.zeros_like(action)), dtype=np.float32)
        residual_action = np.asarray(action, dtype=np.float32)
        executed_action = np.clip(
            expert_hint + self.residual_scale * residual_action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        info["expert_action_hint"] = expert_hint.copy()
        info["policy_residual_action"] = residual_action.copy()
        info["executed_action"] = executed_action.copy()
        return observation, reward, terminated, truncated, info


class PickAndPlaceDenseRewardWrapper(gym.Wrapper):
    """用于 sparse pick-and-place 训练的 dense shaping 包装器。

    为了与 HER relabeling 保持一致，实现了 compute_reward()。
    由于会使用 info dict 中的 ee/object 辅助信号，在 HER 中需要 copy_info_dict=True。
    """

    def __init__(self, env, reward_style: str = "standard"):
        super().__init__(env)
        self.reward_style = str(reward_style)

    def _info_array(self, info, key: str, default: float, target_shape) -> np.ndarray:
        if isinstance(info, dict):
            value = info.get(key, default)
            return np.asarray(value, dtype=np.float32)
        if isinstance(info, (list, tuple)):
            values = [
                (item.get(key, default) if isinstance(item, dict) else default)
                for item in info
            ]
            return np.asarray(values, dtype=np.float32)
        if isinstance(info, np.ndarray) and info.dtype == object:
            flat = [
                (item.get(key, default) if isinstance(item, dict) else default)
                for item in info.reshape(-1)
            ]
            return np.asarray(flat, dtype=np.float32).reshape(info.shape)
        return np.full(target_shape, default, dtype=np.float32)

    def _compute_standard_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        ee_object_distance: np.ndarray,
        object_height: np.ndarray,
    ) -> np.ndarray:
        if is_window_goal(desired_goal):
            return self._compute_window_reward(
                achieved_goal,
                desired_goal,
                ee_object_distance,
                object_height,
                direct_style=False,
            )
        distances = np.linalg.norm(goal_position(achieved_goal) - goal_position(desired_goal), axis=-1)
        reward = -distances.astype(np.float32)
        reward -= 0.25 * ee_object_distance

        base = self.unwrapped
        goal_z_range = max(float(getattr(base, "goal_z_range", 0.2)), 0.0)
        distance_threshold = float(getattr(base, "distance_threshold", 0.05))
        lift_cap = max(goal_z_range, distance_threshold)

        reward += 0.5 * np.minimum(np.maximum(object_height, 0.0), lift_cap)
        reward += 0.25 * (object_height > distance_threshold).astype(np.float32)
        reward += 0.5 * (distances < distance_threshold).astype(np.float32)
        return reward.astype(np.float32)

    def _compute_direct_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        ee_object_distance: np.ndarray,
        object_height: np.ndarray,
    ) -> np.ndarray:
        if is_window_goal(desired_goal):
            return self._compute_window_reward(
                achieved_goal,
                desired_goal,
                ee_object_distance,
                object_height,
                direct_style=True,
            )
        delta = goal_position(desired_goal) - goal_position(achieved_goal)
        xy_distance = np.linalg.norm(delta[..., :2], axis=-1).astype(np.float32)
        z_distance = np.abs(delta[..., 2]).astype(np.float32)
        full_distance = np.linalg.norm(delta, axis=-1).astype(np.float32)

        base = self.unwrapped
        distance_threshold = float(getattr(base, "distance_threshold", 0.05))

        clipped_object_height = np.maximum(object_height, 0.0).astype(np.float32)
        table_height = achieved_goal[..., 2] - clipped_object_height
        goal_height = np.maximum(desired_goal[..., 2] - table_height, 0.0).astype(np.float32)

        carry_height_target = np.clip(
            0.35 * xy_distance + goal_height + 0.035,
            max(distance_threshold, 0.05),
            0.11,
        ).astype(np.float32)
        supported_height = np.minimum(clipped_object_height, carry_height_target)
        excess_height = np.maximum(clipped_object_height - carry_height_target, 0.0)
        near_goal_mask = (xy_distance < (1.5 * distance_threshold)).astype(np.float32)
        overshoot_height = np.maximum(achieved_goal[..., 2] - desired_goal[..., 2], 0.0).astype(np.float32)

        reward = -xy_distance
        reward -= 0.75 * z_distance
        reward -= 0.20 * ee_object_distance
        reward += 0.40 * supported_height
        reward += 0.10 * (clipped_object_height > 0.01).astype(np.float32)
        reward -= 0.20 * excess_height
        reward -= 0.15 * near_goal_mask * overshoot_height
        reward += 0.50 * (full_distance < distance_threshold).astype(np.float32)
        return reward.astype(np.float32)

    def _compute_window_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        ee_object_distance: np.ndarray,
        object_height: np.ndarray,
        direct_style: bool,
    ) -> np.ndarray:
        achieved_pos = goal_position(achieved_goal)
        desired_pos = goal_position(desired_goal)
        distances = np.linalg.norm(achieved_pos - desired_pos, axis=-1).astype(np.float32)
        orientation_alignment = np.asarray(goal_alignment(achieved_goal, desired_goal), dtype=np.float32)
        orientation_error = np.asarray(goal_orientation_error(achieved_goal, desired_goal), dtype=np.float32)

        base = self.unwrapped
        distance_threshold = float(getattr(base, "distance_threshold", 0.03))
        orientation_threshold = float(getattr(base, "orientation_threshold_cos", np.cos(np.deg2rad(20.0))))
        clipped_object_height = np.maximum(object_height, 0.0).astype(np.float32)

        reward = -distances
        reward -= (0.20 if direct_style else 0.15) * ee_object_distance
        reward -= (0.35 if direct_style else 0.25) * orientation_error
        reward += 0.35 * np.minimum(clipped_object_height, 0.12)
        reward += 0.25 * (orientation_alignment >= orientation_threshold).astype(np.float32)
        reward += 0.50 * (
            (distances < distance_threshold) & (orientation_alignment >= orientation_threshold)
        ).astype(np.float32)
        return reward.astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distances = np.linalg.norm(goal_position(achieved_goal) - goal_position(desired_goal), axis=-1)
        ee_object_distance = self._info_array(info, "ee_object_distance", 0.0, distances.shape)
        object_height = self._info_array(info, "object_height", 0.0, distances.shape)

        if self.reward_style == "direct":
            return self._compute_direct_reward(
                achieved_goal,
                desired_goal,
                ee_object_distance,
                object_height,
            )
        return self._compute_standard_reward(
            achieved_goal,
            desired_goal,
            ee_object_distance,
            object_height,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        info = dict(info)
        info["sparse_reward"] = float(reward)
        info["shaped_reward"] = float(np.asarray(shaped_reward).item())
        info["dense_reward_style"] = self.reward_style
        return obs, float(np.asarray(shaped_reward).item()), terminated, truncated, info


class SuccessTrackingWrapper(gym.Wrapper):
    """成功率跟踪包装器"""
    
    def __init__(self, env):
        super().__init__(env)
        self.success_count = 0
        self.any_success_count = 0
        self.episode_count = 0
        self.current_episode_success = False
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get('is_success', False):
            self.current_episode_success = True
        
        if terminated or truncated:
            self.episode_count += 1
            terminal_success = bool(info.get('is_success', False))
            any_success = bool(self.current_episode_success)

            if terminal_success:
                self.success_count += 1
            if any_success:
                self.any_success_count += 1
            
            info['success_rate'] = self.success_count / self.episode_count if self.episode_count > 0 else 0
            info['any_success_rate'] = self.any_success_count / self.episode_count if self.episode_count > 0 else 0
            info['total_episodes'] = self.episode_count
            info['total_successes'] = self.success_count
            info['total_any_successes'] = self.any_success_count
            info['episode_terminal_success'] = terminal_success
            info['episode_any_success'] = any_success
            self.current_episode_success = False
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.current_episode_success = False
        return self.env.reset(**kwargs)
