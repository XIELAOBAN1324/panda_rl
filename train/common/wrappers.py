#/home/dyros/panda_mujoco_gym/train/common/wrappers.py
"""
환경 래퍼 모음
- RewardScalingWrapper: 보상 스케일링
- SuccessTrackingWrapper: 성공률 추적
"""

import gymnasium as gym
import numpy as np


class RewardScalingWrapper(gym.Wrapper):
    """보상 스케일링 래퍼"""
    
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
    """Pick-and-place 진행 상태를 observation에 추가하는 래퍼."""

    phase_names = (
        "approach",
        "descend",
        "grasp",
        "lift",
        "move",
        "place",
        "hold",
    )

    def __init__(self, env):
        super().__init__(env)
        self.phase_name = "approach"
        self.phase_steps = 0
        self.initial_object_height = 0.0
        self.last_expert_hint_action = np.zeros(4, dtype=np.float32)

        observation_space = self.observation_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("PickAndPlaceTaskProgressWrapper expects a Dict observation space")

        base_observation_space = observation_space["observation"]
        extra_dim = 3 + 3 + 1 + 1 + 1 + len(self.phase_names) + 4
        low = np.concatenate([
            np.asarray(base_observation_space.low, dtype=np.float32),
            np.full(3 + 3 + 1 + 1 + 1 + 4, -np.inf, dtype=np.float32),
            np.zeros(len(self.phase_names), dtype=np.float32),
        ])
        high = np.concatenate([
            np.asarray(base_observation_space.high, dtype=np.float32),
            np.full(3 + 3 + 1 + 1 + 1 + 4, np.inf, dtype=np.float32),
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
        ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
        object_position = np.asarray(observation["achieved_goal"], dtype=np.float32)
        goal_position = np.asarray(observation["desired_goal"], dtype=np.float32)
        safe_z = max(float(goal_position[2] + 0.10), float(object_position[2] + 0.10), 0.18)
        targets = {
            "approach": (object_position + np.array([0.0, 0.0, 0.10], dtype=np.float32), 1.0),
            "descend": (object_position + np.array([0.0, 0.0, 0.01], dtype=np.float32), 1.0),
            "grasp": (object_position + np.array([0.0, 0.0, 0.005], dtype=np.float32), -1.0),
            "lift": (np.array([object_position[0], object_position[1], safe_z], dtype=np.float32), -1.0),
            "move": (np.array([goal_position[0], goal_position[1], safe_z], dtype=np.float32), -1.0),
            "place": (goal_position + np.array([0.0, 0.0, 0.02], dtype=np.float32), -1.0),
            "hold": (goal_position + np.array([0.0, 0.0, 0.02], dtype=np.float32), -1.0),
        }
        target_position, gripper_action = targets[self.phase_name]
        delta = np.clip((target_position - ee_position) / 0.05, -1.0, 1.0)
        return np.concatenate([delta, np.array([gripper_action], dtype=np.float32)]).astype(np.float32)

    def _augment_observation(self, observation):
        ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
        object_position = np.asarray(observation["achieved_goal"], dtype=np.float32)
        goal_position = np.asarray(observation["desired_goal"], dtype=np.float32)
        ee_to_object = object_position - ee_position
        object_to_goal = goal_position - object_position
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
        ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
        object_position = np.asarray(observation["achieved_goal"], dtype=np.float32)
        goal_position = np.asarray(observation["desired_goal"], dtype=np.float32)

        horizontal_distance = float(np.linalg.norm((ee_position - object_position)[:2]))
        ee_object_distance = float(np.linalg.norm(ee_position - object_position))
        object_goal_distance = float(np.linalg.norm(object_position - goal_position))

        if self.phase_name == "approach" and horizontal_distance < 0.015 and ee_position[2] > object_position[2] + 0.06:
            self.phase_name = "descend"
            self.phase_steps = 0
        elif self.phase_name == "descend" and ee_object_distance < 0.02:
            self.phase_name = "grasp"
            self.phase_steps = 0
        elif self.phase_name == "grasp" and self.phase_steps > 12:
            self.phase_name = "lift"
            self.phase_steps = 0
        elif self.phase_name == "lift" and object_position[2] > goal_position[2] + 0.03:
            self.phase_name = "move"
            self.phase_steps = 0
        elif self.phase_name == "move" and float(np.linalg.norm((object_position - goal_position)[:2])) < 0.03:
            self.phase_name = "place"
            self.phase_steps = 0
        elif self.phase_name == "place" and object_goal_distance < 0.04:
            self.phase_name = "hold"
            self.phase_steps = 0

        self.phase_steps += 1

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phase_name = "approach"
        self.phase_steps = 0
        self.initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])
        return self._augment_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._update_phase(observation)
        return self._augment_observation(observation), reward, terminated, truncated, info


class PickAndPlaceResidualGuidanceWrapper(gym.Wrapper):
    """Expert hint 위에 residual action만 학습하도록 만드는 래퍼."""

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
    """Sparse pick-and-place 학습용 dense shaping wrapper.

    HER relabeling과 일관성을 맞추기 위해 compute_reward()를 구현한다.
    info dict 안의 ee/object 보조 신호를 사용하므로 HER에서는 copy_info_dict=True가 필요하다.
    """

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

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distances = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        ee_object_distance = self._info_array(info, "ee_object_distance", 0.0, distances.shape)
        object_height = self._info_array(info, "object_height", 0.0, distances.shape)

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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        info = dict(info)
        info["sparse_reward"] = float(reward)
        info["shaped_reward"] = float(np.asarray(shaped_reward).item())
        return obs, float(np.asarray(shaped_reward).item()), terminated, truncated, info


class SuccessTrackingWrapper(gym.Wrapper):
    """성공률 추적 래퍼"""
    
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
