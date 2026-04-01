"""Scripted expert utilities for Franka pick-and-place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass
class ExpertDataset:
    observations: Dict[str, np.ndarray]
    next_observations: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]
    episode_successes: List[bool]

    @property
    def num_transitions(self) -> int:
        return int(self.actions.shape[0])

    @property
    def success_rate(self) -> float:
        if not self.episode_successes:
            return 0.0
        return float(np.mean(np.asarray(self.episode_successes, dtype=np.float32)))


def _staged_expert_action(observation: Dict[str, np.ndarray], phase: str) -> np.ndarray:
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
    target_position, gripper_action = targets[phase]
    delta = np.clip((target_position - ee_position) / 0.05, -1.0, 1.0)
    return np.concatenate([delta, np.array([gripper_action], dtype=np.float32)]).astype(np.float32)


def _direct_expert_action(
    observation: Dict[str, np.ndarray],
    phase: str,
    initial_object_height: float,
) -> np.ndarray:
    ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
    object_position = np.asarray(observation["achieved_goal"], dtype=np.float32)
    goal_position = np.asarray(observation["desired_goal"], dtype=np.float32)

    xy_distance = float(np.linalg.norm((goal_position - object_position)[:2]))
    goal_height = max(float(goal_position[2] - initial_object_height), 0.0)
    carry_height = float(np.clip(goal_height + 0.035 + 0.35 * xy_distance, 0.05, 0.11))
    carry_z = float(max(goal_position[2] + 0.015, initial_object_height + carry_height))

    targets = {
        "approach": (object_position + np.array([0.0, 0.0, 0.075], dtype=np.float32), 1.0),
        "descend": (object_position + np.array([0.0, 0.0, 0.008], dtype=np.float32), 1.0),
        "grasp": (object_position + np.array([0.0, 0.0, 0.004], dtype=np.float32), -1.0),
        "lift": (np.array([object_position[0], object_position[1], carry_z], dtype=np.float32), -1.0),
        "move": (np.array([goal_position[0], goal_position[1], carry_z], dtype=np.float32), -1.0),
        "place": (goal_position + np.array([0.0, 0.0, 0.015], dtype=np.float32), -1.0),
        "hold": (goal_position + np.array([0.0, 0.0, 0.015], dtype=np.float32), -1.0),
    }
    target_position, gripper_action = targets[phase]
    delta = np.clip((target_position - ee_position) / 0.05, -1.0, 1.0)
    return np.concatenate([delta, np.array([gripper_action], dtype=np.float32)]).astype(np.float32)


def collect_pickplace_expert_dataset(
    env_fn: Callable[[], Any],
    num_episodes: int,
    seed_start: int = 0,
    style: str = "staged",
    residual_guidance: bool = False,
    residual_scale: float = 0.1,
) -> ExpertDataset:
    observations = {key: [] for key in ("observation", "achieved_goal", "desired_goal")}
    next_observations = {key: [] for key in ("observation", "achieved_goal", "desired_goal")}
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[float] = []
    infos: List[Dict[str, Any]] = []
    episode_successes: List[bool] = []

    for episode_idx in range(max(int(num_episodes), 0)):
        env = env_fn()
        observation, _ = env.reset(seed=seed_start + episode_idx)
        initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])

        style_name = str(style).lower()
        phase = "approach"
        phase_steps = 0
        episode_success = False

        for _ in range(200):
            ee_position = np.asarray(observation["observation"][:3], dtype=np.float32)
            object_position = np.asarray(observation["achieved_goal"], dtype=np.float32)
            goal_position = np.asarray(observation["desired_goal"], dtype=np.float32)

            horizontal_distance = float(np.linalg.norm((ee_position - object_position)[:2]))
            ee_object_distance = float(np.linalg.norm(ee_position - object_position))
            object_goal_distance = float(np.linalg.norm(object_position - goal_position))
            goal_xy_distance = float(np.linalg.norm((goal_position - object_position)[:2]))

            if style_name == "direct":
                action = _direct_expert_action(observation, phase, initial_object_height)
            else:
                action = _staged_expert_action(observation, phase)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            stored_action = np.asarray(action, dtype=np.float32).copy()
            if residual_guidance:
                hint_action = None
                if hasattr(env, "get_wrapper_attr"):
                    try:
                        hint_action = env.get_wrapper_attr("last_expert_hint_action")
                    except AttributeError:
                        hint_action = None
                hint_action = np.asarray(
                    hint_action if hint_action is not None else np.zeros_like(stored_action),
                    dtype=np.float32,
                )
                safe_scale = max(float(residual_scale), 1e-6)
                stored_action = np.clip(
                    (stored_action - hint_action) / safe_scale,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)

            for key in observations:
                observations[key].append(np.asarray(observation[key], dtype=np.float32).copy())
                next_observations[key].append(np.asarray(next_observation[key], dtype=np.float32).copy())

            actions.append(stored_action)
            rewards.append(float(reward))
            dones.append(float(done))
            infos.append(dict(info))
            episode_success = episode_success or bool(info.get("is_success", False))

            next_object_position = np.asarray(next_observation["achieved_goal"], dtype=np.float32)
            next_goal_xy_distance = float(np.linalg.norm((goal_position - next_object_position)[:2]))
            next_goal_distance = float(np.linalg.norm(next_object_position - goal_position))

            if style_name == "direct":
                goal_height = max(float(goal_position[2] - initial_object_height), 0.0)
                carry_height = float(np.clip(goal_height + 0.035 + 0.35 * object_goal_distance, 0.05, 0.11))
                required_lift = max(0.025, 0.55 * carry_height)
                if phase == "approach" and horizontal_distance < 0.015 and ee_position[2] > object_position[2] + 0.05:
                    phase = "descend"
                    phase_steps = 0
                elif phase == "descend" and ee_object_distance < 0.02:
                    phase = "grasp"
                    phase_steps = 0
                elif phase == "grasp" and phase_steps > 10:
                    phase = "lift"
                    phase_steps = 0
                elif phase == "lift" and next_object_position[2] > initial_object_height + required_lift:
                    phase = "move"
                    phase_steps = 0
                elif phase == "move" and next_goal_xy_distance < 0.03:
                    phase = "place"
                    phase_steps = 0
                elif phase == "place" and next_goal_distance < 0.04:
                    phase = "hold"
                    phase_steps = 0
            else:
                if phase == "approach" and horizontal_distance < 0.015 and ee_position[2] > object_position[2] + 0.06:
                    phase = "descend"
                    phase_steps = 0
                elif phase == "descend" and ee_object_distance < 0.02:
                    phase = "grasp"
                    phase_steps = 0
                elif phase == "grasp" and phase_steps > 12:
                    phase = "lift"
                    phase_steps = 0
                elif phase == "lift" and next_object_position[2] > goal_position[2] + 0.03:
                    phase = "move"
                    phase_steps = 0
                elif phase == "move" and float(np.linalg.norm((next_object_position - goal_position)[:2])) < 0.03:
                    phase = "place"
                    phase_steps = 0
                elif phase == "place" and object_goal_distance < 0.04:
                    phase = "hold"
                    phase_steps = 0

            phase_steps += 1
            observation = next_observation

            if done:
                break

        episode_successes.append(episode_success)
        env.close()

    return ExpertDataset(
        observations={key: np.asarray(values, dtype=np.float32) for key, values in observations.items()},
        next_observations={key: np.asarray(values, dtype=np.float32) for key, values in next_observations.items()},
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        infos=infos,
        episode_successes=episode_successes,
    )
