"""Scripted expert utilities for Franka pick-and-place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from train.common.pickplace_utils import (
    WINDOW_INSERT_PHASE_NAMES,
    is_insertion_task,
    pickplace_expert_action,
    pickplace_next_phase,
    pickplace_phase_names,
)


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return np.zeros_like(axis)
    return axis / norm


def _initial_insert_phase(observation, phase_names: tuple[str, ...]) -> str:
    object_position = np.asarray(observation["achieved_goal"], dtype=np.float32)[:3]
    goal_position = np.asarray(observation["desired_goal"], dtype=np.float32)[:3]
    goal_distance = float(np.linalg.norm(object_position - goal_position))
    achieved_axis = _normalize_axis(np.asarray(observation["achieved_goal"], dtype=np.float32)[3:6])
    goal_axis = _normalize_axis(np.asarray(observation["desired_goal"], dtype=np.float32)[3:6])
    orientation_alignment = float(np.dot(achieved_axis, goal_axis))

    if goal_distance < 0.10 and orientation_alignment > 0.90:
        return "prealign"
    if goal_distance < 0.18:
        return "reorient"
    return phase_names[0]


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


def collect_pickplace_expert_dataset(
    env_fn: Callable[[], Any],
    num_episodes: int,
    seed_start: int = 0,
    style: str = "staged",
    residual_guidance: bool = False,
    residual_scale: float = 0.1,
    keep_failed_episodes: bool = True,
    stop_on_success: bool = True,
    max_steps: int = 200,
) -> ExpertDataset:
    observations = {key: [] for key in ("observation", "achieved_goal", "desired_goal")}
    next_observations = {key: [] for key in ("observation", "achieved_goal", "desired_goal")}
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[float] = []
    infos: List[Dict[str, Any]] = []
    episode_successes: List[bool] = []

    action_dim_seen = 0

    for episode_idx in range(max(int(num_episodes), 0)):
        env = env_fn()
        observation, _ = env.reset(seed=seed_start + episode_idx)
        initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])
        action_dim = int(env.action_space.shape[-1])
        action_dim_seen = action_dim
        phase_names = pickplace_phase_names(
            action_dim=action_dim,
            desired_goal_dim=int(np.asarray(observation["desired_goal"]).shape[-1]),
        )
        if is_insertion_task(observation, initial_object_height):
            phase_names = WINDOW_INSERT_PHASE_NAMES

        style_name = str(style).lower()
        phase = (
            _initial_insert_phase(observation, phase_names)
            if phase_names == WINDOW_INSERT_PHASE_NAMES
            else phase_names[0]
        )
        phase_steps = 0
        episode_success = False
        episode_observations = {key: [] for key in observations}
        episode_next_observations = {key: [] for key in next_observations}
        episode_actions: List[np.ndarray] = []
        episode_rewards: List[float] = []
        episode_dones: List[float] = []
        episode_infos: List[Dict[str, Any]] = []

        for _ in range(max(int(max_steps), 1)):
            ee_forward_axis = None
            ee_rotation_matrix = None
            object_rotation_matrix = None
            if hasattr(env.unwrapped, "get_ee_forward_axis"):
                ee_forward_axis = env.unwrapped.get_ee_forward_axis()
            if hasattr(env.unwrapped, "get_ee_rotation_matrix"):
                ee_rotation_matrix = env.unwrapped.get_ee_rotation_matrix()
            if hasattr(env.unwrapped, "get_object_rotation_matrix"):
                object_rotation_matrix = env.unwrapped.get_object_rotation_matrix()

            action = pickplace_expert_action(
                observation,
                phase,
                initial_object_height,
                action_dim=action_dim,
                hint_style=style_name,
                ee_forward_axis=ee_forward_axis,
                ee_rotation_matrix=ee_rotation_matrix,
                object_rotation_matrix=object_rotation_matrix,
                phase_steps=phase_steps,
            )
            current_hint_action = None
            if residual_guidance and hasattr(env, "get_wrapper_attr"):
                try:
                    current_hint_action = env.get_wrapper_attr("last_expert_hint_action")
                except AttributeError:
                    current_hint_action = None
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            stored_action = np.asarray(action, dtype=np.float32).copy()
            if residual_guidance:
                hint_action = np.asarray(
                    current_hint_action if current_hint_action is not None else np.zeros_like(stored_action),
                    dtype=np.float32,
                )
                safe_scale = max(float(residual_scale), 1e-6)
                stored_action = np.clip(
                    (stored_action - hint_action) / safe_scale,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)

            for key in observations:
                episode_observations[key].append(np.asarray(observation[key], dtype=np.float32).copy())
                episode_next_observations[key].append(np.asarray(next_observation[key], dtype=np.float32).copy())

            episode_success = episode_success or bool(info.get("is_success", False))
            expert_done = bool(done or (stop_on_success and episode_success))
            stored_info = dict(info)
            if expert_done and not done:
                stored_info["expert_stop_on_success"] = True

            episode_actions.append(stored_action)
            episode_rewards.append(float(reward))
            episode_dones.append(float(expert_done))
            episode_infos.append(stored_info)

            next_ee_forward_axis = ee_forward_axis
            next_ee_rotation_matrix = ee_rotation_matrix
            next_object_rotation_matrix = object_rotation_matrix
            if hasattr(env.unwrapped, "get_ee_forward_axis"):
                next_ee_forward_axis = env.unwrapped.get_ee_forward_axis()
            if hasattr(env.unwrapped, "get_ee_rotation_matrix"):
                next_ee_rotation_matrix = env.unwrapped.get_ee_rotation_matrix()
            if hasattr(env.unwrapped, "get_object_rotation_matrix"):
                next_object_rotation_matrix = env.unwrapped.get_object_rotation_matrix()
            phase, phase_steps = pickplace_next_phase(
                next_observation,
                phase,
                phase_steps,
                initial_object_height,
                hint_style=style_name,
                action_dim=action_dim,
                ee_forward_axis=next_ee_forward_axis,
                ee_rotation_matrix=next_ee_rotation_matrix,
                object_rotation_matrix=next_object_rotation_matrix,
            )
            observation = next_observation

            if expert_done:
                break

        episode_successes.append(episode_success)
        if episode_success or keep_failed_episodes:
            for episode_info in episode_infos:
                episode_info["expert_episode_seed"] = int(seed_start + episode_idx)
                episode_info["expert_episode_success"] = bool(episode_success)
            for key in observations:
                observations[key].extend(episode_observations[key])
                next_observations[key].extend(episode_next_observations[key])
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            dones.extend(episode_dones)
            infos.extend(episode_infos)
        env.close()

    if actions:
        actions_array = np.asarray(actions, dtype=np.float32)
    else:
        actions_array = np.zeros((0, int(action_dim_seen)), dtype=np.float32)

    return ExpertDataset(
        observations={key: np.asarray(values, dtype=np.float32) for key, values in observations.items()},
        next_observations={key: np.asarray(values, dtype=np.float32) for key, values in next_observations.items()},
        actions=actions_array,
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        infos=infos,
        episode_successes=episode_successes,
    )
