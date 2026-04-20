"""Scripted expert utilities for Franka pick-and-place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from train.common.pickplace_utils import (
    pickplace_expert_action,
    pickplace_next_phase,
    pickplace_phase_names,
)


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
        action_dim = int(env.action_space.shape[-1])
        phase_names = pickplace_phase_names(
            action_dim=action_dim,
            desired_goal_dim=int(np.asarray(observation["desired_goal"]).shape[-1]),
        )

        style_name = str(style).lower()
        phase = phase_names[0]
        phase_steps = 0
        episode_success = False

        for _ in range(200):
            ee_forward_axis = None
            if hasattr(env.unwrapped, "get_ee_forward_axis"):
                ee_forward_axis = env.unwrapped.get_ee_forward_axis()

            action = pickplace_expert_action(
                observation,
                phase,
                initial_object_height,
                action_dim=action_dim,
                hint_style=style_name,
                ee_forward_axis=ee_forward_axis,
            )
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

            next_ee_forward_axis = ee_forward_axis
            if hasattr(env.unwrapped, "get_ee_forward_axis"):
                next_ee_forward_axis = env.unwrapped.get_ee_forward_axis()
            phase, phase_steps = pickplace_next_phase(
                next_observation,
                phase,
                phase_steps,
                initial_object_height,
                hint_style=style_name,
                action_dim=action_dim,
                ee_forward_axis=next_ee_forward_axis,
            )
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
