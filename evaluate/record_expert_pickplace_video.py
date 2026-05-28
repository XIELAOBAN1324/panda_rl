#!/usr/bin/env python3
"""
录制 scripted expert 在 pick-and-place 环境中的 rollout 视频。
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import gymnasium as gym
import panda_mujoco_gym  # noqa: F401

from evaluate.video_recorder import EpisodeVideoRecorder
from train.common.expert_pickplace import _initial_insert_phase
from train.common.pickplace_utils import (
    WINDOW_INSERT_PHASE_NAMES,
    is_insertion_task,
    pickplace_expert_action,
    pickplace_next_phase,
    pickplace_phase_names,
)


def _capture_frame(env, recorder: EpisodeVideoRecorder) -> None:
    rendered = env.render()
    if rendered is None:
        return
    frame = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
    recorder.add_frame(frame)


def _phase_names_for_observation(observation: Dict[str, np.ndarray], action_dim: int, initial_object_height: float):
    phase_names = pickplace_phase_names(
        action_dim=action_dim,
        desired_goal_dim=int(np.asarray(observation["desired_goal"]).shape[-1]),
    )
    if is_insertion_task(observation, initial_object_height):
        phase_names = WINDOW_INSERT_PHASE_NAMES
    return phase_names


def record_expert_episode(
    env,
    recorder: EpisodeVideoRecorder,
    episode_id: int,
    seed: int,
    style: str,
    max_steps: int,
) -> Dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    initial_object_height = float(np.asarray(observation["achieved_goal"], dtype=np.float32)[2])
    action_dim = int(env.action_space.shape[-1])
    phase_names = _phase_names_for_observation(observation, action_dim, initial_object_height)
    phase = (
        _initial_insert_phase(observation, phase_names)
        if phase_names == WINDOW_INSERT_PHASE_NAMES
        else phase_names[0]
    )
    phase_steps = 0
    total_reward = 0.0
    success = False
    info: Dict[str, Any] = {}

    recorder.start_episode_recording()
    for _ in range(3):
        _capture_frame(env, recorder)

    for step_idx in range(max_steps):
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
            hint_style=style,
            ee_forward_axis=ee_forward_axis,
            ee_rotation_matrix=ee_rotation_matrix,
            object_rotation_matrix=object_rotation_matrix,
            phase_steps=phase_steps,
        )
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        _capture_frame(env, recorder)

        success = success or bool(info.get("is_success", False))
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
            hint_style=style,
            action_dim=action_dim,
            ee_forward_axis=next_ee_forward_axis,
            ee_rotation_matrix=next_ee_rotation_matrix,
            object_rotation_matrix=next_object_rotation_matrix,
        )
        observation = next_observation

        if success:
            for _ in range(15):
                _capture_frame(env, recorder)
            break
        if done:
            break

    episode_info = {
        "stage": "expert",
        "episode_id": int(episode_id),
        "seed": int(seed),
        "reward": float(total_reward),
        "length": int(step_idx + 1),
        "success": bool(success),
        "env_id": str(getattr(env.spec, "id", "unknown")),
        "final_phase": phase,
        "terminal_collision": bool(info.get("collision", False)),
        "terminal_plane_violation": bool(info.get("plane_violation", False)),
        "terminal_position_error": float(info.get("position_error", np.nan)),
        "terminal_orientation_alignment": float(info.get("orientation_alignment", np.nan)),
        "terminal_glass_fits_window": bool(info.get("glass_fits_window", False)),
    }
    video_path = recorder.end_episode_recording(episode_info)
    episode_info["video_path"] = video_path
    return episode_info


def record_env_expert_videos(
    env_id: str,
    save_dir: str,
    num_episodes: int,
    seed_start: int,
    style: str,
    max_steps: int,
    fps: int,
) -> List[Dict[str, Any]]:
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array", target_visible_in_rgb_array=False)
    recorder = EpisodeVideoRecorder(save_dir, fps=fps)
    results: List[Dict[str, Any]] = []
    try:
        for episode_idx in range(num_episodes):
            seed = seed_start + episode_idx
            result = record_expert_episode(
                env=env,
                recorder=recorder,
                episode_id=episode_idx,
                seed=seed,
                style=style,
                max_steps=max_steps,
            )
            results.append(result)
    finally:
        env.close()

    recorder.save_metadata()
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📝 汇总已保存: {summary_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="录制 scripted expert rollout 视频")
    parser.add_argument("--env", required=True, help="环境 ID")
    parser.add_argument("--out-dir", required=True, help="视频输出目录")
    parser.add_argument("--episodes", type=int, default=1, help="录制回合数")
    parser.add_argument("--seed-start", type=int, default=0, help="起始 seed")
    parser.add_argument("--style", type=str, default="insert", help="expert 动作风格")
    parser.add_argument("--max-steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--fps", type=int, default=10, help="视频帧率")
    args = parser.parse_args()

    record_env_expert_videos(
        env_id=args.env,
        save_dir=args.out_dir,
        num_episodes=max(int(args.episodes), 1),
        seed_start=int(args.seed_start),
        style=str(args.style),
        max_steps=max(int(args.max_steps), 1),
        fps=max(int(args.fps), 1),
    )


if __name__ == "__main__":
    main()
