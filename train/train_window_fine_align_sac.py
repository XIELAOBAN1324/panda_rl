"""Train SAC from scratch on the vision-free window fine-alignment task.

This entry point creates a plain Box-observation environment and uses
``MlpPolicy`` for the five fine-alignment controls.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import panda_mujoco_gym  # noqa: F401


ENV_ID = "FrankaWindowFineAlignDense-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--coarse-translation-range", type=float, default=0.015)
    parser.add_argument("--coarse-tilt-range-deg", type=float, default=2.0)
    parser.add_argument("--coarse-yaw-range-deg", type=float, default=4.0)
    parser.add_argument("--preinsert-normal-offset", type=float, default=0.08)
    parser.add_argument("--position-action-scale", type=float, default=0.0015)
    parser.add_argument("--tilt-action-scale-deg", type=float, default=0.25)
    parser.add_argument("--yaw-action-scale-deg", type=float, default=0.4)
    parser.add_argument("--fine-action-sim-steps", type=int, default=4)
    parser.add_argument("--translation-tolerance", type=float, default=0.002)
    parser.add_argument("--tilt-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--normal-gap-tolerance", type=float, default=0.001)
    parser.add_argument("--normal-gap-correction-threshold", type=float, default=0.0005)
    parser.add_argument("--normal-gap-correction-sim-steps", type=int, default=2)
    parser.add_argument("--max-normal-gap-error", type=float, default=0.008)
    parser.add_argument("--insert-action-sim-steps", type=int, default=4)
    parser.add_argument("--insert-translation-abort-tolerance", type=float, default=0.0025)
    parser.add_argument("--insert-tilt-abort-tolerance-deg", type=float, default=0.6)
    parser.add_argument("--insert-yaw-abort-tolerance-deg", type=float, default=0.6)
    parser.add_argument("--insert-alignment-violation-hold-steps", type=int, default=2)
    parser.add_argument("--insert-success-hold-steps", type=int, default=3)
    parser.add_argument("--insert-final-verification-max-steps", type=int, default=20)
    parser.add_argument("--insert-depth-tolerance", type=float, default=0.0005)
    parser.add_argument("--insert-step-size", type=float, default=0.001)
    parser.add_argument("--measurement-noise-translation-std", type=float, default=0.0005)
    parser.add_argument("--measurement-noise-angle-std", type=float, default=0.1)
    parser.add_argument("--eval-measurement-noise-translation-std", type=float, default=None)
    parser.add_argument("--eval-measurement-noise-angle-std", type=float, default=None)
    parser.add_argument("--reward-use-noisy-measurement", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def environment_kwargs(
    args: argparse.Namespace,
    *,
    evaluation: bool = False,
) -> dict[str, Any]:
    translation_noise = args.measurement_noise_translation_std
    angle_noise = args.measurement_noise_angle_std
    if evaluation and args.eval_measurement_noise_translation_std is not None:
        translation_noise = args.eval_measurement_noise_translation_std
    if evaluation and args.eval_measurement_noise_angle_std is not None:
        angle_noise = args.eval_measurement_noise_angle_std
    return {
        "coarse_translation_range": args.coarse_translation_range,
        "coarse_tilt_range_deg": args.coarse_tilt_range_deg,
        "coarse_yaw_range_deg": args.coarse_yaw_range_deg,
        "preinsert_normal_offset": args.preinsert_normal_offset,
        "fine_position_action_scale": args.position_action_scale,
        "fine_tilt_action_scale_deg": args.tilt_action_scale_deg,
        "fine_yaw_action_scale_deg": args.yaw_action_scale_deg,
        "fine_action_sim_steps": args.fine_action_sim_steps,
        "translation_tolerance": args.translation_tolerance,
        "tilt_tolerance_deg": args.tilt_tolerance_deg,
        "yaw_tolerance_deg": args.yaw_tolerance_deg,
        "normal_gap_tolerance": args.normal_gap_tolerance,
        "normal_gap_correction_threshold": args.normal_gap_correction_threshold,
        "normal_gap_correction_sim_steps": args.normal_gap_correction_sim_steps,
        "max_normal_gap_error": args.max_normal_gap_error,
        "insert_action_sim_steps": args.insert_action_sim_steps,
        "insert_translation_abort_tolerance": args.insert_translation_abort_tolerance,
        "insert_tilt_abort_tolerance_deg": args.insert_tilt_abort_tolerance_deg,
        "insert_yaw_abort_tolerance_deg": args.insert_yaw_abort_tolerance_deg,
        "insert_alignment_violation_hold_steps": args.insert_alignment_violation_hold_steps,
        "insert_success_hold_steps": args.insert_success_hold_steps,
        "insert_final_verification_max_steps": args.insert_final_verification_max_steps,
        "insert_depth_tolerance": args.insert_depth_tolerance,
        "insert_step_size": args.insert_step_size,
        "measurement_noise_translation_std": translation_noise,
        "measurement_noise_angle_std_deg": angle_noise,
        "reward_uses_ground_truth": not args.reward_use_noisy_measurement,
    }


def make_environment(
    *,
    seed: int,
    env_kwargs: dict[str, Any],
    monitor_file: Path,
) -> Callable[[], gym.Env]:
    def factory() -> gym.Env:
        env = gym.make(ENV_ID, **env_kwargs)
        env = Monitor(
            env,
            filename=str(monitor_file),
            info_keywords=(
                "is_success",
                "fine_align_success",
                "failure_reason",
                "normal_gap_error",
                "current_normal_gap",
                "ee_target_error",
            ),
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return factory


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0:
        raise ValueError("--timesteps and --n-envs must be positive")
    if (
        args.fine_action_sim_steps <= 0
        or args.normal_gap_correction_sim_steps <= 0
        or args.insert_action_sim_steps <= 0
        or args.insert_alignment_violation_hold_steps <= 0
        or args.insert_success_hold_steps <= 0
        or args.insert_final_verification_max_steps <= 0
    ):
        raise ValueError(
            "--fine-action-sim-steps, --normal-gap-correction-sim-steps, "
            "--insert-action-sim-steps, --insert-alignment-violation-hold-steps, "
            "--insert-success-hold-steps and --insert-final-verification-max-steps "
            "must be positive"
        )
    if args.preinsert_normal_offset <= 0.0:
        raise ValueError("--preinsert-normal-offset must be positive")
    if args.normal_gap_correction_threshold < 0.0:
        raise ValueError("--normal-gap-correction-threshold must be non-negative")
    if args.max_normal_gap_error <= 0.0:
        raise ValueError("--max-normal-gap-error must be positive")
    if args.insert_step_size <= 0.0:
        raise ValueError("--insert-step-size must be positive")
    if args.insert_depth_tolerance < 0.0:
        raise ValueError("--insert-depth-tolerance must be non-negative")
    if args.insert_translation_abort_tolerance < args.translation_tolerance:
        raise ValueError(
            "--insert-translation-abort-tolerance must be at least --translation-tolerance"
        )
    if args.insert_tilt_abort_tolerance_deg < args.tilt_tolerance_deg:
        raise ValueError(
            "--insert-tilt-abort-tolerance-deg must be at least --tilt-tolerance-deg"
        )
    if args.insert_yaw_abort_tolerance_deg < args.yaw_tolerance_deg:
        raise ValueError(
            "--insert-yaw-abort-tolerance-deg must be at least --yaw-tolerance-deg"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(
        args.output_dir
        or Path("outputs") / "window_fine_align" / f"sac-{timestamp}-seed{args.seed}"
    ).resolve()
    model_dir = output_dir / "models"
    checkpoint_dir = model_dir / "checkpoints"
    monitor_dir = output_dir / "monitor"
    evaluation_dir = output_dir / "evaluation"
    tensorboard_dir = output_dir / "tensorboard"
    for directory in (
        model_dir,
        checkpoint_dir,
        monitor_dir,
        evaluation_dir,
        tensorboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    train_kwargs = environment_kwargs(args, evaluation=False)
    eval_kwargs = environment_kwargs(args, evaluation=True)
    configuration = {
        **vars(args),
        "environment_id": ENV_ID,
        "policy": "MlpPolicy",
        "train_environment_kwargs": train_kwargs,
        "evaluation_environment_kwargs": eval_kwargs,
        "git_commit": git_commit(),
        "resolved_output_dir": str(output_dir),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")
    (output_dir / "git_commit.txt").write_text(
        f"{configuration['git_commit']}\n", encoding="utf-8"
    )

    train_env = DummyVecEnv(
        [
            make_environment(
                seed=args.seed + rank,
                env_kwargs=train_kwargs,
                monitor_file=monitor_dir / f"train_env_{rank}.csv",
            )
            for rank in range(args.n_envs)
        ]
    )
    eval_env = DummyVecEnv(
        [
            make_environment(
                seed=args.seed + 100_000,
                env_kwargs=eval_kwargs,
                monitor_file=monitor_dir / "evaluation.csv",
            )
        ]
    )

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(evaluation_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=max(args.save_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="window_fine_align_sac",
                save_replay_buffer=True,
            ),
        ]
    )
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=args.progress_bar,
        )
        latest_path = model_dir / "latest_model"
        model.save(str(latest_path))
        model.save_replay_buffer(str(model_dir / "latest_replay_buffer.pkl"))

        # A save/load/predict check makes the training artifact self-validating.
        reloaded = SAC.load(str(latest_path), env=eval_env, device=args.device)
        observation = eval_env.reset()
        action, _ = reloaded.predict(observation, deterministic=True)
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Reloaded model produced a non-finite action")
        (output_dir / "smoke_reload_ok.txt").write_text("ok\n", encoding="utf-8")
    finally:
        train_env.close()
        eval_env.close()

    print(f"Training complete: {output_dir}")


if __name__ == "__main__":
    main()
