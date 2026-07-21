"""Evaluate the strict nine-stage window assembly pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import panda_mujoco_gym  # noqa: F401
from panda_mujoco_gym.envs.window_assembly_pipeline import WindowAssemblyPipeline


ENV_ID = "FrankaWindowFineAlignDense-v0"

ENV_ARG_DEFAULTS = {
    "measurement_noise_translation_std": 0.0,
    "measurement_noise_angle_std": 0.0,
    "preinsert_normal_offset": 0.20,
    "fine_action_sim_steps": 4,
    "normal_gap_correction_threshold": 0.0005,
    "normal_gap_correction_sim_steps": 2,
    "max_normal_gap_drift": 0.008,
}

ENV_ARG_TO_KWARG = {
    "measurement_noise_translation_std": "measurement_noise_translation_std",
    "measurement_noise_angle_std": "measurement_noise_angle_std_deg",
    "preinsert_normal_offset": "preinsert_normal_offset",
    "fine_action_sim_steps": "fine_action_sim_steps",
    "normal_gap_correction_threshold": "normal_gap_correction_threshold",
    "normal_gap_correction_sim_steps": "normal_gap_correction_sim_steps",
    "max_normal_gap_drift": "max_normal_gap_drift",
}

ENV_ARG_FLAGS = {
    "measurement_noise_translation_std": "--measurement-noise-translation-std",
    "measurement_noise_angle_std": "--measurement-noise-angle-std",
    "preinsert_normal_offset": "--preinsert-normal-offset",
    "fine_action_sim_steps": "--fine-action-sim-steps",
    "normal_gap_correction_threshold": "--normal-gap-correction-threshold",
    "normal_gap_correction_sim_steps": "--normal-gap-correction-sim-steps",
    "max_normal_gap_drift": "--max-normal-gap-drift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--measurement-noise-translation-std", type=float, default=0.0)
    parser.add_argument("--measurement-noise-angle-std", type=float, default=0.0)
    parser.add_argument("--preinsert-normal-offset", type=float, default=0.20)
    parser.add_argument("--fine-action-sim-steps", type=int, default=4)
    parser.add_argument("--normal-gap-correction-threshold", type=float, default=0.0005)
    parser.add_argument("--normal-gap-correction-sim-steps", type=int, default=2)
    parser.add_argument("--max-normal-gap-drift", type=float, default=0.008)
    return parser.parse_args()


class VideoCollector:
    def __init__(self, output_path: Path, fps: int) -> None:
        self.output_path = output_path
        self.fps = int(fps)
        self.frames: list[np.ndarray] = []

    def capture(self, env: gym.Env) -> None:
        frame = env.render()
        if frame is not None:
            self.frames.append(np.asarray(frame, dtype=np.uint8).copy())

    def save(self) -> None:
        if not self.frames:
            return
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("--record-video requires imageio") from exc
        imageio.mimsave(self.output_path, self.frames, fps=self.fps)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _provided_flags(argv: list[str]) -> set[str]:
    flags: set[str] = set()
    for arg in argv:
        if not arg.startswith("--"):
            continue
        flags.add(arg.split("=", 1)[0])
    return flags


def find_training_config(model_path: str) -> Path | None:
    model = Path(model_path).expanduser().resolve()
    start = model.parent if model.suffix or not model.is_dir() else model
    for directory in (start, *start.parents):
        candidate = directory / "training_config.json"
        if candidate.is_file():
            return candidate
    return None


def load_training_environment_kwargs(model_path: str) -> tuple[dict[str, Any], Path | None]:
    config_path = find_training_config(model_path)
    if config_path is None:
        return {}, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_kwargs = config.get("train_environment_kwargs", {})
    if not isinstance(train_kwargs, dict):
        train_kwargs = {}
    return dict(train_kwargs), config_path


def environment_kwargs(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    env_kwargs, config_path = load_training_environment_kwargs(args.model)
    provided = _provided_flags(sys.argv[1:])
    for arg_name, default_value in ENV_ARG_DEFAULTS.items():
        kwarg_name = ENV_ARG_TO_KWARG[arg_name]
        if ENV_ARG_FLAGS[arg_name] in provided or kwarg_name not in env_kwargs:
            env_kwargs[kwarg_name] = getattr(args, arg_name, default_value)
    return env_kwargs, config_path


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(results), 1)
    failures = Counter(
        result["failure_reason"] or "none"
        for result in results
        if not result["full_pipeline_success"]
    )
    final_translation = [
        float(np.hypot(result["final_errors"]["error_u"], result["final_errors"]["error_v"]))
        for result in results
    ]
    final_tilt = [
        float(
            np.hypot(
                result["final_errors"]["tilt_error_t1"],
                result["final_errors"]["tilt_error_t2"],
            )
        )
        for result in results
    ]
    final_yaw = [abs(float(result["final_errors"]["yaw_error"])) for result in results]
    final_normal_gap_error = [
        abs(float(result["final_errors"]["normal_gap_error"]))
        for result in results
    ]
    return {
        "episodes": len(results),
        "fine_align_success_rate": sum(r["fine_align_success"] for r in results) / count,
        "full_pipeline_success_rate": sum(r["full_pipeline_success"] for r in results) / count,
        "insert_success_rate": sum(r["insert_success"] for r in results) / count,
        "hold_success_rate": sum(r["hold_success"] for r in results) / count,
        "mean_fine_align_steps": float(np.mean([r["fine_align_steps"] for r in results])) if results else 0.0,
        "mean_final_translation_error_m": float(np.mean(final_translation)) if results else 0.0,
        "mean_final_tilt_error_rad": float(np.mean(final_tilt)) if results else 0.0,
        "mean_final_yaw_error_rad": float(np.mean(final_yaw)) if results else 0.0,
        "mean_final_normal_gap_error_m": float(np.mean(final_normal_gap_error)) if results else 0.0,
        "max_final_normal_gap_error_m": float(np.max(final_normal_gap_error)) if results else 0.0,
        "failure_reason_counts": dict(sorted(failures.items())),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.fine_action_sim_steps <= 0 or args.normal_gap_correction_sim_steps <= 0:
        raise ValueError("--fine-action-sim-steps and --normal-gap-correction-sim-steps must be positive")
    if args.preinsert_normal_offset <= 0.0:
        raise ValueError("--preinsert-normal-offset must be positive")
    if args.normal_gap_correction_threshold < 0.0:
        raise ValueError("--normal-gap-correction-threshold must be non-negative")
    if args.max_normal_gap_drift <= 0.0:
        raise ValueError("--max-normal-gap-drift must be positive")
    output_dir = Path(
        args.output_dir
        or Path("outputs")
        / "window_assembly_evaluation"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs, training_config_path = environment_kwargs(args)
    render_mode = "rgb_array" if args.record_video else ("human" if args.render else None)
    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        **env_kwargs,
    )
    model = SAC.load(args.model, device="auto")
    pipeline = WindowAssemblyPipeline(env)
    results: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            collector = None
            if args.record_video:
                collector = VideoCollector(
                    output_dir / f"episode_{episode_index:04d}.mp4",
                    fps=int(env.metadata.get("render_fps", 20)),
                )
                env.unwrapped.pipeline_frame_callback = lambda c=collector: c.capture(env)
            elif args.render:
                env.unwrapped.pipeline_frame_callback = env.render
            else:
                env.unwrapped.pipeline_frame_callback = None

            result = pipeline.run_episode(
                model,
                seed=args.seed + episode_index,
                deterministic=args.deterministic,
            )
            results.append(result)
            if collector is not None:
                collector.save()
            print(
                f"episode={episode_index} fine={result['fine_align_success']} "
                f"insert={result['insert_success']} hold={result['hold_success']} "
                f"steps={result['fine_align_steps']} failure={result['failure_reason'] or 'none'}"
            )
    finally:
        env.close()

    summary = summarize(results)
    summary["training_config_path"] = str(training_config_path) if training_config_path is not None else ""
    summary["environment_kwargs"] = env_kwargs
    (output_dir / "episodes.json").write_text(
        json.dumps(results, indent=2, default=json_default), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evaluation results: {output_dir}")


if __name__ == "__main__":
    main()
