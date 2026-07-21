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
        "failure_reason_counts": dict(sorted(failures.items())),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    output_dir = Path(
        args.output_dir
        or Path("outputs")
        / "window_assembly_evaluation"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_mode = "rgb_array" if args.record_video else ("human" if args.render else None)
    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        measurement_noise_translation_std=args.measurement_noise_translation_std,
        measurement_noise_angle_std_deg=args.measurement_noise_angle_std,
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
