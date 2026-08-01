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

import panda_mujoco_gym  # noqa: F401
from panda_mujoco_gym.envs.window_assembly_pipeline import WindowAssemblyPipeline
from utils.fine_align_config import (
    add_final_environment_arguments,
    final_environment_kwargs,
    validate_final_environment_kwargs,
)
from utils.io_utils import load_sac_model


ENV_ID = "FrankaWindowFineAlignDense-v0"

TRACKED_FAILURE_REASONS = (
    "insert_final_error_u_exceeded",
    "insert_final_error_v_exceeded",
    "insert_final_alignment_not_stable",
    "hold_error_u_exceeded",
    "hold_error_v_exceeded",
)

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
    parser.add_argument("--device", default="auto")
    add_final_environment_arguments(parser)
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


def print_effective_environment_configuration(
    env_kwargs: dict[str, Any],
) -> None:
    printed_keys = (
        "preinsert_normal_offset",
        "insert_action_sim_steps",
        "insert_translation_abort_tolerance",
        "insert_tilt_abort_tolerance_deg",
        "insert_yaw_abort_tolerance_deg",
        "insert_alignment_violation_hold_steps",
        "insert_success_hold_steps",
        "insert_final_verification_max_steps",
        "insert_depth_tolerance",
        "insert_step_size",
    )
    print("Effective environment configuration:")
    for key in printed_keys:
        print(f"  {key}: {env_kwargs.get(key)}")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(results), 1)
    failures = Counter(
        result["failure_reason"] or "none"
        for result in results
        if not result["full_pipeline_success"]
    )
    failure_categories = Counter(
        result["failure_category"] or "none"
        for result in results
        if not result["full_pipeline_success"]
    )
    for reason in TRACKED_FAILURE_REASONS:
        failures.setdefault(reason, 0)
    failure_categories.setdefault("final_assembly_verification_failed", 0)
    final_translation = [
        float(np.hypot(result["pipeline_end_errors"]["error_u"], result["pipeline_end_errors"]["error_v"]))
        for result in results
    ]
    final_tilt = [
        float(
            np.hypot(
                result["pipeline_end_errors"]["tilt_error_t1"],
                result["pipeline_end_errors"]["tilt_error_t2"],
            )
        )
        for result in results
    ]
    final_yaw = [abs(float(result["pipeline_end_errors"]["yaw_error"])) for result in results]
    final_normal_gap_error = [
        abs(float(result["pipeline_end_errors"]["normal_gap_error"]))
        for result in results
    ]
    return {
        "episodes": len(results),
        "fine_align_success_rate": sum(r["fine_align_success"] for r in results) / count,
        "full_pipeline_success_rate": sum(r["full_pipeline_success"] for r in results) / count,
        "insert_success_rate": sum(r["insert_success"] for r in results) / count,
        "hold_success_rate": sum(r["hold_success"] for r in results) / count,
        "assembly_verified_rate": sum(r["assembly_verified"] for r in results) / count,
        "mean_fine_align_steps": float(np.mean([r["fine_align_steps"] for r in results])) if results else 0.0,
        "mean_final_translation_error_m": float(np.mean(final_translation)) if results else 0.0,
        "mean_final_tilt_error_rad": float(np.mean(final_tilt)) if results else 0.0,
        "mean_final_yaw_error_rad": float(np.mean(final_yaw)) if results else 0.0,
        "mean_final_normal_gap_error_m": float(np.mean(final_normal_gap_error)) if results else 0.0,
        "max_final_normal_gap_error_m": float(np.max(final_normal_gap_error)) if results else 0.0,
        "failure_reason_counts": dict(sorted(failures.items())),
        "failure_category_counts": dict(sorted(failure_categories.items())),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    model_path = Path(args.model).expanduser().resolve()
    env_kwargs = final_environment_kwargs(args)
    validate_final_environment_kwargs(env_kwargs)
    output_dir = Path(
        args.output_dir
        or Path("outputs")
        / "window_assembly_evaluation"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print_effective_environment_configuration(env_kwargs)
    render_mode = "rgb_array" if args.record_video else ("human" if args.render else None)
    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        **env_kwargs,
    )
    model = load_sac_model(model_path, device=args.device)
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
            insert_start_errors = result.get("insert_start_errors", {})
            insert_max_errors = result.get("insert_max_errors", {})
            print(
                f"episode={episode_index} fine={result['fine_align_success']} "
                f"insert={result['insert_success']} hold={result['hold_success']} "
                f"assembly_verified={result['assembly_verified']} "
                f"full={result['full_pipeline_success']} "
                f"fine_steps={result['fine_align_steps']} insert_steps={result['insert_steps']} "
                f"hold_steps={result['hold_steps']} "
                f"insert_final_verification_steps={result['insert_final_verification_steps']} "
                f"stage={result['stage']} "
                f"start_errors={insert_start_errors} max_errors={insert_max_errors} "
                f"insert_depth={result['insert_depth']:.4f} "
                f"final_errors={result['final_errors']} "
                f"failure={result['failure_reason'] or 'none'} "
                f"category={result['failure_category'] or 'none'}"
            )
    finally:
        env.close()

    summary = summarize(results)
    summary["model_path"] = str(model_path)
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
