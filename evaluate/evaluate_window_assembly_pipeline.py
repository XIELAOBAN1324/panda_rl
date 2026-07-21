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

TRACKED_FAILURE_REASONS = (
    "insert_final_error_u_exceeded",
    "insert_final_error_v_exceeded",
    "insert_final_alignment_not_stable",
    "hold_error_u_exceeded",
    "hold_error_v_exceeded",
)

ENV_ARG_DEFAULTS = {
    "measurement_noise_translation_std": 0.0,
    "measurement_noise_angle_std": 0.0,
    "preinsert_normal_offset": 0.08,
    "fine_action_sim_steps": 4,
    "normal_gap_correction_threshold": 0.0005,
    "normal_gap_correction_sim_steps": 2,
    "max_normal_gap_drift": 0.008,
    "insert_action_sim_steps": 4,
    "insert_translation_abort_tolerance": 0.0025,
    "insert_tilt_abort_tolerance_deg": 0.6,
    "insert_yaw_abort_tolerance_deg": 0.6,
    "insert_alignment_violation_hold_steps": 2,
    "insert_success_hold_steps": 3,
    "insert_final_verification_max_steps": 20,
    "insert_depth_tolerance": 0.0005,
    "insert_step_size": 0.001,
}

ENV_ARG_TO_KWARG = {
    "measurement_noise_translation_std": "measurement_noise_translation_std",
    "measurement_noise_angle_std": "measurement_noise_angle_std_deg",
    "preinsert_normal_offset": "preinsert_normal_offset",
    "fine_action_sim_steps": "fine_action_sim_steps",
    "normal_gap_correction_threshold": "normal_gap_correction_threshold",
    "normal_gap_correction_sim_steps": "normal_gap_correction_sim_steps",
    "max_normal_gap_drift": "max_normal_gap_drift",
    "insert_action_sim_steps": "insert_action_sim_steps",
    "insert_translation_abort_tolerance": "insert_translation_abort_tolerance",
    "insert_tilt_abort_tolerance_deg": "insert_tilt_abort_tolerance_deg",
    "insert_yaw_abort_tolerance_deg": "insert_yaw_abort_tolerance_deg",
    "insert_alignment_violation_hold_steps": "insert_alignment_violation_hold_steps",
    "insert_success_hold_steps": "insert_success_hold_steps",
    "insert_final_verification_max_steps": "insert_final_verification_max_steps",
    "insert_depth_tolerance": "insert_depth_tolerance",
    "insert_step_size": "insert_step_size",
}

ENV_ARG_FLAGS = {
    "measurement_noise_translation_std": "--measurement-noise-translation-std",
    "measurement_noise_angle_std": "--measurement-noise-angle-std",
    "preinsert_normal_offset": "--preinsert-normal-offset",
    "fine_action_sim_steps": "--fine-action-sim-steps",
    "normal_gap_correction_threshold": "--normal-gap-correction-threshold",
    "normal_gap_correction_sim_steps": "--normal-gap-correction-sim-steps",
    "max_normal_gap_drift": "--max-normal-gap-drift",
    "insert_action_sim_steps": "--insert-action-sim-steps",
    "insert_translation_abort_tolerance": "--insert-translation-abort-tolerance",
    "insert_tilt_abort_tolerance_deg": "--insert-tilt-abort-tolerance-deg",
    "insert_yaw_abort_tolerance_deg": "--insert-yaw-abort-tolerance-deg",
    "insert_alignment_violation_hold_steps": "--insert-alignment-violation-hold-steps",
    "insert_success_hold_steps": "--insert-success-hold-steps",
    "insert_final_verification_max_steps": "--insert-final-verification-max-steps",
    "insert_depth_tolerance": "--insert-depth-tolerance",
    "insert_step_size": "--insert-step-size",
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
    parser.add_argument("--preinsert-normal-offset", type=float, default=0.08)
    parser.add_argument("--fine-action-sim-steps", type=int, default=4)
    parser.add_argument("--normal-gap-correction-threshold", type=float, default=0.0005)
    parser.add_argument("--normal-gap-correction-sim-steps", type=int, default=2)
    parser.add_argument("--max-normal-gap-drift", type=float, default=0.008)
    parser.add_argument("--insert-action-sim-steps", type=int, default=4)
    parser.add_argument("--insert-translation-abort-tolerance", type=float, default=0.0025)
    parser.add_argument("--insert-tilt-abort-tolerance-deg", type=float, default=0.6)
    parser.add_argument("--insert-yaw-abort-tolerance-deg", type=float, default=0.6)
    parser.add_argument("--insert-alignment-violation-hold-steps", type=int, default=2)
    parser.add_argument("--insert-success-hold-steps", type=int, default=3)
    parser.add_argument("--insert-final-verification-max-steps", type=int, default=20)
    parser.add_argument("--insert-depth-tolerance", type=float, default=0.0005)
    parser.add_argument("--insert-step-size", type=float, default=0.001)
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


def _values_differ(first: Any, second: Any) -> bool:
    try:
        return not bool(np.isclose(float(first), float(second), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return first != second


def build_environment_kwargs(
    model_path: str,
    cli_values: dict[str, Any],
    provided_flags: set[str],
) -> tuple[dict[str, Any], Path | None, dict[str, tuple[Any, Any]]]:
    env_kwargs, config_path = load_training_environment_kwargs(model_path)
    loaded_kwargs = dict(env_kwargs)
    explicit_overrides: dict[str, tuple[Any, Any]] = {}
    for arg_name, default_value in ENV_ARG_DEFAULTS.items():
        kwarg_name = ENV_ARG_TO_KWARG[arg_name]
        cli_value = cli_values.get(arg_name, default_value)
        if ENV_ARG_FLAGS[arg_name] in provided_flags:
            if kwarg_name in loaded_kwargs and _values_differ(
                loaded_kwargs[kwarg_name],
                cli_value,
            ):
                explicit_overrides[kwarg_name] = (loaded_kwargs[kwarg_name], cli_value)
            env_kwargs[kwarg_name] = cli_value
        elif kwarg_name not in env_kwargs:
            env_kwargs[kwarg_name] = cli_value
    return env_kwargs, config_path, explicit_overrides


def environment_kwargs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None, dict[str, tuple[Any, Any]]]:
    return build_environment_kwargs(
        args.model,
        vars(args),
        _provided_flags(sys.argv[1:]),
    )


def validate_environment_kwargs(env_kwargs: dict[str, Any]) -> None:
    if int(env_kwargs.get("fine_action_sim_steps", 4)) <= 0:
        raise ValueError("fine_action_sim_steps must be positive")
    if int(env_kwargs.get("normal_gap_correction_sim_steps", 2)) <= 0:
        raise ValueError("normal_gap_correction_sim_steps must be positive")
    if int(env_kwargs.get("insert_action_sim_steps", 4)) <= 0:
        raise ValueError("insert_action_sim_steps must be positive")
    if int(env_kwargs.get("insert_alignment_violation_hold_steps", 2)) <= 0:
        raise ValueError("insert_alignment_violation_hold_steps must be positive")
    if int(env_kwargs.get("insert_success_hold_steps", 3)) <= 0:
        raise ValueError("insert_success_hold_steps must be positive")
    if int(env_kwargs.get("insert_final_verification_max_steps", 20)) <= 0:
        raise ValueError("insert_final_verification_max_steps must be positive")
    if float(env_kwargs.get("preinsert_normal_offset", 0.08)) <= 0.0:
        raise ValueError("preinsert_normal_offset must be positive")
    if float(env_kwargs.get("normal_gap_correction_threshold", 0.0005)) < 0.0:
        raise ValueError("normal_gap_correction_threshold must be non-negative")
    if float(env_kwargs.get("max_normal_gap_drift", 0.008)) <= 0.0:
        raise ValueError("max_normal_gap_drift must be positive")
    if float(env_kwargs.get("insert_step_size", 0.001)) <= 0.0:
        raise ValueError("insert_step_size must be positive")
    if float(env_kwargs.get("insert_depth_tolerance", 0.0005)) < 0.0:
        raise ValueError("insert_depth_tolerance must be non-negative")
    translation_tolerance = float(env_kwargs.get("translation_tolerance", 0.002))
    tilt_tolerance_deg = float(env_kwargs.get("tilt_tolerance_deg", 0.5))
    yaw_tolerance_deg = float(env_kwargs.get("yaw_tolerance_deg", 0.5))
    if float(env_kwargs.get("insert_translation_abort_tolerance", 0.0025)) < translation_tolerance:
        raise ValueError(
            "insert_translation_abort_tolerance must be at least translation_tolerance"
        )
    if float(env_kwargs.get("insert_tilt_abort_tolerance_deg", 0.6)) < tilt_tolerance_deg:
        raise ValueError(
            "insert_tilt_abort_tolerance_deg must be at least tilt_tolerance_deg"
        )
    if float(env_kwargs.get("insert_yaw_abort_tolerance_deg", 0.6)) < yaw_tolerance_deg:
        raise ValueError(
            "insert_yaw_abort_tolerance_deg must be at least yaw_tolerance_deg"
        )


def print_effective_environment_configuration(
    env_kwargs: dict[str, Any],
    *,
    explicit_overrides: dict[str, tuple[Any, Any]],
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
    if explicit_overrides:
        print("Evaluating model with overridden environment configuration.")
        for key, (old_value, new_value) in sorted(explicit_overrides.items()):
            print(f"  override {key}: {old_value} -> {new_value}")
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
    if args.max_normal_gap_drift <= 0.0:
        raise ValueError("--max-normal-gap-drift must be positive")
    if args.insert_step_size <= 0.0:
        raise ValueError("--insert-step-size must be positive")
    if args.insert_depth_tolerance < 0.0:
        raise ValueError("--insert-depth-tolerance must be non-negative")
    output_dir = Path(
        args.output_dir
        or Path("outputs")
        / "window_assembly_evaluation"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs, training_config_path, explicit_overrides = environment_kwargs(args)
    validate_environment_kwargs(env_kwargs)
    print_effective_environment_configuration(
        env_kwargs,
        explicit_overrides=explicit_overrides,
    )
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
