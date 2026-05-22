#!/usr/bin/env python3
"""Record rollout videos from a trained SAC policy checkpoint."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import panda_mujoco_gym  # noqa: F401
from evaluate.video_recorder import EpisodeVideoRecorder, _to_json_safe
from train.train_sac import create_env


DEFAULT_ENV_ID = "FrankaPickAndPlaceWindowSparse-v0"


def _infer_exp_dir_from_model_path(model_path: Path) -> Optional[Path]:
    for parent in model_path.parents:
        if parent.name == "models":
            return parent.parent
    return None


def _load_training_summary(exp_dir: Optional[Path]) -> Dict[str, Any]:
    if exp_dir is None:
        return {}

    for relative_path in ("training_summary.json", "logs/training_summary.json"):
        summary_path = exp_dir / relative_path
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _ensure_zip_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix != ".zip":
        zip_path = Path(str(path) + ".zip")
        if zip_path.exists():
            return zip_path
    return path


def _resolve_model_path(args: argparse.Namespace, exp_dir: Optional[Path]) -> Path:
    if args.model_path is not None:
        return _ensure_zip_path(Path(args.model_path).expanduser().resolve())

    if exp_dir is None:
        raise ValueError("Pass --exp-dir or --model-path.")

    models_dir = exp_dir / "models"
    requested = args.model_name or "best_model"
    candidate = _ensure_zip_path(models_dir / requested)
    if candidate.exists():
        return candidate.resolve()

    if requested == "best_model":
        fallback = _ensure_zip_path(models_dir / "final_model")
        if fallback.exists():
            return fallback.resolve()

    raise FileNotFoundError(f"Model checkpoint not found: {candidate}")


def _candidate_vecnormalize_paths(exp_dir: Optional[Path], model_path: Path) -> List[Path]:
    candidates: List[Path] = []
    model_stem = model_path.stem

    candidates.append(model_path.with_name(f"{model_stem}_vecnormalize.pkl"))

    checkpoint_match = re.match(r"(.+)_([0-9]+)_steps$", model_stem)
    if checkpoint_match:
        prefix, steps = checkpoint_match.groups()
        candidates.append(model_path.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl"))

    if exp_dir is not None:
        models_dir = exp_dir / "models"
        candidates.append(models_dir / f"{model_stem}_vecnormalize.pkl")
        if checkpoint_match:
            prefix, steps = checkpoint_match.groups()
            candidates.append(models_dir / "checkpoints" / f"{prefix}_vecnormalize_{steps}_steps.pkl")
        candidates.append(models_dir / "vec_normalize.pkl")

    seen = set()
    unique_candidates = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique_candidates.append(path)
            seen.add(resolved)
    return unique_candidates


def _resolve_vecnormalize_path(
    args: argparse.Namespace,
    exp_dir: Optional[Path],
    model_path: Path,
    config: Dict[str, Any],
) -> Optional[Path]:
    if args.no_vecnormalize:
        return None

    if args.vecnormalize_path is not None:
        path = Path(args.vecnormalize_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"VecNormalize file not found: {path}")
        return path

    normalize_env = bool(config.get("normalize_env", False))
    if not normalize_env:
        return None

    for candidate in _candidate_vecnormalize_paths(exp_dir, model_path):
        if candidate.exists():
            return candidate.resolve()

    searched = "\n  ".join(str(path) for path in _candidate_vecnormalize_paths(exp_dir, model_path))
    raise FileNotFoundError(
        "Training summary says normalize_env=True, but no VecNormalize stats were found.\n"
        f"Searched:\n  {searched}\n"
        "Pass --vecnormalize-path explicitly, or use --no-vecnormalize if this checkpoint was not normalized."
    )


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    effective = dict(config)

    scalar_overrides = {
        "env_name": args.env,
        "reward_scale": args.reward_scale,
        "expert_hint_style": args.expert_hint_style,
        "residual_action_scale": args.residual_action_scale,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            effective[key] = value

    bool_overrides = {
        "task_geometry_features": args.task_geometry_features,
        "task_stage_features": args.task_stage_features,
        "task_progress_features": args.task_progress_features,
        "residual_guidance": args.residual_guidance,
    }
    for key, value in bool_overrides.items():
        if value is not None:
            effective[key] = bool(value)

    effective.setdefault("env_name", DEFAULT_ENV_ID)
    effective.setdefault("reward_scale", 1.0)
    effective.setdefault("dense_reward_style", "standard")
    effective.setdefault("task_geometry_features", False)
    effective.setdefault("task_stage_features", False)
    effective.setdefault("task_progress_features", False)
    effective.setdefault("expert_hint_style", "staged")
    effective.setdefault("residual_guidance", False)
    effective.setdefault("residual_action_scale", 0.1)
    effective.setdefault("safe_curriculum", False)
    return effective


def _checkpoint_spaces(model_path: Path):
    data, _, _ = load_from_zip_file(str(model_path), load_data=True, device="cpu")
    if not data:
        return None, None
    return data.get("observation_space"), data.get("action_space")


def _spaces_match(left, right) -> bool:
    if left is None or right is None:
        return False
    return left == right


def _checkpoint_target_env_ids(config: Dict[str, Any], args: argparse.Namespace, model_path: Path) -> List[str]:
    env_ids = []
    if args.env is not None:
        env_ids.append(str(args.env))
    if config.get("env_name"):
        env_ids.append(str(config["env_name"]))

    try:
        expected_obs_space, expected_action_space = _checkpoint_spaces(model_path)
    except Exception:
        expected_obs_space, expected_action_space = None, None

    candidate_env_ids = [
        DEFAULT_ENV_ID,
        "FrankaPickAndPlaceWindowInsertSparse-v0",
        "FrankaPickAndPlaceWindowPrealignSparse-v0",
        "FrankaPickAndPlaceWindowDense-v0",
        "FrankaPickAndPlaceWindowInsertDense-v0",
        "FrankaPickAndPlaceWindowPrealignDense-v0",
    ]

    for env_id in candidate_env_ids:
        if env_id in env_ids:
            continue
        if expected_action_space is None:
            env_ids.append(env_id)
            continue
        try:
            probe_env = gym.make(env_id)
            action_matches = _spaces_match(probe_env.action_space, expected_action_space)
            probe_env.close()
        except Exception:
            action_matches = False
        if action_matches:
            env_ids.append(env_id)

    return list(dict.fromkeys(env_ids))


def _candidate_wrapper_configs(config: Dict[str, Any], args: argparse.Namespace, model_path: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = [dict(config)]
    if (
        args.task_geometry_features is not None
        or args.task_stage_features is not None
        or args.task_progress_features is not None
        or args.residual_guidance is not None
    ):
        return candidates

    for env_id in _checkpoint_target_env_ids(config, args, model_path):
        for feature_key in (None, "task_geometry_features", "task_stage_features", "task_progress_features"):
            candidate = dict(config)
            candidate["env_name"] = env_id
            candidate["task_geometry_features"] = False
            candidate["task_stage_features"] = False
            candidate["task_progress_features"] = False
            if feature_key is not None:
                candidate[feature_key] = True
            candidate.setdefault("expert_hint_style", "staged")
            candidate.setdefault("residual_guidance", False)
            candidates.append(candidate)

    unique = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate.get("env_name"),
            bool(candidate.get("task_geometry_features", False)),
            bool(candidate.get("task_stage_features", False)),
            bool(candidate.get("task_progress_features", False)),
            bool(candidate.get("residual_guidance", False)),
            str(candidate.get("expert_hint_style", "staged")),
            float(candidate.get("residual_action_scale", 0.1)),
        )
        if key in seen:
            continue
        unique.append(candidate)
        seen.add(key)
    return unique


def _auto_match_checkpoint_config(
    model_path: Path,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        expected_obs_space, expected_action_space = _checkpoint_spaces(model_path)
    except Exception:
        return config

    if expected_obs_space is None or expected_action_space is None:
        return config

    errors = []
    for candidate in _candidate_wrapper_configs(config, args, model_path):
        env = None
        try:
            env = _make_policy_env(
                config=candidate,
                render_mode=None,
                seed=None,
                eval_dense_reward=bool(args.eval_dense_reward),
            )
            if _spaces_match(env.observation_space, expected_obs_space) and _spaces_match(
                env.action_space,
                expected_action_space,
            ):
                changed = {
                    key: candidate.get(key)
                    for key in (
                        "env_name",
                        "task_geometry_features",
                        "task_stage_features",
                        "task_progress_features",
                        "residual_guidance",
                    )
                    if candidate.get(key) != config.get(key)
                }
                if changed:
                    print(f"Auto-matched checkpoint environment config: {changed}")
                return candidate
            errors.append(
                f"{candidate.get('env_name')} obs={env.observation_space} action={env.action_space}"
            )
        except Exception as exc:
            errors.append(f"{candidate.get('env_name')}: {exc}")
        finally:
            if env is not None:
                env.close()

    searched = "\n  ".join(errors[:12])
    raise ValueError(
        "Could not create an environment matching the checkpoint spaces.\n"
        f"Checkpoint observation_space: {expected_obs_space}\n"
        f"Checkpoint action_space: {expected_action_space}\n"
        f"Tried:\n  {searched}"
    )


def _make_policy_env(
    config: Dict[str, Any],
    render_mode: Optional[str],
    seed: Optional[int],
    eval_dense_reward: bool,
):
    return create_env(
        config["env_name"],
        render_mode=render_mode,
        reward_scale=float(config.get("reward_scale", 1.0)),
        seed=seed,
        dense_reward_shaping=bool(eval_dense_reward),
        dense_reward_style=str(config.get("dense_reward_style", "standard")),
        task_geometry_features=bool(config.get("task_geometry_features", False)),
        task_stage_features=bool(config.get("task_stage_features", False)),
        task_progress_features=bool(config.get("task_progress_features", False)),
        expert_hint_style=str(config.get("expert_hint_style", "staged")),
        residual_guidance=bool(config.get("residual_guidance", False)),
        residual_action_scale=float(config.get("residual_action_scale", 0.1)),
        safe_curriculum=False,
    )


def _load_vec_normalizer(
    vecnormalize_path: Optional[Path],
    config: Dict[str, Any],
    eval_dense_reward: bool,
) -> Optional[VecNormalize]:
    if vecnormalize_path is None:
        return None

    def make_stats_env():
        return _make_policy_env(
            config=config,
            render_mode=None,
            seed=None,
            eval_dense_reward=eval_dense_reward,
        )

    stats_env = DummyVecEnv([make_stats_env])
    normalizer = VecNormalize.load(str(vecnormalize_path), stats_env)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def _normalize_observation(obs: Any, normalizer: Optional[VecNormalize]) -> Any:
    if normalizer is None:
        return obs
    return normalizer.normalize_obs(obs)


def _resize_frame(frame: np.ndarray, width: Optional[int], height: Optional[int]) -> np.ndarray:
    if width is None and height is None:
        return frame

    original_height, original_width = frame.shape[:2]
    if width is None:
        width = int(round(original_width * float(height) / float(original_height)))
    if height is None:
        height = int(round(original_height * float(width) / float(original_width)))
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _capture_frame(
    env,
    recorder: EpisodeVideoRecorder,
    width: Optional[int],
    height: Optional[int],
) -> None:
    rendered = env.render()
    if rendered is None:
        return

    frame = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    recorder.add_frame(_resize_frame(frame, width=width, height=height))


def _episode_max_steps(env, requested_max_steps: Optional[int]) -> int:
    if requested_max_steps is not None:
        return max(int(requested_max_steps), 1)
    spec = getattr(env, "spec", None)
    spec_steps = getattr(spec, "max_episode_steps", None)
    if spec_steps is not None:
        return int(spec_steps)
    return 250


def _safe_float(info: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = info.get(key, default)
    try:
        return float(np.asarray(value).item())
    except Exception:
        return float(default)


def _safe_bool(info: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = info.get(key, default)
    try:
        return bool(np.asarray(value).item())
    except Exception:
        return bool(default)


def record_policy_episode(
    env,
    model: SAC,
    normalizer: Optional[VecNormalize],
    recorder: EpisodeVideoRecorder,
    episode_id: int,
    seed: int,
    model_label: str,
    max_steps: int,
    deterministic: bool,
    stop_on_success: bool,
    initial_frames: int,
    terminal_hold_frames: int,
    success_hold_frames: int,
    frame_width: Optional[int],
    frame_height: Optional[int],
) -> Dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminal_info: Dict[str, Any] = {}
    terminal_success = False
    any_success = False
    terminated = False
    truncated = False

    recorder.start_episode_recording()
    for _ in range(max(initial_frames, 0)):
        _capture_frame(env, recorder, width=frame_width, height=frame_height)

    step_count = 0
    for step_idx in range(max_steps):
        model_observation = _normalize_observation(observation, normalizer)
        action, _ = model.predict(model_observation, deterministic=deterministic)

        observation, reward, terminated, truncated, terminal_info = env.step(action)
        step_count = step_idx + 1
        total_reward += float(np.asarray(reward).item())

        _capture_frame(env, recorder, width=frame_width, height=frame_height)

        terminal_success = _safe_bool(terminal_info, "is_success", False)
        any_success = any_success or terminal_success or _safe_bool(terminal_info, "episode_any_success", False)
        done = bool(terminated or truncated)

        if stop_on_success and any_success:
            for _ in range(max(success_hold_frames, 0)):
                _capture_frame(env, recorder, width=frame_width, height=frame_height)
            break

        if done:
            for _ in range(max(terminal_hold_frames, 0)):
                _capture_frame(env, recorder, width=frame_width, height=frame_height)
            break

    episode_info = {
        "stage": model_label,
        "model_label": model_label,
        "episode_id": int(episode_id),
        "seed": int(seed),
        "reward": float(total_reward),
        "length": int(step_count),
        "success": bool(any_success),
        "terminal_success": bool(terminal_success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "env_id": str(configured_env_id(env)),
        "terminal_collision": _safe_bool(terminal_info, "collision", False),
        "terminal_plane_violation": _safe_bool(terminal_info, "plane_violation", False),
        "terminal_position_error": _safe_float(terminal_info, "position_error"),
        "terminal_orientation_alignment": _safe_float(terminal_info, "orientation_alignment"),
        "terminal_glass_fits_window": _safe_bool(terminal_info, "glass_fits_window", False),
    }
    video_path = recorder.end_episode_recording(episode_info)
    episode_info["video_path"] = video_path
    return episode_info


def configured_env_id(env) -> str:
    spec = getattr(env, "spec", None)
    return getattr(spec, "id", "unknown")


def _default_output_dir(exp_dir: Optional[Path], model_path: Path) -> Path:
    if exp_dir is not None:
        return exp_dir / "evaluate" / model_path.stem
    return PROJECT_ROOT / "outputs" / "evaluate" / model_path.stem


def _write_summary(
    output_dir: Path,
    model_path: Path,
    vecnormalize_path: Optional[Path],
    config: Dict[str, Any],
    episodes: List[Dict[str, Any]],
) -> Path:
    rewards = np.asarray([episode["reward"] for episode in episodes], dtype=np.float64)
    successes = np.asarray([episode["success"] for episode in episodes], dtype=np.float64)
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.float64)

    summary = {
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path is not None else None,
        "env_name": str(config.get("env_name", "unknown")),
        "total_episodes": int(len(episodes)),
        "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        "std_reward": float(np.std(rewards)) if rewards.size else 0.0,
        "success_rate": float(np.mean(successes)) if successes.size else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths.size else 0.0,
        "episodes": episodes,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_safe(summary), f, indent=2, ensure_ascii=False)
    return summary_path


def record_policy_videos(args: argparse.Namespace) -> Path:
    exp_dir = Path(args.exp_dir).expanduser().resolve() if args.exp_dir is not None else None
    model_path = _resolve_model_path(args, exp_dir)
    if exp_dir is None:
        inferred_exp_dir = _infer_exp_dir_from_model_path(model_path)
        if inferred_exp_dir is not None:
            exp_dir = inferred_exp_dir

    summary = _load_training_summary(exp_dir)
    training_config = summary.get("config", {}) if isinstance(summary.get("config", {}), dict) else {}
    if "env_name" in summary and "env_name" not in training_config:
        training_config["env_name"] = summary["env_name"]

    config = _apply_cli_overrides(training_config, args)
    config = _auto_match_checkpoint_config(model_path, config, args)
    vecnormalize_path = _resolve_vecnormalize_path(args, exp_dir, model_path, config)
    output_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_output_dir(exp_dir, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Env: {config['env_name']}")
    print(f"Videos: {output_dir}")
    if vecnormalize_path is not None:
        print(f"VecNormalize: {vecnormalize_path}")

    normalizer = _load_vec_normalizer(
        vecnormalize_path=vecnormalize_path,
        config=config,
        eval_dense_reward=bool(args.eval_dense_reward),
    )
    load_env = _make_policy_env(
        config=config,
        render_mode=None,
        seed=args.seed_start,
        eval_dense_reward=bool(args.eval_dense_reward),
    )
    try:
        model = SAC.load(str(model_path), env=load_env, device=args.device)
    except ValueError as exc:
        load_env.close()
        raise ValueError(
            "Failed to load the checkpoint with the environment created from the current code.\n"
            "This usually means the checkpoint was trained with a different observation/action wrapper "
            "or an older version of the environment code.\n"
            f"Checkpoint: {model_path}\n"
            f"Env: {config['env_name']}\n"
            f"Current observation_space: {load_env.observation_space}\n"
            f"Current action_space: {load_env.action_space}\n"
            "Use the same code/config that produced the checkpoint, or override the wrapper flags "
            "with --task-geometry-features/--task-stage-features/--task-progress-features/"
            "--residual-guidance if the checkpoint was trained with them."
        ) from exc
    env = _make_policy_env(
        config=config,
        render_mode="rgb_array",
        seed=args.seed_start,
        eval_dense_reward=bool(args.eval_dense_reward),
    )
    max_steps = _episode_max_steps(env, args.max_steps)
    recorder = EpisodeVideoRecorder(str(output_dir), fps=max(int(args.fps), 1))
    model_label = args.model_label or model_path.stem

    episodes: List[Dict[str, Any]] = []
    try:
        for episode_idx in range(max(int(args.episodes), 1)):
            seed = int(args.seed_start) + episode_idx
            result = record_policy_episode(
                env=env,
                model=model,
                normalizer=normalizer,
                recorder=recorder,
                episode_id=episode_idx,
                seed=seed,
                model_label=model_label,
                max_steps=max_steps,
                deterministic=bool(args.deterministic),
                stop_on_success=bool(args.stop_on_success),
                initial_frames=int(args.initial_frames),
                terminal_hold_frames=int(args.terminal_hold_frames),
                success_hold_frames=int(args.success_hold_frames),
                frame_width=args.width,
                frame_height=args.height,
            )
            episodes.append(result)
            status = "SUCCESS" if result["success"] else "FAIL"
            print(
                f"Episode {episode_idx:03d}: {status}, "
                f"reward={result['reward']:.3f}, length={result['length']}, "
                f"video={result.get('video_path')}"
            )
    finally:
        env.close()
        model_env = model.get_env()
        if model_env is not None:
            model_env.close()
        if normalizer is not None:
            normalizer.close()

    recorder.save_metadata()
    summary_path = _write_summary(
        output_dir=output_dir,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        config=config,
        episodes=episodes,
    )
    print(f"Summary: {summary_path}")
    return summary_path


def evaluate_experiment(
    exp_dir: str,
    model_name: str = "best_model",
    episodes: int = 3,
    seed_start: int = 0,
    out_dir: Optional[str] = None,
    **kwargs,
) -> Path:
    """Compatibility wrapper for older imports of evaluate.evaluate_experiment."""

    parser = build_parser()
    args = parser.parse_args([])
    args.exp_dir = exp_dir
    args.model_name = model_name
    args.episodes = episodes
    args.seed_start = seed_start
    args.out_dir = out_dir
    for key, value in kwargs.items():
        if not hasattr(args, key):
            raise TypeError(f"Unknown evaluate_experiment option: {key}")
        setattr(args, key, value)
    return record_policy_videos(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record policy rollout videos from a trained SAC checkpoint.")
    parser.add_argument("--exp-dir", type=str, default=None, help="Training output directory.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to a SAC .zip checkpoint.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="best_model",
        help="Model name under <exp-dir>/models, e.g. best_model, final_model, stage_4_80percent.",
    )
    parser.add_argument("--model-label", type=str, default=None, help="Label written into episode metadata.")
    parser.add_argument("--vecnormalize-path", type=str, default=None, help="Optional VecNormalize .pkl path.")
    parser.add_argument("--no-vecnormalize", action="store_true", help="Do not load VecNormalize stats.")
    parser.add_argument("--env", type=str, default=None, help="Override env id from training_summary.json.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for mp4 files and summaries.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--seed-start", type=int, default=0, help="First episode seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Episode step limit; defaults to env spec.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--width", type=int, default=None, help="Resize recorded frames to this width.")
    parser.add_argument("--height", type=int, default=None, help="Resize recorded frames to this height.")
    parser.add_argument("--device", type=str, default="auto", help="SB3 load device: auto/cpu/cuda/cuda:0.")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Sample stochastic actions.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
    parser.set_defaults(stop_on_success=True)
    parser.add_argument("--initial-frames", type=int, default=3, help="Frames to record before the first action.")
    parser.add_argument("--terminal-hold-frames", type=int, default=5, help="Extra frames after termination.")
    parser.add_argument("--success-hold-frames", type=int, default=15, help="Extra frames after first success.")
    parser.add_argument("--reward-scale", type=float, default=None, help="Override reward scaling wrapper value.")
    parser.add_argument(
        "--eval-dense-reward",
        action="store_true",
        help="Use the dense reward wrapper during video evaluation.",
    )
    parser.add_argument("--expert-hint-style", type=str, default=None, choices=["staged", "direct", "insert"])
    parser.add_argument("--residual-action-scale", type=float, default=None)

    parser.set_defaults(task_geometry_features=None, task_stage_features=None, task_progress_features=None)
    parser.add_argument("--task-geometry-features", dest="task_geometry_features", action="store_true")
    parser.add_argument("--no-task-geometry-features", dest="task_geometry_features", action="store_false")
    parser.add_argument("--task-stage-features", dest="task_stage_features", action="store_true")
    parser.add_argument("--no-task-stage-features", dest="task_stage_features", action="store_false")
    parser.add_argument("--task-progress-features", dest="task_progress_features", action="store_true")
    parser.add_argument("--no-task-progress-features", dest="task_progress_features", action="store_false")
    parser.set_defaults(residual_guidance=None)
    parser.add_argument("--residual-guidance", dest="residual_guidance", action="store_true")
    parser.add_argument("--no-residual-guidance", dest="residual_guidance", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    record_policy_videos(args)


if __name__ == "__main__":
    main()
