#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture:
    00_start.png
    01_approach_end.png
    02_descend_end.png
    03_suction_grasp_end.png
    04_lift_end.png
    05_reorient_end.png
    06_coarse_align_end.png
    07_fine_align_end.png
    08_insert_end.png
    09_hold_end.png

The script uses a fixed 3:2 camera view and captures images directly
with mujoco.Renderer. It does not register env.render() as the reset-time
pipeline callback, avoiding interference with reset/goal generation.

Example:
    MUJOCO_GL=egl python capture_stage_snapshots.py \
        --model outputs/window_fine_align/sac-20260721-205122-seed0/models/checkpoints/window_fine_align_sac_150000_steps.zip \
        --seed 0 \
        --output-dir outputs/stage_snapshots_seed0
"""

from __future__ import annotations

import os

# Select headless EGL automatically when no desktop DISPLAY is available.
# This must be set before importing mujoco.
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import SAC


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import panda_mujoco_gym  # noqa: E402,F401


ENV_ID = "FrankaWindowFineAlignDense-v0"

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.880000,
    "azimuth": -263.500000,
    "elevation": -18.000000,
    "lookat": np.array(
        [0.000000, -0.130000, 0.230000],
        dtype=np.float64,
    ),
}

RESET_STAGE_FILES = {
    "approach": "01_approach_end.png",
    "descend": "02_descend_end.png",
    "suction_grasp": "03_suction_grasp_end.png",
    "lift": "04_lift_end.png",
    "reorient": "05_reorient_end.png",
    "coarse_align": "06_coarse_align_end.png",
}

EXPECTED_IMAGE_FILES = (
    "00_start.png",
    "01_approach_end.png",
    "02_descend_end.png",
    "03_suction_grasp_end.png",
    "04_lift_end.png",
    "05_reorient_end.png",
    "06_coarse_align_end.png",
    "07_fine_align_end.png",
    "08_insert_end.png",
    "09_hold_end.png",
)

# Current evaluation defaults. Values loaded from training_config.json
# take priority when present, except for the explicit overrides below.
DEFAULT_ENV_KWARGS: dict[str, Any] = {
    "measurement_noise_translation_std": 0.0,
    "measurement_noise_angle_std_deg": 0.0,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the start state and nine stage-end images."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to an SAC .zip model or checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Episode seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage_snapshots",
        help="Directory used to save PNG images and summary.json.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Output image width. Default: 1200.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Output image height. Default: 800.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Stable-Baselines3 model device.",
    )

    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use deterministic SAC actions.",
    )
    policy_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample stochastic SAC actions.",
    )
    parser.set_defaults(deterministic=True)

    parser.add_argument(
        "--preinsert-normal-offset",
        type=float,
        default=0.08,
        help="Force the pre-insertion normal offset. Default: 0.08 m.",
    )
    parser.add_argument(
        "--measurement-noise-translation-std",
        type=float,
        default=0.0,
        help="Evaluation translation measurement noise.",
    )
    parser.add_argument(
        "--measurement-noise-angle-std-deg",
        type=float,
        default=0.0,
        help="Evaluation angular measurement noise in degrees.",
    )

    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Cannot JSON-encode value of type {type(value).__name__}"
    )


def find_training_config(model_path: str | Path) -> Path | None:
    model = Path(model_path).expanduser().resolve()

    start_dir = model if model.is_dir() else model.parent

    for directory in (start_dir, *start_dir.parents):
        candidate = directory / "training_config.json"

        if candidate.is_file():
            return candidate

    return None


def load_environment_kwargs(
    model_path: str | Path,
    *,
    preinsert_normal_offset: float,
    measurement_noise_translation_std: float,
    measurement_noise_angle_std_deg: float,
) -> tuple[dict[str, Any], Path | None]:
    """
    Load environment parameters from the model run when available.

    Priority:
        1. Script defaults
        2. training_config.json
        3. Explicit snapshot-script overrides
    """

    env_kwargs = dict(DEFAULT_ENV_KWARGS)
    config_path = find_training_config(model_path)

    if config_path is not None:
        config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        configured_kwargs = config.get(
            "evaluation_environment_kwargs"
        )

        if not isinstance(configured_kwargs, dict):
            configured_kwargs = config.get(
                "train_environment_kwargs",
                {},
            )

        if isinstance(configured_kwargs, dict):
            env_kwargs.update(configured_kwargs)

    # Explicit overrides for this visualization script.
    env_kwargs["preinsert_normal_offset"] = float(
        preinsert_normal_offset
    )
    env_kwargs["measurement_noise_translation_std"] = float(
        measurement_noise_translation_std
    )
    env_kwargs["measurement_noise_angle_std_deg"] = float(
        measurement_noise_angle_std_deg
    )

    return env_kwargs, config_path


def validate_image_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {width}x{height}"
        )

    # width : height = 3 : 2
    if width * 2 != height * 3:
        raise ValueError(
            "Image dimensions must have an exact 3:2 ratio, "
            f"got {width}x{height}"
        )


def build_free_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    camera.distance = float(
        DEFAULT_CAMERA_CONFIG["distance"]
    )
    camera.azimuth = float(
        DEFAULT_CAMERA_CONFIG["azimuth"]
    )
    camera.elevation = float(
        DEFAULT_CAMERA_CONFIG["elevation"]
    )
    camera.lookat[:] = np.asarray(
        DEFAULT_CAMERA_CONFIG["lookat"],
        dtype=np.float64,
    )

    return camera


def make_renderer(
    base_env: Any,
    *,
    width: int,
    height: int,
) -> tuple[mujoco.Renderer, mujoco.MjvCamera]:
    """
    Increase the model offscreen framebuffer before Renderer creation.
    """

    validate_image_size(width, height)

    model = base_env.model

    old_width = int(model.vis.global_.offwidth)
    old_height = int(model.vis.global_.offheight)

    model.vis.global_.offwidth = max(
        old_width,
        int(width),
    )
    model.vis.global_.offheight = max(
        old_height,
        int(height),
    )

    print(
        "[capture_stage_snapshots] offscreen framebuffer: "
        f"{old_width}x{old_height} -> "
        f"{int(model.vis.global_.offwidth)}x"
        f"{int(model.vis.global_.offheight)}"
    )

    renderer = mujoco.Renderer(
        model,
        height=int(height),
        width=int(width),
    )

    camera = build_free_camera()

    return renderer, camera


def close_renderer_safely(
    renderer: mujoco.Renderer | None,
) -> None:
    """
    Some older MuJoCo Renderer versions do not expose close().
    """

    if renderer is None:
        return

    close_method = getattr(renderer, "close", None)

    if callable(close_method):
        close_method()

    del renderer
    gc.collect()


def render_rgb(
    base_env: Any,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
) -> np.ndarray:
    # Ensure all derived body and geometry transforms are up to date.
    mujoco.mj_forward(base_env.model, base_env.data)

    renderer.update_scene(
        base_env.data,
        camera=camera,
    )

    frame = np.asarray(
        renderer.render(),
        dtype=np.uint8,
    )

    expected_shape = (
        int(renderer.height),
        int(renderer.width),
        3,
    )

    # Older versions may not expose width/height as public attributes.
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(
            f"Unexpected rendered frame shape: {frame.shape}"
        )

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    return frame.copy()


def save_frame(
    *,
    base_env: Any,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = render_rgb(
        base_env,
        renderer,
        camera,
    )

    imageio.imwrite(
        output_path,
        frame,
    )

    print(
        f"[capture_stage_snapshots] saved: {output_path}"
    )


def stage_to_key(stage: Any) -> str:
    value = getattr(stage, "value", None)

    if isinstance(value, str):
        return value.strip().lower()

    name = getattr(stage, "name", None)

    if isinstance(name, str):
        return name.strip().lower()

    return str(stage).strip().lower()


def extract_error_summary(base_env: Any) -> dict[str, float]:
    errors = base_env.get_fine_alignment_errors()

    return {
        "error_u": float(errors["error_u"]),
        "error_v": float(errors["error_v"]),
        "tilt_error_t1": float(
            errors["tilt_error_t1"]
        ),
        "tilt_error_t2": float(
            errors["tilt_error_t2"]
        ),
        "yaw_error": float(errors["yaw_error"]),
        "current_normal_gap": float(
            errors["current_normal_gap"]
        ),
        "normal_gap_error": float(
            errors["normal_gap_error"]
        ),
    }


def main() -> None:
    args = parse_args()

    validate_image_size(
        args.width,
        args.height,
    )

    model_path = Path(args.model).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env_kwargs, training_config_path = (
        load_environment_kwargs(
            model_path,
            preinsert_normal_offset=(
                args.preinsert_normal_offset
            ),
            measurement_noise_translation_std=(
                args.measurement_noise_translation_std
            ),
            measurement_noise_angle_std_deg=(
                args.measurement_noise_angle_std_deg
            ),
        )
    )

    print(
        "[capture_stage_snapshots] MUJOCO_GL =",
        os.environ.get("MUJOCO_GL", "<default>"),
    )
    print(
        "[capture_stage_snapshots] model =",
        model_path,
    )
    print(
        "[capture_stage_snapshots] seed =",
        args.seed,
    )
    print(
        "[capture_stage_snapshots] training config =",
        training_config_path or "<not found>",
    )
    print(
        "[capture_stage_snapshots] preinsert_normal_offset =",
        env_kwargs.get("preinsert_normal_offset"),
    )

    env = gym.make(
        ENV_ID,
        render_mode=None,
        **env_kwargs,
    )

    base = env.unwrapped

    renderer: mujoco.Renderer | None = None
    camera: mujoco.MjvCamera | None = None

    saved_files: dict[str, str] = {}
    stage_end_errors: dict[str, dict[str, float]] = {}

    fine_info: dict[str, Any] = {}
    fine_steps = 0
    fine_success = False
    insert_success = False
    hold_success = False
    assembly_verified = False

    insert_diagnostics: dict[str, Any] = {}
    assembly_diagnostics: dict[str, Any] = {}

    failure_reason = ""
    failure_stage = ""

    def save_named_image(filename: str) -> None:
        if filename in saved_files:
            return

        if renderer is None or camera is None:
            raise RuntimeError(
                "Renderer has not been initialized"
            )

        path = output_dir / filename

        save_frame(
            base_env=base,
            renderer=renderer,
            camera=camera,
            output_path=path,
        )

        saved_files[filename] = str(path)

    try:
        renderer, camera = make_renderer(
            base,
            width=args.width,
            height=args.height,
        )

        model = SAC.load(
            str(model_path),
            device=args.device,
        )

        # -----------------------------------------------------
        # Capture reset-time stages.
        #
        # Do not assign env.render as pipeline_frame_callback
        # before reset. Captures are made directly with the
        # standalone Renderer, avoiding reset/render callbacks.
        # -----------------------------------------------------

        original_run_scripted_stage = (
            base._run_scripted_stage
        )

        def wrapped_run_scripted_stage(
            stage: Any,
            operation: Any,
        ) -> Any:
            stage_key = stage_to_key(stage)

            # Start state: window and object have been sampled,
            # but APPROACH has not yet executed.
            if (
                stage_key == "approach"
                and "00_start.png" not in saved_files
            ):
                save_named_image("00_start.png")

            result = original_run_scripted_stage(
                stage,
                operation,
            )

            filename = RESET_STAGE_FILES.get(
                stage_key
            )

            if filename is not None:
                save_named_image(filename)

                try:
                    stage_end_errors[stage_key] = (
                        extract_error_summary(base)
                    )
                except Exception:
                    # Early stages do not necessarily have
                    # meaningful final fine-alignment diagnostics.
                    pass

            return result

        base._run_scripted_stage = (
            wrapped_run_scripted_stage
        )

        try:
            obs, reset_info = env.reset(
                seed=args.seed
            )
        finally:
            # Restore the environment method immediately after
            # reset so the temporary capture hook does not affect
            # later pipeline logic.
            base._run_scripted_stage = (
                original_run_scripted_stage
            )

        missing_reset_images = [
            filename
            for filename in EXPECTED_IMAGE_FILES[:7]
            if filename not in saved_files
        ]

        if missing_reset_images:
            raise RuntimeError(
                "Failed to capture one or more reset-stage "
                f"images: {missing_reset_images}"
            )

        # -----------------------------------------------------
        # Stage 7: fine alignment
        # -----------------------------------------------------

        terminated = False
        truncated = False
        fine_info = dict(reset_info)

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=args.deterministic,
            )

            (
                obs,
                _reward,
                terminated,
                truncated,
                fine_info,
            ) = env.step(action)

            fine_steps += 1

        save_named_image("07_fine_align_end.png")

        stage_end_errors["fine_align"] = (
            extract_error_summary(base)
        )

        fine_success = bool(
            fine_info.get(
                "fine_align_success",
                False,
            )
        )

        if not fine_success:
            failure_stage = "fine_align"
            failure_reason = str(
                fine_info.get(
                    "failure_reason",
                    "fine_align_failed",
                )
            )

        # -----------------------------------------------------
        # Stage 8: scripted insert
        # -----------------------------------------------------

        if fine_success:
            insert_success = bool(
                base.run_scripted_insert()
            )

            insert_diagnostics = dict(
                base.get_last_insert_diagnostics()
            )

            save_named_image("08_insert_end.png")

            stage_end_errors["insert"] = (
                extract_error_summary(base)
            )

            if not insert_success:
                failure_stage = "insert"
                failure_reason = str(
                    insert_diagnostics.get(
                        "failure_reason",
                        "insert_failed",
                    )
                )

        # -----------------------------------------------------
        # Stage 9: scripted hold
        # -----------------------------------------------------

        if insert_success:
            hold_success = bool(
                base.run_scripted_hold()
            )

            insert_diagnostics = dict(
                base.get_last_insert_diagnostics()
            )

            save_named_image("09_hold_end.png")

            stage_end_errors["hold"] = (
                extract_error_summary(base)
            )

            if not hold_success:
                failure_stage = "hold"
                failure_reason = str(
                    insert_diagnostics.get(
                        "failure_reason",
                        "hold_failed",
                    )
                )

        # -----------------------------------------------------
        # Final strict assembly verification
        # -----------------------------------------------------

        if hold_success:
            (
                assembly_verified,
                assembly_diagnostics,
            ) = base.verify_final_assembly()

            assembly_verified = bool(
                assembly_verified
            )
            assembly_diagnostics = dict(
                assembly_diagnostics
            )

            if not assembly_verified:
                failure_stage = (
                    "final_assembly_verification"
                )
                failure_reason = str(
                    assembly_diagnostics.get(
                        "failure_reason",
                        "final_assembly_verification_failed",
                    )
                )

        full_pipeline_success = bool(
            fine_success
            and insert_success
            and hold_success
            and assembly_verified
        )

        summary = {
            "model": str(model_path),
            "seed": int(args.seed),
            "deterministic": bool(
                args.deterministic
            ),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "aspect_ratio": "3:2",
            "camera_config": {
                "distance": float(
                    DEFAULT_CAMERA_CONFIG["distance"]
                ),
                "azimuth": float(
                    DEFAULT_CAMERA_CONFIG["azimuth"]
                ),
                "elevation": float(
                    DEFAULT_CAMERA_CONFIG["elevation"]
                ),
                "lookat": np.asarray(
                    DEFAULT_CAMERA_CONFIG["lookat"]
                ).tolist(),
            },
            "mujoco_gl": os.environ.get(
                "MUJOCO_GL",
                "",
            ),
            "training_config_path": (
                str(training_config_path)
                if training_config_path is not None
                else ""
            ),
            "environment_kwargs": env_kwargs,
            "saved_files": saved_files,
            "stage_end_errors": stage_end_errors,
            "fine_align_steps": int(fine_steps),
            "fine_align_success": bool(
                fine_success
            ),
            "insert_success": bool(
                insert_success
            ),
            "hold_success": bool(
                hold_success
            ),
            "assembly_verified": bool(
                assembly_verified
            ),
            "full_pipeline_success": bool(
                full_pipeline_success
            ),
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "fine_info": fine_info,
            "insert_diagnostics": (
                insert_diagnostics
            ),
            "assembly_diagnostics": (
                assembly_diagnostics
            ),
        }

        summary_path = (
            output_dir / "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print("Stage snapshot result")
        print("=" * 72)
        print(
            f"fine={fine_success} "
            f"insert={insert_success} "
            f"hold={hold_success} "
            f"assembly_verified={assembly_verified} "
            f"full={full_pipeline_success}"
        )
        print(
            f"fine_steps={fine_steps}"
        )

        if failure_reason:
            print(
                f"failure_stage={failure_stage}"
            )
            print(
                f"failure_reason={failure_reason}"
            )
        else:
            print("failure=none")

        print()
        print("Saved images:")

        for filename in EXPECTED_IMAGE_FILES:
            path = saved_files.get(filename)

            if path is None:
                print(
                    f"  [missing] {filename}"
                )
            else:
                print(
                    f"  [saved]   {filename}"
                )

        print()
        print(
            f"Summary: {summary_path}"
        )
        print("=" * 72)

        # Do not silently create fake images for stages that did
        # not actually run.
        if not full_pipeline_success:
            missing = [
                filename
                for filename in EXPECTED_IMAGE_FILES
                if filename not in saved_files
            ]

            raise RuntimeError(
                "The selected episode did not complete the "
                "strict nine-stage pipeline. "
                f"failure_stage={failure_stage or 'unknown'}, "
                f"failure_reason={failure_reason or 'unknown'}, "
                f"missing_images={missing}. "
                f"See {summary_path}"
            )

    finally:
        close_renderer_safely(renderer)
        env.close()


if __name__ == "__main__":
    main()