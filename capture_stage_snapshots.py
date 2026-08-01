#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture:
    00_start.png
    01_approach_end.png
    02_descend_end.png
    03_grasp_end.png
    04_lift_end.png
    05_reorient_end.png
    06_coarse_align_end.png
    07_fine_align_end.png
    08_insert_end.png
    09_hold_end.png

Additionally save the robot 13D state for the 9 stage-end images:
    approach
    descend
    grasp
    lift
    reorient
    coarse_align
    fine_align
    insert
    hold

13D state definition:
    [q1, q2, q3, q4, q5, q6, q7,
     ee_x, ee_y, ee_z,
     ee_roll, ee_pitch, ee_yaw]

Example:
    MUJOCO_GL=egl python capture_stage_snapshots.py \
        --model outputs/window_fine_align/sac-20260721-205122-seed0/models/checkpoints/window_fine_align_sac_150000_steps.zip \
        --seed 0 \
        --output-dir outputs/stage_snapshots_seed0
"""

from __future__ import annotations

import os

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
from utils.fine_align_config import (  # noqa: E402
    add_final_environment_arguments,
    final_environment_kwargs,
    validate_final_environment_kwargs,
)


ENV_ID = "FrankaWindowFineAlignDense-v0"

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.76,
    "azimuth": -1.0,
    "elevation": -25.5,
    "lookat": np.array([0.18, -0.04, 0.185], dtype=np.float64),
}

RESET_STAGE_FILES = {
    "approach": "01_approach_end.png",
    "descend": "02_descend_end.png",
    "grasp": "03_grasp_end.png",
    "lift": "04_lift_end.png",
    "reorient": "05_reorient_end.png",
    "coarse_align": "06_coarse_align_end.png",
}

EXPECTED_IMAGE_FILES = (
    "00_start.png",
    "01_approach_end.png",
    "02_descend_end.png",
    "03_grasp_end.png",
    "04_lift_end.png",
    "05_reorient_end.png",
    "06_coarse_align_end.png",
    "07_fine_align_end.png",
    "08_insert_end.png",
    "09_hold_end.png",
)

STAGE_ORDER_9 = (
    "approach",
    "descend",
    "grasp",
    "lift",
    "reorient",
    "coarse_align",
    "fine_align",
    "insert",
    "hold",
)

STAGE_TO_IMAGE_9 = {
    "approach": "01_approach_end.png",
    "descend": "02_descend_end.png",
    "grasp": "03_grasp_end.png",
    "lift": "04_lift_end.png",
    "reorient": "05_reorient_end.png",
    "coarse_align": "06_coarse_align_end.png",
    "fine_align": "07_fine_align_end.png",
    "insert": "08_insert_end.png",
    "hold": "09_hold_end.png",
}

STATE_KEYS_13D = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_roll",
    "ee_pitch",
    "ee_yaw",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture stage snapshots and export 13D robot states."
    )

    parser.add_argument(
        "--model",
        default=None,
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
        help="Directory to save PNG images and JSON files.",
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
        "--camera-distance",
        type=float,
        default=float(DEFAULT_CAMERA_CONFIG["distance"]),
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=float(DEFAULT_CAMERA_CONFIG["azimuth"]),
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=float(DEFAULT_CAMERA_CONFIG["elevation"]),
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=np.asarray(DEFAULT_CAMERA_CONFIG["lookat"]).tolist(),
    )
    parser.add_argument(
        "--tune-camera",
        action="store_true",
        help="Open the interactive camera tuner instead of capturing.",
    )
    add_final_environment_arguments(parser)

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


def validate_image_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {width}x{height}"
        )

    if width * 2 != height * 3:
        raise ValueError(
            "Image dimensions must have an exact 3:2 ratio, "
            f"got {width}x{height}"
        )


def build_free_camera(camera_config: dict[str, Any]) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(camera_config["distance"])
    camera.azimuth = float(camera_config["azimuth"])
    camera.elevation = float(camera_config["elevation"])
    camera.lookat[:] = np.asarray(
        camera_config["lookat"],
        dtype=np.float64,
    )
    return camera


def make_renderer(
    base_env: Any,
    *,
    width: int,
    height: int,
    camera_config: dict[str, Any],
) -> tuple[mujoco.Renderer, mujoco.MjvCamera]:
    validate_image_size(width, height)

    model = base_env.model

    old_width = int(model.vis.global_.offwidth)
    old_height = int(model.vis.global_.offheight)

    model.vis.global_.offwidth = max(old_width, int(width))
    model.vis.global_.offheight = max(old_height, int(height))

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

    camera = build_free_camera(camera_config)
    return renderer, camera


def close_renderer_safely(
    renderer: mujoco.Renderer | None,
) -> None:
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
    mujoco.mj_forward(base_env.model, base_env.data)

    renderer.update_scene(
        base_env.data,
        camera=camera,
    )

    frame = np.asarray(
        renderer.render(),
        dtype=np.uint8,
    )

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

    imageio.imwrite(output_path, frame)

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
        "tilt_error_t1": float(errors["tilt_error_t1"]),
        "tilt_error_t2": float(errors["tilt_error_t2"]),
        "yaw_error": float(errors["yaw_error"]),
        "current_normal_gap": float(errors["current_normal_gap"]),
        "normal_gap_error": float(errors["normal_gap_error"]),
    }


def rotation_matrix_to_rpy_xyz(rotation: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to roll, pitch, yaw (radians).
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-8

    if not singular:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)


def find_arm_joint_names(base_env: Any) -> list[str]:
    """
    Resolve the 7 Franka arm joint names.
    """
    candidate_attrs = [
        "arm_joint_names",
        "panda_joint_names",
        "robot_joint_names",
        "joint_names",
    ]

    for attr_name in candidate_attrs:
        names = getattr(base_env, attr_name, None)
        if isinstance(names, (list, tuple)) and len(names) >= 7:
            return [str(name) for name in list(names)[:7]]

    model = base_env.model
    names: list[str] = []

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if joint_name is None:
            continue

        if joint_name.startswith("panda_joint"):
            names.append(joint_name)

    if len(names) >= 7:
        def _joint_index(name: str) -> int:
            digits = "".join(ch for ch in name if ch.isdigit())
            return int(digits) if digits else 999

        names = sorted(names, key=_joint_index)
        return names[:7]

    raise RuntimeError(
        "Could not resolve the 7 Franka joint names."
    )


def extract_robot_state_13d(base_env: Any) -> dict[str, Any]:
    """
    Extract the 13D Franka robot state.

    State definition:
        [
            q1, q2, q3, q4, q5, q6, q7,
            ee_x, ee_y, ee_z,
            ee_roll, ee_pitch, ee_yaw,
        ]

    Units:
        joint positions: rad
        end-effector position: m
        end-effector RPY: rad
    """

    model = base_env.model
    data = base_env.data

    # Update all derived kinematic quantities before reading them.
    mujoco.mj_forward(model, data)

    # The current environment already resolves the seven Franka
    # arm joint names during initialization.
    joint_names = list(
        getattr(base_env, "arm_joint_names", [])
    )

    if len(joint_names) != 7:
        # Retain the fallback for model variants.
        joint_names = find_arm_joint_names(base_env)

    if len(joint_names) != 7:
        raise RuntimeError(
            "Expected exactly 7 Franka arm joints, "
            f"but resolved {len(joint_names)}: {joint_names}"
        )

    joint_positions: list[float] = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"Joint does not exist in MuJoCo model: "
                f"{joint_name}"
            )

        qpos_address = int(
            model.jnt_qposadr[joint_id]
        )

        joint_positions.append(
            float(data.qpos[qpos_address])
        )

    # Use the environment's official end-effector accessors.
    # In the current model these access ee_center_site.
    ee_position = np.asarray(
        base_env.get_ee_position(),
        dtype=np.float64,
    ).reshape(3).copy()

    ee_rotation = np.asarray(
        base_env.get_ee_rotation_matrix(),
        dtype=np.float64,
    ).reshape(3, 3).copy()

    ee_rpy = rotation_matrix_to_rpy_xyz(
        ee_rotation
    )

    state = np.concatenate(
        [
            np.asarray(
                joint_positions,
                dtype=np.float64,
            ),
            ee_position,
            ee_rpy,
        ],
        axis=0,
    )

    if state.shape != (13,):
        raise RuntimeError(
            "Robot state dimension mismatch: "
            f"expected (13,), got {state.shape}"
        )

    if not np.all(np.isfinite(state)):
        raise RuntimeError(
            f"Robot 13D state contains NaN or Inf: {state}"
        )

    named_state = {
        key: float(value)
        for key, value in zip(
            STATE_KEYS_13D,
            state,
        )
    }

    # Keep the actual site id as auxiliary metadata.
    ee_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ee_center_site",
    )

    return {
        "state": [
            float(value)
            for value in state
        ],
        "named_state": named_state,
        "joint_names": joint_names,
        "joint_position_unit": "rad",
        "ee_position_unit": "m",
        "ee_orientation_type": "roll_pitch_yaw_xyz",
        "ee_orientation_unit": "rad",
        "ee_reference_site": "ee_center_site",
        "ee_site_id": int(ee_site_id),
    }




def main() -> None:
    args = parse_args()

    if args.tune_camera:
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("--tune-camera requires a desktop DISPLAY")
        from evaluate.tune_camera_view import InteractiveCameraTuner

        tuner = InteractiveCameraTuner(env_id=ENV_ID, seed=args.seed)
        try:
            tuner.run()
        finally:
            tuner.close()
        return

    if args.model is None:
        raise ValueError("--model is required unless --tune-camera is used")

    validate_image_size(
        args.width,
        args.height,
    )

    model_path = Path(args.model).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env_kwargs = final_environment_kwargs(args)
    validate_final_environment_kwargs(env_kwargs)
    camera_config = {
        "distance": float(args.camera_distance),
        "azimuth": float(args.camera_azimuth),
        "elevation": float(args.camera_elevation),
        "lookat": np.asarray(args.camera_lookat, dtype=np.float64),
    }

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

    # New: 9 stage robot states
    stage_robot_states_13d: dict[str, dict[str, Any]] = {}

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

    def save_stage_robot_state(
        stage_name: str,
    ) -> None:
        if stage_name not in STAGE_ORDER_9:
            return

        state_info = extract_robot_state_13d(
            base
        )

        stage_robot_states_13d[stage_name] = {
            "stage": stage_name,
            "image_file": STAGE_TO_IMAGE_9[
                stage_name
            ],
            "state_dim": 13,
            "state_definition": (
                "7 joint positions + "
                "end-effector xyz + "
                "end-effector roll/pitch/yaw"
            ),
            "state_keys": list(
                STATE_KEYS_13D
            ),
            "state": state_info["state"],
            "named_state": (
                state_info["named_state"]
            ),
            "joint_names": (
                state_info["joint_names"]
            ),
            "joint_position_unit": (
                state_info["joint_position_unit"]
            ),
            "ee_position_unit": (
                state_info["ee_position_unit"]
            ),
            "ee_orientation_type": (
                state_info[
                    "ee_orientation_type"
                ]
            ),
            "ee_orientation_unit": (
                state_info[
                    "ee_orientation_unit"
                ]
            ),
            "ee_reference_site": (
                state_info[
                    "ee_reference_site"
                ]
            ),
            "ee_site_id": (
                state_info["ee_site_id"]
            ),
            "simulation_time": float(
                base.data.time
            ),
        }

        print(
            "[capture_stage_snapshots] "
            f"captured 13D state: {stage_name}"
        )

    try:
        renderer, camera = make_renderer(
            base,
            width=args.width,
            height=args.height,
            camera_config=camera_config,
        )

        model = SAC.load(
            str(model_path),
            device=args.device,
        )

        controller = base.script_controller
        original_run_stage = controller.run_stage

        def wrapped_run_stage(
            stage: Any,
            operation: Any,
        ) -> Any:
            stage_key = stage_to_key(stage)

            if (
                stage_key == "approach"
                and "00_start.png" not in saved_files
            ):
                save_named_image("00_start.png")

            result = original_run_stage(
                stage,
                operation,
            )

            filename = RESET_STAGE_FILES.get(stage_key)

            if filename is not None:
                save_named_image(filename)

                try:
                    stage_end_errors[stage_key] = (
                        extract_error_summary(base)
                    )
                except Exception:
                    pass

                # New: save 13D state for reset-time stage end
                save_stage_robot_state(stage_key)

            return result

        controller.run_stage = wrapped_run_stage

        try:
            obs, reset_info = env.reset(seed=args.seed)
        finally:
            controller.run_stage = original_run_stage

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

        # Stage 7: fine alignment
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
        stage_end_errors["fine_align"] = extract_error_summary(base)
        save_stage_robot_state("fine_align")

        fine_success = bool(
            fine_info.get("fine_align_success", False)
        )

        if not fine_success:
            failure_stage = "fine_align"
            failure_reason = str(
                fine_info.get(
                    "failure_reason",
                    "fine_align_failed",
                )
            )

        # Stage 8: scripted insert
        if fine_success:
            insert_success = bool(
                controller.run_insert()
            )

            insert_diagnostics = dict(
                controller.get_last_insert_diagnostics()
            )

            save_named_image("08_insert_end.png")
            stage_end_errors["insert"] = extract_error_summary(base)
            save_stage_robot_state("insert")

            if not insert_success:
                failure_stage = "insert"
                failure_reason = str(
                    insert_diagnostics.get(
                        "failure_reason",
                        "insert_failed",
                    )
                )

        # Stage 9: scripted hold
        if insert_success:
            hold_success = bool(
                controller.run_hold()
            )

            insert_diagnostics = dict(
                controller.get_last_insert_diagnostics()
            )

            save_named_image("09_hold_end.png")
            stage_end_errors["hold"] = extract_error_summary(base)
            save_stage_robot_state("hold")

            if not hold_success:
                failure_stage = "hold"
                failure_reason = str(
                    insert_diagnostics.get(
                        "failure_reason",
                        "hold_failed",
                    )
                )

        # Final strict assembly verification
        if hold_success:
            (
                assembly_verified,
                assembly_diagnostics,
            ) = controller.verify_final_assembly()

            assembly_verified = bool(assembly_verified)
            assembly_diagnostics = dict(assembly_diagnostics)

            if not assembly_verified:
                failure_stage = "final_assembly_verification"
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

        # New: save dedicated 13D stage-state json
        stage_state_payload = {
            "model": str(model_path),
            "seed": int(args.seed),
            "state_definition": "13D = 7 joint positions + ee_xyz + ee_rpy_xyz(rad)",
            "state_keys": STATE_KEYS_13D,
            "stage_order": list(STAGE_ORDER_9),
            "states": stage_robot_states_13d,
        }

        stage_state_json_path = output_dir / "stage_robot_states_13d.json"
        stage_state_json_path.write_text(
            json.dumps(
                stage_state_payload,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            ),
            encoding="utf-8",
        )

        summary = {
            "model": str(model_path),
            "seed": int(args.seed),
            "deterministic": bool(args.deterministic),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "aspect_ratio": "3:2",
            "camera_config": {
                "distance": float(camera_config["distance"]),
                "azimuth": float(camera_config["azimuth"]),
                "elevation": float(camera_config["elevation"]),
                "lookat": np.asarray(
                    camera_config["lookat"]
                ).tolist(),
            },
            "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
            "environment_kwargs": env_kwargs,
            "saved_files": saved_files,
            "stage_end_errors": stage_end_errors,
            "stage_robot_states_json": str(stage_state_json_path),
            "fine_align_steps": int(fine_steps),
            "fine_align_success": bool(fine_success),
            "insert_success": bool(insert_success),
            "hold_success": bool(hold_success),
            "assembly_verified": bool(assembly_verified),
            "full_pipeline_success": bool(full_pipeline_success),
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "fine_info": fine_info,
            "insert_diagnostics": insert_diagnostics,
            "assembly_diagnostics": assembly_diagnostics,
        }

        summary_path = output_dir / "summary.json"
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
        print(f"fine_steps={fine_steps}")

        if failure_reason:
            print(f"failure_stage={failure_stage}")
            print(f"failure_reason={failure_reason}")
        else:
            print("failure=none")

        print()
        print("Saved images:")
        for filename in EXPECTED_IMAGE_FILES:
            path = saved_files.get(filename)
            if path is None:
                print(f"  [missing] {filename}")
            else:
                print(f"  [saved]   {filename}")

        print()
        print(f"Stage-state JSON: {stage_state_json_path}")
        print(f"Summary: {summary_path}")
        print("=" * 72)

        missing_stage_states = [
            stage_name
            for stage_name in STAGE_ORDER_9
            if stage_name not in stage_robot_states_13d
        ]

        if missing_stage_states:
            raise RuntimeError(
                "Missing 13D robot states for one or more stages: "
                f"{missing_stage_states}. "
                f"See {stage_state_json_path}"
            )

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
