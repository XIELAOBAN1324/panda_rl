"""Explicit final-environment arguments shared by evaluation tools."""

from __future__ import annotations

import argparse
from typing import Any


FINAL_ENV_PARAMETERS = (
    ("coarse_translation_range", "coarse_translation_range", float, 0.015),
    ("coarse_tilt_range_deg", "coarse_tilt_range_deg", float, 2.0),
    ("coarse_yaw_range_deg", "coarse_yaw_range_deg", float, 4.0),
    ("preinsert_normal_offset", "preinsert_normal_offset", float, 0.08),
    ("position_action_scale", "fine_position_action_scale", float, 0.0015),
    ("tilt_action_scale_deg", "fine_tilt_action_scale_deg", float, 0.25),
    ("yaw_action_scale_deg", "fine_yaw_action_scale_deg", float, 0.4),
    ("fine_action_sim_steps", "fine_action_sim_steps", int, 4),
    ("translation_tolerance", "translation_tolerance", float, 0.002),
    ("tilt_tolerance_deg", "tilt_tolerance_deg", float, 0.5),
    ("yaw_tolerance_deg", "yaw_tolerance_deg", float, 0.5),
    ("normal_gap_tolerance", "normal_gap_tolerance", float, 0.001),
    (
        "normal_gap_correction_threshold",
        "normal_gap_correction_threshold",
        float,
        0.0005,
    ),
    (
        "normal_gap_correction_sim_steps",
        "normal_gap_correction_sim_steps",
        int,
        2,
    ),
    ("max_normal_gap_drift", "max_normal_gap_drift", float, 0.008),
    ("insert_action_sim_steps", "insert_action_sim_steps", int, 4),
    (
        "insert_translation_abort_tolerance",
        "insert_translation_abort_tolerance",
        float,
        0.0025,
    ),
    (
        "insert_tilt_abort_tolerance_deg",
        "insert_tilt_abort_tolerance_deg",
        float,
        0.6,
    ),
    (
        "insert_yaw_abort_tolerance_deg",
        "insert_yaw_abort_tolerance_deg",
        float,
        0.6,
    ),
    (
        "insert_alignment_violation_hold_steps",
        "insert_alignment_violation_hold_steps",
        int,
        2,
    ),
    ("insert_success_hold_steps", "insert_success_hold_steps", int, 3),
    (
        "insert_final_verification_max_steps",
        "insert_final_verification_max_steps",
        int,
        20,
    ),
    ("insert_depth_tolerance", "insert_depth_tolerance", float, 0.0005),
    ("insert_step_size", "insert_step_size", float, 0.001),
)


def add_final_environment_arguments(
    parser: argparse.ArgumentParser,
    *,
    measurement_noise_translation_std: float = 0.0,
    measurement_noise_angle_std_deg: float = 0.0,
) -> None:
    for destination, _, value_type, default in FINAL_ENV_PARAMETERS:
        parser.add_argument(
            "--" + destination.replace("_", "-"),
            dest=destination,
            type=value_type,
            default=default,
        )
    parser.add_argument(
        "--measurement-noise-translation-std",
        type=float,
        default=measurement_noise_translation_std,
    )
    parser.add_argument(
        "--measurement-noise-angle-std-deg",
        type=float,
        default=measurement_noise_angle_std_deg,
    )


def final_environment_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        environment_key: getattr(args, destination)
        for destination, environment_key, _, _ in FINAL_ENV_PARAMETERS
    }
    values.update(
        {
            "measurement_noise_translation_std": (
                args.measurement_noise_translation_std
            ),
            "measurement_noise_angle_std_deg": (
                args.measurement_noise_angle_std_deg
            ),
        }
    )
    return values


def validate_final_environment_kwargs(values: dict[str, Any]) -> None:
    positive_integer_keys = (
        "fine_action_sim_steps",
        "normal_gap_correction_sim_steps",
        "insert_action_sim_steps",
        "insert_alignment_violation_hold_steps",
        "insert_success_hold_steps",
        "insert_final_verification_max_steps",
    )
    for key in positive_integer_keys:
        if int(values[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    positive_float_keys = (
        "preinsert_normal_offset",
        "max_normal_gap_drift",
        "insert_step_size",
    )
    for key in positive_float_keys:
        if float(values[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if float(values["normal_gap_correction_threshold"]) < 0.0:
        raise ValueError("normal_gap_correction_threshold must be non-negative")
    if float(values["insert_depth_tolerance"]) < 0.0:
        raise ValueError("insert_depth_tolerance must be non-negative")
    if (
        float(values["insert_translation_abort_tolerance"])
        < float(values["translation_tolerance"])
    ):
        raise ValueError(
            "insert_translation_abort_tolerance must be at least translation_tolerance"
        )
    if (
        float(values["insert_tilt_abort_tolerance_deg"])
        < float(values["tilt_tolerance_deg"])
    ):
        raise ValueError(
            "insert_tilt_abort_tolerance_deg must be at least tilt_tolerance_deg"
        )
    if (
        float(values["insert_yaw_abort_tolerance_deg"])
        < float(values["yaw_tolerance_deg"])
    ):
        raise ValueError(
            "insert_yaw_abort_tolerance_deg must be at least yaw_tolerance_deg"
        )
