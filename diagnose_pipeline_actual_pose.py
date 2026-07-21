from __future__ import annotations

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

import panda_mujoco_gym  # noqa: F401

from panda_mujoco_gym.envs.window_assembly_geometry import (
    project_to_rotation_matrix,
    so3_log,
)
from panda_mujoco_gym.envs.window_assembly_pipeline import (
    WindowAssemblyPipeline,
)
from evaluate.evaluate_window_assembly_pipeline import (
    ENV_ARG_DEFAULTS,
    build_environment_kwargs,
    validate_environment_kwargs,
)


MODEL_PATH = (
    "outputs/window_fine_align/"
    "sac-20260721-205122-seed0/"
    "models/best_model.zip"
)


def actual_alignment_errors(base) -> dict[str, float]:
    actual_center = np.asarray(
        base.data.xpos[base.obj_body_id],
        dtype=np.float64,
    ).copy()

    actual_object_rotation = project_to_rotation_matrix(
        np.asarray(
            base.data.xmat[base.obj_body_id],
            dtype=np.float64,
        ).reshape(3, 3)
    )

    actual_glass_rotation = project_to_rotation_matrix(
        actual_object_rotation @ base.GLASS_ASSEMBLY_FRAME
    )

    frame_center, frame_rotation = base._current_fine_frame_pose()
    orientation_frame = base._current_orientation_frame()

    center_delta = actual_center - frame_center

    relative_rotation = (
        orientation_frame.T @ actual_glass_rotation
    )
    rotation_error_world = (
        orientation_frame @ so3_log(relative_rotation)
    )

    return {
        "error_u": float(
            np.dot(frame_rotation[:, 0], center_delta)
        ),
        "error_v": float(
            np.dot(frame_rotation[:, 1], center_delta)
        ),
        "tilt_error_t1": float(
            np.dot(frame_rotation[:, 0], rotation_error_world)
        ),
        "tilt_error_t2": float(
            np.dot(frame_rotation[:, 1], rotation_error_world)
        ),
        "yaw_error": float(
            np.dot(frame_rotation[:, 2], rotation_error_world)
        ),
        "current_normal_gap": float(
            np.dot(frame_rotation[:, 2], center_delta)
        ),
    }


environment_kwargs, training_config_path, _ = build_environment_kwargs(
    MODEL_PATH,
    ENV_ARG_DEFAULTS,
    set(),
)
validate_environment_kwargs(environment_kwargs)
print(f"training_config_path={training_config_path or 'none'}")
print(f"environment_kwargs={environment_kwargs}")

env = gym.make(
    "FrankaWindowFineAlignDense-v0",
    render_mode=None,
    **environment_kwargs,
)

model = SAC.load(MODEL_PATH, device="auto")
pipeline = WindowAssemblyPipeline(env)

try:
    for seed in range(5):
        result = pipeline.run_episode(
            model,
            seed=seed,
            deterministic=True,
        )

        base = env.unwrapped

        virtual_errors = base.get_fine_alignment_errors()
        actual_errors = actual_alignment_errors(base)

        virtual_center = np.asarray(
            base.get_object_position(),
            dtype=np.float64,
        )
        actual_center = np.asarray(
            base.data.xpos[base.obj_body_id],
            dtype=np.float64,
        )

        virtual_rotation = project_to_rotation_matrix(
            base.get_object_rotation_matrix()
        )
        actual_rotation = project_to_rotation_matrix(
            np.asarray(
                base.data.xmat[base.obj_body_id],
                dtype=np.float64,
            ).reshape(3, 3)
        )

        position_discrepancy = float(
            np.linalg.norm(virtual_center - actual_center)
        )
        rotation_discrepancy_deg = float(
            np.rad2deg(
                np.linalg.norm(
                    so3_log(
                        virtual_rotation.T @ actual_rotation
                    )
                )
            )
        )

        virtual_gap = float(
            virtual_errors["current_normal_gap"]
        )
        actual_gap = float(
            actual_errors["current_normal_gap"]
        )

        commanded_depth = float(
            result.get("insert_depth", 0.0)
        )
        actual_depth = float(
            result["insert_start_errors"][
                "insert_start_normal_gap"
            ]
            - actual_gap
        )

        print()
        print("=" * 68)
        print(
            f"seed={seed} "
            f"fine={result['fine_align_success']} "
            f"insert={result['insert_success']} "
            f"hold={result['hold_success']} "
            f"full={result['full_pipeline_success']}"
        )
        print("failure:", result["failure_reason"] or "none")
        print()

        print("[Depth]")
        print(f"commanded depth : {commanded_depth:.6f} m")
        print(f"actual depth    : {actual_depth:.6f} m")
        print(f"virtual gap     : {virtual_gap:.6f} m")
        print(f"actual gap      : {actual_gap:.6f} m")

        print()
        print("[Virtual vs actual]")
        print(
            f"position discrepancy: "
            f"{position_discrepancy * 1000:.3f} mm"
        )
        print(
            f"rotation discrepancy: "
            f"{rotation_discrepancy_deg:.6f} deg"
        )

        print()
        print("[Actual final errors]")
        for key, value in actual_errors.items():
            if "tilt" in key or "yaw" in key:
                print(
                    f"{key:20s}: "
                    f"{np.rad2deg(value): .6f} deg"
                )
            else:
                print(
                    f"{key:20s}: "
                    f"{value * 1000: .3f} mm"
                )
finally:
    env.close()
