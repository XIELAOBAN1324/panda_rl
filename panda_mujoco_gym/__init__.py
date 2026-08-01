"""Final subway window-glass fine-alignment environment registration."""

from gymnasium.envs.registration import register


FINE_ALIGN_ENV_ID = "FrankaWindowFineAlignDense-v0"
ENV_IDS = (FINE_ALIGN_ENV_ID,)

register(
    id=FINE_ALIGN_ENV_ID,
    entry_point=(
        "panda_mujoco_gym.envs.window_fine_align:"
        "FrankaWindowFineAlignEnv"
    ),
    kwargs={
        "reward_type": "dense",
        "max_episode_steps": 100,
        "coarse_translation_range": 0.015,
        "coarse_tilt_range_deg": 2.0,
        "coarse_yaw_range_deg": 4.0,
        "preinsert_normal_offset": 0.08,
        "fine_position_action_scale": 0.0015,
        "fine_tilt_action_scale_deg": 0.25,
        "fine_yaw_action_scale_deg": 0.4,
        "fine_action_sim_steps": 4,
        "translation_tolerance": 0.002,
        "tilt_tolerance_deg": 0.5,
        "yaw_tolerance_deg": 0.5,
        "normal_gap_tolerance": 0.001,
        "normal_gap_correction_threshold": 0.0005,
        "normal_gap_correction_sim_steps": 2,
        "max_normal_gap_drift": 0.008,
        "insert_action_sim_steps": 4,
        "insert_translation_abort_tolerance": 0.0025,
        "insert_tilt_abort_tolerance_deg": 0.6,
        "insert_yaw_abort_tolerance_deg": 0.6,
        "insert_alignment_violation_hold_steps": 2,
        "insert_step_size": 0.001,
        "success_hold_steps": 5,
    },
    max_episode_steps=100,
)


__all__ = ["ENV_IDS", "FINE_ALIGN_ENV_ID"]
