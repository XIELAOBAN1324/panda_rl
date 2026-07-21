from gymnasium.envs.registration import register

ENV_IDS = []

for reward_type in ["sparse", "dense"]:
    reward_suffix = "Dense" if reward_type == "dense" else "Sparse"
    env_id = f"FrankaPickAndPlaceWindow{reward_suffix}-v0"
    register(
        id=env_id,
        entry_point="panda_mujoco_gym.envs:FrankaPickAndPlaceWindowEnv",
        kwargs={
            "reward_type": reward_type,
            "reset_mode": "full_task",
            "success_position_threshold": 0.03,
            "success_orientation_deg": 20.0,
            "glass_fit_orientation_deg": 12.0,
            "collision_termination": False,
            "glass_plane_tolerance": 0.01,
        },
        max_episode_steps=250,
    )
    ENV_IDS.append(env_id)

    insert_env_id = f"FrankaPickAndPlaceWindowInsert{reward_suffix}-v0"
    register(
        id=insert_env_id,
        entry_point="panda_mujoco_gym.envs:FrankaPickAndPlaceWindowEnv",
        kwargs={
            "reward_type": reward_type,
            "reset_mode": "post_grasp_lifted",
            "success_position_threshold": 0.005,
            "success_orientation_deg": 5.0,
            "glass_fit_orientation_deg": 5.0,
            "collision_termination": True,
            "glass_plane_tolerance": 0.005,
        },
        max_episode_steps=200,
    )
    ENV_IDS.append(insert_env_id)

    prealign_env_id = f"FrankaPickAndPlaceWindowPrealign{reward_suffix}-v0"
    register(
        id=prealign_env_id,
        entry_point="panda_mujoco_gym.envs:FrankaPickAndPlaceWindowEnv",
        kwargs={
            "reward_type": reward_type,
            "reset_mode": "post_grasp_prealign",
            "success_position_threshold": 0.01,
            "success_orientation_deg": 8.0,
            "glass_fit_orientation_deg": 8.0,
            "collision_termination": True,
            "glass_plane_tolerance": 0.01,
        },
        max_episode_steps=200,
    )
    ENV_IDS.append(prealign_env_id)


FINE_ALIGN_ENV_ID = "FrankaWindowFineAlignDense-v0"
register(
    id=FINE_ALIGN_ENV_ID,
    entry_point="panda_mujoco_gym.envs:FrankaWindowFineAlignEnv",
    kwargs={
        "reward_type": "dense",
        "max_episode_steps": 100,
        "coarse_translation_range": 0.015,
        "coarse_tilt_range_deg": 2.0,
        "coarse_yaw_range_deg": 4.0,
        "fine_position_action_scale": 0.0015,
        "fine_tilt_action_scale_deg": 0.25,
        "fine_yaw_action_scale_deg": 0.4,
        "translation_tolerance": 0.002,
        "tilt_tolerance_deg": 0.5,
        "yaw_tolerance_deg": 0.5,
        "normal_gap_tolerance": 0.001,
        "success_hold_steps": 5,
    },
    max_episode_steps=100,
)
ENV_IDS.append(FINE_ALIGN_ENV_ID)
