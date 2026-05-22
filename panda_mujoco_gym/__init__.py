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
