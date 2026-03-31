import os
from gymnasium.envs.registration import register

ENV_IDS = []

# 对 PickAndPlace 给予更长的探索窗口。
# Slide / Push 保持原先 100 步，避免影响原本已稳定收敛的任务。
MAX_EPISODE_STEPS = {
    "Slide": 100,
    "Push": 100,
    "PickAndPlace": 200,
}

for task in ["Slide", "Push", "PickAndPlace"]:
    for reward_type in ["sparse", "dense"]:
        reward_suffix = "Dense" if reward_type == "dense" else "Sparse"
        env_id = f"Franka{task}{reward_suffix}-v0"
        register(
            id=env_id,
            entry_point=f"panda_mujoco_gym.envs:Franka{task}Env",
            kwargs={"reward_type": reward_type},
            max_episode_steps=MAX_EPISODE_STEPS[task],
        )
        ENV_IDS.append(env_id)
