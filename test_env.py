import gymnasium as gym
import panda_mujoco_gym
import imageio

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
obs, info = env.reset()

frames = []
for i in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
imageio.mimsave("test_video.mp4", frames, fps=20)
print("saved test_video.mp4")