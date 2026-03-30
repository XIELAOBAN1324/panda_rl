# panda_mujoco_gym/my_test_env.py

import sys
import time
import gymnasium as gym
import panda_mujoco_gym
import os

def test_environment():
    """
    测试 Franka Panda 机器人环境的函数
    """
    try:
        # 首先以 headless 模式测试
        print("Testing environment in headless mode...")
        env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
        
        observation, info = env.reset()
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # 输出 Dict observation 的每个键的 shape
        if isinstance(observation, dict):
            print(f"Observation keys: {list(observation.keys())}")
            for key, value in observation.items():
                print(f"  {key} shape: {value.shape}")
        else:
            print(f"Initial observation shape: {observation.shape}")
        
        # 执行几步测试
        for step in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 将 terminated 和 truncated 转换为布尔值
            terminated = bool(terminated)
            truncated = bool(truncated)
            
            print(f"Step {step}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                observation, info = env.reset()
                print("Episode reset!")
        
        env.close()
        print("Headless test completed successfully!")
        
        # 测试图形模式
        print("\nTesting environment with graphics...")
        env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
        
        observation, info = env.reset()
        
        for step in range(100):  # 用更少的步数测试
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                observation, info = env.reset()
            
            time.sleep(0.05)  # 更快的执行
        
        env.close()
        print("Graphics test completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Trying alternative solutions...")
        
        # 替代方案：使用虚拟显示
        try:
            # 清理现有的 X 服务器进程
            os.system('pkill -f "Xvfb :99"')
            time.sleep(1)
            
            # 清理临时文件
            os.system('rm -f /tmp/.X99-lock')
            
            # 启动新的虚拟显示
            os.environ['DISPLAY'] = ':99'
            os.system('Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &')
            time.sleep(2)  # 等待显示启动
            
            env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
            observation, info = env.reset()
            
            for step in range(50):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                
                # 转换为布尔值
                terminated = bool(terminated)
                truncated = bool(truncated)
                
                if terminated or truncated:
                    observation, info = env.reset()
                    
                if step % 10 == 0:
                    print(f"Alternative step {step}: reward={reward:.4f}")
            
            env.close()
            print("Virtual display test completed!")
            
        except Exception as e2:
            print(f"Alternative solution also failed: {e2}")
            print("Please check your OpenGL installation and graphics drivers.")

if __name__ == "__main__":
    # 设置环境变量
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    
    test_environment()
