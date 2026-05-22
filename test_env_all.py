#/home/minjun/panda_mujoco_gym/test_env_all.py
import sys
import time
import gymnasium as gym
import panda_mujoco_gym

def test_single_env(env_id, steps=500, sleep_time=0.1):
    """Test a single environment and visualize"""
    print(f"\n🎮 Start testing {env_id}!")
    print("=" * 60)
    
    # Descriptions of each environment
    descriptions = {
        "FrankaPickAndPlaceWindowSparse-v0": "🪟 Window Pick & Place (Sparse Reward): Attach object and insert through the window",
        "FrankaPickAndPlaceWindowDense-v0": "🪟 Window Pick & Place (Dense Reward): Position and orientation shaping for window insertion",
    }
    
    if env_id in descriptions:
        print(f"📝 {descriptions[env_id]}")
    print()
    
    env = gym.make(env_id, render_mode="human")
    observation, info = env.reset()
    
    episode_count = 0
    success_count = 0
    total_reward = 0
    
    for i in range(steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Check if successful
        if 'is_success' in info and info['is_success']:
            success_count += 1
        
        # Periodic state output
        if i % 100 == 0:
            print(f"  Step {i:3d}: reward={reward:6.2f}, total_reward={total_reward:8.2f}")
        
        if terminated or truncated:
            episode_count += 1
            observation, info = env.reset()
            print(f"  🔄 Episode {episode_count} Complete! (Step {i})")
            total_reward = 0  # Reset each episode
        
        time.sleep(sleep_time)
    
    env.close()
    
    print(f"\n✅ {env_id} Test Complete!")
    print(f"   📊 Episodes: {episode_count}, Successes: {success_count}")
    print(f"   ⏱️  Total steps: {steps}")
    
    return episode_count, success_count

def main():
    """Main execution function"""
    print("🚀 Panda MuJoCo Gym - Test Window Environments")
    print("=" * 70)
    print("💡 Test each environment in order.")
    print("💡 Press Ctrl+C to interrupt.")
    print("💡 Press Enter to continue to the next environment.\n")
    
    # Output all available environments
    print(f"📋 Available environments ({len(panda_mujoco_gym.ENV_IDS)} 个):")
    for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS, 1):
        print(f"  {i}. {env_id}")
    
    # User choice
    print("\n🎯 Test options:")
    print("  1. Automatic test all environments")
    print("  2. Test specific environments")
    print("  3. Quick test (200 steps per environment)")
    
    try:
        choice = input("\nPlease select (1/2/3, default: 1): ").strip()
        
        if choice == "2":
            # Select specific environment
            print("\nPlease enter environment number:")
            for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS, 1):
                print(f"  {i}. {env_id}")
            
            env_num = int(input("Input number: ")) - 1
            if 0 <= env_num < len(panda_mujoco_gym.ENV_IDS):
                selected_env = panda_mujoco_gym.ENV_IDS[env_num]
                test_single_env(selected_env, steps=1000, sleep_time=0.1)
            else:
                print("❌ Invalid number.")
                return
                
        elif choice == "3":
            # Quick test
            print("\n⚡ Quick test mode (200 steps per environment)")
            for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS):
                print(f"\n[{i+1}/{len(panda_mujoco_gym.ENV_IDS)}] {env_id}")
                test_single_env(env_id, steps=200, sleep_time=0.05)
                
                if i < len(panda_mujoco_gym.ENV_IDS) - 1:
                    input("\n⏸️  Press Enter to continue...")
        
        else:
            # Test all environments (default)
            print("\n🎬 Start testing all environments!")
            
            results = []
            for i, env_id in enumerate(panda_mujoco_gym.ENV_IDS):
                print(f"\n[{i+1}/{len(panda_mujoco_gym.ENV_IDS)}] {env_id}")
                episodes, successes = test_single_env(env_id, steps=500, sleep_time=0.1)
                results.append((env_id, episodes, successes))
                
                if i < len(panda_mujoco_gym.ENV_IDS) - 1:
                    input("\n⏸️  Press Enter to continue to next environment...")
            
            # Final results summary
            print("\n" + "=" * 70)
            print("📊 Final test results summary")
            print("=" * 70)
            for env_id, episodes, successes in results:
                print(f"{env_id:30} | Episodes: {episodes:2d} | Successes: {successes:3d}")
        
        print("\n🎉 All tests complete!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 User interrupted the test.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")

if __name__ == "__main__":
    main()
