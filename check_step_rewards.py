"""检查step级奖励"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("Step级奖励检查")
print("="*80)

step_rewards_list = []

# 运行一个episode
for step in range(60):
    actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES  # RSU动作
    next_obs_list, rewards, terminated, truncated, info = env.step(actions)
    
    step_r = sum(rewards) / len(rewards)
    step_rewards_list.append(step_r)
    
    if step < 10:
        print(f"Step {step}: step_r={step_r:.6f}, rewards={[f'{r:.4f}' for r in rewards[:3]]}")
    
    obs_list = next_obs_list
    if terminated or truncated:
        break

print(f"\n{'='*80}")
print("Episode统计:")
print(f"{'='*80}")
print(f"  总步数: {len(step_rewards_list)}")
print(f"  Episode平均奖励: {np.mean(step_rewards_list):.6f}")
print(f"  Step奖励标准差: {np.std(step_rewards_list):.6f}")
print(f"  Step奖励最小值: {np.min(step_rewards_list):.6f}")
print(f"  Step奖励最大值: {np.max(step_rewards_list):.6f}")

print("\n" + "="*80)

