"""检查奖励细节"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("奖励细节检查")
print("="*80)

all_rewards = []

# 运行一个episode
for step in range(60):
    actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES  # RSU动作
    next_obs_list, rewards, terminated, truncated, info = env.step(actions)
    
    all_rewards.extend(rewards)
    
    if step < 5:
        print(f"\nStep {step}:")
        print(f"  Rewards: {[f'{r:.6f}' for r in rewards[:3]]}")
        print(f"  Mean: {np.mean(rewards):.6f}")
        print(f"  Std: {np.std(rewards):.6f}")
    
    obs_list = next_obs_list
    if terminated or truncated:
        print(f"\nEpisode结束于step {step}")
        break

print(f"\n{'='*80}")
print("Episode统计:")
print(f"{'='*80}")
print(f"  总步数: {step+1}")
print(f"  总奖励样本: {len(all_rewards)}")
print(f"  奖励均值: {np.mean(all_rewards):.6f}")
print(f"  奖励标准差: {np.std(all_rewards):.6f}")
print(f"  奖励最小值: {np.min(all_rewards):.6f}")
print(f"  奖励最大值: {np.max(all_rewards):.6f}")
print(f"  奖励中位数: {np.median(all_rewards):.6f}")

# 检查奖励分布
unique_rewards = np.unique(np.round(all_rewards, 6))
print(f"\n不同的奖励值数量: {len(unique_rewards)}")
if len(unique_rewards) <= 10:
    print(f"所有不同的奖励值: {unique_rewards}")

print("\n" + "="*80)

