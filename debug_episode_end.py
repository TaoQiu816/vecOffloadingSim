"""诊断episode结束时的统计"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg

np.random.seed(42)
env = VecOffloadingEnv(Cfg)
obs_list = env.reset(seed=42)

print("="*80)
print("Episode结束统计诊断")
print("="*80)

# 执行一个完整episode（只执行10步）
actions = [{"target": 0, "power": 1.0}] * env.config.NUM_VEHICLES

for step in range(10):
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break

print(f"\nEpisode运行了{step+1}步")
print(f"\n执行结束前的P2统计:")
print(f"  _p2_active_time: {env._p2_active_time:.3f}s")
print(f"  _p2_idle_time: {env._p2_idle_time:.3f}s")
print(f"  _p2_deltaW_active: {env._p2_deltaW_active/1e9:.3f}G cycles")

# 手动调用_log_episode_stats
env._log_episode_stats(terminated, truncated)

print(f"\n_last_episode_metrics:")
for key, value in env._last_episode_metrics.items():
    if 'rate' in key.lower() or 'fraction' in key.lower() or 'idle' in key.lower():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

print("\n" + "="*80)

