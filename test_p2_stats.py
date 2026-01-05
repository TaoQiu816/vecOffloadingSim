"""测试P2统计"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("P2统计测试")
print("="*80)

# 运行一个完整episode，使用Local动作
for step in range(100):
    actions = [{"target": 0, "power": 1.0}] * env.config.NUM_VEHICLES
    next_obs_list, rewards, terminated, truncated, info = env.step(actions)
    
    if step % 10 == 0 or terminated or truncated:
        print(f"\nStep {step}:")
        print(f"  _p2_active_time: {env._p2_active_time:.3f}s")
        print(f"  _p2_idle_time: {env._p2_idle_time:.3f}s")
        print(f"  _p2_deltaW_active: {env._p2_deltaW_active/1e9:.3f}G")
        
        if hasattr(env, '_last_episode_metrics'):
            metrics = env._last_episode_metrics
            print(f"  service_rate_when_active: {metrics.get('service_rate_when_active', 0)/1e9:.3f}G/s")
            print(f"  idle_fraction: {metrics.get('idle_fraction', 0):.3f}")
    
    obs_list = next_obs_list
    if terminated or truncated:
        print(f"\nEpisode结束于step {step}")
        break

print("\n" + "="*80)

