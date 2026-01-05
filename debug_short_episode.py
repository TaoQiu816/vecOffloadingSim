"""诊断为什么episode这么短"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.train_config import TrainConfig as TC
from agents.mappo_agent import MAPPOAgent
from models.offloading_policy import OffloadingPolicyNetwork

np.random.seed(42)
env = VecOffloadingEnv()

# 创建agent
network = OffloadingPolicyNetwork(
    d_model=TC.EMBED_DIM,
    num_heads=TC.NUM_HEADS,
    num_layers=TC.NUM_LAYERS
)
agent = MAPPOAgent(network, device='cpu')

print("="*80)
print("短Episode诊断")
print("="*80)

obs_list = env.reset(seed=42)
done = False
step = 0

while not done and step < 50:
    # 使用agent选择动作
    actions, _, _ = agent.select_actions(obs_list, deterministic=False)
    
    # 执行
    next_obs_list, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    if step < 5 or done:
        print(f"\nStep {step}:")
        print(f"  terminated={terminated}, truncated={truncated}")
        print(f"  rewards: {[f'{r:.3f}' for r in rewards]}")
        print(f"  _p2_active_time: {env._p2_active_time:.3f}s")
        print(f"  _p2_deltaW_active: {env._p2_deltaW_active/1e9:.3f}G")
        
        # 检查车辆状态
        for i, v in enumerate(env.vehicles):
            if i < 2:  # 只看前2辆车
                print(f"  Vehicle {i}: finished={v.task_dag.is_finished}, status={v.task_dag.status[:3]}")
    
    obs_list = next_obs_list
    step += 1

print(f"\nEpisode结束: 共{step}步")
print(f"最终P2统计:")
print(f"  _p2_active_time: {env._p2_active_time:.3f}s")
print(f"  _p2_idle_time: {env._p2_idle_time:.3f}s")
print(f"  _p2_deltaW_active: {env._p2_deltaW_active/1e9:.3f}G")

print(f"\n_last_episode_metrics:")
print(f"  service_rate_when_active: {env._last_episode_metrics.get('service_rate_when_active', 0)/1e9:.3f}G/s")
print(f"  idle_fraction: {env._last_episode_metrics.get('idle_fraction', 0):.3f}")

print("\n" + "="*80)

