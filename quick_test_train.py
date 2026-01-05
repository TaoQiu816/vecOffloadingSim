"""快速训练测试 - 只运行5个episode验证统计数据"""
import sys
import numpy as np
import torch
from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg
from agents.mappo_agent import MAPPOAgent

# 固定seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 创建环境
env = VecOffloadingEnv(Cfg)
obs_list = env.reset(seed=SEED)

# 创建agent
agent = MAPPOAgent(
    obs_space=env.observation_space,
    act_space=env.action_space,
    config=Cfg,
    device='cpu'
)

print("="*120)
print("|    Ep |  Time(s) |   Reward |   V_SR |   T_SR |   S_SR | Deadlock |  D_Miss |   TX |  NoTX |  Local |    RSU |    V2V |   SvcRate |   Idle |")
print("="*120)

# 运行5个episode
for episode in range(1, 6):
    obs_list = env.reset(seed=SEED + episode)
    done = False
    episode_reward = 0.0
    steps = 0
    
    while not done and steps < 200:
        # 选择动作
        actions, _, _ = agent.select_actions(obs_list, deterministic=False)
        
        # 执行动作
        next_obs_list, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        
        episode_reward += np.mean(rewards)
        obs_list = next_obs_list
        steps += 1
    
    # 获取统计数据
    metrics = env._last_episode_metrics
    
    # 提取关键指标
    reward_mean = metrics.get('reward_mean', 0.0)
    vehicle_sr = metrics.get('vehicle_success_rate', 0.0)
    task_sr = metrics.get('task_success_rate', 0.0)
    subtask_sr = metrics.get('subtask_success_rate', 0.0)
    deadlock = metrics.get('deadlock_vehicle_count', 0)
    deadline_miss = metrics.get('deadline_miss_count', 0)
    tx_created = metrics.get('tx_tasks_created_count', 0)
    same_node_no_tx = metrics.get('same_node_no_tx_count', 0)
    
    frac_local = metrics.get('decision_frac_local', 0.0)
    frac_rsu = metrics.get('decision_frac_rsu', 0.0)
    frac_v2v = metrics.get('decision_frac_v2v', 0.0)
    
    service_rate = metrics.get('service_rate_when_active', 0.0)
    idle_frac = metrics.get('idle_fraction', 0.0)
    
    duration = steps * Cfg.DT
    
    print(f"| {episode:5d} | {duration:8.2f} | {reward_mean:8.2f} | {vehicle_sr:6.2%} | {task_sr:6.2%} | {subtask_sr:6.2%} | {deadlock:8d} | {deadline_miss:7d} | {tx_created:4d} | {same_node_no_tx:5d} | {frac_local:6.2%} | {frac_rsu:6.2%} | {frac_v2v:6.2%} | {service_rate/1e9:9.3f}G | {idle_frac:6.2%} |")

print("="*120)
print("\n✅ 测试完成！检查以下指标是否正常：")
print("   1. SvcRate (服务率) 应该 > 0")
print("   2. Idle (空闲率) 应该 < 100%")
print("   3. TX (传输任务数) 应该 > 0 (如果有RSU/V2V决策)")
print("   4. Local/RSU/V2V 决策分布应该有变化")

