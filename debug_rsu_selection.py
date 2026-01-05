"""诊断RSU选择问题"""
import numpy as np
import torch
from envs.vec_offloading_env import VecOffloadingEnv
from configs.train_config import TrainConfig as TC
from agents.mappo_agent import MAPPOAgent
from models.offloading_policy import OffloadingPolicyNetwork

np.random.seed(42)
torch.manual_seed(42)

env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

# 创建agent
network = OffloadingPolicyNetwork(
    d_model=TC.EMBED_DIM,
    num_heads=TC.NUM_HEADS,
    num_layers=TC.NUM_LAYERS
)
agent = MAPPOAgent(network, device='cpu')

print("="*80)
print("RSU选择诊断")
print("="*80)

print(f"\n配置:")
print(f"  LOGIT_BIAS_RSU: {TC.LOGIT_BIAS_RSU}")
print(f"  LOGIT_BIAS_LOCAL: {TC.LOGIT_BIAS_LOCAL}")
print(f"  NUM_VEHICLES: {env.config.NUM_VEHICLES}")
print(f"  RSU数量: {len(env.rsus)}")

# 检查第一个观测
v0 = env.vehicles[0]
obs0 = obs_list[0]

print(f"\n第一辆车的观测:")
print(f"  obs类型: {type(obs0)}")
if isinstance(obs0, dict):
    print(f"  obs keys: {obs0.keys()}")
    if 'action_mask' in obs0:
        print(f"  action_mask: {obs0['action_mask']}")
else:
    print(f"  obs是list，长度: {len(obs0) if isinstance(obs0, list) else 'N/A'}")

# 选择动作
actions, values, log_probs = agent.select_action(obs_list, deterministic=False)

print(f"\n选择的动作:")
for i, act in enumerate(actions[:3]):  # 只看前3辆车
    print(f"  Vehicle {i}: target={act['target']}, power={act['power']:.3f}")
    target_name = ['Local', 'RSU', 'V2V'][act['target']] if act['target'] < 2 else f"V2V-{act['target']-2}"
    print(f"    -> {target_name}")

# 手动测试不同的动作
print(f"\n测试不同动作的执行:")

# 测试1: 全部Local
actions_local = [{"target": 0, "power": 1.0}] * env.config.NUM_VEHICLES
plans_local = env._plan_actions_snapshot(actions_local)
print(f"\n1. 全部Local动作:")
for i, plan in enumerate(plans_local[:3]):
    print(f"   Vehicle {i}: planned_kind={plan['planned_kind']}, illegal={plan['illegal_reason']}")

# 测试2: 全部RSU
actions_rsu = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
plans_rsu = env._plan_actions_snapshot(actions_rsu)
print(f"\n2. 全部RSU动作:")
for i, plan in enumerate(plans_rsu[:3]):
    print(f"   Vehicle {i}: planned_kind={plan['planned_kind']}, illegal={plan['illegal_reason']}")
    if plan['illegal_reason']:
        print(f"      原因: {plan['illegal_reason']}")

# 测试3: 全部V2V (target=2)
actions_v2v = [{"target": 2, "power": 1.0}] * env.config.NUM_VEHICLES
plans_v2v = env._plan_actions_snapshot(actions_v2v)
print(f"\n3. 全部V2V动作 (target=2):")
for i, plan in enumerate(plans_v2v[:3]):
    print(f"   Vehicle {i}: planned_kind={plan['planned_kind']}, illegal={plan['illegal_reason']}")
    if plan['illegal_reason']:
        print(f"      原因: {plan['illegal_reason']}")

# 检查RSU是否存在且可用
print(f"\nRSU状态:")
print(f"  RSU数量: {len(env.rsus)}")
if len(env.rsus) > 0:
    rsu = env.rsus[0]
    print(f"  RSU[0] 位置: {rsu.position}")
    print(f"  RSU[0] CPU频率: {rsu.cpu_freq/1e9:.2f} GHz")
    print(f"  RSU[0] 覆盖范围: {env.config.RSU_RANGE}m")
    
    # 检查车辆是否在RSU覆盖范围内
    for i, v in enumerate(env.vehicles[:3]):
        dist = np.linalg.norm(v.position - rsu.position)
        in_range = dist <= env.config.RSU_RANGE
        print(f"  Vehicle {i} 距离RSU: {dist:.2f}m, 在范围内: {in_range}")

# 检查_last_rsu_choice缓存
print(f"\n_last_rsu_choice缓存:")
if hasattr(env, '_last_rsu_choice'):
    for i in range(min(3, len(env._last_rsu_choice))):
        print(f"  Vehicle {i}: {env._last_rsu_choice[i]}")

# 执行一步RSU动作，看看实际效果
print(f"\n执行RSU动作:")
obs_list = env.reset(seed=42)
actions_rsu = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
next_obs, rewards, terminated, truncated, info = env.step(actions_rsu)

print(f"  决策统计: {env._decision_counts}")
print(f"  RSU队列长度: {sum(len(q) for proc_dict in env.rsu_cpu_q.values() for q in proc_dict.values())}")

print("\n" + "="*80)

