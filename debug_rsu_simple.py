"""简化的RSU选择诊断"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("RSU选择诊断")
print("="*80)

print(f"\n环境配置:")
print(f"  NUM_VEHICLES: {env.config.NUM_VEHICLES}")
print(f"  RSU数量: {len(env.rsus)}")
print(f"  RSU_RANGE: {env.config.RSU_RANGE}m")

if len(env.rsus) > 0:
    print(f"\nRSU信息:")
    for i, rsu in enumerate(env.rsus):
        print(f"  RSU[{i}]: 位置={rsu.position}, CPU={rsu.cpu_freq/1e9:.2f}GHz")

print(f"\n车辆位置和RSU可达性:")
for i, v in enumerate(env.vehicles[:5]):  # 只看前5辆
    print(f"  Vehicle {i}: 位置={v.pos}")
    if len(env.rsus) > 0:
        for j, rsu in enumerate(env.rsus):
            dist = np.linalg.norm(v.pos - rsu.position)
            in_range = dist <= env.config.RSU_RANGE
            print(f"    到RSU[{j}]距离: {dist:.2f}m, 在范围内: {in_range}")

# 测试不同动作
print(f"\n测试动作映射:")

# 1. Local动作
print(f"\n1. Local动作 (target=0):")
actions = [{"target": 0, "power": 1.0}] * env.config.NUM_VEHICLES
next_obs, rewards, terminated, truncated, info = env.step(actions)
print(f"   决策统计: {env._decision_counts}")

# 2. RSU动作
print(f"\n2. RSU动作 (target=1):")
obs_list = env.reset(seed=42)
actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
next_obs, rewards, terminated, truncated, info = env.step(actions)
print(f"   决策统计: {env._decision_counts}")
print(f"   RSU队列: {[(rid, len(q)) for rid, proc_dict in env.rsu_cpu_q.items() for pid, q in proc_dict.items()]}")

# 检查action planning
print(f"\n3. 详细检查action planning:")
obs_list = env.reset(seed=42)
actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
plans = env._plan_actions_snapshot(actions)

for i, plan in enumerate(plans[:5]):
    print(f"   Vehicle {i}:")
    print(f"     planned_kind: {plan['planned_kind']}")
    print(f"     planned_target: {plan['planned_target']}")
    print(f"     illegal_reason: {plan['illegal_reason']}")
    if plan['illegal_reason']:
        print(f"     详细: {plan['illegal_reason']}")

# 检查_last_rsu_choice
print(f"\n4. _last_rsu_choice缓存:")
if hasattr(env, '_last_rsu_choice'):
    for i in range(min(5, len(env._last_rsu_choice))):
        print(f"   Vehicle {i}: {env._last_rsu_choice[i]}")
else:
    print(f"   _last_rsu_choice不存在")

print("\n" + "="*80)

