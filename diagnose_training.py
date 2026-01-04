#!/usr/bin/env python3
"""
诊断训练问题的脚本
"""
import numpy as np
import sys
from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv

print("="*80)
print("  训练诊断：检查环境和动作执行")
print("="*80)

# 1. 创建环境
env = VecOffloadingEnv()
obs_list, _ = env.reset(seed=42)

print(f"\n1. 环境初始化:")
print(f"   车辆数: {len(env.vehicles)}")
print(f"   RSU数: {len(env.rsus)}")

# 检查第一辆车
v = env.vehicles[0]
print(f"\n2. 第一辆车:")
print(f"   ID: {v.id}")
print(f"   任务节点数: {v.task_dag.num_subtasks}")
print(f"   Top priority task: {v.task_dag.get_top_priority_task()}")
print(f"   CPU: {v.cpu_freq/1e9:.2f} GHz")

# 检查观测
obs = obs_list[0]
print(f"\n3. 观测:")
print(f"   action_mask: {obs['action_mask']}")
print(f"   target_mask: {obs['target_mask']}")
print(f"   resource_ids: {obs.get('resource_ids', 'N/A')}")

# 检查缓存
print(f"\n4. 决策缓存:")
print(f"   _last_rsu_choice[{v.id}]: {env._last_rsu_choice.get(v.id)}")
print(f"   _last_candidates[{v.id}]: {env._last_candidates.get(v.id)}")

# 5. 尝试不同的动作
print(f"\n5. 测试不同动作:")

# Test Local
print(f"\n   [Test 1] Local动作:")
actions = [{'target': 0, 'power': 0.5} for _ in range(len(env.vehicles))]
plans = env._plan_actions_snapshot(actions)
plan = plans[0]
print(f"      subtask_idx: {plan['subtask_idx']}")
print(f"      planned_target: {plan['planned_target']}")
print(f"      illegal_reason: {plan['illegal_reason']}")

# Test RSU
print(f"\n   [Test 2] RSU动作:")
actions = [{'target': 1, 'power': 0.5} for _ in range(len(env.vehicles))]
plans = env._plan_actions_snapshot(actions)
plan = plans[0]
print(f"      subtask_idx: {plan['subtask_idx']}")
print(f"      planned_target: {plan['planned_target']}")
print(f"      illegal_reason: {plan['illegal_reason']}")

# Test V2V
print(f"\n   [Test 3] V2V动作 (target=2):")
actions = [{'target': 2, 'power': 0.5} for _ in range(len(env.vehicles))]
plans = env._plan_actions_snapshot(actions)
plan = plans[0]
print(f"      subtask_idx: {plan['subtask_idx']}")
print(f"      planned_target: {plan['planned_target']}")
print(f"      illegal_reason: {plan['illegal_reason']}")

# 6. 执行一个完整的step
print(f"\n6. 执行完整step (Local动作):")
env.reset(seed=42)
obs_list, _ = env.reset()
actions = [{'target': 0, 'power': 0.5} for _ in range(len(env.vehicles))]

next_obs_list, rewards, terminated, truncated, info = env.step(actions)

print(f"   Rewards: {[f'{r:.6f}' for r in rewards[:3]]}")
print(f"   决策统计: {env._decision_counts}")
print(f"   P2活跃时间: {env._p2_active_time:.4f}s")
print(f"   P2空闲时间: {env._p2_idle_time:.4f}s")

# 检查第一辆车的状态
v = env.vehicles[0]
print(f"\n7. 第一辆车状态变化:")
print(f"   exec_locations[0]: {v.task_dag.exec_locations[0]}")
print(f"   task_dag.status[0]: {v.task_dag.status[0]}")
print(f"   active_tasks: {v.active_task_manager.get_num_active_tasks()}")

# 检查队列
print(f"\n8. 队列状态:")
print(f"   veh_cpu_q[{v.id}] 长度: {len(env.veh_cpu_q[v.id])}")
print(f"   txq_v2i 总长度: {sum(len(q) for q in env.txq_v2i.values())}")
print(f"   txq_v2v 总长度: {sum(len(q) for q in env.txq_v2v.values())}")

# 9. 连续执行多步
print(f"\n9. 连续执行20步 (Local动作):")
for step in range(20):
    actions = [{'target': 0, 'power': 0.5} for _ in range(len(env.vehicles))]
    next_obs_list, rewards, terminated, truncated, info = env.step(actions)
    
    if step % 5 == 0:
        total_active = env._get_total_active_tasks()
        total_finished = sum([1 for v in env.vehicles if v.task_dag.is_finished])
        print(f"   Step {step}: active={total_active}, finished={total_finished}/{len(env.vehicles)}, "
              f"reward_mean={np.mean(rewards):.6f}")

print(f"\n10. 最终统计:")
print(f"   决策统计: {env._decision_counts}")
print(f"   P2活跃时间: {env._p2_active_time:.4f}s")
print(f"   P2空闲时间: {env._p2_idle_time:.4f}s")
total_active = env._get_total_active_tasks()
print(f"   当前活跃任务: {total_active}")

print("\n" + "="*80)

