"""诊断RSU任务为什么没有入队"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("RSU任务入队诊断")
print("="*80)

# 执行RSU动作
actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES

print(f"\n执行前状态:")
v0 = env.vehicles[0]
print(f"  Vehicle 0:")
print(f"    task_dag.status[0]: {v0.task_dag.status[0]}")
print(f"    task_dag.rem_data[0]: {v0.task_dag.rem_data[0]/1e6:.3f}MB")
print(f"    task_dag.total_comp[0]: {v0.task_dag.total_comp[0]/1e9:.3f}G cycles")
print(f"    task_dag.exec_locations[0]: {v0.task_dag.exec_locations[0]}")

# 执行step
next_obs, rewards, terminated, truncated, info = env.step(actions)

print(f"\n执行后状态:")
print(f"  Vehicle 0:")
print(f"    task_dag.status[0]: {v0.task_dag.status[0]}")
print(f"    task_dag.rem_data[0]: {v0.task_dag.rem_data[0]/1e6:.3f}MB")
print(f"    task_dag.exec_locations[0]: {v0.task_dag.exec_locations[0]}")

print(f"\n队列状态:")
print(f"  veh_cpu_q[0]长度: {len(env.veh_cpu_q[0])}")
print(f"  RSU队列总长度: {sum(len(q) for proc_dict in env.rsu_cpu_q.values() for q in proc_dict.values())}")
for rid, proc_dict in env.rsu_cpu_q.items():
    for pid, q in proc_dict.items():
        if len(q) > 0:
            print(f"    RSU[{rid}] proc[{pid}]: {len(q)}个任务")

print(f"\n传输队列状态:")
print(f"  txq_v2i总长度: {sum(len(q) for q in env.txq_v2i.values())}")
for key, q in env.txq_v2i.items():
    if len(q) > 0:
        print(f"    {key}: {len(q)}个任务")

print(f"\n决策统计: {env._decision_counts}")

# 再执行几步，看传输是否完成
print(f"\n继续执行5步...")
for i in range(5):
    actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    rsu_queue_len = sum(len(q) for proc_dict in env.rsu_cpu_q.values() for q in proc_dict.values())
    tx_queue_len = sum(len(q) for q in env.txq_v2i.values())
    
    print(f"  Step {i+1}: RSU队列={rsu_queue_len}, 传输队列={tx_queue_len}, status[0]={v0.task_dag.status[0]}")

print("\n" + "="*80)

