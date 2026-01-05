"""诊断传输进度"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("传输进度诊断")
print("="*80)

# 执行RSU动作
actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
next_obs, rewards, terminated, truncated, info = env.step(actions)

print(f"\n第一步后:")
print(f"  传输队列长度: {sum(len(q) for q in env.txq_v2i.values())}")

# 检查第一个传输任务
v0_key = ('VEH', 0)
if v0_key in env.txq_v2i and len(env.txq_v2i[v0_key]) > 0:
    job = env.txq_v2i[v0_key][0]
    print(f"\n  Vehicle 0的传输任务:")
    print(f"    owner_vehicle_id: {job.owner_vehicle_id}")
    print(f"    subtask_id: {job.subtask_id}")
    print(f"    rem_bytes: {job.rem_bytes/1e6:.3f}MB")
    print(f"    dst_node: {job.dst_node}")
    print(f"    start_time: {job.start_time}")
    print(f"    step_bytes_sent: {job.step_bytes_sent/1e6:.3f}MB")

# 再执行几步，观察进度
print(f"\n继续执行步骤:")
for i in range(10):
    actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    if v0_key in env.txq_v2i and len(env.txq_v2i[v0_key]) > 0:
        job = env.txq_v2i[v0_key][0]
        print(f"  Step {i+2}: rem_bytes={job.rem_bytes/1e6:.3f}MB, step_bytes_sent={job.step_bytes_sent/1e6:.3f}MB")
    else:
        print(f"  Step {i+2}: 传输队列已空")
        break

# 检查RSU队列
print(f"\n最终状态:")
print(f"  传输队列长度: {sum(len(q) for q in env.txq_v2i.values())}")
print(f"  RSU队列长度: {sum(len(q) for proc_dict in env.rsu_cpu_q.values() for q in proc_dict.values())}")
print(f"  Vehicle 0 status[0]: {env.vehicles[0].task_dag.status[0]}")

print("\n" + "="*80)

