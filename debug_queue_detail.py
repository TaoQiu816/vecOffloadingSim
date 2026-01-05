"""详细检查队列和任务执行"""
import sys
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg

# 固定seed
np.random.seed(42)

# 创建环境
env = VecOffloadingEnv(Cfg)
obs_list = env.reset(seed=42)

print("="*80)
print("队列和任务执行详细诊断")
print("="*80)

v = env.vehicles[0]
print(f"\n1. 初始状态:")
print(f"   车辆ID: {v.id}")
print(f"   任务子任务数: {v.task_dag.num_subtasks}")
print(f"   Top priority task: {v.task_dag.get_top_priority_task()}")
print(f"   CPU频率: {v.cpu_freq/1e9:.2f} GHz")

# 执行一个Local动作
actions = [{"target": 0, "power": 1.0}] * env.config.NUM_VEHICLES
print(f"\n2. 执行Local动作...")

# 执行step
next_obs, rewards, terminated, truncated, info = env.step(actions)

print(f"\n3. Step后状态:")
print(f"   Reward: {rewards[0]:.6f}")
print(f"   exec_locations[0]: {v.task_dag.exec_locations[0]}")
print(f"   status[0]: {v.task_dag.status[0]}")
print(f"   total_comp[0]: {v.task_dag.total_comp[0]/1e9:.3f}G cycles")
print(f"   rem_comp[0]: {v.task_dag.rem_comp[0]/1e9:.3f}G cycles")
print(f"   rem_data[0]: {v.task_dag.rem_data[0]/1e6:.3f}MB")

# 检查队列
print(f"\n4. 队列详情:")
veh_queue = env.veh_cpu_q[v.id]
print(f"   veh_cpu_q[{v.id}] 长度: {len(veh_queue)}")

if len(veh_queue) > 0:
    job = veh_queue[0]
    print(f"   队头任务:")
    print(f"     owner_vehicle_id: {job.owner_vehicle_id}")
    print(f"     subtask_id: {job.subtask_id}")
    print(f"     rem_cycles: {job.rem_cycles/1e9:.3f}G cycles")
    print(f"     exec_node: {job.exec_node}")
    print(f"     start_time: {job.start_time}")
    print(f"     finish_time: {job.finish_time}")
    print(f"     enqueue_time: {job.enqueue_time:.3f}s")

# 检查active_tasks
print(f"\n5. Active任务:")
print(f"   active_task_manager.get_num_active_tasks(): {v.active_task_manager.get_num_active_tasks()}")

# 再执行一步，看队列是否被处理
print(f"\n6. 再执行一步...")
next_obs, rewards, terminated, truncated, info = env.step(actions)

print(f"\n7. 第二步后状态:")
print(f"   Reward: {rewards[0]:.6f}")
print(f"   status[0]: {v.task_dag.status[0]}")
print(f"   veh_cpu_q[{v.id}] 长度: {len(veh_queue)}")

if len(veh_queue) > 0:
    job = veh_queue[0]
    print(f"   队头任务 rem_cycles: {job.rem_cycles/1e9:.3f}G cycles")
    print(f"   队头任务 start_time: {job.start_time}")
    print(f"   队头任务 step_cycles_done: {job.step_cycles_done/1e9:.3f}G cycles")
    print(f"   队头任务 step_time_used: {job.step_time_used:.6f}s")

# 检查CPU服务是否被调用
print(f"\n8. 环境时间:")
print(f"   env.time: {env.time:.3f}s")
print(f"   env.config.DT: {env.config.DT:.3f}s")

# 多步执行，看任务是否完成
print(f"\n9. 连续执行10步...")
for i in range(10):
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    if i % 2 == 0:
        print(f"   Step {i+2}: queue_len={len(veh_queue)}, status[0]={v.task_dag.status[0]}, reward={rewards[0]:.6f}")

print(f"\n10. 最终状态:")
print(f"   veh_cpu_q[{v.id}] 长度: {len(veh_queue)}")
print(f"   status[0]: {v.task_dag.status[0]}")
print(f"   active_tasks: {v.active_task_manager.get_num_active_tasks()}")

# 检查P2统计
if hasattr(env, '_p2_active_time'):
    print(f"\n11. P2统计:")
    print(f"   _p2_active_time: {env._p2_active_time:.6f}s")
    print(f"   _p2_idle_time: {env._p2_idle_time:.6f}s")
    print(f"   _p2_deltaW_active: {env._p2_deltaW_active/1e9:.6f}G cycles")

print("\n" + "="*80)

