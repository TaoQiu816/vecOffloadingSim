"""诊断P2统计问题"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg

np.random.seed(42)
env = VecOffloadingEnv(Cfg)
obs_list = env.reset(seed=42)

print("="*80)
print("P2统计诊断")
print("="*80)

# 执行Local动作
actions = [{"target": 0, "power": 1.0}] * env.config.NUM_VEHICLES

for step in range(5):
    print(f"\n--- Step {step} ---")
    
    # 执行前状态
    W_before = env._get_total_W_remaining()
    active_before = env._get_total_active_tasks()
    
    print(f"执行前:")
    print(f"  W_remaining: {W_before/1e9:.3f}G cycles")
    print(f"  active_tasks: {active_before}")
    print(f"  veh_cpu_q[0]长度: {len(env.veh_cpu_q[0])}")
    if len(env.veh_cpu_q[0]) > 0:
        job = env.veh_cpu_q[0][0]
        print(f"  队头任务 rem_cycles: {job.rem_cycles/1e9:.3f}G")
    
    # 执行step
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    # 执行后状态
    W_after = env._get_total_W_remaining()
    active_after = env._get_total_active_tasks()
    deltaW = W_before - W_after
    
    print(f"执行后:")
    print(f"  W_remaining: {W_after/1e9:.3f}G cycles")
    print(f"  active_tasks: {active_after}")
    print(f"  deltaW: {deltaW/1e9:.3f}G cycles")
    print(f"  veh_cpu_q[0]长度: {len(env.veh_cpu_q[0])}")
    if len(env.veh_cpu_q[0]) > 0:
        job = env.veh_cpu_q[0][0]
        print(f"  队头任务 rem_cycles: {job.rem_cycles/1e9:.3f}G")
        print(f"  队头任务 step_cycles_done: {job.step_cycles_done/1e9:.3f}G")
    
    print(f"  P2统计:")
    print(f"    _p2_active_time: {env._p2_active_time:.3f}s")
    print(f"    _p2_idle_time: {env._p2_idle_time:.3f}s")
    print(f"    _p2_deltaW_active: {env._p2_deltaW_active/1e9:.3f}G")
    
    # 检查DAG状态
    v = env.vehicles[0]
    print(f"  DAG状态:")
    for i in range(min(3, v.task_dag.num_subtasks)):
        print(f"    subtask[{i}]: status={v.task_dag.status[i]}, rem_comp={v.task_dag.rem_comp[i]/1e9:.3f}G")

print("\n" + "="*80)

