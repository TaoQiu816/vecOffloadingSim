#!/usr/bin/env python3
import numpy as np
from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv

env = VecOffloadingEnv()
obs_list, _ = env.reset(seed=42)

v = env.vehicles[0]
print(f"检查exec_locations:")
print(f"  v.task_dag.exec_locations[0]: {v.task_dag.exec_locations[0]}")

# 尝试assign
result = v.task_dag.assign_task(0, 'Local')
print(f"\nassign_task返回: {result}")
print(f"  v.task_dag.exec_locations[0]: {v.task_dag.exec_locations[0]}")

# 尝试入队
print(f"\n尝试入计算队列:")
enqueue_result = env._dag_handler._try_enqueue_compute_if_ready(
    v, 0, env.time, env.veh_cpu_q, env.rsu_cpu_q, env.rsus
)
print(f"  入队结果: {enqueue_result}")
print(f"  队列长度: {len(env.veh_cpu_q[v.id])}")
print(f"  active_tasks: {v.active_task_manager.get_num_active_tasks()}")
