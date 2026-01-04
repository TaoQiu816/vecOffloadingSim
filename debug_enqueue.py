#!/usr/bin/env python3
import numpy as np
from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv

env = VecOffloadingEnv()
obs_list, _ = env.reset(seed=42)

v = env.vehicles[0]
subtask_id = 0

# Assign
v.task_dag.assign_task(subtask_id, 'Local')
print(f"After assign_task:")
print(f"  exec_locations[{subtask_id}]: {v.task_dag.exec_locations[subtask_id]}")
print(f"  status[{subtask_id}]: {v.task_dag.status[subtask_id]}")
print(f"  rem_data[{subtask_id}]: {v.task_dag.rem_data[subtask_id]}")
print(f"  in_degree[{subtask_id}]: {v.task_dag.in_degree[subtask_id]}")

# 检查入队条件
print(f"\n检查入队条件:")
print(f"  1. exec_locations已确定? {v.task_dag.exec_locations[subtask_id] is not None}")
print(f"  2. input_ready (rem_data <= 1e-9)? {v.task_dag.rem_data[subtask_id] <= 1e-9}")
print(f"  3. in_degree == 0? {v.task_dag.in_degree[subtask_id] == 0}")
print(f"  4. status == 1? {v.task_dag.status[subtask_id] == 1}")
print(f"  5. status < 2? {v.task_dag.status[subtask_id] < 2}")

# 手动设置rem_data为0（模拟Local路径）
v.task_dag.rem_data[subtask_id] = 0.0
print(f"\n设置rem_data=0后:")
print(f"  rem_data[{subtask_id}]: {v.task_dag.rem_data[subtask_id]}")

# 再次尝试入队
enqueue_result = env._dag_handler._try_enqueue_compute_if_ready(
    v, subtask_id, env.time, env.veh_cpu_q, env.rsu_cpu_q, env.rsus
)
print(f"\n入队结果: {enqueue_result}")
print(f"队列长度: {len(env.veh_cpu_q[v.id])}")

