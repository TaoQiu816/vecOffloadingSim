#!/usr/bin/env python3
import numpy as np
from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv

env = VecOffloadingEnv()
obs_list, _ = env.reset(seed=42)

v = env.vehicles[0]
print(f"车辆 {v.id}:")
print(f"  任务节点数: {v.task_dag.num_subtasks}")
print(f"  Top priority: {v.task_dag.get_top_priority_task()}")
print(f"  status[0]: {v.task_dag.status[0]}")
print(f"  exec_locations[0]: {v.exec_locations[0]}")

# 尝试assign
subtask_id = 0
target = 'Local'
print(f"\n尝试assign_task({subtask_id}, '{target}'):")
print(f"  检查1 - 越界: {subtask_id} < 0 or >= {v.task_dag.num_subtasks}? {subtask_id < 0 or subtask_id >= v.task_dag.num_subtasks}")
print(f"  检查2 - 重复分配: exec_locations[{subtask_id}] is None? {v.exec_locations[subtask_id] is None}")
print(f"  检查3 - 状态检查: status[{subtask_id}] == 1? {v.task_dag.status[subtask_id] == 1}")

result = v.task_dag.assign_task(subtask_id, target)
print(f"  结果: {result}")
print(f"  执行后 exec_locations[0]: {v.exec_locations[0]}")
print(f"  执行后 status[0]: {v.task_dag.status[0]}")

