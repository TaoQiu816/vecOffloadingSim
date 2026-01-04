# 训练问题修复总结

## 问题诊断

训练时所有指标为0，包括：
- 成功率：0%
- 服务率：0 G/s  
- 空闲率：100%
- TX任务：0
- Deadlock：0

## 根本原因

1. **`assign_task`过早改变status**
   - `assign_task`将status从1(READY)改为2(RUNNING)
   - 但`_try_enqueue_compute_if_ready`要求status==1
   - 导致任务永远无法入队

2. **`exec_locations`引用不一致**
   - `reset()`中创建了新的`v.exec_locations`列表
   - 覆盖了`v.task_dag.exec_locations`的引用
   - 导致`assign_task`写入的位置无法被其他模块读取

3. **缺失方法**
   - `_log_episode_stats`未实现
   - `_record_illegal_action`未实现

## 修复内容

### 1. task_dag.py
```python
# 修改前
self.exec_locations[subtask_id] = target
self.status[subtask_id] = 2  # READY -> RUNNING

# 修改后  
self.exec_locations[subtask_id] = target
# 不在这里改变status，让入队逻辑负责状态转换
```

### 2. vec_offloading_env.py
```python
# 修改前
v.exec_locations = [None] * v.task_dag.num_subtasks

# 修改后
# 移除这行，直接使用v.task_dag.exec_locations
```

### 3. 全局替换
- `v.exec_locations` → `v.task_dag.exec_locations`
- `vehicle.exec_locations` → `vehicle.task_dag.exec_locations`
- 影响文件：
  - `envs/vec_offloading_env.py`
  - `envs/services/dag_completion_handler.py`

### 4. 添加缺失方法

#### `_log_episode_stats()`
- 计算成功率、子任务成功率
- 统计决策分布
- 计算P2性能指标（服务率、空闲率）
- 统计死锁、传输任务等
- 写入JSONL文件

#### 替换`_record_illegal_action()`
```python
# 修改前
r = self._record_illegal_action(i, v.illegal_reason)

# 修改后
r = self.config.REWARD_MIN
```

## 测试结果

修复后：
- ✅ 任务可以成功分配（`exec_locations`正确设置）
- ✅ 任务可以入队（`veh_cpu_q`长度>0）
- ⚠️  任务尚未被激活执行（需要进一步检查`CpuQueueService`）

## 下一步

如果任务仍未执行，需要检查：
1. `CpuQueueService.step()`实现
2. 队列到`active_tasks`的转换逻辑
3. Phase4的推进机制

## 文件清单

修改的文件：
- `envs/entities/task_dag.py`
- `envs/vec_offloading_env.py`
- `envs/services/dag_completion_handler.py`

新增的诊断脚本：
- `diagnose_training.py`
- `debug_assign.py`
- `debug_exec_loc.py`
- `debug_enqueue.py`

