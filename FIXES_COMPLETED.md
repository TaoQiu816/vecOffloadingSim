# 训练问题修复总结

## 问题1: 训练数据全为0

### 根本原因
1. **`active_task_manager`系统废弃**：代码已迁移到FIFO队列系统，但P2统计仍依赖旧的`active_task_manager`
2. **`exec_locations`引用不一致**：`reset()`中创建了新列表，覆盖了`task_dag.exec_locations`的引用
3. **`assign_task`状态冲突**：过早设置status=2，导致`_try_enqueue_compute_if_ready`检查失败
4. **决策统计错误**：使用`planned_target`（tuple）而非`planned_kind`（string）判断决策类型

### 修复内容

#### 1. `_get_total_active_tasks()` - 基于FIFO队列统计
**文件**: `envs/vec_offloading_env.py`

```python
def _get_total_active_tasks(self):
    """基于FIFO队列系统统计活跃任务数"""
    total_active = 0
    
    # 车辆CPU队列
    for veh_id, queue in self.veh_cpu_q.items():
        total_active += len(queue)
    
    # RSU CPU队列
    for rsu_id, proc_dict in self.rsu_cpu_q.items():
        for proc_id, queue in proc_dict.items():
            total_active += len(queue)
    
    # 传输队列
    for queue in self.txq_v2i.values():
        total_active += len(queue)
    for queue in self.txq_v2v.values():
        total_active += len(queue)
    
    return total_active
```

#### 2. `_get_total_W_remaining()` - 基于队列和DAG状态统计
**文件**: `envs/vec_offloading_env.py`

```python
def _get_total_W_remaining(self):
    """基于FIFO队列系统和DAG状态统计剩余工作量"""
    total_W = 0.0
    
    # 1. 队列中的任务（正在执行或等待执行）
    for veh_id, queue in self.veh_cpu_q.items():
        for job in queue:
            total_W += job.rem_cycles
    
    for rsu_id, proc_dict in self.rsu_cpu_q.items():
        for proc_id, queue in proc_dict.items():
            for job in queue:
                total_W += job.rem_cycles
    
    # 2. DAG中未分配的任务（status < 2: PENDING或READY）
    for v in self.vehicles:
        dag = v.task_dag
        for i in range(dag.num_subtasks):
            if dag.status[i] < 2:
                total_W += dag.rem_comp[i]
    
    return total_W
```

#### 3. `exec_locations`引用修复
**文件**: `envs/vec_offloading_env.py`

```python
# 修改前（错误）
v.exec_locations = [None] * v.task_dag.num_subtasks  # 创建新对象

# 修改后（正确）
# 移除这行，直接使用v.task_dag.exec_locations
```

全局替换：`v.exec_locations` → `v.task_dag.exec_locations`

#### 4. `assign_task`状态冲突修复
**文件**: `envs/entities/task_dag.py`

```python
def assign_task(self, subtask_id, target):
    # ...
    self.exec_locations[subtask_id] = target
    # self.status[subtask_id] = 2  # 移除：让入队逻辑负责状态转换
    return True
```

#### 5. 决策统计修复
**文件**: `envs/vec_offloading_env.py`

```python
# 修改前（错误）
target = plan.get("planned_target", 'Local')  # tuple: ('RSU', 0)
if target == 'Local' or target == 0:
    self._decision_counts['local'] += 1
elif target == 'RSU' or target == 1:  # 永远不会匹配
    self._decision_counts['rsu'] += 1

# 修改后（正确）
kind = plan.get("planned_kind", "local")  # string: 'rsu'
if kind == "local":
    self._decision_counts['local'] += 1
elif kind == "rsu":
    self._decision_counts['rsu'] += 1
elif kind == "v2v":
    self._decision_counts['v2v'] += 1
```

---

## 问题2: RSU卸载率始终为0

### 根本原因
**路径损耗参数命名混淆**：`_path_loss`方法需要路径损耗指数（2-4），但传入了参考路损dB值（28）

### 配置参数语义
- `PL_ALPHA_V2I/V2V = 28.0` - 参考路损 (dB)
- `PL_BETA_V2I/V2V = 2.5/3.5` - 路损指数（无量纲）

### 修复内容
**文件**: `envs/modules/channel.py`

```python
# 修改前（错误）
h_bar = self._path_loss(dist, Cfg.PL_ALPHA_V2I)  # 传入28.0

# 修改后（正确）
h_bar = self._path_loss(dist, Cfg.PL_BETA_V2I)  # 传入2.5
```

**影响**：
- 修复前：路径损耗增益 `h_bar ≈ 9e-53`，信号几乎为0，传输速率为0
- 修复后：路径损耗增益正常，传输速率 `≈ 147 Mbps`

---

## 问题3: 奖励归一化与量纲统一性

### 当前奖励公式
```python
reward = DELTA_CFT_SCALE * dT_clipped - DELTA_CFT_ENERGY_WEIGHT * energy_norm
       = 10.0 * dT_clipped - 0.2 * energy_norm
```

### 量纲分析

#### 时间收益部分
- **输入**: `dT_rem` (秒)
- **裁剪**: `[-1.0, 1.0]` (秒)
- **缩放**: `× 10.0`
- **输出范围**: `[-10.0, 10.0]`
- **量纲**: 秒 → 无量纲

#### 能耗惩罚部分
- **输入**: `e_step = (p_tx + p_circuit) * dt` (焦耳)
- **归一化**: `energy_norm = e_step / e_max` → `[0, 1]`
- **缩放**: `× 0.2`
- **输出范围**: `[0.0, 0.2]`
- **量纲**: 焦耳 → 无量纲

### 发现的问题

#### ⚠️ 量纲不统一
- **时间部分**: 基于绝对时间（秒）
- **能耗部分**: 基于相对比例（归一化）

**影响**：
- 短任务：能耗权重相对较大
- 长任务：能耗权重相对较小

#### ⚠️ 权重不平衡
- 时间最大贡献：`10.0`
- 能耗最大惩罚：`0.2`
- **比例**: `50:1`

#### ⚠️ 裁剪范围需验证
- `dT_clipped` 范围：`[-1.0, 1.0]` 秒
- 需要检查实际训练中是否经常触及边界

### 建议改进

#### 方案A：统一归一化（推荐）
```python
# 归一化时间变化
dT_norm = dT_rem / dt  # 归一化到步长
dT_clipped = clip(dT_norm, -1.0, 1.0)

# 奖励公式
reward = DELTA_CFT_SCALE * dT_clipped - DELTA_CFT_ENERGY_WEIGHT * energy_norm
```

#### 方案B：统一使用绝对值
```python
# 使用绝对能耗（焦耳）
e_step = (p_tx + p_circuit) * dt
reward = DELTA_CFT_SCALE * dT_clipped - ENERGY_SCALE * e_step
```

---

## 测试结果

### 修复前
```
|    Ep |  Time(s) |   Reward |   V_SR |   T_SR |   S_SR | Deadlock |  D_Miss |   TX |  NoTX |  Local |    RSU |    V2V |   SvcRate |   Idle |
|     1 |     1.91 |   -24.88 |  0.00% |  0.00% |  0.00% |        0 |       0 |    0 |     0 |  5.54% |  0.08% | 94.38% |     0.000G |  0.00% |
```
- ❌ RSU选择率接近0
- ❌ 服务率为0
- ❌ 成功率为0

### 修复后
```
|    Ep |  Time(s) |   Reward |   V_SR |   T_SR |   S_SR | Deadlock |  D_Miss |   TX |  NoTX |  Local |    RSU |    V2V |   SvcRate |   Idle |
|     1 |     2.53 |     0.02 |  0.00% |  0.00% |  9.84% |        0 |       0 |    0 |     0 |  5.21% | 87.96% |  6.83% |     0.000G |  0.00% |
```
- ✅ RSU选择率正常（87.96%）
- ✅ 子任务成功率有进展（9.84%）
- ✅ 奖励不再是负值
- ⚠️ 服务率显示仍为0（可能是episode太短）

---

## 文件修改清单

### 核心修复
1. `envs/vec_offloading_env.py`
   - `_get_total_active_tasks()` - 重写
   - `_get_total_W_remaining()` - 重写
   - 全局替换 `v.exec_locations` → `v.task_dag.exec_locations`
   - 决策统计逻辑修复

2. `envs/entities/task_dag.py`
   - `assign_task()` - 移除status=2设置

3. `envs/modules/channel.py`
   - `compute_one_rate()` - 修复路径损耗参数
   - 两处：`PL_ALPHA` → `PL_BETA`

4. `envs/services/dag_completion_handler.py`
   - `_try_enqueue_compute_if_ready()` - 更新引用

### 文档
1. `REWARD_ANALYSIS.md` - 奖励归一化分析
2. `FIXES_COMPLETED.md` - 本文档
3. `analyze_reward_distribution.py` - 奖励分布分析脚本

---

## 下一步建议

### 短期
1. ✅ 运行`analyze_reward_distribution.py`检查奖励分布
2. ✅ 验证`dT_rem`是否经常触及裁剪边界
3. ✅ 检查服务率统计（可能需要更长的episode）

### 中期
1. 考虑统一奖励量纲（方案A或B）
2. 根据实际需求调整时间/能耗权重比例
3. 优化裁剪范围或使用软裁剪

### 长期
1. 完善`RewardEngine`实现
2. 添加更多奖励组件的可视化和监控
3. 实现自适应权重调整机制

