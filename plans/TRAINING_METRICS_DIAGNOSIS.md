# 训练过程指标统计诊断报告

## 问题描述

根据训练日志输出，发现以下指标显示异常：
```
Ep    Wall    R/step     T_SR     V_SR     S_SR            L/R/V   Lat(s)       En     Miss      Ill      Ent       KL     Clip
     1    13.8s   -0.0108     5.0%     5.0%    51.0%      33%/14%/53%    0.000        -     0.0%     0.0%   2.3285   0.0041     9.7%
```

**关键问题**：
1. `Lat(s)` 显示为 `0.000` - 任务完成时间异常为0
2. `En` 显示为 `-` - 能耗指标缺失
3. 部分物理指标可能未正确记录

## 诊断分析

### 1. 代码架构检查

#### 1.1 指标记录流程
```
train.py (主循环)
  ├─> env.step() → 返回 info 字典
  ├─> 从 info 提取 episode_metrics
  ├─> 写入 training_stats.csv
  └─> 写入 TensorBoard
```

#### 1.2 关键文件
- [`train.py`](../train.py) - 主训练循环，行3487-3629记录training_stats
- [`envs/vec_offloading_env.py`](../envs/vec_offloading_env.py) - 环境统计收集
- [`utils/data_recorder.py`](../utils/data_recorder.py) - 数据记录和可视化

### 2. 指标缺失原因分析

#### 2.1 任务完成时间 (Lat/task_duration_mean)

**问题定位**：
- 控制台显示 `Lat(s)=0.000`
- 对应字段：`task_duration_mean` (training_stats.csv 第3528行)
- 数据来源：`env_stats.get("task_duration_mean")`

**根本原因**：
```python
# train.py 第2744行
task_duration_mean = env_stats.get("task_duration_mean") if env_stats else 0.0
```
环境未正确计算或返回 `task_duration_mean`。

**验证方法**：
检查 `envs/vec_offloading_env.py` 中 `_log_episode_stats()` 方法是否计算了该指标。

#### 2.2 能耗指标 (En/energy_norm_mean)

**问题定位**：
- 控制台显示 `En=-`
- 对应字段：`energy_norm_mean` (training_stats.csv 第3537行)
- 数据来源：`env_stats.get("energy_norm_mean")`

**根本原因**：
```python
# train.py 第2740行
energy_norm_mean = _pick_stat("energy_norm_mean", "energy_norm.mean")
```
环境的 reward_stats 或 episode_metrics 未记录能耗指标。

### 3. 完整性检查清单

#### 3.1 CSV文件字段 (training_stats.csv)

**必需字段**（TRAINING_STATS_FIELDS，train.py 第76-108行）：
- ✅ episode, steps, physical_steps, decision_steps
- ✅ reward_mean, reward_total, vehicle_sr, task_sr, subtask_sr
- ⚠️ task_duration_mean, task_duration_p95 - **可能缺失**
- ⚠️ energy_mean, energy_p95 - **可能缺失**
- ✅ decision_frac_local, decision_frac_rsu, decision_frac_v2v
- ✅ actor_loss, critic_loss, entropy, approx_kl, clip_frac

#### 3.2 环境统计模块

**StatsCollector** ([`envs/audit/stats_collector.py`](../envs/audit/stats_collector.py))：
- ✅ 基础统计：decisions, tasks_completed, dags_completed
- ⚠️ **缺失**：task_duration 统计
- ⚠️ **缺失**：energy_norm 统计

**环境 episode_metrics**：
需要在 `vec_offloading_env.py` 的 `_log_episode_stats()` 中添加：
```python
"task_duration_mean": float,  # 已完成任务的平均完成时间
"task_duration_p95": float,   # P95完成时间
"energy_norm_mean": float,    # 归一化能耗均值
"energy_norm_p95": float,     # P95能耗
```

## 修复方案

### 方案1：环境统计增强（推荐）

**目标**：在环境层面完整记录物理指标

**实施步骤**：

1. **增强 StatsCollector**
```python
# envs/audit/stats_collector.py
def reset_episode_stats(self):
    self.episode_stats = {
        # ... 现有字段 ...
        "task_durations": [],      # 新增：任务完成时间列表
        "energy_values": [],       # 新增：能耗值列表
    }

def on_task_complete(self, vehicle_id: int, subtask_id: int, 
                     makespan: float, energy: float):
    if not self.enabled:
        return
    self.episode_stats["tasks_completed"] += 1
    self.episode_stats["task_durations"].append(makespan)
    self.episode_stats["energy_values"].append(energy)

def get_episode_stats(self) -> Dict[str, Any]:
    stats = self.episode_stats.copy()
    
    # 计算任务完成时间统计
    if stats["task_durations"]:
        stats["task_duration_mean"] = np.mean(stats["task_durations"])
        stats["task_duration_p95"] = np.percentile(stats["task_durations"], 95)
    else:
        stats["task_duration_mean"] = 0.0
        stats["task_duration_p95"] = 0.0
    
    # 计算能耗统计
    if stats["energy_values"]:
        stats["energy_norm_mean"] = np.mean(stats["energy_values"])
        stats["energy_norm_p95"] = np.percentile(stats["energy_values"], 95)
    else:
        stats["energy_norm_mean"] = 0.0
        stats["energy_norm_p95"] = 0.0
    
    return stats
```

2. **环境调用统计收集器**
```python
# envs/vec_offloading_env.py
def _log_episode_stats(self, terminated, truncated):
    # ... 现有逻辑 ...
    
    # 收集已完成任务的物理指标
    task_durations = []
    energy_values = []
    for v in self.vehicles:
        if v.task_dag.is_finished and not v.task_dag.is_failed:
            # 计算任务完成时间
            duration = v.task_dag.completion_time - v.task_dag.arrival_time
            task_durations.append(duration)
            
            # 计算能耗（如果有记录）
            if hasattr(v.task_dag, 'total_energy'):
                energy_values.append(v.task_dag.total_energy)
    
    # 添加到 episode_metrics
    if task_durations:
        episode_metrics["task_duration_mean"] = float(np.mean(task_durations))
        episode_metrics["task_duration_p95"] = float(np.percentile(task_durations, 95))
        episode_metrics["completed_tasks_count"] = len(task_durations)
    else:
        episode_metrics["task_duration_mean"] = 0.0
        episode_metrics["task_duration_p95"] = 0.0
        episode_metrics["completed_tasks_count"] = 0
    
    if energy_values:
        episode_metrics["energy_norm_mean"] = float(np.mean(energy_values))
        episode_metrics["energy_norm_p95"] = float(np.percentile(energy_values, 95))
```

### 方案2：训练循环补充计算（临时方案）

**目标**：在 train.py 中从现有数据推导缺失指标

**实施步骤**：

```python
# train.py 第2740行附近
# 如果环境未提供，从其他指标推导
if task_duration_mean is None or task_duration_mean == 0.0:
    # 使用 mean_cft_completed 作为替代
    task_duration_mean = mean_cft_completed if mean_cft_completed is not None else 0.0

if energy_norm_mean is None:
    # 从 reward_stats 中提取能耗组件
    energy_norm_mean = env_metrics.get("energy_norm.mean", 0.0)
```

### 方案3：数据后处理（最后手段）

**目标**：训练后修复CSV文件中的缺失值

**实施步骤**：

```python
# scripts/fix_missing_metrics.py
import pandas as pd
import numpy as np

def fix_training_stats(csv_path):
    df = pd.read_csv(csv_path)
    
    # 修复 task_duration_mean：使用 mean_cft_completed
    if 'task_duration_mean' in df.columns and 'mean_cft_completed' in df.columns:
        mask = (df['task_duration_mean'] == 0.0) | df['task_duration_mean'].isna()
        df.loc[mask, 'task_duration_mean'] = df.loc[mask, 'mean_cft_completed']
    
    # 修复 energy_norm_mean：使用前后值插值
    if 'energy_norm_mean' in df.columns:
        df['energy_norm_mean'] = df['energy_norm_mean'].replace(0.0, np.nan)
        df['energy_norm_mean'] = df['energy_norm_mean'].interpolate(method='linear')
    
    df.to_csv(csv_path, index=False)
    print(f"Fixed metrics in {csv_path}")
```

## 验证方案

### 1. 单元测试

```python
# tests/test_metrics_collection.py
def test_task_duration_collection():
    """验证任务完成时间是否正确记录"""
    env = VecOffloadingEnv()
    obs, _ = env.reset()
    
    # 运行一个完整episode
    for _ in range(100):
        actions = [env.action_space.sample() for _ in range(env.num_vehicles)]
        obs, rewards, done, truncated, info = env.step(actions)
        if done or truncated:
            break
    
    # 检查 episode_metrics
    assert "task_duration_mean" in info.get("episode_metrics", {})
    assert info["episode_metrics"]["task_duration_mean"] > 0.0

def test_energy_collection():
    """验证能耗指标是否正确记录"""
    env = VecOffloadingEnv()
    obs, _ = env.reset()
    
    # ... 同上 ...
    
    assert "energy_norm_mean" in info.get("episode_metrics", {})
```

### 2. 集成测试

```bash
# 运行短期训练验证
python train.py --max-episodes 10 --seed 42

# 检查输出
tail -n 20 runs/run_*/logs/training_stats.csv | grep -E "task_duration_mean|energy_norm_mean"
```

### 3. 日志验证

**预期输出**：
```
Ep    Wall    R/step     T_SR     V_SR     S_SR            L/R/V   Lat(s)       En     Miss      Ill      Ent       KL     Clip
     1    13.8s   -0.0108     5.0%     5.0%    51.0%      33%/14%/53%    2.345    0.123     0.0%     0.0%   2.3285   0.0041     9.7%
                                                                          ^^^^     ^^^^
                                                                          应显示具体数值
```

## 优先级建议

### 高优先级（必须修复）
1. ✅ **task_duration_mean** - 任务完成时间是核心性能指标
2. ✅ **energy_norm_mean** - 能耗是优化目标之一
3. ✅ **completed_tasks_count** - 用于验证统计有效性

### 中优先级（建议修复）
4. ⚠️ **deadline_miss_rate** - 约束满足情况
5. ⚠️ **illegal_action_rate** - 策略合法性
6. ⚠️ **v2v_link_break_rate** - 通信可靠性

### 低优先级（可选）
7. ℹ️ **oracle_match_rate** - 策略质量诊断
8. ℹ️ **action_regret_mean** - 决策后悔度

## 实施时间线

### 第1阶段：紧急修复（1-2小时）
- [ ] 实施方案2（训练循环补充计算）
- [ ] 验证下一次训练的日志输出
- [ ] 确认 CSV 文件字段完整

### 第2阶段：根本修复（3-4小时）
- [ ] 实施方案1（环境统计增强）
- [ ] 添加单元测试
- [ ] 更新文档

### 第3阶段：数据修复（1小时）
- [ ] 实施方案3（后处理脚本）
- [ ] 修复历史训练数据
- [ ] 重新生成可视化图表

## 附录：完整字段清单

### training_stats.csv 必需字段（108个）

**基础信息**（9个）：
- episode, steps, physical_steps, decision_steps, active_decision_ratio
- wall_time, sim_time, termination_reason_raw, termination_reason_bucket

**奖励指标**（13个）：
- reward_mean, reward_total, episode_reward, reward_p95, reward_abs_mean
- mean_r_prog, mean_r_term, mean_cost_power, mean_cost_trust
- abs_ratio_r_time, abs_ratio_r_energy, abs_ratio_r_interf, abs_ratio_r_term

**成功率**（3个）：
- vehicle_sr, task_sr, subtask_sr

**物理性能**（6个）：
- ⚠️ task_duration_mean, task_duration_p95, completed_tasks
- mean_cft_completed, episode_time_seconds
- ⚠️ energy_mean, energy_p95, t_tx_mean, dT_eff_mean

**约束与安全**（7个）：
- deadline_misses, deadline_miss_rate
- time_limit_rate, illegal_action_rate, illegal_action_ratio
- no_task_rate, on_task_rate, has_task_available_rate
- unified_illegal_trigger_rate, hard_trigger_rate, v2v_link_break_rate

**决策分布**（6个）：
- ratio_local, ratio_rsu, ratio_v2v
- decision_frac_local, decision_frac_rsu, decision_frac_v2v

**系统负载**（10个）：
- avg_power, avg_rsu_queue, rsu_queue_p95
- power_ratio_mean, power_ratio_p95
- I_total_p50, I_total_p95, I_caused_mean, I_caused_p95
- tx_created, same_node_no_tx, service_rate_ghz, idle_fraction

**训练诊断**（20个）：
- actor_loss, critic_loss, critic_loss_raw_mean, normalized_value_loss
- entropy, approx_kl, clip_frac
- cost_power_value_loss, cost_trust_value_loss, lambda_power, lambda_trust
- grad_norm, grad_norm_preclip, grad_norm_postclip
- active_ratio, actor_update_active_frac, value_clip_fraction
- critic_loss_active, critic_loss_inactive
- mode_aux_loss, mode_aux_acc
- ppo_epochs_executed, num_minibatches_executed
- mb_kl_max, mb_kl_p95
- early_stop_epoch_idx, early_stop_batch_idx
- skipped_update_count, early_stop, lr

**Bias状态**（2个）：
- bias_rsu, bias_local

**其他**（32个）：
- reward_clip_hit_count, reward_clip_hit_rate, abs_ratio_basis
- r_prog_on_task_mean, r_prog_on_task_abs_mean
- abs_ratio_on_task_r_prog, abs_ratio_on_task_r_term
- ... (详见 TRAINING_STATS_FIELDS)

---

**总结**：训练过程的指标统计架构完整，但环境层面的物理指标收集存在缺失。建议优先实施方案1（环境统计增强），确保所有关键指标都能正确记录和显示。
