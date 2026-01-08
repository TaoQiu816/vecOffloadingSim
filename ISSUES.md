# 代码重构计划 (Code Refactoring Plan)

> 创建时间: 2026-01-08
> 目标: 提升代码可维护性，不改变现有逻辑

---

## 一、文件规模分析 (File Size Analysis)

| 文件 | 行数 | 优先级 | 状态 |
|------|------|--------|------|
| `envs/vec_offloading_env.py` | 3288 | P0-最高 | 部分重构(-181行) |
| `train.py` | 1655 | P1-高 | 部分重构(-111行) |
| `utils/data_recorder.py` | 1078 | P2-中 | 待评估 |
| `generate_all_plots.py` | 677 | P3-低 | 独立脚本 |
| `models/actor_critic.py` | 589 | - | 暂不处理 |
| `envs/entities/task_dag.py` | 578 | - | 暂不处理 |

---

## 二、vec_offloading_env.py 重构方案 (P0)

### 2.1 当前结构问题

**问题1: 数据类定义应独立**
- `TransferJob` (lines 26-75) - 通信任务数据结构
- `ComputeJob` (lines 77-111) - 计算任务数据结构
- 建议: 移至 `envs/jobs/__init__.py`

**问题2: 独立函数应模块化**
- `compute_absolute_reward()` (lines 113-145) - 奖励计算函数
- 建议: 移至 `envs/rl/reward_engine.py`

**问题3: 主类VecOffloadingEnv过大 (~3300行)**

### 2.2 方法分类与提取方案

#### A. 队列管理方法 → `envs/modules/queue_helpers.py` (新建)
```python
# 待提取方法:
_get_veh_queue_load()      # line 364
_get_veh_queue_wait_time() # line 374
_get_rsu_queue_load()      # line 388
_get_rsu_queue_wait_time() # line 411
_is_veh_queue_full()       # line 437
_is_rsu_queue_full()       # line 446
_get_node_delay()          # line 466
```

#### B. RSU管理方法 → 整合到 `envs/entities/rsu.py`
```python
# 待提取方法:
_init_rsus()              # line 699
_get_nearest_rsu()        # line 751
_get_all_rsus_in_range()  # line 776
_select_best_rsu()        # line 788
_is_rsu_location()        # line 844
_get_rsu_id_from_location() # line 856
_get_rsu_position()       # line 873
```

#### C. CFT计算方法 → `envs/services/cft_calculator.py` (新建)
```python
# 待提取方法:
_compute_mean_cft_pi0()           # line 1004
_compute_vehicle_cfts_snapshot()  # line 1034
_calculate_global_cft_critical_path() # line 1912
_calc_critical_path_local()       # line 2663
```

#### D. 速率估算方法 → 整合到 `envs/modules/channel.py`
```python
# 待提取方法:
_estimate_rate()           # line 2842
_get_comm_rate()           # line 2537
_get_inter_task_comm_rate() # line 2541
_update_rate_norm()        # line 960
_get_norm_rate()           # line 965
```

#### E. 奖励计算方法 → 扩展 `envs/rl/reward_engine.py`
```python
# 待提取方法:
calculate_agent_reward()       # line 3188
_calculate_efficiency_gain()   # line 2701
_estimate_execution_time()     # line 2741
_calculate_congestion_penalty() # line 2872
_calculate_constraint_penalty() # line 2928
_compute_cost_components()     # line 3042
_clip_reward()                 # line 3172
```

#### F. 执行时间估算 → `envs/services/time_estimator.py` (新建)
```python
# 待提取方法:
_calculate_local_execution_time() # line 2631
_estimate_execution_time()        # line 2741
```

### 2.3 重构后目标结构

```
envs/
├── vec_offloading_env.py      # 核心环境类 (~1500行)
├── jobs/
│   └── __init__.py            # TransferJob, ComputeJob (新建)
├── modules/
│   ├── channel.py             # 通信模型 (扩展)
│   ├── queue_helpers.py       # 队列辅助函数 (新建)
│   └── ...
├── services/
│   ├── cft_calculator.py      # CFT计算服务 (新建)
│   ├── time_estimator.py      # 时间估算服务 (新建)
│   └── ...
├── rl/
│   ├── reward_engine.py       # 奖励引擎 (扩展)
│   └── ...
└── entities/
    ├── rsu.py                 # RSU实体 (扩展)
    └── ...
```

---

## 三、train.py 重构方案 (P1)

### 3.1 当前结构问题

**问题1: 工具函数散落在文件头部**
- `_ensure_dir()`, `_read_last_jsonl()` (lines 58-77)
- `_format_table_*()` (lines 79-110)
- `_compute_time_limit_penalty()` (lines 111-127)
- 建议: 移至 `utils/train_helpers.py`

**问题2: 环境变量处理函数分散**
- `_env_int()`, `_env_float()`, `_env_bool()`, `_env_str()` (lines 146-228)
- `apply_env_overrides()` (lines 173-219)
- 建议: 移至 `utils/env_config.py`

**问题3: Baseline评估逻辑**
- `evaluate_baselines()` (lines 313-367)
- `evaluate_single_baseline_episode()` (lines 368-472)
- 建议: 移至 `baselines/evaluator.py`

**问题4: main()函数过长 (~1190行)**
- 建议: 拆分为多个子函数

### 3.2 重构后目标结构

```
train.py                       # 主入口 (~300行)
utils/
├── train_helpers.py           # 训练辅助函数 (新建)
├── env_config.py              # 环境变量配置 (新建)
└── ...
baselines/
├── evaluator.py               # Baseline评估器 (新建)
└── ...
```

---

## 四、data_recorder.py 重构方案 (P2)

### 4.1 当前结构问题

**问题1: 绘图方法过多 (10+个方法)**
- `auto_plot()` (line 187) - 主绘图入口
- `plot_agent_reward_distribution()` (line 398)
- `plot_latency_energy_tradeoff()` (line 448)
- `plot_performance_radar()` (line 495)
- `plot_resource_utilization()` (line 562)
- `plot_training_stability()` (line 608)
- `plot_completion_time_distribution()` (line 641)
- `plot_rsu_load_balance()` (line 683)
- `plot_episode_duration_analysis()` (line 715)
- `plot_reward_decomposition()` (line 753)
- `plot_success_rate_comparison()` (line 811)
- `plot_training_stats()` (line 853)

### 4.2 重构方案

**方案A: 提取绘图模块**
```
utils/
├── data_recorder.py           # 数据记录核心 (~300行)
├── plotting/
│   ├── __init__.py
│   ├── training_plots.py      # 训练相关图表
│   ├── performance_plots.py   # 性能分析图表
│   └── comparison_plots.py    # 对比分析图表
```

---

## 五、待清理的冗余内容 (Cleanup)

### 5.1 潜在冗余文件
经检查，以下文件未被导入，可能为独立脚本或遗留代码:
- [x] `generate_all_plots.py` - **独立脚本**，读取episode_log.csv生成图表，与`plot_results.py`功能部分重复
- [x] `utils/plot_baseline_comparison.py` - **未被导入**，独立baseline可视化脚本

**建议**: 保留作为独立工具，但在文档中说明其用途。

### 5.2 已知废弃代码
- [ ] `ResourceFeatureEncoder` 中标记为废弃的 `max_vehicle_id` 参数
- [ ] 配置文件中的旧注释需要清理

### 5.3 待检查的重复逻辑
- [ ] `_estimate_rate()` vs `ChannelModel.estimate_*()` - 可能存在重复
- [ ] CFT计算逻辑分散在多处

---

## 六、执行进度 (Execution Progress)

| Phase | 任务 | 状态 | 备注 |
|-------|------|------|------|
| 1 | 提取 TransferJob/ComputeJob | ✅ 完成 | 创建 envs/jobs/__init__.py |
| 2 | 创建 RSUSelector 服务 | ✅ 完成 | 创建 envs/services/rsu_selector.py，委托5个方法 |
| 3 | 提取 compute_absolute_reward | ✅ 完成 | 创建 envs/rl/reward_functions.py |
| 4 | 重构 train.py 工具函数 | ✅ 完成 | 创建 utils/train_helpers.py，减少~111行 |
| 5 | 创建 queue_helpers.py | ⏭️ 跳过 | 队列方法访问self.veh_cpu_q等，深度耦合 |
| 6 | 创建 cft_calculator.py | ⏭️ 跳过 | 已有time_calculator.py，但env方法有缓存逻辑 |
| 7 | 扩展 reward_engine.py | ⏭️ 跳过 | 奖励引擎ObsBuilder为框架阶段 |
| 8 | 模块化 data_recorder.py | ⏭️ 跳过 | 绘图方法与实例属性耦合深 |
| 9 | 清理冗余代码和文件 | ✅ 完成 | 识别独立脚本，无需删除 |

---

## 6.1 vec_offloading_env.py 为何仍然较大？

### 方法长度分析（Top 10）

| 方法名 | 行数 | 深度耦合原因 |
|--------|------|--------------|
| `_get_obs` | 393 | 访问所有环境状态，生成多维观测 |
| `step` | 319 | 核心仿真循环，调用5个Phase |
| `_plan_actions_snapshot` | 211 | 动作冲突解决，访问队列/RSU/邻居 |
| `__init__` | 188 | 初始化所有组件 |
| `_calculate_global_cft_critical_path` | 170 | CFT计算+缓存逻辑 |
| `_log_episode_stats` | 138 | 审计统计，访问所有车辆状态 |
| `_compute_cost_components` | 129 | 奖励组件计算 |
| `_phase1_commit_offload_decisions` | 118 | 决策提交阶段 |
| `_calculate_constraint_penalty` | 113 | 约束惩罚计算 |
| `_phase2_activate_edge_transfers` | 106 | EDGE激活阶段 |

### 难以进一步提取的原因

1. **状态耦合**：多数方法需要访问 `self.vehicles`, `self.rsus`, `self.time`, `self.veh_cpu_q`, `self.rsu_cpu_q` 等多个环境属性
2. **缓存机制**：CFT计算包含状态哈希+缓存逻辑，与环境状态紧密绑定
3. **副作用链**：Phase方法修改队列状态，需要即时生效
4. **框架已存在**：`ObsBuilder`, `RewardEngine` 存在但尚未实现（标记为NotImplementedError）

### 可行的后续优化方向

1. **实现ObsBuilder**：将393行的`_get_obs`逻辑迁移（需要全面测试）
2. **实现RewardEngine**：将奖励相关方法迁移
3. **参数对象模式**：创建EnvState数据类，减少方法参数

---

## 七、风险控制

### 7.1 测试策略
- 每次重构后运行 `pytest tests/ -v`
- 特别关注 `test_integration.py` 和 `test_fixes.py`
- 运行短训练验证: `python train.py --max-episodes 5`

### 7.2 回滚策略
- 每个Phase完成后提交一个独立commit
- 保留原始代码备份直到完整验证

---

## 八、已发现的现有问题 (待后续修复)

### 8.1 设计问题
- [ ] `vec_offloading_env.py` 的 `_get_obs()` 方法过长 (~400行)
- [ ] 部分魔法数字未配置化

### 8.2 潜在Bug
- [ ] 暂未发现

### 8.3 性能问题
- [ ] `data_recorder.py` 的 `plot_agent_reward_distribution()` 读取大CSV效率问题 (已有优化代码)

---

*此文档将随重构进度更新*
