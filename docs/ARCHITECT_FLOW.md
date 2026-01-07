# 架构与逻辑流 (Architecture & Data Flow)

本文档详细描述系统架构、模块交互、以及从训练脚本到仿真环境的完整数据流。

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              train.py (训练入口)                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ MAPPOAgent  │◄──►│RolloutBuffer│    │DataRecorder │    │  Baselines  │  │
│  └──────┬──────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              OffloadingPolicyNetwork (策略网络)                       │    │
│  │  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────┐     │    │
│  │  │DAGEmbedding  │─►│EdgeEnhancedTransf. │─►│ResourceEncoder   │     │    │
│  │  └──────────────┘  └────────────────────┘  └────────┬─────────┘     │    │
│  │                                                      │               │    │
│  │                                          ┌───────────▼───────────┐  │    │
│  │                                          │  ActorCriticNetwork   │  │    │
│  │                                          │  (target + power)     │  │    │
│  │                                          └───────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VecOffloadingEnv (仿真环境)                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           实体层 (Entities)                             │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐                │ │
│  │  │ Vehicle │  │   RSU   │  │ TaskDAG │  │ TransferJob │                │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           服务层 (Services)                             │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │
│  │  │CommQueueService │  │ CpuQueueService │  │ DAGCompletionHandler   │ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           RL接口层 (RL Interface)                       │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐                    │ │
│  │  │ ObsBuilder  │  │ RewardEngine │  │ActionHandler│                    │ │
│  │  └─────────────┘  └──────────────┘  └─────────────┘                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心文件索引

### 2.1 仿真环境（Env）

| 文件 | 行数 | 职责 |
|------|------|------|
| `envs/vec_offloading_env.py` | ~3400 | 主环境类（Gymnasium接口），5阶段step推进 |
| `envs/entities/vehicle.py` | - | 车辆实体：位置、速度、CPU频率、任务队列 |
| `envs/entities/rsu.py` | - | RSU边缘服务器：多处理器队列、覆盖范围检测 |
| `envs/entities/task_dag.py` | - | DAG任务：邻接矩阵、依赖关系、状态机转换 |
| `envs/modules/channel.py` | - | C-V2X信道模型：瑞利衰落、路径损耗、干扰计算 |
| `envs/modules/queue_system.py` | - | FIFO队列系统（work-conserving特性） |
| `envs/services/comm_queue_service.py` | - | 通信队列服务（Phase 3调用） |
| `envs/services/cpu_queue_service.py` | - | 计算队列服务（Phase 4调用） |
| `envs/services/dag_completion_handler.py` | - | DAG完成/超时处理（Phase 5调用） |
| `envs/rl/obs_builder.py` | - | 观测空间构造器 |
| `envs/rl/reward_engine.py` | - | 奖励引擎（Delta-CFT计算） |

### 2.2 智能体（Agent）

| 文件 | 职责 |
|------|------|
| `agents/mappo_agent.py` | MAPPO封装：`select_action`、`evaluate_actions`、`update` |
| `agents/rollout_buffer.py` | 经验缓冲区：存储轨迹、计算GAE优势估计 |

### 2.3 神经网络模型（Models）

| 文件 | 职责 |
|------|------|
| `models/offloading_policy.py` | 完整策略网络（端到端，整合所有子模块） |
| `models/dag_embedding.py` | DAG节点嵌入：状态、位置、前向/后向层级 |
| `models/edge_enhanced_transformer.py` | 边增强Transformer：多头注意力+边偏置+空间偏置 |
| `models/resource_features.py` | 资源特征编码：14维物理特征+角色嵌入 |
| `models/actor_critic.py` | Actor-Critic头：Target选择（Categorical）+Power控制（Beta） |

### 2.4 配置系统（Configs）

| 文件 | 职责 |
|------|------|
| `configs/config.py` | `SystemConfig`：物理参数（道路、通信、计算、DAG生成） |
| `configs/train_config.py` | `TrainConfig`：训练超参（网络结构、PPO参数、学习率） |

### 2.5 基准策略（Baselines）

| 文件 | 职责 |
|------|------|
| `baselines/random_policy.py` | 随机策略：均匀随机选择可用目标 |
| `baselines/local_only_policy.py` | 全本地策略：始终选择target=0 |
| `baselines/greedy_policy.py` | 贪婪策略：选择预估完成时间最短的目标 |

---

## 3. 训练主循环（train.py）

```python
# 简化的训练流程伪代码
for episode in range(MAX_EPISODES):
    obs_list, _ = env.reset()                    # ① 环境重置
    buffer.clear()

    for step in range(MAX_STEPS):
        # ② 策略推理
        action_dict = agent.select_action(obs_list, deterministic=False)
        actions = action_dict['actions']         # [{target, power}, ...]
        log_probs = action_dict['log_probs']
        values = action_dict['values']

        # ③ 环境交互
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # ④ 存储经验
        buffer.add(obs_list, actions, rewards, values, log_probs,
                   done=(terminated or truncated))

        obs_list = next_obs
        if terminated or truncated:
            break

    # ⑤ PPO更新（每episode结束后）
    last_value = agent.get_value(obs_list)
    buffer.compute_returns_and_advantages(last_value)
    loss = agent.update(buffer, batch_size=MINI_BATCH_SIZE)
```

---

## 4. 环境Step流程（5阶段推进）

`VecOffloadingEnv.step(actions)` 的执行分为5个严格有序的阶段：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: _phase1_commit_offload_decisions()                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  输入: actions = [{target: int, power: float}, ...]                         │
│  处理:                                                                       │
│    1. 解析每个车辆的动作（target=0本地/1RSU/2+V2V）                           │
│    2. 写入 vehicle.exec_locations[subtask_idx] = target                     │
│    3. 如果需要传输，创建 INPUT TransferJob 并加入通信队列                      │
│    4. 如果本地执行，直接创建 ComputeJob 加入计算队列                           │
│  输出: 更新了各队列状态                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: _phase2_activate_edge_transfers()                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  处理:                                                                       │
│    1. 扫描所有进行中的任务，检查 inter_task_transfers（边传输）              │
│    2. 如果前驱子任务完成且需要传输结果到下一节点，创建 EDGE TransferJob       │
│    3. EDGE传输：V2V或V2I通信                                                 │
│  注意: 只有前驱完成后才能激活后继任务的数据传输                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: _phase3_advance_comm_queues()                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  处理:                                                                       │
│    1. FIFO并行服务所有通信队列（V2I队列 + V2V队列）                           │
│    2. 调用 CommQueueService.serve_dt(DT)                                    │
│    3. 计算传输速率（Shannon公式）                                            │
│    4. 更新 remaining_bits，完成后触发回调                                    │
│  特性: Work-Conserving - 队头完成后用剩余时间推进下一个                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 4: _phase4_advance_cpu_queues()                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  处理:                                                                       │
│    1. FIFO并行服务所有计算队列（车辆队列 + RSU多核队列）                       │
│    2. 调用 CpuQueueService.serve_dt(DT)                                     │
│    3. 计算处理进度 = cpu_freq × dt                                          │
│    4. 更新 remaining_cycles，完成后触发回调                                  │
│  RSU特性: 4核并行，负载均衡分配                                               │
│  任务完成: 调用_mark_done()，可能创建新的inter_task_transfers                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 4.5: _phase2_activate_edge_transfers() [P01修复-补偿激活]             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  目的: 激活Phase4中新产生的EDGE传输，消除1时隙延迟                             │
│  处理:                                                                       │
│    1. 扫描所有DAG的inter_task_transfers                                     │
│    2. 对于child已分配且parent已完成的边，创建EDGE TransferJob                 │
│    3. 同位置边瞬时清零，不入队列                                              │
│  幂等性: 通过active_edge_keys去重，不会重复创建                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 5: 时间推进 + 车辆移动 + 终止判断 + 奖励计算                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  处理:                                                                       │
│    1. self.time += DT                                                       │
│    2. 更新所有车辆位置: vehicle.update_position(DT)                          │
│    3. 检查DAG完成状态: DAGCompletionHandler                                  │
│    4. 判断终止条件: all_finished / time_limit                                │
│    5. 计算奖励: RewardEngine.compute_rewards()                              │
│    6. 构建观测: ObsBuilder.build_obs()                                      │
│  输出: next_obs, rewards, terminated, truncated, info                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 观测空间构建详解

`_get_obs()` 为每个车辆构建独立的观测字典：

### 5.1 DAG相关特征

| 字段 | 形状 | 说明 |
|------|------|------|
| `node_x` | `[MAX_NODES, 7]` | 节点连续特征 |
| `adj` | `[MAX_NODES, MAX_NODES]` | DAG邻接矩阵 |
| `data_matrix` | `[MAX_NODES, MAX_NODES]` | 边数据量矩阵 |
| `Delta` | `[MAX_NODES, MAX_NODES]` | 最短路径距离矩阵 |
| `status` | `[MAX_NODES]` | 节点状态ID（0-4） |
| `location` | `[MAX_NODES]` | 执行位置ID |
| `L_fwd` | `[MAX_NODES]` | 前向层级 |
| `L_bwd` | `[MAX_NODES]` | 后向层级 |
| `task_mask` | `[MAX_NODES]` | 有效节点掩码 |
| `subtask_index` | `int` | 当前调度的子任务索引 |

**node_x的7个维度：**
```
[0] comp_norm      - 归一化计算量 (cycles / MAX_COMP)
[1] data_norm      - 归一化数据量 (bits / MAX_DATA)
[2] status_float   - 状态归一化 (status / 4.0)
[3] in_degree      - 入度
[4] out_degree     - 出度
[5] t_rem_norm     - 剩余时间归一化
[6] urgency        - 紧急度 = 1 / (t_rem + ε)
```

### 5.2 资源特征

| 字段 | 形状 | 说明 |
|------|------|------|
| `resource_raw` | `[MAX_TARGETS, 14]` | 资源物理特征（无ID泄漏） |
| `resource_ids` | `[MAX_TARGETS]` | 资源角色ID（0=pad, 1=Local, 2=RSU, 3+=V2V） |
| `action_mask` | `[MAX_TARGETS]` | 动作掩码（True=可选） |

**resource_raw的14个维度：**
```
[0]  cpu_freq_norm     - 归一化CPU频率
[1]  queue_wait_norm   - 归一化队列等待时间
[2]  distance_norm     - 归一化距离
[3]  rate_norm         - 归一化传输速率
[4]  rel_pos_x         - 相对位置X
[5]  rel_pos_y         - 相对位置Y
[6]  vel_x             - 目标速度X
[7]  vel_y             - 目标速度Y
[8]  node_type         - 节点类型（0=Local, 1=RSU, 2=V2V）
[9]  slack_norm        - 归一化松弛时间
[10] contact_time_norm - 归一化接触时间
[11] est_exec_time     - 预估执行时间
[12] est_comm_time     - 预估通信时间
[13] est_wait_time     - 预估等待时间
```

### 5.3 自身信息

| 字段 | 形状 | 说明 |
|------|------|------|
| `self_info` | `[7]` | 车辆自身状态 |

**self_info的7个维度：**
```
[0] vel_x_norm   - 归一化速度X
[1] vel_y_norm   - 归一化速度Y（通常为0）
[2] wait_time    - 归一化等待时间
[3] cpu_freq     - 归一化CPU频率
[4] v2i_rate     - 归一化V2I速率
[5] pos_x_norm   - 归一化位置X
[6] pos_y_norm   - 归一化位置Y
```

---

## 6. 策略网络前向传播

`OffloadingPolicyNetwork.forward()` 的处理流程：

```
输入张量
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DAG节点嵌入 (DAGNodeEmbedding)                                           │
│     node_emb = continuous_proj(node_x)                                       │
│               + status_emb[status]                                           │
│               + location_emb[location]                                       │
│               + L_fwd_emb[L_fwd]                                             │
│               + L_bwd_emb[L_bwd]                                             │
│     输出: [B, MAX_NODES, d_model]                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. 边偏置计算                                                               │
│     edge_bias = EdgeFeatureEncoder(data_matrix)   # [B, H, N, N]            │
│     spatial_bias = SpatialDistanceEncoder(Delta)  # [B, H, N, N]            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. 边增强Transformer (EdgeEnhancedTransformer)                             │
│     for layer in transformer_layers:                                         │
│         attn_out = MultiHeadAttention(Q, K, V,                               │
│                                       bias=edge_bias + spatial_bias)         │
│         x = LayerNorm(x + attn_out)                                          │
│         ff_out = FeedForward(x)                                              │
│         x = LayerNorm(x + ff_out)                                            │
│     输出: dag_features [B, MAX_NODES, d_model]                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. 资源特征编码 (ResourceFeatureEncoder)                                    │
│     resource_encoded = MLP(resource_raw) + role_embedding[resource_ids]      │
│     输出: [B, MAX_TARGETS, d_model]                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. Actor-Critic输出 (ActorCriticNetwork)                                   │
│     # 提取当前子任务特征                                                      │
│     subtask_feat = dag_features[batch_idx, subtask_index]  # [B, d_model]   │
│                                                                              │
│     # Target选择（交叉注意力）                                                │
│     target_logits = CrossAttention(subtask_feat, resource_encoded)          │
│     target_logits = masked_softmax(target_logits, action_mask)              │
│                                                                              │
│     # Power控制（Beta分布参数）                                              │
│     alpha = softplus(power_head_alpha(subtask_feat)) + 1                    │
│     beta = softplus(power_head_beta(subtask_feat)) + 1                      │
│                                                                              │
│     # 状态价值（全局池化）                                                    │
│     global_feat = mean_pool(dag_features, task_mask)                         │
│     value = value_head(global_feat)                                          │
│                                                                              │
│     输出: target_logits [B, M], alpha [B, 1], beta [B, 1], value [B, 1]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键设计决策

### 7.1 5阶段Step的因果一致性

为什么不能合并Phase 3和Phase 4？
- 传输完成后才能开始计算
- 如果同时推进，可能出现"数据还在传输中就开始计算"的逻辑错误
- 严格的阶段划分保证了仿真的物理正确性

### 7.2 Work-Conserving队列

FIFO队列具有"work-conserving"特性：
```python
def serve_dt(self, dt):
    remaining_dt = dt
    while remaining_dt > 0 and self.queue:
        job = self.queue[0]
        time_needed = job.remaining / self.service_rate
        if time_needed <= remaining_dt:
            # 完成当前job，用剩余时间推进下一个
            remaining_dt -= time_needed
            self.complete_job(job)
        else:
            # 部分推进
            job.remaining -= self.service_rate * remaining_dt
            remaining_dt = 0
```

### 7.3 TransferJob类型

| 类型 | 触发时机 | 传输内容 |
|------|----------|----------|
| `INPUT` | Phase 1决策时 | 子任务输入数据（从源节点到执行节点） |
| `EDGE` | Phase 2激活时 | 子任务输出数据（从执行节点到后继节点） |

### 7.4 观测的obs_stamp机制

每个观测携带时间戳 `obs_stamp`，用于验证动作与观测的时序一致性：
```python
# 在step()中检查
if action.get('obs_stamp') != obs.get('obs_stamp'):
    # 警告：动作基于过时的观测
```

---

## 8. 输出目录结构

```
runs/run_YYYYMMDD_HHMMSS/
├── logs/
│   ├── training_stats.csv      # 绘图用指标（episode级）
│   ├── metrics.csv             # 详细指标（episode级）
│   ├── metrics.jsonl           # JSON格式详细指标
│   ├── step_metrics.csv        # Step级指标（可选，--step-metrics启用）
│   ├── config_snapshot.json    # 配置快照
│   └── env_reward.jsonl        # 环境奖励日志
├── models/
│   ├── best_model.pth          # 最佳模型（基于成功率）
│   ├── best_model_reward.pth   # 最佳模型（基于累积奖励）
│   └── model_ep*.pth           # 周期性检查点
├── plots/
│   ├── fig_convergence.png     # 收敛曲线
│   ├── fig_policy_evolution.png # 策略演化
│   ├── fig_physics.png         # 物理指标
│   └── fig_training.png        # 训练诊断
└── tensorboard/                # TensorBoard日志
```
