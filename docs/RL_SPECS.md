# 强化学习规范 (Reinforcement Learning Specifications)

本文档详细记录MAPPO算法的状态空间、动作空间、奖励函数设计，以及DAG任务依赖的处理逻辑。

---

## 1. 算法概述

### 1.1 MAPPO (Multi-Agent PPO)

本项目采用参数共享的MAPPO算法：
- 所有智能体（车辆）共享同一策略网络
- 分散执行、集中训练（CTDE范式）
- 每个智能体基于自身观测独立决策

### 1.2 算法框架

```
┌────────────────────────────────────────────────────────────────────────┐
│                          MAPPO Training Loop                            │
├────────────────────────────────────────────────────────────────────────┤
│  for episode in range(MAX_EPISODES):                                    │
│      # Rollout Phase (采集经验)                                          │
│      for step in range(MAX_STEPS):                                      │
│          actions = policy(observations)                                  │
│          next_obs, rewards, dones = env.step(actions)                   │
│          buffer.add(obs, actions, rewards, values, log_probs)           │
│                                                                          │
│      # Update Phase (策略更新)                                           │
│      advantages = GAE(rewards, values, gamma, gae_lambda)               │
│      for epoch in range(PPO_EPOCH):                                     │
│          for batch in buffer.get_batches():                             │
│              # PPO Clip Loss                                             │
│              ratio = exp(new_log_prob - old_log_prob)                   │
│              surr1 = ratio * advantages                                  │
│              surr2 = clip(ratio, 1-ε, 1+ε) * advantages                 │
│              policy_loss = -min(surr1, surr2).mean()                    │
│                                                                          │
│              # Value Loss                                                │
│              value_loss = MSE(values, returns)                          │
│                                                                          │
│              # Entropy Bonus                                             │
│              entropy_loss = -entropy.mean()                              │
│                                                                          │
│              # Total Loss                                                │
│              loss = policy_loss + vf_coef*value_loss + ent_coef*entropy │
│              optimizer.step(loss)                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 状态空间（Observation Space）

每个智能体（车辆）接收一个字典形式的观测，包含以下字段：

### 2.1 DAG结构特征

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `node_x` | `[N, 7]` | float32 | 节点连续特征 |
| `adj` | `[N, N]` | float32 | DAG邻接矩阵 |
| `data_matrix` | `[N, N]` | float32 | 边数据量矩阵 |
| `Delta` | `[N, N]` | int64 | 最短路径距离矩阵 |
| `status` | `[N]` | int64 | 节点状态（0-4） |
| `location` | `[N]` | int64 | 执行位置ID |
| `L_fwd` | `[N]` | int64 | 前向拓扑层级 |
| `L_bwd` | `[N]` | int64 | 后向拓扑层级 |
| `task_mask` | `[N]` | bool | 有效节点掩码 |
| `subtask_index` | scalar | int64 | 当前调度的子任务索引 |

**node_x的7个维度详解：**

| 索引 | 名称 | 计算方式 | 范围 |
|------|------|----------|------|
| 0 | comp_norm | `comp / MAX_COMP` | [0, 1] |
| 1 | data_norm | `data / MAX_DATA` | [0, 1] |
| 2 | status_float | `status / 4.0` | [0, 1] |
| 3 | in_degree | DAG入度 | [0, N-1] |
| 4 | out_degree | DAG出度 | [0, N-1] |
| 5 | t_rem_norm | `t_remaining / deadline` | [0, 1+] |
| 6 | urgency | `1 / (t_remaining + ε)` | [0, ∞) |

### 2.2 资源特征

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `resource_raw` | `[M, 14]` | float32 | 资源物理特征 |
| `resource_ids` | `[M]` | int64 | 资源角色ID |
| `action_mask` | `[M]` | bool | 动作掩码 |

**resource_raw的14个维度详解：**

| 索引 | 名称 | 说明 | 归一化方式 |
|------|------|------|------------|
| 0 | cpu_freq | 处理器频率 | / MAX_CPU |
| 1 | queue_wait | 队列等待时间 | / MAX_WAIT |
| 2 | distance | 与目标距离 | / MAX_DIST |
| 3 | rate | 传输速率 | / MAX_RATE |
| 4 | rel_pos_x | 相对位置X | / ROAD_LEN |
| 5 | rel_pos_y | 相对位置Y | / ROAD_WIDTH |
| 6 | vel_x | 目标速度X | / VEL_MAX |
| 7 | vel_y | 目标速度Y | / VEL_MAX |
| 8 | node_type | 节点类型 | 0=L, 1=R, 2=V |
| 9 | slack | 松弛时间 | / deadline |
| 10 | contact_time | 接触时间 | / MAX_CONTACT |
| 11 | est_exec_time | 预估执行时间 | / deadline |
| 12 | est_comm_time | 预估通信时间 | / deadline |
| 13 | est_wait_time | 预估等待时间 | / deadline |

**resource_ids映射：**

| ID | 含义 |
|------|------|
| 0 | Padding（无效） |
| 1 | Local（本地执行） |
| 2 | RSU（边缘服务器） |
| 3+ | V2V邻居（ID = 3 + neighbor_index） |

### 2.3 自身信息

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `self_info` | `[7]` | float32 | 车辆自身状态 |

**self_info的7个维度：**

| 索引 | 名称 | 说明 |
|------|------|------|
| 0 | vel_x | 归一化速度X |
| 1 | vel_y | 归一化速度Y |
| 2 | wait_time | 归一化队列等待时间 |
| 3 | cpu_freq | 归一化CPU频率 |
| 4 | v2i_rate | 归一化V2I速率 |
| 5 | pos_x | 归一化位置X |
| 6 | pos_y | 归一化位置Y |

---

## 3. 动作空间（Action Space）

采用**混合动作空间**：离散目标选择 + 连续功率控制。

### 3.1 Target动作（离散）

| 索引 | 含义 | 说明 |
|------|------|------|
| 0 | Local | 本地执行，无需传输 |
| 1 | RSU | 卸载到边缘服务器 |
| 2+ | V2V | 卸载到第k个邻居（k = index - 2） |

**动作掩码（action_mask）规则：**
- Local：始终可选（mask[0] = True）
- RSU：车辆在RSU覆盖范围内时可选
- V2V：邻居在V2V通信范围内且有足够接触时间时可选

### 3.2 Power动作（连续）

| 参数 | 范围 | 说明 |
|------|------|------|
| power_ratio | [0, 1] | 功率比例 |

映射到实际功率：
$$
P_{tx} = P_{min} + power\_ratio \times (P_{max} - P_{min})
$$

### 3.3 动作采样

**Target采样（Categorical分布）：**
```python
# 应用Logit Bias（解决动作空间不平衡）
if USE_LOGIT_BIAS:
    logits[0] += LOGIT_BIAS_LOCAL  # Local偏置
    logits[1] += LOGIT_BIAS_RSU    # RSU偏置

# 应用action_mask
masked_logits = where(action_mask, logits, -1e10)

# 采样
probs = softmax(masked_logits)
target = Categorical(probs).sample()
```

**Power采样（Beta分布）：**
```python
# 网络输出alpha, beta参数
alpha = softplus(alpha_head(features)) + 1
beta = softplus(beta_head(features)) + 1

# 采样
power = Beta(alpha, beta).sample()  # [0, 1]
```

### 3.4 Logit Bias退火

解决动作空间不平衡问题（V2V选项数量远多于Local/RSU）：

| 参数 | 初始值 | 最终值 | 说明 |
|------|--------|--------|------|
| LOGIT_BIAS_LOCAL | 1.0 | 0.0 | Local偏置 |
| LOGIT_BIAS_RSU | 2.0 | 0.0 | RSU偏置 |
| BIAS_DECAY_EVERY_EP | 100 | - | 每100 episode衰减一次 |

---

## 4. 奖励函数（Reward Function）

### 4.1 主奖励：Delta-CFT

基于**关键完成时间（Critical Finish Time）变化量**设计奖励：

$$
r_{cft} = \alpha_{cft} \times \Delta T
$$

$$
\Delta T = T_{cft,prev}^{rem} - T_{cft,curr}^{rem}
$$

其中：
- $T_{cft,prev}^{rem}$：决策前的剩余关键路径时间
- $T_{cft,curr}^{rem}$：决策后的剩余关键路径时间
- $\alpha_{cft}$：缩放系数（DELTA_CFT_SCALE）

**直觉**：选择好的卸载目标应该减少剩余完成时间，产生正奖励。

### 4.2 能耗惩罚

$$
r_{energy} = -\beta_{energy} \times E_{norm}
$$

$$
E_{norm} = \frac{(P_{tx} + P_{circuit}) \times \Delta t}{E_{max}}
$$

其中：
- $\beta_{energy}$：能耗权重（DELTA_CFT_ENERGY_WEIGHT）
- $E_{max}$：参考最大能耗

### 4.3 总奖励

$$
r = r_{cft} + r_{energy}
$$

### 4.4 约束处理

**硬约束触发（非法动作）：**
- 选择超出通信范围的目标 → $r = R_{min}$
- 选择不可用的RSU → $r = R_{min}$

**软约束（距离预警）：**
- V2V目标接近通信边界 → 距离惩罚项

**时间截断惩罚：**
- Episode因时间限制截断且任务未完成 → 末步附加惩罚

---

## 5. 策略网络架构

### 5.1 整体结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OffloadingPolicyNetwork                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐                                                       │
│  │ DAGEmbedding  │◄── node_x, status, location, L_fwd, L_bwd            │
│  │  - Linear(7→d)│                                                       │
│  │  - Embeddings │                                                       │
│  └───────┬───────┘                                                       │
│          │ [B, N, d_model]                                               │
│          ▼                                                                │
│  ┌────────────────────────┐                                              │
│  │ EdgeEnhancedTransformer │◄── edge_bias, spatial_bias                 │
│  │  - L layers            │                                              │
│  │  - H attention heads   │                                              │
│  └───────┬────────────────┘                                              │
│          │ [B, N, d_model]  (dag_features)                               │
│          │                                                                │
│          │     ┌──────────────────┐                                      │
│          │     │ResourceEncoder   │◄── resource_raw, resource_ids        │
│          │     │  - MLP(14→d)     │                                      │
│          │     │  - RoleEmbedding │                                      │
│          │     └────────┬─────────┘                                      │
│          │              │ [B, M, d_model]  (resource_encoded)            │
│          │              │                                                │
│          ▼              ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ActorCriticNetwork                            │    │
│  │  ┌─────────────────────────────────────────────────────────────┐│    │
│  │  │ Actor Head (Target)                                          ││    │
│  │  │  subtask_feat = dag_features[:, subtask_index]              ││    │
│  │  │  logits = CrossAttention(subtask_feat, resource_encoded)    ││    │
│  │  │  target = Categorical(softmax(logits + bias)).sample()      ││    │
│  │  └─────────────────────────────────────────────────────────────┘│    │
│  │  ┌─────────────────────────────────────────────────────────────┐│    │
│  │  │ Actor Head (Power)                                           ││    │
│  │  │  alpha = softplus(linear(subtask_feat)) + 1                 ││    │
│  │  │  beta = softplus(linear(subtask_feat)) + 1                  ││    │
│  │  │  power = Beta(alpha, beta).sample()                         ││    │
│  │  └─────────────────────────────────────────────────────────────┘│    │
│  │  ┌─────────────────────────────────────────────────────────────┐│    │
│  │  │ Critic Head (Value)                                          ││    │
│  │  │  global_feat = mean_pool(dag_features, task_mask)           ││    │
│  │  │  value = MLP(global_feat)                                   ││    │
│  │  └─────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 网络参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 128 | 嵌入维度 |
| num_heads | 4 | 注意力头数 |
| num_layers | 3 | Transformer层数 |
| d_ff | 512 | 前馈层维度 |
| dropout | 0.1 | Dropout率 |

---

## 6. PPO超参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 学习率（Actor） | $\alpha_{actor}$ | 3e-4 | 策略网络学习率 |
| 学习率（Critic） | $\alpha_{critic}$ | 1e-3 | 价值网络学习率 |
| 折扣因子 | $\gamma$ | 0.99 | 未来奖励折扣 |
| GAE参数 | $\lambda$ | 0.95 | 优势估计参数 |
| Clip参数 | $\epsilon$ | 0.2 | PPO裁剪范围 |
| 价值损失系数 | $c_{vf}$ | 0.5 | Value loss权重 |
| 熵正则系数 | $c_{ent}$ | 0.01 | Entropy bonus权重 |
| PPO Epoch | - | 4 | 每次更新的epoch数 |
| Mini-batch大小 | - | 64 | 批次大小 |
| 最大梯度范数 | - | 0.5 | 梯度裁剪阈值 |

---

## 7. DAG依赖处理

### 7.1 子任务调度顺序

采用**拓扑排序 + 就绪优先**策略：

```python
def select_next_subtask(dag):
    ready_tasks = [i for i in range(dag.num_nodes)
                   if dag.status[i] == READY]

    if not ready_tasks:
        return None

    # 优先选择：1) 关键路径上的任务 2) 前向层级小的任务
    return min(ready_tasks, key=lambda i: (
        -dag.is_on_critical_path[i],
        dag.L_fwd[i]
    ))
```

### 7.2 依赖检查

子任务状态转换逻辑：

```python
def update_task_status(dag, completed_task_id):
    # 标记完成
    dag.status[completed_task_id] = COMPLETED

    # 检查后继任务
    for succ in dag.successors[completed_task_id]:
        # 所有前驱都完成 → 变为READY
        if all(dag.status[pred] == COMPLETED
               for pred in dag.predecessors[succ]):
            dag.status[succ] = READY
```

### 7.3 数据依赖传输

当子任务在不同节点执行时，需要传输中间结果：

```
Task A (在Vehicle 1执行)
    │
    │ 输出数据: 2 MB
    │
    ▼
Task B (在RSU执行)
    需要先传输A的输出到RSU
```

传输类型：
- **INPUT传输**：将子任务输入数据传到执行节点
- **EDGE传输**：将子任务输出数据传到后继任务的执行节点

### 7.4 同位置优化

如果前驱和后继在同一节点执行，跳过EDGE传输：

```python
if exec_location[pred] == exec_location[succ]:
    # 无需传输，直接标记数据就绪
    data_ready[succ][pred] = True
else:
    # 创建EDGE TransferJob
    create_edge_transfer(pred, succ)
```

---

## 8. 经验回放与GAE

### 8.1 RolloutBuffer结构

```python
class RolloutBuffer:
    obs_buffer: List[Dict]       # 观测列表
    actions_buffer: List[Dict]   # 动作列表
    rewards_buffer: List[float]  # 奖励
    values_buffer: List[float]   # 状态价值
    log_probs_buffer: List[float]# 动作log概率
    dones_buffer: List[bool]     # 终止标志
```

### 8.2 GAE计算

$$
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}
$$

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

```python
def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

---

## 9. 训练配置

所有训练超参数定义在 `configs/train_config.py` 的 `TrainConfig` 类中：

```python
class TrainConfig:
    # 网络结构
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3

    # PPO参数
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    # 优化器
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    PPO_EPOCH = 4
    MINI_BATCH_SIZE = 64

    # 训练控制
    MAX_EPISODES = 5000
    MAX_STEPS = 500
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 100
    EVAL_INTERVAL = 50

    # Logit Bias
    USE_LOGIT_BIAS = True
    LOGIT_BIAS_LOCAL = 1.0
    LOGIT_BIAS_RSU = 2.0
    BIAS_DECAY_EVERY_EP = 100
    BIAS_DECAY_LOCAL = 0.1
    BIAS_DECAY_RSU = 0.2
    BIAS_MIN_LOCAL = 0.0
    BIAS_MIN_RSU = 0.0
```
