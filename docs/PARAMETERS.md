# 系统参数快照 (System Parameters Snapshot)

**更新时间 (Last Updated)**: 2026-01-02  
**配置文件 (Config Files)**: `configs/config.py`, `configs/train_config.py`

---

## 📋 目录 (Table of Contents)

1. [物理环境参数 (Physics)](#1-物理环境参数-physics)
2. [资源与任务参数 (Resources & Tasks)](#2-资源与任务参数-resources--tasks)
3. [强化学习超参数 (RL Hyperparameters)](#3-强化学习超参数-rl-hyperparameters)
4. [训练控制参数 (Training Control)](#4-训练控制参数-training-control)
5. [网络结构参数 (Model Architecture)](#5-网络结构参数-model-architecture)
6. [归一化参数 (Normalization)](#6-归一化参数-normalization)

---

## 1. 物理环境参数 (Physics)

### 1.1 地图与拓扑 (Map & Topology)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `MAP_SIZE` | 1000.0 | m | 道路长度 | 仿真区域大小，影响车辆密度和RSU覆盖 |
| `NUM_LANES` | 2 | - | 车道数量 | 车辆横向分布，减少换道干扰 |
| `LANE_WIDTH` | 3.5 | m | 车道宽度 | 标准车道宽度，影响V2V距离计算 |
| `DT` | 0.05 | s | 时间步长 | 精度与计算开销权衡，保证 v_max * DT << R_rsu |
| `MAX_STEPS` | 200 | - | Episode步数 | Episode总时长 = MAX_STEPS × DT = 10秒 |

### 1.2 车辆参数 (Vehicle Parameters)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `NUM_VEHICLES` | 12 | - | 初始车辆数 | 网络负载和V2V候选数量 |
| `V2V_TOP_K` | 11 | - | V2V候选数上限 | 限制每个Agent的V2V目标数 |
| `VEHICLE_ARRIVAL_RATE` | 0.2 | veh/s | 泊松到达率 | 动态车辆生成，0表示禁用 |
| `VEL_MEAN` | 11.1 | m/s | 速度均值 | ≈ 40 km/h，城市道路标准 |
| `VEL_STD` | 2.0 | m/s | 速度标准差 | 速度异构性 |
| `VEL_MIN` | 5.0 | m/s | 最小速度 | ≈ 18 km/h，防止静止 |
| `VEL_MAX` | 16.6 | m/s | 最大速度 | ≈ 60 km/h，城市限速 |

### 1.3 RSU部署 (RSU Deployment)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `NUM_RSU` | 3 | - | RSU数量 | 覆盖率和卸载能力 |
| `RSU_Y_DIST` | 10.0 | m | RSU离路边距离 | 影响V2I路径损耗 |
| `RSU_RANGE` | 400.0 | m | RSU覆盖半径 | V2I通信范围 |

### 1.4 通信物理层 (Communication Physical Layer)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `FC` | 5.9e9 | Hz | 载波频率 | C-V2X标准 (5.9 GHz) |
| `BW_V2I` | 20e6 | Hz | V2I带宽 | 20 MHz，影响上行速率 |
| `BW_V2V` | 10e6 | Hz | V2V带宽 | 10 MHz，影响V2V速率 |
| `TX_POWER_UP_DBM` | 23.0 | dBm | 上行发射功率 | 200mW，FCC限制 |
| `TX_POWER_V2V_DBM` | 23.0 | dBm | V2V发射功率 | 200mW |
| `TX_POWER_MIN_DBM` | 20.0 | dBm | 功控下限 | 能耗优化范围 |
| `TX_POWER_MAX_DBM` | 23.0 | dBm | 功控上限 | 能耗优化范围 |
| `NOISE_POWER_DBM` | -95.0 | dBm | 热噪声功率 | 影响SNR计算 |
| `PL_ALPHA_V2I` | 28.0 | dB | V2I参考路损 | Log-Distance模型 |
| `PL_BETA_V2I` | 2.5 | - | V2I路损指数 | LOS环境 |
| `PL_ALPHA_V2V` | 28.0 | dB | V2V参考路损 | Log-Distance模型 |
| `PL_BETA_V2V` | 3.5 | - | V2V路损指数 | NLOS环境，高衰减 |
| `V2V_INTERFERENCE_DBM` | -90.0 | dBm | V2V背景干扰 | 影响V2V SINR |
| `V2V_RANGE` | 300.0 | m | V2V通信半径 | 邻居发现范围 |

---

## 2. 资源与任务参数 (Resources & Tasks)

### 2.1 计算资源 (Computing Resources)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `MIN_VEHICLE_CPU_FREQ` | 1.0e9 | Hz | 车辆最小CPU频率 | 1 GHz，异构性下界 |
| `MAX_VEHICLE_CPU_FREQ` | 3.0e9 | Hz | 车辆最大CPU频率 | 3 GHz，异构性上界 |
| `F_RSU` | 10.0e9 | Hz | RSU CPU频率 | 10 GHz，强于车辆 |
| `RSU_NUM_PROCESSORS` | 4 | - | RSU处理器核心数 | 多核并行处理 |
| `K_ENERGY` | 1e-28 | - | 能耗系数 | E = k·f²·cycles |

### 2.2 队列限制 (Queue Limits)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `VEHICLE_QUEUE_CYCLES_LIMIT` | 4.0e9 | cycles | 车辆队列上限 | 约20个平均任务 |
| `RSU_QUEUE_CYCLES_LIMIT` | 20.0e9 | cycles | RSU队列上限 | 约100个平均任务 |

### 2.3 DAG任务生成 (DAG Task Generation)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `MIN_NODES` | 4 | - | DAG最小节点数 | 降低依赖链复杂度 |
| `MAX_NODES` | 6 | - | DAG最大节点数 | 降低依赖链深度 |
| `MIN_COMP` | 1.0e9 | cycles | 子任务最小计算量 | 本地执行约0.33s @3GHz |
| `MAX_COMP` | 2.0e9 | cycles | 子任务最大计算量 | 本地执行约0.67s @3GHz |
| `MIN_DATA` | 1.0e6 | bits | 子任务最小数据量 | 1 Mbit，V2I传输约0.02s |
| `MAX_DATA` | 3.0e6 | bits | 子任务最大数据量 | 3 Mbit，V2I传输约0.06s |
| `MIN_EDGE_DATA` | 0.2e6 | bits | DAG边最小数据量 | 200 Kbit，依赖传输开销 |
| `MAX_EDGE_DATA` | 0.6e6 | bits | DAG边最大数据量 | 600 Kbit，依赖传输开销 |
| `DAG_FAT` | 0.6 | - | DAG宽度参数 | 影响并行度 |
| `DAG_DENSITY` | 0.2 | - | DAG连接密度 | 影响依赖复杂度 |
| `DAG_REGULAR` | 0.5 | - | DAG规则性 | 影响结构类型 |
| `DAG_CCR` | 0.2 | - | 通信计算比 | 结构生成参数 |

### 2.4 Deadline计算 (Deadline Calculation)

| 参数名 | 默认值 | 单位 | 说明 | 影响 |
|--------|--------|------|------|------|
| `DEADLINE_TIGHTENING_FACTOR` | 0.85 | - | Deadline紧缩因子 | <1强制卸载，>1放宽 |
| `DEADLINE_TIGHTENING_MIN` | 5.0 | s | γ最小值 | 大幅放宽以容纳排队 |
| `DEADLINE_TIGHTENING_MAX` | 8.0 | s | γ最大值 | 必须看到V2V/RSU成功 |
| `DEADLINE_SLACK_SECONDS` | 0.0 | s | 额外松弛时间 | 在关键路径基础上附加 |

### 2.5 优先级调度 (Priority Scheduling)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| `PRIORITY_W1` | 100.0 | 后向层级权重 | 主导关键路径 |
| `PRIORITY_W2` | 1.0 | 计算量权重 | 同层级tie-breaking |
| `PRIORITY_W3` | 1.0 | 出度权重 | 同计算量tie-breaking |

---

## 3. 强化学习超参数 (RL Hyperparameters)

### 3.1 优化器参数 (Optimizer)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| `LR_ACTOR` | 3e-4 | Actor学习率 | PPO标准，控制策略更新速度 |
| `LR_CRITIC` | 1e-3 | Critic学习率 | 略快于Actor，加速值函数收敛 |
| `USE_LR_DECAY` | True | 是否启用学习率衰减 | 后期稳定训练 |
| `LR_DECAY_STEPS` | 500 | 学习率衰减间隔 (Episodes) | 每500 Episodes衰减一次 |
| `LR_DECAY_RATE` | 0.92 | 学习率衰减率 | 指数衰减系数 |
| `MAX_GRAD_NORM` | 0.5 | 梯度裁剪阈值 | 防止梯度爆炸 |

### 3.2 PPO算法参数 (PPO Algorithm)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| `GAMMA` | 0.95 | 折扣因子 | 关注短期Deadline |
| `GAE_LAMBDA` | 0.95 | GAE平滑因子 | 优势估计偏差-方差权衡 |
| `CLIP_PARAM` | 0.2 | PPO裁剪阈值 ε | 限制策略更新幅度 |
| `PPO_EPOCH` | 4 | 每次采样更新轮数 | 重复利用经验 |
| `MINI_BATCH_SIZE` | 64 | 小批次大小 | 适应动态图结构 |
| `ENTROPY_COEF` | 0.02 | 熵正则化系数 | 增加探索，应对动态环境 |
| `VF_COEF` | 0.5 | 价值函数损失系数 | 平衡Actor-Critic训练 |

### 3.3 动作空间平衡 (Action Space Balancing)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| `USE_LOGIT_BIAS` | True | 是否启用Logit偏置 | 对抗V2V数量优势 |
| `LOGIT_BIAS_RSU` | 8.0 | RSU的Logit偏置 | 强制探索RSU |
| `LOGIT_BIAS_LOCAL` | 8.0 | Local的Logit偏置 | 强制探索本地执行 |

### 3.4 奖励函数参数 (Reward Function - Delta CFT Mode)

**奖励模式**: Delta Counterfactual Time (固定)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| **Delta CFT参数** |
| `DELTA_CFT_SCALE` | 10.0 | 时间节省缩放系数 | 0.1s节省 = +1.0奖励 |
| `DELTA_CFT_ENERGY_WEIGHT` | 0.5 | 能耗惩罚权重 | 最大功率≈0.05s时延惩罚 |
| `DELTA_CFT_CLIP_MIN` | -1.0 | 奖励下限裁剪 | 防止梯度爆炸 |
| `DELTA_CFT_CLIP_MAX` | 1.0 | 奖励上限裁剪 | 防止梯度爆炸 |
| **距离预警惩罚** |
| `DIST_PENALTY_WEIGHT` | 2.0 | 距离预警权重 | 最大产生-2.0惩罚 |
| `DIST_SAFE_FACTOR` | 0.8 | 安全距离因子 | 通信半径的80%为安全区 |
| `DIST_SENSITIVITY` | 2.0 | 距离敏感度 | 平方增长 |
| **超时惩罚** |
| `TIMEOUT_PENALTY_WEIGHT` | 1.0 | 超时惩罚上限 | 避免淹没terminal奖励 |
| `TIMEOUT_STEEPNESS` | 3.0 | 超时陡峭度 | 微小超时也有显著惩罚 |
| **硬约束惩罚** |
| `PENALTY_LINK_BREAK` | -10.0 | 链路断开惩罚 | 直接终止Episode |
| `PENALTY_OVERFLOW` | -10.0 | 队列溢出惩罚 | 直接终止Episode |
| `PENALTY_FAILURE` | -10.0 | 任务失败惩罚 | 超时终止 |
| `TIME_LIMIT_PENALTY` | -1.0 | 时间截断惩罚 | Episode超时额外惩罚 |
| **成功奖励** |
| `SUCCESS_BONUS` | 20.0 | 任务成功完成奖励 | 稀疏奖励强化 |
| `SUBTASK_SUCCESS_BONUS` | 2.0 | 子任务成功完成奖励 | 计件工资 |
| **奖励范围** |
| `REWARD_MAX` | 30.0 | 奖励上限 | 容纳SUCCESS_BONUS |
| `REWARD_MIN` | -15.0 | 奖励下限 | 容纳大惩罚 |

---

## 4. 训练控制参数 (Training Control)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| `MAX_EPISODES` | 5000 | 总训练Episodes | 控制训练总量 |
| `LOG_INTERVAL` | 10 | 日志打印间隔 (Episodes) | 控制输出频率 |
| `SAVE_INTERVAL` | 100 | 模型保存间隔 (Episodes) | 控制检查点频率 |
| `EVAL_INTERVAL` | 50 | 评估间隔 (Episodes) | 控制验证频率 |
| `DEVICE_NAME` | "cuda" | 训练设备 | CPU/CUDA/MPS |
| `SEED` | 42 | 随机种子 | 可复现性 |

---

## 5. 网络结构参数 (Model Architecture)

| 参数名 | 默认值 | 说明 | 影响 |
|--------|--------|------|------|
| `TASK_INPUT_DIM` | 7 | 任务特征维度 | comp, data, status等 |
| `VEH_INPUT_DIM` | 7 | 车辆特征维度 | vx, vy, queue等 |
| `EDGE_INPUT_DIM` | 2 | 边特征维度 | DAG边权重 |
| `RSU_INPUT_DIM` | 1 | RSU特征维度 | 负载等 |
| `EMBED_DIM` | 128 | 嵌入维度 | Transformer & GNN |
| `NUM_HEADS` | 4 | 注意力头数 | Multi-Head Attention |
| `NUM_LAYERS` | 3 | Transformer层数 | 模型深度 |
| `RESOURCE_RAW_DIM` | 14 | 资源原始特征维度 | 11原始 + 3时间预估 |
| `MAX_NEIGHBORS` | min(11, NUM_VEHICLES-1) | 最大邻居数 | 派生值 |
| `MAX_TARGETS` | 2 + MAX_NEIGHBORS | 最大目标数 | Local+RSU+Neighbors |

---

## 6. 归一化参数 (Normalization)

| 参数名 | 默认值 | 单位 | 说明 |
|--------|--------|------|------|
| `NORM_MAX_CPU` | 25.0e9 | Hz | CPU频率归一化基准 |
| `NORM_MAX_COMP` | 2.0e9 | cycles | 计算量归一化基准 |
| `NORM_MAX_DATA` | 5.0e6 | bits | 数据量归一化基准 |
| `NORM_MAX_RATE_V2I` | 50e6 | bps | V2I速率归一化基准 |
| `NORM_MAX_RATE_V2V` | 20e6 | bps | V2V速率归一化基准 |

---

## 📝 修改历史 (Change Log)

- **2026-01-02**: 删除所有Profile预设和模式参数，固化为单一配置
  - 删除 `PROFILE_REGISTRY`
  - 删除 `REWARD_MODE`, `BONUS_MODE`, `DIST_PENALTY_MODE`, `TIME_LIMIT_PENALTY_MODE`, `NORM_RATE_MODE`
  - 固化奖励函数为Delta CFT模式
  - 固化归一化为Static模式
  - 固化距离预警为启用状态
  - 添加详细的中英文注释

---

## 🔗 参考文献 (References)

1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
2. **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016
3. **VEC**: Mao et al., "A Survey on Mobile Edge Computing: The Communication Perspective", 2017
4. **DAG Offloading**: Chen et al., "Dependency-Aware Task Offloading and Service Caching in Mobile Edge Computing", 2020
5. **C-V2X**: 3GPP TS 36.213, ETSI EN 302 637-2

---

## 💡 调参建议 (Tuning Recommendations)

### 加快训练收敛 (Speed Up Convergence)
- 增大 `LR_ACTOR` 和 `LR_CRITIC` (但注意稳定性)
- 减小 `ENTROPY_COEF` (减少探索)
- 增大 `PPO_EPOCH` (更充分利用样本)

### 提高探索能力 (Increase Exploration)
- 增大 `ENTROPY_COEF` (0.03-0.05)
- 增大 `LOGIT_BIAS_RSU` 和 `LOGIT_BIAS_LOCAL` (强制探索)
- 减小 `CLIP_PARAM` (更保守的策略更新)

### 适应更难的任务 (Adapt to Harder Tasks)
- 增大 `DEADLINE_TIGHTENING_MIN` 和 `DEADLINE_TIGHTENING_MAX`
- 增大 `F_RSU` 和 `RSU_NUM_PROCESSORS` (提升RSU能力)
- 增大 `BW_V2I` 和 `BW_V2V` (提升通信能力)
- 减小 `MIN_COMP` 和 `MAX_COMP` (减轻任务负载)

### 增强模型容量 (Increase Model Capacity)
- 增大 `EMBED_DIM` (128 → 256)
- 增大 `NUM_LAYERS` (3 → 4 or 5)
- 增大 `NUM_HEADS` (4 → 8)
- **注意**: 需要更多训练时间和GPU内存

### 提高样本效率 (Improve Sample Efficiency)
- 增大 `GAMMA` (0.95 → 0.98 or 0.99)
- 增大 `GAE_LAMBDA` (0.95 → 0.98)
- 增大 `PPO_EPOCH` (4 → 6 or 8)
- 减小 `MINI_BATCH_SIZE` (增加更新频率)

