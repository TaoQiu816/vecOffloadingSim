# VEC Task Offloading Simulator

**车联网边缘计算任务卸载仿真器 (Vehicular Edge Computing Task Offloading Simulator)**

基于MAPPO的DAG任务卸载决策系统，用于动态车联网环境中的多智能体协作决策。

---

## 项目概述 (Project Overview)

本项目实现了一个完整的车联网边缘计算（VEC）仿真环境，用于研究和评估基于深度强化学习的DAG任务卸载策略。系统支持：

- **动态拓扑**：车辆移动、RSU覆盖、V2V/V2I通信
- **DAG任务**：有向无环图任务建模，支持依赖关系和关键路径
- **多智能体协作**：基于MAPPO的分布式决策
- **物理真实性**：C-V2X通信模型、瑞利衰落、干扰计算、队列系统

---

## 核心特性 (Key Features)

### 1. 物理仿真 (Physical Simulation)
- **通信模型**：C-V2X标准（V2I 20MHz, V2V 10MHz）
- **信道模型**：瑞利衰落 + 路径损耗 + 干扰
- **移动性模型**：截断正态分布速度（13.8±3.0 m/s）
- **资源异构**：车辆CPU 1-3 GHz，RSU 12 GHz（4核并行）

### 2. 强化学习 (Reinforcement Learning)
- **算法**：MAPPO (Multi-Agent PPO)
- **网络架构**：
  - DAG编码：边增强Transformer（Edge-Enhanced Transformer）
  - 资源编码：14维物理特征 + 角色嵌入（无ID泄漏）
  - Actor-Critic：交叉注意力 + 物理偏置
- **动作空间**：混合动作（离散Target + 连续Power）
- **奖励函数**：Delta-CFT（关键路径时间增量）+ 能耗正则化

### 3. 训练监控 (Training Monitoring)
- **参数自检**：启动时验证关键配置（RESOURCE_RAW_DIM, DEADLINE, LOGIT_BIAS等）
- **全指标记录**：CSV格式记录训练过程（reward, success_rate, loss, entropy等）
- **最佳模型保存**：基于成功率（50-ep滑动平均）自动保存
- **自动可视化**：训练结束后自动生成4张论文级图表

---

## 项目结构 (Project Structure)

```
vecOffloadingSim/
├── configs/                    # 配置文件
│   ├── config.py              # 系统参数（物理环境、资源、任务）
│   └── train_config.py        # 训练超参数（网络结构、PPO参数）
├── envs/                       # 仿真环境
│   ├── vec_offloading_env.py  # 主环境类
│   ├── entities/              # 实体类（Vehicle, RSU, TaskDAG）
│   └── modules/               # 功能模块（Channel, Queue, TimeCalculator）
├── models/                     # 神经网络模型
│   ├── offloading_policy.py   # 完整策略网络
│   ├── actor_critic.py        # Actor-Critic模块
│   ├── dag_embedding.py       # DAG特征嵌入
│   ├── edge_enhanced_transformer.py  # 边增强Transformer
│   └── resource_features.py   # 资源特征编码
├── agents/                     # 智能体
│   ├── mappo_agent.py         # MAPPO智能体
│   └── rollout_buffer.py      # 经验回放缓冲区（GAE）
├── baselines/                  # 基准策略
│   ├── random_policy.py       # 随机策略
│   ├── local_only_policy.py   # 全本地执行
│   └── greedy_policy.py       # 贪婪策略
├── utils/                      # 工具函数
│   ├── dag_generator.py       # DAG任务生成器
│   ├── data_recorder.py       # TensorBoard记录器
│   └── data_utils.py          # 数据处理工具
├── scripts/                    # 辅助脚本
│   ├── plot_key_metrics_v4.py # 绘图脚本（旧版）
│   └── tensorboard_ctl.sh     # TensorBoard控制脚本
├── tests/                      # 单元测试（53个测试文件）
├── train.py                    # 训练主脚本
├── eval_baselines.py           # 基准策略评估
├── plot_results.py             # 绘图脚本（新版）
└── README.md                   # 本文件
```

---

## 快速开始 (Quick Start)

### 1. 环境配置

**依赖安装：**
```bash
pip install -r requirements.txt
```

**主要依赖：**
- Python >= 3.8
- PyTorch >= 1.10
- NumPy, Pandas, Matplotlib, Seaborn
- TensorBoard（可选，用于可视化）

### 2. 训练模型

**基础训练：**
```bash
python train.py --max-episodes 5000 --device cuda --seed 42
```

**自定义参数：**
```bash
python train.py \
  --max-episodes 1000 \
  --log-interval 10 \
  --save-interval 100 \
  --device cuda \
  --seed 42
```

**启动时会打印参数验证：**
```
================================================================================
  STARTUP PARAMETER VERIFICATION
================================================================================
  RESOURCE_RAW_DIM:             14 (Expected: 14)
  DEADLINE_TIGHTENING_MIN:      0.70 (Expected: 0.70)
  DEADLINE_TIGHTENING_MAX:      0.80 (Expected: 0.80)
  LOGIT_BIAS_LOCAL:             1.0 (Expected: 1.0)
  LOGIT_BIAS_RSU:               2.0 (Expected: 2.0)
  F_RSU:                        12.0 GHz (Expected: 12.0 GHz)
  Device:                       cuda
================================================================================
```

### 3. 评估基准策略

```bash
python eval_baselines.py \
  --model-path runs/run_XXX/models/best_model.pth \
  --num-episodes 100
```

### 4. 可视化结果

**自动绘图（训练结束后自动执行）：**
```bash
python plot_results.py \
  --log-file logs/run_XXX/training_stats.csv \
  --output-dir plots/
```

**生成图表：**
- `fig_convergence.png` - 收敛曲线（Reward, Task Success Rate, Subtask Success Rate）
- `fig_policy_evolution.png` - 策略演化（Local/RSU/V2V比例堆叠面积图）
- `fig_physics.png` - 物理指标（Latency + Energy双轴图）
- `fig_training.png` - 训练诊断（Actor Loss, Critic Loss, Entropy）

### 5. TensorBoard监控

```bash
tensorboard --logdir runs/run_XXX/logs
```

---

## 配置说明 (Configuration)

### 关键参数 (Key Parameters)

#### 物理环境 (`configs/config.py`)
```python
MAP_SIZE = 1000.0              # 道路长度 (m)
NUM_VEHICLES = 12              # 车辆数
VEL_MEAN = 13.8                # 平均速度 (m/s, ~50km/h)
DT = 0.05                      # 时间步长 (s)
MAX_STEPS = 200                # Episode最大步数 (10秒)

BW_V2I = 20e6                  # V2I带宽 (20MHz)
BW_V2V = 10e6                  # V2V带宽 (10MHz)
V2V_RANGE = 300.0              # V2V通信范围 (m)
RSU_RANGE = 400.0              # RSU覆盖范围 (m)

MIN_VEHICLE_CPU_FREQ = 1.0e9   # 车辆CPU最小频率 (1GHz)
MAX_VEHICLE_CPU_FREQ = 3.0e9   # 车辆CPU最大频率 (3GHz)
F_RSU = 12.0e9                 # RSU CPU频率 (12GHz)

MIN_COMP = 0.8e9               # 子任务最小计算量 (0.8 Gcycles)
MAX_COMP = 2.5e9               # 子任务最大计算量 (2.5 Gcycles)
MIN_DATA = 1.0e6               # 子任务最小数据量 (1 Mbit)
MAX_DATA = 4.0e6               # 子任务最大数据量 (4 Mbits)

DEADLINE_TIGHTENING_MIN = 0.70 # Deadline紧缩系数最小值
DEADLINE_TIGHTENING_MAX = 0.80 # Deadline紧缩系数最大值
```

#### 训练超参数 (`configs/train_config.py`)
```python
EMBED_DIM = 128                # 嵌入维度
NUM_HEADS = 4                  # 注意力头数
NUM_LAYERS = 3                 # Transformer层数

LR_ACTOR = 3e-4                # Actor学习率
LR_CRITIC = 1e-3               # Critic学习率
GAMMA = 0.98                   # 折扣因子
GAE_LAMBDA = 0.95              # GAE平滑因子
CLIP_PARAM = 0.2               # PPO裁剪阈值
PPO_EPOCH = 5                  # PPO更新轮数
MINI_BATCH_SIZE = 128          # Mini-batch大小
ENTROPY_COEF = 0.03            # 熵正则化系数

LOGIT_BIAS_LOCAL = 1.0         # Local动作偏置
LOGIT_BIAS_RSU = 2.0           # RSU动作偏置
```

---

## 输出文件 (Output Files)

训练过程会在 `runs/run_YYYYMMDD_HHMMSS/` 目录下生成以下文件：

```
runs/run_20260102_120000/
├── logs/
│   ├── training_stats.csv     # 训练指标（用于绘图）
│   ├── metrics.csv            # 详细指标（包含物理量和诊断信息）
│   ├── config_snapshot.json   # 配置快照
│   └── env_reward.jsonl       # 环境奖励日志
├── models/
│   ├── best_model.pth         # 最佳模型（基于成功率）
│   ├── best_model_reward.pth  # 最佳模型（基于奖励）
│   └── model_ep*.pth          # 定期保存的检查点
├── plots/
│   ├── fig_convergence.png    # 收敛曲线
│   ├── fig_policy_evolution.png  # 策略演化
│   ├── fig_physics.png        # 物理指标
│   └── fig_training.png       # 训练诊断
└── tensorboard/               # TensorBoard日志
```

---

## 测试 (Testing)

项目包含53个单元测试，覆盖环境逻辑、网络模块、训练流程等。

**运行所有测试：**
```bash
pytest tests/ -v
```

**运行特定测试：**
```bash
pytest tests/test_env_overrides_take_effect.py -v
```

---

## 参考文献 (References)

### 强化学习算法
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using GAE" (2016)

### 网络架构
- **Transformer**: Vaswani et al., "Attention Is All You Need" (2017)
- **Actor-Critic**: Mnih et al., "Asynchronous Methods for Deep RL" (2016)

### 车联网与边缘计算
- **VEC**: Mao et al., "A Survey on Mobile Edge Computing" (2017)
- **DAG Offloading**: Chen et al., "Dependency-Aware Task Offloading" (2020)
- **C-V2X**: 3GPP TS 36.213, ETSI EN 302 637-2

---

## 常见问题 (FAQ)

**Q1: 训练收敛慢怎么办？**
- 检查 `LOGIT_BIAS` 是否合理（推荐 Local=1.0, RSU=2.0）
- 增加 `ENTROPY_COEF` 增强探索（推荐 0.03-0.05）
- 减小 `MINI_BATCH_SIZE` 增加随机性（推荐 64-128）

**Q2: 如何调整任务难度？**
- 修改 `DEADLINE_TIGHTENING_MIN/MAX` 控制Deadline紧迫性
- 修改 `MIN_COMP/MAX_COMP` 控制计算负载
- 修改 `MIN_DATA/MAX_DATA` 控制通信负载

**Q3: 如何禁用动态车辆到达？**
- 设置 `VEHICLE_ARRIVAL_RATE = 0` 禁用泊松到达

**Q4: 如何使用CPU训练？**
- 设置 `--device cpu` 或修改 `train_config.py` 中的 `DEVICE_NAME = "cpu"`

---

## 许可证 (License)

本项目仅供学术研究使用。

---

## 联系方式 (Contact)

如有问题或建议，请通过以下方式联系：
- Issue: 在GitHub上提交Issue
- Email: [您的邮箱]

---

**最后更新 (Last Updated)**: 2026-01-02
