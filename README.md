# 车联网任务卸载仿真系统 (Vehicular Offloading Simulation)

基于MAPPO的多车辆协同DAG任务卸载强化学习系统

## 项目结构

```
vecOffloadingSim/
├── configs/              # 配置文件
│   ├── config.py        # 系统参数配置
│   └── train_config.py  # 训练超参数配置
├── envs/                # 仿真环境
│   ├── entities/        # 实体类（车辆、RSU、任务）
│   ├── modules/         # 功能模块（信道、队列、时间计算）
│   └── vec_offloading_env.py  # 主环境类
├── models/              # 神经网络模型
│   ├── offloading_policy.py   # 完整策略网络
│   ├── actor_critic.py        # Actor-Critic架构
│   ├── dag_embedding.py       # DAG特征嵌入
│   ├── edge_enhanced_transformer.py  # 边增强Transformer
│   └── resource_features.py   # 资源特征编码
├── baselines/           # 基准策略
│   ├── random_policy.py       # 随机策略
│   ├── local_only_policy.py   # 全本地执行
│   └── greedy_policy.py       # 贪婪策略
├── utils/               # 工具函数
│   ├── dag_generator.py       # DAG任务生成器
│   ├── data_recorder.py       # 数据记录与可视化
│   └── plot_baseline_comparison.py  # 基准对比绘图
├── train.py             # 训练脚本
└── eval_baselines.py    # 基准策略评估脚本
```

## 核心特性

### 仿真环境
- **多车辆协同**: 20辆车辆，支持V2V/V2I通信
- **DAG任务模型**: 8-12节点的有向无环图任务，考虑依赖关系
- **真实信道模型**: 路径损耗、瑞利衰落、干扰建模
- **队列系统**: 基于计算量(cycles)的队列管理，科学准确
- **奖励函数**: 效率收益 + 拥塞惩罚 + 约束惩罚

### 神经网络架构
- **DAG特征嵌入**: 位置编码、状态编码、拓扑位置编码
- **边增强Transformer**: 融合边特征和空间距离的注意力机制
- **资源特征编码**: 统一的9维资源节点特征 + ID嵌入
- **混合动作空间**: 离散卸载目标(Categorical) + 连续功率控制(Beta分布)
- **Actor-Critic**: 跨注意力融合 + 物理偏置注入

### 基准策略
- **Random**: 随机卸载策略
- **Local-Only**: 全本地执行
- **Greedy**: 选择计算能力最强的节点

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python train.py
```
训练过程会：
- 自动评估基准策略
- 保存训练数据到 `data/` 目录
- 生成TensorBoard日志（如果已安装tensorboard）
- 定期保存模型检查点

### 3. 评估基准策略
```bash
python eval_baselines.py
```
生成评估结果到 `eval_results/baseline_comparison.json`

### 4. 生成对比图表
```bash
python utils/plot_baseline_comparison.py
```
生成对比图表到 `eval_results/plots/`

## 主要参数

### 仿真环境参数 (`configs/config.py`)
- `NUM_VEHICLES`: 车辆数量 (20)
- `NUM_RSU`: RSU数量 (3)
- `MAP_SIZE`: 道路长度 (1000m)
- `MAX_STEPS`: 每回合步数 (200)
- `DT`: 仿真步长 (0.1s)

### 通信参数
- `BW_V2I`: V2I带宽 (20MHz)
- `BW_V2V`: V2V带宽 (10MHz)
- `TX_POWER_UP_DBM`: 上行发射功率 (23dBm)
- `V2V_INTERFERENCE_DBM`: V2V背景干扰 (-90dBm)

### DAG任务参数
- `MIN_NODES/MAX_NODES`: 节点数 (8-12)
- `MIN_COMP/MAX_COMP`: 计算量 (0.1e9-0.3e9 cycles)
- `DEADLINE_TIGHTENING_FACTOR`: Deadline紧缩因子 (0.8)

### 训练参数 (`configs/train_config.py`)
- `LR_ACTOR/LR_CRITIC`: 学习率 (3e-4 / 1e-3)
- `GAMMA`: 折扣因子 (0.95)
- `ENTROPY_COEF`: 熵系数 (0.02)
- `MAX_EPISODES`: 训练回合数 (5000)

## 输出文件

### 训练输出
- `data/MAPPO_DAG_*/`
  - `config.json`: 配置参数
  - `episode_log.csv`: 每回合详细数据
  - `step_log.csv`: 每步详细数据
  - `models/`: 模型检查点
  - `plots/`: 训练曲线（含基准策略对比）
  - `tb_logs/`: TensorBoard日志

### 评估输出
- `eval_results/`
  - `baseline_comparison.json`: 基准策略评估结果
  - `plots/`: 对比图表（奖励、成功率、决策分布等）

## 关键改进

1. **队列系统**: 从任务个数限制改为计算量限制，更科学准确
2. **V2V信道**: 从相对因子改为绝对干扰功率(dBm)，符合物理实际
3. **Beta分布**: 连续功率控制使用Beta分布，确保可学习性
4. **基准对比**: 训练收敛图中自动绘制基准策略水平线
5. **TensorBoard**: 可选的实时监控支持

## 系统要求

- Python 3.8+
- PyTorch 1.10+
- CUDA (可选，用于GPU加速)
- 8GB+ RAM
- 建议使用GPU进行训练（RTX 3090或更高）

## 注意事项

1. 首次运行会生成大量数据，确保有足够磁盘空间
2. TensorBoard为可选依赖，未安装不影响训练
3. 训练过程中会定期保存模型，可随时中断恢复
4. 基准策略评估在训练开始时自动执行


