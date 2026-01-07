# CLAUDE.md

本文件为 Claude Code 提供项目导航，详细文档请参阅 `docs/` 目录。

---

## 项目概述

**VEC DAG任务卸载仿真器** - 基于MAPPO多智能体强化学习的车联网边缘计算任务卸载系统。

**核心目标**：训练智能体学会将DAG结构计算任务卸载到最优执行位置（本地/RSU/邻近车辆），优化完成时间与能耗。

---

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python train.py --max-episodes 5000 --device cuda --seed 42

# 评估
python eval_baselines.py --model-path runs/run_XXX/models/best_model.pth

# 绘图
python plot_results.py --log-file runs/run_XXX/logs/training_stats.csv

# 测试
pytest tests/ -v

# TensorBoard
tensorboard --logdir runs/run_XXX/logs
```

---

## 文档地图

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECT_FLOW.md](docs/ARCHITECT_FLOW.md) | 系统架构、数据流、5阶段Step逻辑、核心文件索引 |
| [docs/PHYSICAL_MODELS.md](docs/PHYSICAL_MODELS.md) | C-V2X通信公式、计算模型、移动性模型、能耗模型 |
| [docs/RL_SPECS.md](docs/RL_SPECS.md) | 状态空间、动作空间、奖励函数、MAPPO超参数 |

---

## 核心文件速查

| 模块 | 入口文件 | 说明 |
|------|----------|------|
| 训练 | `train.py` | MAPPO训练主循环 |
| 环境 | `envs/vec_offloading_env.py` | Gymnasium环境（~3400行） |
| 智能体 | `agents/mappo_agent.py` | PPO更新逻辑 |
| 策略网络 | `models/offloading_policy.py` | 端到端神经网络 |
| 物理配置 | `configs/config.py` | SystemConfig类 |
| 训练配置 | `configs/train_config.py` | TrainConfig类 |

---

## 关键设计要点

1. **混合动作空间**：离散Target（Categorical）+ 连续Power（Beta分布）
2. **5阶段Step**：决策提交 → 边传输激活 → 通信服务 → 计算服务 → 时间推进
3. **Delta-CFT奖励**：基于关键路径剩余时间变化量设计
4. **Logit Bias退火**：解决Local/RSU vs V2V动作空间不平衡
5. **14维资源特征**：物理特征编码，无车辆ID泄漏

---

## 输出结构

```
runs/run_YYYYMMDD_HHMMSS/
├── logs/           # 训练日志（CSV/JSONL）
├── models/         # 模型检查点（best_model.pth）
├── plots/          # 可视化图表
└── tensorboard/    # TensorBoard日志
```

---

## 代码风格

- 保持响应简洁，仅输出代码
- 不生成任务后文档或变更总结
- 优先使用中文注释（专有名词可用英文）
