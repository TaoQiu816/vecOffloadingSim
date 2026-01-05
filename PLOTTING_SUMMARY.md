# 训练数据可视化总结

## 生成的图表

### 原有图表（plot_results.py）
1. **fig_convergence.png** - 收敛曲线
   - Reward曲线
   - Success Rate曲线
   
2. **fig_policy_evolution.png** - 策略演化
   - 决策分布随时间变化
   
3. **fig_training.png** - 训练诊断
   - Loss曲线
   - Entropy曲线

### 新增详细图表（generate_all_plots.py）

4. **fig_training_overview.png** - 训练总览（4×2）
   - Total Reward
   - Task Success Rate
   - Subtask Success Rate
   - V2V Subtask Success Rate
   - Local Decision Ratio
   - RSU Decision Ratio
   - V2V Decision Ratio
   - Training Loss

5. **fig_success_rate_analysis.png** - 成功率详细分析（2×2）
   - Task vs Subtask Success Rate对比
   - Vehicle-level Success Rate
   - Success Rate分布直方图
   - Success Rate按训练阶段分析

6. **fig_decision_distribution.png** - 决策分布分析（2×2）
   - Stacked Area Chart（堆叠面积图）
   - 各决策趋势曲线
   - 按训练阶段的Box Plot
   - 最终决策分布（最后50 episodes）

7. **fig_reward_analysis.png** - 奖励分析（2×2）
   - Total Reward with variance band
   - Average Step Reward
   - Reward分布直方图
   - Reward vs Success Rate散点图（带相关性）

8. **fig_multi_agent_metrics.png** - 多智能体指标（2×2）
   - Jain's Fairness Index
   - Reward Gap (Max - Min)
   - Collaboration Score
   - Agent Reward Range

9. **fig_system_metrics.png** - 系统指标（2×2）
   - Average Power Consumption
   - Average Queue Length
   - Episode Duration
   - Vehicle & Task Count

10. **fig_convergence_analysis.png** - 收敛性分析（2×2）
    - Reward收敛（Mean ± Std）
    - Reward方差趋势
    - Success Rate波动性
    - Policy Diversity (Decision Entropy)

11. **fig_baseline_comparison.png** - Baseline对比（如有数据）
    - Task SR对比
    - Total Reward对比
    - Subtask SR对比
    - 最终性能柱状图

## 总计

- **原有图表**: 3张
- **新增图表**: 8张
- **总计**: 11张

## 使用方法

### 为新训练run生成所有图表：

```bash
python generate_all_plots.py --run-dir runs/run_YYYYMMDD_HHMMSS
```

### 训练结束后自动生成：

训练脚本(train.py)会在训练结束时自动调用plot_results.py生成基础图表。

如需生成详细图表，手动运行：

```bash
python generate_all_plots.py --run-dir runs/run_YYYYMMDD_HHMMSS
```

## 图表说明

### 1. 训练总览
快速浏览所有关键指标，包括奖励、成功率、决策分布和Loss。

### 2. 成功率分析
深入分析Task和Subtask成功率，包括分布和训练阶段对比。

### 3. 决策分布
分析Local/RSU/V2V决策随训练的演化，帮助理解策略收敛。

### 4. 奖励分析
分析奖励信号的分布、方差和与成功率的相关性。

### 5. 多智能体指标
评估多智能体的公平性、协作和个体差异。

### 6. 系统指标
监控功耗、队列长度等物理系统指标。

### 7. 收敛性分析
评估训练的稳定性和策略多样性。

### 8. Baseline对比
与Random/Local-Only/Greedy baseline对比性能。

## 关键发现（当前训练run_20260105_112351）

从生成的图表可以看出：

1. **Task SR = 0%**: 整个训练过程无任务成功
2. **Subtask SR ~10%**: 90%子任务失败
3. **快速收敛到100% RSU**: Episode 27开始
4. **Reward信号弱**: 波动小，学习困难
5. **V2V未被充分探索**: 很少选择V2V

**根本原因**: 12车RSU队列拥塞导致deadline超时

**已应用修复**:
- BW_V2I: 50Mbps → 100Mbps
- DEADLINE_TIGHTENING: [2-3x] → [4-6x]

**预期改善**: Task SR 0% → 50-70%
