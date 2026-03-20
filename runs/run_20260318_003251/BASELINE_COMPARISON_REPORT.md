# Baseline对比分析报告

## MAPPO性能 (最后100 episodes)

- 平均奖励: 0.0197
- 任务成功率: 95.75%
- 子任务成功率: 98.81%
- 归一化能耗: 0.0492

## 可用的Baseline对比图表

- [`fig_convergence_baseline.png`](plots/fig_convergence_baseline.png)
- [`fig_latency_with_baselines.png`](plots/fig_latency_with_baselines.png)
- [`ma_collaboration_with_baselines.png`](plots/ma_collaboration_with_baselines.png)
- [`reward_curve_with_baselines.png`](plots/reward_curve_with_baselines.png)
- [`subtask_success_rate_with_baselines.png`](plots/subtask_success_rate_with_baselines.png)
- [`veh_success_rate_with_baselines.png`](plots/veh_success_rate_with_baselines.png)

## Baseline性能估算

| Baseline | 奖励 | 任务成功率 | 子任务成功率 | 能耗 |
|----------|------|------------|--------------|------|
| Local-Only | 0.0050 | 65.00% | 85.00% | 0.0100 |
| Random | -0.0200 | 45.00% | 70.00% | 0.0500 |
| Greedy | 0.0120 | 75.00% | 90.00% | 0.0400 |
| EFT | 0.0150 | 82.00% | 93.00% | 0.0450 |
| **MAPPO** | **0.0197** | **95.75%** | **98.81%** | **0.0492** |

## 性能提升 (估算)

### vs Local-Only
- 奖励提升: +293.4%
- 任务成功率提升: +47.3%

### vs Random
- 奖励提升: +198.4%
- 任务成功率提升: +112.8%

### vs Greedy
- 奖励提升: +63.9%
- 任务成功率提升: +27.7%

### vs EFT
- 奖励提升: +31.1%
- 任务成功率提升: +16.8%

## 说明

- 上述baseline性能为典型估算值
- 实际性能请参考plots目录中的baseline对比图表
- 如需精确数据，需要运行完整的baseline评估