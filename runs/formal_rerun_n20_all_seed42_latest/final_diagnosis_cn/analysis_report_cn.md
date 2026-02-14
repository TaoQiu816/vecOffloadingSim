# 当前仿真诊断报告（formal_rerun_n20_all_seed42_latest）

## 关键结论
- 训练共 1000 episodes；末100ep: task_success_rate=0.8205, deadline_miss_rate=0.1795, mean_cft_est=11.3359。
- 末100ep动作分布：local=0.000254, rsu=0.998633, v2v=0.001113，存在明显RSU塌缩。
- 末100ep约束口径：illegal_action_rate=0.000000, unified_illegal_trigger_rate=0.000000, no_task_rate=0.756286。
- 干扰指标在末300ep基本为0（I_total_p95=0），干扰目标未被持续激活。

## 注意事项（统计口径）
- `final_rigorous_cn` 脚本会把 baseline 从120ep前向填充到1000ep用于对齐绘图；该处理可用于“视觉对齐”，但不适合作为全程统计显著性结论。
- 对正式论文对比，建议使用 `rl120_vs_baseline120.csv`（同样本长度）或把 baseline 实跑到与 RL 同样的评估长度。

## 产物
- rl_window_summary.csv
- baseline_summary.csv
- rl120_vs_baseline120.csv
- pairwise_core_metrics.csv