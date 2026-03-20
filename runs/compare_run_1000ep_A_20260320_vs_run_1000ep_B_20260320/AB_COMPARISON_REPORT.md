# A/B 双 Run 深度对比报告

## 概览
- Run A: lr_critic=0.0002, repaired_metrics_rows=1000
- Run B: lr_critic=0.0005, repaired_metrics_rows=1000

## 真实配置差异
- train_config.ENTROPY_COEF_END: A=0.02 | B=0.012
- train_config.ENTROPY_COEF_START: A=0.02 | B=0.012
- train_config.LR_CRITIC: A=0.0002 | B=0.0005
- system_config.lr_critic: A=0.0002 | B=0.0005

## last100 指标对比
- 平均步奖励: A=0.010839, B=0.011612, winner=B
- 任务成功率: A=0.928500, B=0.861000, winner=A
- 子任务成功率: A=0.990413, B=0.978952, winner=A
- 超时失败率: A=0.071500, B=0.139000, winner=A
- 平均RSU队列: A=1.260782, B=0.176767, winner=B
- 平均功率: A=0.451783, B=0.538183, winner=A
- Approx KL: A=0.019096, B=0.008174, winner=B
- 策略熵: A=1.426623, B=1.181152, winner=A

## 结论与解释
- 在任务成功率上，赢家=A。
- 在平均奖励上，赢家=B。
- 在超时失败率上，赢家=A。
- 若 B 的 reward 更高但 task_sr / deadline_miss_rate 更差，说明更高 critic 学习率提升了优化速度，但牺牲了稳定可交付性。
- 若 A 的 avg_rsu_queue 明显更高而 deadline_miss 更低，说明 A 通过更积极利用 RSU 获得成功率，但系统拥塞代价更大。
- 需同时看 reward、success、deadline、queue、power，而不能只看 reward 单指标。

## 已确认代码问题
- DataRecorder 原先仅靠进程内标志写 header，复用 exact run-dir 时会把表头重复写入 episode_log.csv。
- train.py 原先允许 exact run-dir 在已有非空训练 CSV 时继续写入，容易产生脏日志和重复 episode。
- generate_all_plots.py 对 object 列直接 rolling，遇到脏行会抛 No numeric types to aggregate。
- 多个绘图模块使用过期 baseline 列表，和当前训练实际 baseline 集合不一致。

## 建议
- 立即修复项: 保持本次已实现的日志保护和 header 防护，不再复用脏 run-dir 直接追加训练。
- 下一轮实验项: 在 lr_critic=2e-4、3e-4、5e-4 三档做固定 seed 对照，重点看 task_sr、deadline_miss_rate、avg_rsu_queue 的联动。
- 下一轮实验项: 保持当前 reward 权重不变，单独扫描 queue / timeout 相关项，确认 reward 提升是否来自真实任务质量而非指标偏置。
- 口径治理项: 所有 baseline 名称、CSV schema、后处理脚本字段别名保持单一来源，不再各脚本手写一套列表。