# 不同学习率训练对比报告

算法首次标注：TERA-MAPPO (Topology-Enhanced and Resource-Aware MAPPO)

## 对比设置
- 比较对象：TERA-MAPPO 在不同评论器学习率下的训练过程。
- 固定设置：actor 学习率固定为 2×10⁻⁴。
- 统计窗口：尾部 100 回合。

## 核心结论
- 尾部平均奖励最高：lr_c=2×10⁻⁴，对应奖励 0.0108。
- 尾部任务成功率最高：lr_c=2×10⁻⁴，对应成功率 0.9285。
- 尾部超期率最低：lr_c=2×10⁻⁴，对应超期率 0.0715。
- 收敛速度最快：lr_c=3×10⁻⁴，稳定达到尾部性能 95% 的回合约为 46。

## 各学习率尾部统计
- lr_c=2×10⁻⁴: reward=0.0108, task_sr=0.9285, deadline_miss=0.0715, avg_rsu_queue=1.2608, task_sr_convergence_ep=176
- lr_c=3×10⁻⁴: reward=0.0106, task_sr=0.8960, deadline_miss=0.1040, avg_rsu_queue=1.5143, task_sr_convergence_ep=46
- lr_c=5×10⁻⁴: reward=0.0091, task_sr=0.9130, deadline_miss=0.0870, avg_rsu_queue=7.1309, task_sr_convergence_ep=337
