# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-333], 中期[334-666], 后期[667-1000]

## 末100关键指标排名（均值）
### 平均步奖励
- Greedy: mean=0.0472, rank=1
- CP-EFT: mean=0.0390, rank=2
- 当前算法(MAPPO): mean=0.0376, rank=3
- Static: mean=0.0374, rank=4
- EFT: mean=0.0327, rank=5
- LB-Greedy: mean=0.0326, rank=6
- Oracle-Min: mean=0.0314, rank=7
- Local-Only: mean=0.0002, rank=8
- Random: mean=-0.0021, rank=9

### 任务成功率
- Greedy: mean=85.0000, rank=1
- CP-EFT: mean=80.0000, rank=2
- Static: mean=78.5000, rank=3
- 当前算法(MAPPO): mean=74.0000, rank=4
- EFT: mean=65.5000, rank=5
- LB-Greedy: mean=65.5000, rank=5
- Oracle-Min: mean=65.5000, rank=5
- Local-Only: mean=20.5000, rank=6
- Random: mean=0.5000, rank=7

### 超时失败率
- Greedy: mean=15.0000, rank=1
- CP-EFT: mean=20.0000, rank=2
- Static: mean=21.5000, rank=3
- 当前算法(MAPPO): mean=26.0000, rank=4
- EFT: mean=34.5000, rank=5
- LB-Greedy: mean=34.5000, rank=5
- Oracle-Min: mean=34.5000, rank=5
- Local-Only: mean=79.5000, rank=6
- Random: mean=99.0000, rank=7

### 平均完工时间估计
- Greedy: mean=2.6720, rank=1
- 当前算法(MAPPO): mean=2.9485, rank=2
- EFT: mean=2.9534, rank=3
- LB-Greedy: mean=2.9707, rank=4
- Oracle-Min: mean=2.9775, rank=5
- CP-EFT: mean=3.0026, rank=6
- Static: mean=3.0282, rank=7
- Local-Only: mean=5.5254, rank=8
- Random: mean=21.9319, rank=9

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=44.44%, sign-p=1.000e+00, welch-p=1.833e-01, boot95=[-14.5, 1.5] (不显著)
- vs EFT: 胜率=66.67%, sign-p=5.078e-01, welch-p=1.077e-01, boot95=[-0.5, 17.5] (不显著)
- vs Greedy: 胜率=0.00%, sign-p=1.563e-02, welch-p=1.515e-02, boot95=[-19, -4] (显著)
- vs LB-Greedy: 胜率=66.67%, sign-p=5.078e-01, welch-p=1.077e-01, boot95=[-0.5, 17.5] (不显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.953e-03, welch-p=5.702e-10, boot95=[45, 61.5] (显著)
- vs Oracle-Min: 胜率=70.00%, sign-p=3.437e-01, welch-p=1.370e-01, boot95=[-1.5, 19] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.953e-03, welch-p=1.142e-09, boot95=[66.24, 78.5] (显著)
- vs Static: 胜率=33.33%, sign-p=5.078e-01, welch-p=2.847e-01, boot95=[-12.5, 3] (不显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=60.00%, sign-p=7.539e-01, welch-p=8.239e-01, boot95=[-0.3737, 0.5185] (不显著)
- vs EFT: 胜率=40.00%, sign-p=7.539e-01, welch-p=9.694e-01, boot95=[-0.213, 0.2384] (不显著)
- vs Greedy: 胜率=20.00%, sign-p=1.094e-01, welch-p=1.833e-02, boot95=[-0.477, -0.08617] (不显著)
- vs LB-Greedy: 胜率=50.00%, sign-p=1.000e+00, welch-p=8.617e-01, boot95=[-0.2059, 0.2653] (不显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.953e-03, welch-p=4.699e-11, boot95=[2.292, 2.874] (显著)
- vs Oracle-Min: 胜率=50.00%, sign-p=1.000e+00, welch-p=8.155e-01, boot95=[-0.1905, 0.2589] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.953e-03, welch-p=3.571e-05, boot95=[14.98, 24.09] (显著)
- vs Static: 胜率=40.00%, sign-p=7.539e-01, welch-p=6.353e-01, boot95=[-0.1999, 0.392] (不显著)

### 平均步奖励
- vs CP-EFT: 胜率=40.00%, sign-p=7.539e-01, welch-p=7.944e-01, boot95=[-0.01115, 0.007602] (不显著)
- vs EFT: 胜率=70.00%, sign-p=3.437e-01, welch-p=2.002e-01, boot95=[-0.002095, 0.01184] (不显著)
- vs Greedy: 胜率=10.00%, sign-p=2.148e-02, welch-p=1.574e-02, boot95=[-0.01601, -0.003305] (显著)
- vs LB-Greedy: 胜率=70.00%, sign-p=3.437e-01, welch-p=1.976e-01, boot95=[-0.002011, 0.01172] (不显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.953e-03, welch-p=1.460e-07, boot95=[0.03198, 0.04258] (显著)
- vs Oracle-Min: 胜率=70.00%, sign-p=3.437e-01, welch-p=1.062e-01, boot95=[-0.0005786, 0.01319] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.953e-03, welch-p=1.768e-07, boot95=[0.03452, 0.0449] (显著)
- vs Static: 胜率=50.00%, sign-p=1.000e+00, welch-p=9.477e-01, boot95=[-0.006284, 0.006795] (不显著)

### 任务成功率
- vs CP-EFT: 胜率=44.44%, sign-p=1.000e+00, welch-p=1.833e-01, boot95=[-15, 2] (不显著)
- vs EFT: 胜率=66.67%, sign-p=5.078e-01, welch-p=1.077e-01, boot95=[-1, 17.5] (不显著)
- vs Greedy: 胜率=0.00%, sign-p=1.563e-02, welch-p=1.515e-02, boot95=[-19, -4] (显著)
- vs LB-Greedy: 胜率=66.67%, sign-p=5.078e-01, welch-p=1.077e-01, boot95=[-1.5, 17.76] (不显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.953e-03, welch-p=5.702e-10, boot95=[44.5, 61] (显著)
- vs Oracle-Min: 胜率=70.00%, sign-p=3.437e-01, welch-p=1.370e-01, boot95=[-2.5, 18.5] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.953e-03, welch-p=1.646e-09, boot95=[66.5, 78.5] (显著)
- vs Static: 胜率=33.33%, sign-p=5.078e-01, welch-p=2.847e-01, boot95=[-12.5, 2.5] (不显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- Greedy: on_task_rate=0.4354, P(local|on_task)=0.5749, P(rsu|on_task)=0.4233, P(v2v|on_task)=0.0017
- 当前算法(MAPPO): on_task_rate=0.4155, P(local|on_task)=0.0004, P(rsu|on_task)=0.9983, P(v2v|on_task)=0.0013
- CP-EFT: on_task_rate=0.4081, P(local|on_task)=0.6734, P(rsu|on_task)=0.2917, P(v2v|on_task)=0.0349
- EFT: on_task_rate=0.3915, P(local|on_task)=0.6159, P(rsu|on_task)=0.3667, P(v2v|on_task)=0.0174
- Oracle-Min: on_task_rate=0.3903, P(local|on_task)=0.6165, P(rsu|on_task)=0.3650, P(v2v|on_task)=0.0185
- LB-Greedy: on_task_rate=0.3892, P(local|on_task)=0.6181, P(rsu|on_task)=0.3652, P(v2v|on_task)=0.0167
- Static: on_task_rate=0.3819, P(local|on_task)=0.0250, P(rsu|on_task)=0.9750, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.2055, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- Random: on_task_rate=0.0418, P(local|on_task)=0.1552, P(rsu|on_task)=0.1632, P(v2v|on_task)=0.6816
