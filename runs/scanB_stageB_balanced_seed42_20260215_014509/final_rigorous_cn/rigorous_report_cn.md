# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-266], 中期[267-533], 后期[534-800]

## 末100关键指标排名（均值）
### 平均步奖励
- Greedy: mean=-0.0076, rank=1
- 当前算法(MAPPO): mean=-0.0112, rank=2
- Static: mean=-0.0478, rank=3
- Local-Only: mean=-0.3038, rank=4
- EFT: mean=-0.8509, rank=5
- CP-EFT: mean=-0.8955, rank=6
- LB-Greedy: mean=-1.1543, rank=7
- Random: mean=-1.7463, rank=8

### 任务成功率
- Local-Only: mean=73.9500, rank=1
- 当前算法(MAPPO): mean=65.2000, rank=2
- Greedy: mean=55.6000, rank=3
- Static: mean=53.3500, rank=4
- EFT: mean=24.6000, rank=5
- CP-EFT: mean=24.0000, rank=6
- LB-Greedy: mean=15.6000, rank=7
- Random: mean=0.2000, rank=8

### 超时失败率
- Local-Only: mean=26.0500, rank=1
- 当前算法(MAPPO): mean=34.8000, rank=2
- Greedy: mean=44.4000, rank=3
- Static: mean=46.6500, rank=4
- EFT: mean=75.4000, rank=5
- CP-EFT: mean=76.0000, rank=6
- LB-Greedy: mean=84.4000, rank=7
- Random: mean=99.8000, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=13.8006, rank=1
- Greedy: mean=14.4971, rank=2
- Static: mean=15.0908, rank=3
- Local-Only: mean=16.2167, rank=4
- CP-EFT: mean=22.2949, rank=5
- EFT: mean=22.9816, rank=6
- LB-Greedy: mean=23.5823, rank=7
- Random: mean=25.1215, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=96.49%, sign-p=6.664e-28, welch-p=8.736e-39, boot95=[36.44, 45.71] (显著)
- vs EFT: 胜率=94.12%, sign-p=1.796e-25, welch-p=6.786e-39, boot95=[36.41, 45.79] (显著)
- vs Greedy: 胜率=61.26%, sign-p=2.230e-02, welch-p=2.693e-03, boot95=[3.103, 14.87] (显著)
- vs LB-Greedy: 胜率=98.28%, sign-p=1.634e-31, welch-p=6.195e-48, boot95=[45.19, 54.23] (显著)
- vs Local-Only: 胜率=44.64%, sign-p=2.986e-01, welch-p=1.127e-04, boot95=[-13.96, -4.77] (不显著)
- vs Random: 胜率=100.00%, sign-p=3.009e-36, welch-p=2.302e-57, boot95=[60.77, 69.15] (显著)
- vs Static: 胜率=59.82%, sign-p=4.674e-02, welch-p=3.603e-04, boot95=[4.708, 16.56] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=99.17%, sign-p=1.821e-34, welch-p=2.014e-99, boot95=[8.126, 9.032] (显著)
- vs EFT: 胜率=99.17%, sign-p=1.821e-34, welch-p=1.388e-103, boot95=[8.845, 9.802] (显著)
- vs Greedy: 胜率=60.83%, sign-p=2.208e-02, welch-p=5.282e-03, boot95=[0.2154, 1.079] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.315e-114, boot95=[9.302, 10.11] (显著)
- vs Local-Only: 胜率=80.83%, sign-p=5.264e-12, welch-p=1.966e-21, boot95=[1.915, 2.793] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.563e-105, boot95=[10.96, 11.66] (显著)
- vs Static: 胜率=69.75%, sign-p=1.961e-05, welch-p=3.856e-07, boot95=[0.7885, 1.705] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.033e-41, boot95=[0.8078, 0.9742] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.078e-34, boot95=[0.7267, 0.9115] (显著)
- vs Greedy: 胜率=40.83%, sign-p=5.478e-02, welch-p=4.701e-02, boot95=[-0.00676, -6.864e-05] (不显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.299e-41, boot95=[1.031, 1.252] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.752e-81, boot95=[0.283, 0.3089] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.872e-46, boot95=[1.567, 1.846] (显著)
- vs Static: 胜率=85.83%, sign-p=3.407e-16, welch-p=9.289e-22, boot95=[0.0306, 0.04352] (显著)

### 任务成功率
- vs CP-EFT: 胜率=96.49%, sign-p=6.664e-28, welch-p=8.736e-39, boot95=[36.31, 45.75] (显著)
- vs EFT: 胜率=94.12%, sign-p=1.796e-25, welch-p=6.786e-39, boot95=[36.58, 45.46] (显著)
- vs Greedy: 胜率=61.26%, sign-p=2.230e-02, welch-p=2.693e-03, boot95=[3.061, 14.88] (显著)
- vs LB-Greedy: 胜率=98.28%, sign-p=1.634e-31, welch-p=6.195e-48, boot95=[45.25, 54.06] (显著)
- vs Local-Only: 胜率=44.64%, sign-p=2.986e-01, welch-p=1.127e-04, boot95=[-13.96, -4.561] (不显著)
- vs Random: 胜率=100.00%, sign-p=3.009e-36, welch-p=2.302e-57, boot95=[60.67, 69.06] (显著)
- vs Static: 胜率=59.82%, sign-p=4.674e-02, welch-p=3.603e-04, boot95=[5.311, 16.81] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.1955, P(local|on_task)=0.0004, P(rsu|on_task)=0.9982, P(v2v|on_task)=0.0014
- Greedy: on_task_rate=0.1783, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1673, P(local|on_task)=0.1328, P(rsu|on_task)=0.8672, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1311, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0839, P(local|on_task)=0.9266, P(rsu|on_task)=0.0364, P(v2v|on_task)=0.0370
- CP-EFT: on_task_rate=0.0817, P(local|on_task)=0.9396, P(rsu|on_task)=0.0306, P(v2v|on_task)=0.0298
- LB-Greedy: on_task_rate=0.0713, P(local|on_task)=0.9362, P(rsu|on_task)=0.0299, P(v2v|on_task)=0.0339
- Random: on_task_rate=0.0271, P(local|on_task)=0.0899, P(rsu|on_task)=0.1053, P(v2v|on_task)=0.8048
