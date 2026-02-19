# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-200], 中期[201-400], 后期[401-600]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=0.0864, rank=1
- Greedy: mean=-0.0066, rank=2
- Static: mean=-0.0476, rank=3
- Local-Only: mean=-0.3038, rank=4
- EFT: mean=-0.8092, rank=5
- CP-EFT: mean=-1.0019, rank=6
- LB-Greedy: mean=-1.0172, rank=7
- Random: mean=-1.7739, rank=8

### 任务成功率
- 当前算法(MAPPO): mean=88.6500, rank=1
- Local-Only: mean=73.9500, rank=2
- Greedy: mean=57.9500, rank=3
- Static: mean=51.7000, rank=4
- EFT: mean=24.9500, rank=5
- CP-EFT: mean=21.3000, rank=6
- LB-Greedy: mean=18.1500, rank=7
- Random: mean=0.3500, rank=8

### 超时失败率
- 当前算法(MAPPO): mean=11.3500, rank=1
- Local-Only: mean=26.0500, rank=2
- Greedy: mean=42.0500, rank=3
- Static: mean=48.3000, rank=4
- EFT: mean=75.0500, rank=5
- CP-EFT: mean=78.7000, rank=6
- LB-Greedy: mean=81.8500, rank=7
- Random: mean=99.6500, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=1.5844, rank=1
- Greedy: mean=14.4069, rank=2
- Static: mean=15.3447, rank=3
- Local-Only: mean=16.2167, rank=4
- CP-EFT: mean=22.7850, rank=5
- EFT: mean=22.8722, rank=6
- LB-Greedy: mean=23.3295, rank=7
- Random: mean=25.1262, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.675e-131, boot95=[64.48, 68.92] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.448e-118, boot95=[61.58, 66.4] (显著)
- vs Greedy: 胜率=86.61%, sign-p=7.122e-16, welch-p=6.757e-28, boot95=[24.58, 33.19] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.166e-131, boot95=[68.23, 72.86] (显著)
- vs Local-Only: 胜率=92.73%, sign-p=6.838e-22, welch-p=1.179e-28, boot95=[11.83, 15.92] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.604e-139, boot95=[86.75, 89.29] (显著)
- vs Static: 胜率=93.10%, sign-p=1.649e-23, welch-p=8.365e-34, boot95=[30.69, 39.29] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.100e-142, boot95=[20.91, 21.43] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=9.206e-122, boot95=[20.96, 21.71] (显著)
- vs Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.414e-104, boot95=[12.49, 13.16] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.625e-140, boot95=[21.43, 21.98] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.554e-116, boot95=[14.31, 14.87] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=9.872e-200, boot95=[23.42, 23.62] (显著)
- vs Static: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.430e-105, boot95=[13.29, 13.94] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=7.997e-49, boot95=[1.005, 1.172] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.433e-44, boot95=[0.8162, 0.9689] (显著)
- vs Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.340e-85, boot95=[0.08868, 0.09792] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.004e-38, boot95=[0.9815, 1.208] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.297e-104, boot95=[0.3806, 0.407] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.283e-49, boot95=[1.716, 2.01] (显著)
- vs Static: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.231e-97, boot95=[0.128, 0.1419] (显著)

### 任务成功率
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.675e-131, boot95=[64.6, 69.08] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.448e-118, boot95=[61.77, 66.54] (显著)
- vs Greedy: 胜率=86.61%, sign-p=7.122e-16, welch-p=6.757e-28, boot95=[24.96, 33.04] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.166e-131, boot95=[68, 72.58] (显著)
- vs Local-Only: 胜率=92.73%, sign-p=6.838e-22, welch-p=1.179e-28, boot95=[11.75, 15.92] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.604e-139, boot95=[86.71, 89.25] (显著)
- vs Static: 胜率=93.10%, sign-p=1.649e-23, welch-p=8.365e-34, boot95=[30.64, 39.38] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.7529, P(local|on_task)=0.9977, P(rsu|on_task)=0.0013, P(v2v|on_task)=0.0011
- Greedy: on_task_rate=0.1792, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1675, P(local|on_task)=0.1348, P(rsu|on_task)=0.8652, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1311, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0852, P(local|on_task)=0.9255, P(rsu|on_task)=0.0376, P(v2v|on_task)=0.0369
- CP-EFT: on_task_rate=0.0805, P(local|on_task)=0.9405, P(rsu|on_task)=0.0299, P(v2v|on_task)=0.0296
- LB-Greedy: on_task_rate=0.0739, P(local|on_task)=0.9337, P(rsu|on_task)=0.0311, P(v2v|on_task)=0.0352
- Random: on_task_rate=0.0279, P(local|on_task)=0.0899, P(rsu|on_task)=0.1052, P(v2v|on_task)=0.8049
