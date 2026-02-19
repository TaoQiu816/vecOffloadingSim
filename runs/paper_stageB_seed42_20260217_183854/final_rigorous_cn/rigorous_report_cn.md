# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-200], 中期[201-400], 后期[401-600]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=0.0847, rank=1
- Greedy: mean=-0.0062, rank=2
- Static: mean=-0.0452, rank=3
- Local-Only: mean=-0.3038, rank=4
- EFT: mean=-0.7245, rank=5
- CP-EFT: mean=-0.9870, rank=6
- LB-Greedy: mean=-0.9905, rank=7
- Random: mean=-1.6732, rank=8

### 任务成功率
- 当前算法(MAPPO): mean=87.4000, rank=1
- Local-Only: mean=73.9500, rank=2
- Greedy: mean=58.2500, rank=3
- Static: mean=55.6500, rank=4
- EFT: mean=23.6500, rank=5
- CP-EFT: mean=23.4000, rank=6
- LB-Greedy: mean=15.2500, rank=7
- Random: mean=0.3000, rank=8

### 超时失败率
- 当前算法(MAPPO): mean=12.6000, rank=1
- Local-Only: mean=26.0500, rank=2
- Greedy: mean=41.7500, rank=3
- Static: mean=44.3500, rank=4
- EFT: mean=76.3500, rank=5
- CP-EFT: mean=76.6000, rank=6
- LB-Greedy: mean=84.7500, rank=7
- Random: mean=99.7000, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=1.5856, rank=1
- Greedy: mean=14.3969, rank=2
- Static: mean=15.1601, rank=3
- Local-Only: mean=16.2167, rank=4
- CP-EFT: mean=22.4310, rank=5
- EFT: mean=23.1266, rank=6
- LB-Greedy: mean=23.4535, rank=7
- Random: mean=25.1237, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.029e-123, boot95=[61.79, 66.25] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.915e-119, boot95=[62.27, 67.04] (显著)
- vs Greedy: 胜率=85.96%, sign-p=1.477e-15, welch-p=2.007e-26, boot95=[25.79, 34.25] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.896e-161, boot95=[70.88, 74.79] (显著)
- vs Local-Only: 胜率=85.84%, sign-p=2.545e-15, welch-p=1.216e-26, boot95=[11, 15.23] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.404e-136, boot95=[86.1, 88.75] (显著)
- vs Static: 胜率=85.34%, sign-p=2.957e-15, welch-p=2.707e-27, boot95=[26.23, 35.19] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.059e-133, boot95=[20.52, 21.12] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.763e-131, boot95=[21.26, 21.9] (显著)
- vs Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.934e-103, boot95=[12.49, 13.15] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.673e-141, boot95=[21.61, 22.15] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.786e-116, boot95=[14.31, 14.87] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=8.367e-201, boot95=[23.44, 23.63] (显著)
- vs Static: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.246e-101, boot95=[13.11, 13.82] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=9.979e-43, boot95=[0.9894, 1.189] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.855e-37, boot95=[0.7957, 0.9807] (显著)
- vs Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.274e-90, boot95=[0.08683, 0.09628] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.312e-43, boot95=[0.9983, 1.191] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.240e-103, boot95=[0.3788, 0.4055] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.146e-52, boot95=[1.645, 1.906] (显著)
- vs Static: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.024e-92, boot95=[0.1221, 0.1364] (显著)

### 任务成功率
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=6.029e-123, boot95=[61.69, 66.21] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.915e-119, boot95=[62.23, 66.92] (显著)
- vs Greedy: 胜率=85.96%, sign-p=1.477e-15, welch-p=2.007e-26, boot95=[25.54, 34.36] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.896e-161, boot95=[70.85, 74.96] (显著)
- vs Local-Only: 胜率=85.84%, sign-p=2.545e-15, welch-p=1.216e-26, boot95=[11.12, 15.27] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.404e-136, boot95=[86.12, 88.62] (显著)
- vs Static: 胜率=85.34%, sign-p=2.957e-15, welch-p=2.707e-27, boot95=[26.25, 35.38] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.7522, P(local|on_task)=0.9998, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0002
- Greedy: on_task_rate=0.1773, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1649, P(local|on_task)=0.1337, P(rsu|on_task)=0.8663, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1311, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0847, P(local|on_task)=0.9262, P(rsu|on_task)=0.0375, P(v2v|on_task)=0.0363
- CP-EFT: on_task_rate=0.0825, P(local|on_task)=0.9390, P(rsu|on_task)=0.0304, P(v2v|on_task)=0.0306
- LB-Greedy: on_task_rate=0.0728, P(local|on_task)=0.9347, P(rsu|on_task)=0.0307, P(v2v|on_task)=0.0345
- Random: on_task_rate=0.0276, P(local|on_task)=0.0899, P(rsu|on_task)=0.1053, P(v2v|on_task)=0.8048
