# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-266], 中期[267-533], 后期[534-800]

## 末100关键指标排名（均值）
### 平均步奖励
- Greedy: mean=-0.0038, rank=1
- Static: mean=-0.0539, rank=2
- 当前算法(MAPPO): mean=-0.2320, rank=3
- EFT: mean=-0.2560, rank=4
- Random: mean=-0.2629, rank=5
- CP-EFT: mean=-0.2816, rank=6
- LB-Greedy: mean=-0.2901, rank=7
- Local-Only: mean=-0.3214, rank=8

### 任务成功率
- Greedy: mean=72.5500, rank=1
- Local-Only: mean=70.4000, rank=2
- Static: mean=68.7500, rank=3
- EFT: mean=38.0000, rank=4
- CP-EFT: mean=35.5000, rank=5
- LB-Greedy: mean=28.7000, rank=6
- Random: mean=0.5000, rank=7
- 当前算法(MAPPO): mean=0.1500, rank=8

### 超时失败率
- Greedy: mean=27.4500, rank=1
- Local-Only: mean=29.6000, rank=2
- Static: mean=31.2500, rank=3
- EFT: mean=62.0000, rank=4
- CP-EFT: mean=64.5000, rank=5
- LB-Greedy: mean=71.3000, rank=6
- Random: mean=99.5000, rank=7
- 当前算法(MAPPO): mean=99.8500, rank=8

### 平均完工时间估计
- Greedy: mean=12.5213, rank=1
- Static: mean=13.5308, rank=2
- Local-Only: mean=15.6156, rank=3
- CP-EFT: mean=21.7483, rank=4
- EFT: mean=22.3352, rank=5
- LB-Greedy: mean=22.7154, rank=6
- Random: mean=25.1150, rank=7
- 当前算法(MAPPO): mean=25.8056, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=0.00%, sign-p=1.505e-36, welch-p=5.837e-57, boot95=[-38.58, -33.77] (显著)
- vs EFT: 胜率=0.00%, sign-p=1.505e-36, welch-p=1.313e-59, boot95=[-40.58, -35.92] (显著)
- vs Greedy: 胜率=0.00%, sign-p=1.505e-36, welch-p=1.661e-78, boot95=[-75.98, -69.79] (显著)
- vs LB-Greedy: 胜率=0.00%, sign-p=1.505e-36, welch-p=3.228e-47, boot95=[-30.12, -25.75] (显著)
- vs Local-Only: 胜率=0.00%, sign-p=1.505e-36, welch-p=7.720e-105, boot95=[-72.75, -69.25] (显著)
- vs Random: 胜率=21.43%, sign-p=5.737e-02, welch-p=2.788e-02, boot95=[-0.625, -0.04167] (不显著)
- vs Static: 胜率=0.00%, sign-p=1.505e-36, welch-p=8.797e-68, boot95=[-73.71, -66.64] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=0.00%, sign-p=1.505e-36, welch-p=5.862e-41, boot95=[-4.602, -3.759] (显著)
- vs EFT: 胜率=5.00%, sign-p=5.796e-27, welch-p=2.472e-27, boot95=[-4.007, -2.947] (显著)
- vs Greedy: 胜率=0.00%, sign-p=1.505e-36, welch-p=7.545e-149, boot95=[-13.59, -12.99] (显著)
- vs LB-Greedy: 胜率=2.50%, sign-p=4.335e-31, welch-p=2.738e-25, boot95=[-3.34, -2.457] (显著)
- vs Local-Only: 胜率=0.00%, sign-p=1.505e-36, welch-p=4.138e-120, boot95=[-10.53, -9.898] (显著)
- vs Random: 胜率=25.83%, sign-p=1.119e-07, welch-p=1.088e-11, boot95=[-0.8382, -0.485] (显著)
- vs Static: 胜率=0.00%, sign-p=1.505e-36, welch-p=5.413e-124, boot95=[-12.7, -12.01] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=67.50%, sign-p=1.580e-04, welch-p=8.373e-02, boot95=[-0.005283, 0.08559] (显著)
- vs EFT: 胜率=62.50%, sign-p=7.847e-03, welch-p=4.051e-01, boot95=[-0.02603, 0.06069] (显著)
- vs Greedy: 胜率=0.00%, sign-p=1.505e-36, welch-p=2.116e-22, boot95=[-0.2687, -0.1949] (显著)
- vs LB-Greedy: 胜率=68.33%, sign-p=7.292e-05, welch-p=4.496e-02, boot95=[-0.003403, 0.1037] (显著)
- vs Local-Only: 胜率=75.00%, sign-p=3.773e-08, welch-p=1.959e-05, boot95=[0.04826, 0.1279] (显著)
- vs Random: 胜率=55.83%, sign-p=2.352e-01, welch-p=2.371e-01, boot95=[-0.0211, 0.08579] (不显著)
- vs Static: 胜率=10.00%, sign-p=1.780e-20, welch-p=4.240e-16, boot95=[-0.2221, -0.1454] (显著)

### 任务成功率
- vs CP-EFT: 胜率=0.00%, sign-p=1.505e-36, welch-p=5.837e-57, boot95=[-38.54, -33.67] (显著)
- vs EFT: 胜率=0.00%, sign-p=1.505e-36, welch-p=1.313e-59, boot95=[-40.62, -35.75] (显著)
- vs Greedy: 胜率=0.00%, sign-p=1.505e-36, welch-p=1.661e-78, boot95=[-76.15, -69.85] (显著)
- vs LB-Greedy: 胜率=0.00%, sign-p=1.505e-36, welch-p=3.228e-47, boot95=[-30.29, -25.67] (显著)
- vs Local-Only: 胜率=0.00%, sign-p=1.505e-36, welch-p=7.720e-105, boot95=[-72.73, -69.21] (显著)
- vs Random: 胜率=21.43%, sign-p=5.737e-02, welch-p=2.788e-02, boot95=[-0.625, -0.04167] (不显著)
- vs Static: 胜率=0.00%, sign-p=1.505e-36, welch-p=8.797e-68, boot95=[-73.9, -66.46] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- Greedy: on_task_rate=0.1953, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1787, P(local|on_task)=0.1949, P(rsu|on_task)=0.8051, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1361, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0930, P(local|on_task)=0.9166, P(rsu|on_task)=0.0491, P(v2v|on_task)=0.0343
- CP-EFT: on_task_rate=0.0881, P(local|on_task)=0.9309, P(rsu|on_task)=0.0407, P(v2v|on_task)=0.0284
- LB-Greedy: on_task_rate=0.0813, P(local|on_task)=0.9261, P(rsu|on_task)=0.0418, P(v2v|on_task)=0.0321
- Random: on_task_rate=0.0325, P(local|on_task)=0.0849, P(rsu|on_task)=0.1505, P(v2v|on_task)=0.7645
- 当前算法(MAPPO): on_task_rate=0.0298, P(local|on_task)=0.0351, P(rsu|on_task)=0.0864, P(v2v|on_task)=0.8785
