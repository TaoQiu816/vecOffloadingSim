# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-333], 中期[334-666], 后期[667-1000]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=-0.0053, rank=1
- Greedy: mean=-0.0075, rank=2
- Static: mean=-0.0746, rank=3
- Local-Only: mean=-0.3205, rank=4
- EFT: mean=-0.7616, rank=5
- CP-EFT: mean=-1.0089, rank=6
- LB-Greedy: mean=-1.0308, rank=7
- Random: mean=-1.8127, rank=8

### 任务成功率
- 当前算法(MAPPO): mean=82.0500, rank=1
- Local-Only: mean=70.4000, rank=2
- Greedy: mean=70.0000, rank=3
- Static: mean=68.5000, rank=4
- EFT: mean=28.5000, rank=5
- CP-EFT: mean=25.1500, rank=6
- LB-Greedy: mean=21.4000, rank=7
- Random: mean=0.1500, rank=8

### 超时失败率
- 当前算法(MAPPO): mean=17.9500, rank=1
- Local-Only: mean=29.6000, rank=2
- Greedy: mean=30.0000, rank=3
- Static: mean=31.5000, rank=4
- EFT: mean=71.5000, rank=5
- CP-EFT: mean=74.8500, rank=6
- LB-Greedy: mean=78.6000, rank=7
- Random: mean=99.8500, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=11.3359, rank=1
- Greedy: mean=12.8368, rank=2
- Static: mean=13.9240, rank=3
- Local-Only: mean=15.6156, rank=4
- CP-EFT: mean=22.7229, rank=5
- EFT: mean=23.2357, rank=6
- LB-Greedy: mean=23.4042, rank=7
- Random: mean=25.4754, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=99.15%, sign-p=7.162e-34, welch-p=1.602e-68, boot95=[53.33, 61.46] (显著)
- vs EFT: 胜率=96.67%, sign-p=1.279e-29, welch-p=8.894e-65, boot95=[49.73, 58.11] (显著)
- vs Greedy: 胜率=66.96%, sign-p=3.510e-04, welch-p=7.286e-06, boot95=[6.853, 16.92] (显著)
- vs LB-Greedy: 胜率=99.15%, sign-p=1.420e-33, welch-p=2.714e-73, boot95=[57.38, 65.69] (显著)
- vs Local-Only: 胜率=74.78%, sign-p=9.759e-08, welch-p=4.624e-08, boot95=[7.478, 15.29] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.190e-77, boot95=[78.85, 85.83] (显著)
- vs Static: 胜率=74.34%, sign-p=2.201e-07, welch-p=1.477e-07, boot95=[8.373, 18.5] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.037e-117, boot95=[10.79, 11.81] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.940e-119, boot95=[11.5, 12.51] (显著)
- vs Greedy: 胜率=67.50%, sign-p=1.580e-04, welch-p=4.855e-08, boot95=[0.9064, 1.888] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.891e-126, boot95=[11.63, 12.55] (显著)
- vs Local-Only: 胜率=95.00%, sign-p=5.796e-27, welch-p=3.391e-45, boot95=[3.749, 4.667] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.035e-113, boot95=[13.71, 14.48] (显著)
- vs Static: 胜率=83.33%, sign-p=5.508e-14, welch-p=3.448e-21, boot95=[2.071, 3.041] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.872e-34, boot95=[0.8937, 1.119] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.063e-40, boot95=[0.6903, 0.8337] (显著)
- vs Greedy: 胜率=52.50%, sign-p=6.483e-01, welch-p=2.964e-01, boot95=[-0.002051, 0.006644] (不显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=8.888e-37, boot95=[0.9527, 1.172] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.974e-82, boot95=[0.3046, 0.332] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.690e-51, boot95=[1.689, 1.956] (显著)
- vs Static: 胜率=93.33%, sign-p=1.360e-24, welch-p=8.343e-33, boot95=[0.06013, 0.07771] (显著)

### 任务成功率
- vs CP-EFT: 胜率=99.15%, sign-p=7.162e-34, welch-p=1.602e-68, boot95=[53.17, 61.11] (显著)
- vs EFT: 胜率=96.67%, sign-p=1.279e-29, welch-p=8.894e-65, boot95=[49.52, 58.04] (显著)
- vs Greedy: 胜率=66.96%, sign-p=3.510e-04, welch-p=7.286e-06, boot95=[6.708, 17.04] (显著)
- vs LB-Greedy: 胜率=99.15%, sign-p=1.420e-33, welch-p=2.714e-73, boot95=[57.64, 65.75] (显著)
- vs Local-Only: 胜率=74.78%, sign-p=9.759e-08, welch-p=4.624e-08, boot95=[7.333, 15.08] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.190e-77, boot95=[78.79, 85.79] (显著)
- vs Static: 胜率=74.34%, sign-p=2.201e-07, welch-p=1.477e-07, boot95=[8.583, 18.42] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.2437, P(local|on_task)=0.0003, P(rsu|on_task)=0.9986, P(v2v|on_task)=0.0011
- Greedy: on_task_rate=0.2111, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1852, P(local|on_task)=0.2468, P(rsu|on_task)=0.7532, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1361, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0892, P(local|on_task)=0.9204, P(rsu|on_task)=0.0468, P(v2v|on_task)=0.0328
- CP-EFT: on_task_rate=0.0838, P(local|on_task)=0.9366, P(rsu|on_task)=0.0377, P(v2v|on_task)=0.0257
- LB-Greedy: on_task_rate=0.0769, P(local|on_task)=0.9296, P(rsu|on_task)=0.0396, P(v2v|on_task)=0.0307
- Random: on_task_rate=0.0253, P(local|on_task)=0.0764, P(rsu|on_task)=0.1191, P(v2v|on_task)=0.8044
