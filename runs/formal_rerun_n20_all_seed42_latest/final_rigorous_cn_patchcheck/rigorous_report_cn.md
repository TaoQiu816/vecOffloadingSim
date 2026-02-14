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
- Greedy: mean=-0.0092, rank=2
- Static: mean=-0.0742, rank=3
- Local-Only: mean=-0.3205, rank=4
- EFT: mean=-2.9477, rank=5
- LB-Greedy: mean=-3.0077, rank=6
- CP-EFT: mean=-3.3500, rank=7
- Random: mean=-7.6306, rank=8

### 任务成功率
- 当前算法(MAPPO): mean=82.0500, rank=1
- Local-Only: mean=70.4000, rank=2
- Greedy: mean=68.1000, rank=3
- Static: mean=67.2000, rank=4
- CP-EFT: mean=27.2500, rank=5
- EFT: mean=25.3000, rank=6
- LB-Greedy: mean=22.2500, rank=7
- Random: mean=0.0000, rank=8

### 超时失败率
- 当前算法(MAPPO): mean=17.9500, rank=1
- Local-Only: mean=29.6000, rank=2
- Greedy: mean=31.9000, rank=3
- Static: mean=32.8000, rank=4
- CP-EFT: mean=72.7500, rank=5
- EFT: mean=74.7000, rank=6
- LB-Greedy: mean=77.7500, rank=7
- Random: mean=100.0000, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=11.3359, rank=1
- Greedy: mean=13.1011, rank=2
- Static: mean=14.0483, rank=3
- Local-Only: mean=15.6156, rank=4
- CP-EFT: mean=22.6129, rank=5
- EFT: mean=23.5698, rank=6
- LB-Greedy: mean=23.6109, rank=7
- Random: mean=25.5725, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=97.41%, sign-p=6.265e-30, welch-p=1.582e-66, boot95=[51.27, 60.06] (显著)
- vs EFT: 胜率=99.16%, sign-p=3.611e-34, welch-p=1.254e-68, boot95=[53.42, 61.52] (显著)
- vs Greedy: 胜率=72.73%, sign-p=2.018e-06, welch-p=1.506e-06, boot95=[7.917, 17.94] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=3.009e-36, welch-p=1.744e-72, boot95=[56.88, 65.4] (显著)
- vs Local-Only: 胜率=74.78%, sign-p=9.759e-08, welch-p=4.624e-08, boot95=[7.478, 15.29] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.336e-77, boot95=[78.98, 85.94] (显著)
- vs Static: 胜率=75.65%, sign-p=3.184e-08, welch-p=7.142e-08, boot95=[9.125, 19.42] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=8.307e-109, boot95=[10.66, 11.72] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.553e-127, boot95=[11.85, 12.79] (显著)
- vs Greedy: 胜率=72.50%, sign-p=8.680e-07, welch-p=2.709e-09, boot95=[1.054, 2.115] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.630e-128, boot95=[11.87, 12.82] (显著)
- vs Local-Only: 胜率=95.00%, sign-p=5.796e-27, welch-p=3.391e-45, boot95=[3.749, 4.667] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.539e-115, boot95=[13.79, 14.55] (显著)
- vs Static: 胜率=85.00%, sign-p=1.976e-15, welch-p=1.310e-21, boot95=[2.153, 3.128] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.599e-45, boot95=[3.068, 3.646] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=7.092e-48, boot95=[2.748, 3.224] (显著)
- vs Greedy: 胜率=57.50%, sign-p=1.203e-01, welch-p=1.249e-01, boot95=[-0.0008415, 0.00747] (不显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.221e-46, boot95=[2.829, 3.356] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.974e-82, boot95=[0.3046, 0.332] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.907e-66, boot95=[7.206, 8.002] (显著)
- vs Static: 胜率=95.00%, sign-p=5.796e-27, welch-p=2.826e-33, boot95=[0.05866, 0.07557] (显著)

### 任务成功率
- vs CP-EFT: 胜率=97.41%, sign-p=6.265e-30, welch-p=1.582e-66, boot95=[51.38, 59.62] (显著)
- vs EFT: 胜率=99.16%, sign-p=3.611e-34, welch-p=1.254e-68, boot95=[53.35, 61.38] (显著)
- vs Greedy: 胜率=72.73%, sign-p=2.018e-06, welch-p=1.506e-06, boot95=[7.583, 18.23] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=3.009e-36, welch-p=1.744e-72, boot95=[57.23, 65.36] (显著)
- vs Local-Only: 胜率=74.78%, sign-p=9.759e-08, welch-p=4.624e-08, boot95=[7.333, 15.08] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.336e-77, boot95=[78.94, 85.92] (显著)
- vs Static: 胜率=75.65%, sign-p=3.184e-08, welch-p=7.142e-08, boot95=[9.25, 19.17] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.2437, P(local|on_task)=0.0003, P(rsu|on_task)=0.9986, P(v2v|on_task)=0.0011
- Greedy: on_task_rate=0.2095, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1863, P(local|on_task)=0.2476, P(rsu|on_task)=0.7524, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1361, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- CP-EFT: on_task_rate=0.0826, P(local|on_task)=0.9389, P(rsu|on_task)=0.0363, P(v2v|on_task)=0.0248
- EFT: on_task_rate=0.0825, P(local|on_task)=0.9280, P(rsu|on_task)=0.0429, P(v2v|on_task)=0.0291
- LB-Greedy: on_task_rate=0.0755, P(local|on_task)=0.9308, P(rsu|on_task)=0.0389, P(v2v|on_task)=0.0303
- Random: on_task_rate=0.0244, P(local|on_task)=0.0764, P(rsu|on_task)=0.1192, P(v2v|on_task)=0.8044
