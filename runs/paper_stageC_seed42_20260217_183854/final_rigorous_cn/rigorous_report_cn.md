# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-200], 中期[201-400], 后期[401-600]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=0.1184, rank=1
- Greedy: mean=-0.0032, rank=2
- Static: mean=-0.0564, rank=3
- EFT: mean=-0.2376, rank=4
- Random: mean=-0.2838, rank=5
- CP-EFT: mean=-0.2838, rank=6
- Local-Only: mean=-0.3214, rank=7
- LB-Greedy: mean=-0.3385, rank=8

### 任务成功率
- 当前算法(MAPPO): mean=88.6000, rank=1
- Greedy: mean=72.5500, rank=2
- Local-Only: mean=70.4000, rank=3
- Static: mean=68.8000, rank=4
- EFT: mean=36.5500, rank=5
- CP-EFT: mean=36.3000, rank=6
- LB-Greedy: mean=28.1000, rank=7
- Random: mean=0.5500, rank=8

### 超时失败率
- 当前算法(MAPPO): mean=11.4000, rank=1
- Greedy: mean=27.4500, rank=2
- Local-Only: mean=29.6000, rank=3
- Static: mean=31.2000, rank=4
- EFT: mean=63.4500, rank=5
- CP-EFT: mean=63.7000, rank=6
- LB-Greedy: mean=71.9000, rank=7
- Random: mean=99.4500, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=1.5825, rank=1
- Greedy: mean=12.4187, rank=2
- Static: mean=13.5311, rank=3
- Local-Only: mean=15.6156, rank=4
- CP-EFT: mean=21.5663, rank=5
- EFT: mean=22.3186, rank=6
- LB-Greedy: mean=22.6907, rank=7
- Random: mean=25.1318, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=99.17%, sign-p=1.821e-34, welch-p=1.014e-84, boot95=[49.33, 55.13] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=7.510e-85, boot95=[49.69, 55.37] (显著)
- vs Greedy: 胜率=81.90%, sign-p=2.304e-11, welch-p=2.745e-16, boot95=[12.56, 19.73] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.132e-108, boot95=[57.58, 62.56] (显著)
- vs Local-Only: 胜率=92.73%, sign-p=6.838e-22, welch-p=1.230e-36, boot95=[15.14, 19.4] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.739e-141, boot95=[86.54, 89.12] (显著)
- vs Static: 胜率=82.24%, sign-p=8.481e-12, welch-p=6.417e-18, boot95=[15.58, 22.96] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.844e-110, boot95=[19.54, 20.44] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.219e-108, boot95=[20.36, 21.29] (显著)
- vs Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.519e-111, boot95=[10.64, 11.1] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.647e-115, boot95=[20.7, 21.54] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.943e-114, boot95=[13.73, 14.28] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=8.538e-192, boot95=[23.44, 23.66] (显著)
- vs Static: 胜率=100.00%, sign-p=1.505e-36, welch-p=9.583e-106, boot95=[11.61, 12.21] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.552e-50, boot95=[0.3651, 0.425] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.730e-74, boot95=[0.335, 0.3714] (显著)
- vs Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.785e-86, boot95=[0.1167, 0.1274] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.042e-41, boot95=[0.406, 0.4909] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.394e-107, boot95=[0.4282, 0.4563] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.254e-28, boot95=[0.3551, 0.4656] (显著)
- vs Static: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.421e-113, boot95=[0.1666, 0.1821] (显著)

### 任务成功率
- vs CP-EFT: 胜率=99.17%, sign-p=1.821e-34, welch-p=1.014e-84, boot95=[49.42, 55.25] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=7.510e-85, boot95=[49.71, 55.31] (显著)
- vs Greedy: 胜率=81.90%, sign-p=2.304e-11, welch-p=2.745e-16, boot95=[12.81, 19.5] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.132e-108, boot95=[57.54, 62.61] (显著)
- vs Local-Only: 胜率=92.73%, sign-p=6.838e-22, welch-p=1.230e-36, boot95=[15.04, 19.5] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.739e-141, boot95=[86.5, 89.17] (显著)
- vs Static: 胜率=82.24%, sign-p=8.481e-12, welch-p=6.417e-18, boot95=[15.42, 23.12] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.7523, P(local|on_task)=0.9996, P(rsu|on_task)=0.0002, P(v2v|on_task)=0.0002
- Greedy: on_task_rate=0.1953, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1791, P(local|on_task)=0.1948, P(rsu|on_task)=0.8052, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1361, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0914, P(local|on_task)=0.9183, P(rsu|on_task)=0.0487, P(v2v|on_task)=0.0330
- CP-EFT: on_task_rate=0.0901, P(local|on_task)=0.9290, P(rsu|on_task)=0.0416, P(v2v|on_task)=0.0293
- LB-Greedy: on_task_rate=0.0812, P(local|on_task)=0.9262, P(rsu|on_task)=0.0415, P(v2v|on_task)=0.0323
- Random: on_task_rate=0.0325, P(local|on_task)=0.0849, P(rsu|on_task)=0.1501, P(v2v|on_task)=0.7649
