# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-266], 中期[267-533], 后期[534-800]

## 末100关键指标排名（均值）
### 平均步奖励
- Greedy: mean=-0.0085, rank=1
- 当前算法(MAPPO): mean=-0.0109, rank=2
- Static: mean=-0.0455, rank=3
- Local-Only: mean=-0.3038, rank=4
- EFT: mean=-0.7827, rank=5
- CP-EFT: mean=-0.9296, rank=6
- LB-Greedy: mean=-1.0278, rank=7
- Random: mean=-1.7929, rank=8

### 任务成功率
- Local-Only: mean=73.9500, rank=1
- 当前算法(MAPPO): mean=63.7500, rank=2
- Greedy: mean=54.3000, rank=3
- Static: mean=54.2500, rank=4
- EFT: mean=25.2000, rank=5
- CP-EFT: mean=23.1000, rank=6
- LB-Greedy: mean=16.1000, rank=7
- Random: mean=0.2000, rank=8

### 超时失败率
- Local-Only: mean=26.0500, rank=1
- 当前算法(MAPPO): mean=36.2500, rank=2
- Greedy: mean=45.7000, rank=3
- Static: mean=45.7500, rank=4
- EFT: mean=74.8000, rank=5
- CP-EFT: mean=76.9000, rank=6
- LB-Greedy: mean=83.9000, rank=7
- Random: mean=99.8000, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=13.7417, rank=1
- Greedy: mean=14.5238, rank=2
- Static: mean=14.9808, rank=3
- Local-Only: mean=16.2167, rank=4
- CP-EFT: mean=22.6069, rank=5
- EFT: mean=23.1248, rank=6
- LB-Greedy: mean=23.3984, rank=7
- Random: mean=25.1271, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=94.83%, sign-p=7.547e-26, welch-p=1.989e-39, boot95=[37.27, 46.9] (显著)
- vs EFT: 胜率=92.37%, sign-p=5.855e-23, welch-p=1.113e-37, boot95=[36.33, 45.31] (显著)
- vs Greedy: 胜率=64.60%, sign-p=2.456e-03, welch-p=1.082e-03, boot95=[4.083, 15.13] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=4.815e-35, welch-p=1.653e-46, boot95=[44.81, 53.69] (显著)
- vs Local-Only: 胜率=41.07%, sign-p=7.213e-02, welch-p=1.705e-04, boot95=[-13.67, -4.52] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.048e-56, boot95=[60.6, 69.33] (显著)
- vs Static: 胜率=62.16%, sign-p=1.323e-02, welch-p=2.143e-03, boot95=[3.292, 15.36] (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.029e-107, boot95=[8.544, 9.425] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=7.908e-112, boot95=[9.124, 9.999] (显著)
- vs Greedy: 胜率=61.67%, sign-p=1.338e-02, welch-p=7.362e-04, boot95=[0.343, 1.261] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=4.598e-115, boot95=[9.395, 10.21] (显著)
- vs Local-Only: 胜率=84.17%, sign-p=1.075e-14, welch-p=7.102e-24, boot95=[2.113, 2.965] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=9.385e-106, boot95=[11.13, 11.8] (显著)
- vs Static: 胜率=70.83%, sign-p=5.733e-06, welch-p=6.973e-07, boot95=[0.7562, 1.708] (显著)

### 平均步奖励
- vs CP-EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.785e-43, boot95=[0.8555, 1.019] (显著)
- vs EFT: 胜率=100.00%, sign-p=1.505e-36, welch-p=3.014e-43, boot95=[0.7556, 0.9079] (显著)
- vs Greedy: 胜率=44.17%, sign-p=2.352e-01, welch-p=1.527e-01, boot95=[-0.005881, 0.0007881] (不显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=1.505e-36, welch-p=5.021e-36, boot95=[0.8823, 1.103] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.505e-36, welch-p=9.847e-82, boot95=[0.2838, 0.3092] (显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=2.357e-46, boot95=[1.628, 1.923] (显著)
- vs Static: 胜率=89.17%, sign-p=1.496e-19, welch-p=4.284e-24, boot95=[0.0295, 0.04055] (显著)

### 任务成功率
- vs CP-EFT: 胜率=94.83%, sign-p=7.547e-26, welch-p=1.989e-39, boot95=[37.71, 46.98] (显著)
- vs EFT: 胜率=92.37%, sign-p=5.855e-23, welch-p=1.113e-37, boot95=[36.12, 45.23] (显著)
- vs Greedy: 胜率=64.60%, sign-p=2.456e-03, welch-p=1.082e-03, boot95=[4.456, 15.25] (显著)
- vs LB-Greedy: 胜率=100.00%, sign-p=4.815e-35, welch-p=1.653e-46, boot95=[44.39, 53.48] (显著)
- vs Local-Only: 胜率=41.07%, sign-p=7.213e-02, welch-p=1.705e-04, boot95=[-13.67, -4.333] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.505e-36, welch-p=1.048e-56, boot95=[60.73, 69.42] (显著)
- vs Static: 胜率=62.16%, sign-p=1.323e-02, welch-p=2.143e-03, boot95=[3.667, 15.4] (显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- 当前算法(MAPPO): on_task_rate=0.1957, P(local|on_task)=0.0003, P(rsu|on_task)=0.9986, P(v2v|on_task)=0.0011
- Greedy: on_task_rate=0.1801, P(local|on_task)=0.0000, P(rsu|on_task)=1.0000, P(v2v|on_task)=0.0000
- Static: on_task_rate=0.1682, P(local|on_task)=0.1323, P(rsu|on_task)=0.8677, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.1311, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- EFT: on_task_rate=0.0839, P(local|on_task)=0.9270, P(rsu|on_task)=0.0369, P(v2v|on_task)=0.0361
- CP-EFT: on_task_rate=0.0813, P(local|on_task)=0.9402, P(rsu|on_task)=0.0303, P(v2v|on_task)=0.0295
- LB-Greedy: on_task_rate=0.0729, P(local|on_task)=0.9347, P(rsu|on_task)=0.0306, P(v2v|on_task)=0.0347
- Random: on_task_rate=0.0279, P(local|on_task)=0.0899, P(rsu|on_task)=0.1052, P(v2v|on_task)=0.8049
