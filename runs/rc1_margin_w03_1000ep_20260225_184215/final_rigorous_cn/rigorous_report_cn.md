# 严谨对比报告（中文）

## 方法说明
- 绘图趋势允许 baseline 曲线延展（仅用于视觉对齐）。
- 统计与显著性严格使用 baseline 原始样本（raw），不使用 forward-fill 样本。
- 多窗口统计采用 matched-tail：K=min(窗口长度, RL样本数, baseline原始样本数)。
- 显著性同时报告 Sign Test 与 Welch t-test，并给出 bootstrap 均值差95%区间。
- 训练阶段划分: 前期[1-333], 中期[334-666], 后期[667-1000]

## 末100关键指标排名（均值）
### 平均步奖励
- Greedy: mean=0.0510, rank=1
- 当前算法(MAPPO): mean=0.0438, rank=2
- CP-EFT: mean=0.0412, rank=3
- Static: mean=0.0400, rank=4
- EFT: mean=0.0368, rank=5
- LB-Greedy: mean=0.0367, rank=6
- Oracle-Min: mean=0.0367, rank=7
- Local-Only: mean=-0.0013, rank=8
- Random: mean=-0.0022, rank=9

### 任务成功率
- Greedy: mean=82.7500, rank=1
- CP-EFT: mean=79.7500, rank=2
- 当前算法(MAPPO): mean=77.5000, rank=3
- Static: mean=76.7500, rank=4
- Oracle-Min: mean=69.7500, rank=5
- EFT: mean=69.2500, rank=6
- LB-Greedy: mean=69.0000, rank=7
- Local-Only: mean=16.5000, rank=8
- Random: mean=0.5000, rank=9

### 超时失败率
- Greedy: mean=17.2500, rank=1
- CP-EFT: mean=20.2500, rank=2
- 当前算法(MAPPO): mean=22.5000, rank=3
- Static: mean=23.2500, rank=4
- Oracle-Min: mean=30.2500, rank=5
- EFT: mean=30.7500, rank=6
- LB-Greedy: mean=31.0000, rank=7
- Local-Only: mean=83.5000, rank=8
- Random: mean=98.7500, rank=9

### 平均完工时间估计
- Greedy: mean=2.6502, rank=1
- Oracle-Min: mean=2.8770, rank=2
- 当前算法(MAPPO): mean=2.8796, rank=3
- EFT: mean=2.9014, rank=4
- LB-Greedy: mean=2.9036, rank=5
- CP-EFT: mean=2.9114, rank=6
- Static: mean=3.1144, rank=7
- Local-Only: mean=5.5345, rank=8
- Random: mean=21.4760, rank=9

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=43.75%, sign-p=8.036e-01, welch-p=5.171e-01, boot95=[-8.5, 4.131] (不显著)
- vs EFT: 胜率=76.47%, sign-p=4.904e-02, welch-p=2.497e-02, boot95=[1.5, 14.88] (显著)
- vs Greedy: 胜率=31.25%, sign-p=2.101e-01, welch-p=8.535e-02, boot95=[-11, 0.25] (不显著)
- vs LB-Greedy: 胜率=76.47%, sign-p=4.904e-02, welch-p=2.269e-02, boot95=[1.25, 15.75] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.907e-06, welch-p=2.427e-21, boot95=[54.75, 66.75] (显著)
- vs Oracle-Min: 胜率=65.00%, sign-p=2.632e-01, welch-p=4.713e-02, boot95=[0.5, 14.25] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.907e-06, welch-p=3.163e-19, boot95=[71.75, 80.25] (显著)
- vs Static: 胜率=47.06%, sign-p=1.000e+00, welch-p=8.135e-01, boot95=[-5, 6.75] (不显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=45.00%, sign-p=8.238e-01, welch-p=8.672e-01, boot95=[-0.3046, 0.423] (不显著)
- vs EFT: 胜率=50.00%, sign-p=1.000e+00, welch-p=8.332e-01, boot95=[-0.1825, 0.2275] (不显著)
- vs Greedy: 胜率=35.00%, sign-p=2.632e-01, welch-p=1.913e-02, boot95=[-0.4001, -0.06998] (不显著)
- vs LB-Greedy: 胜率=50.00%, sign-p=1.000e+00, welch-p=8.160e-01, boot95=[-0.1746, 0.2226] (不显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.907e-06, welch-p=1.849e-22, boot95=[2.401, 2.89] (显著)
- vs Oracle-Min: 胜率=50.00%, sign-p=1.000e+00, welch-p=9.799e-01, boot95=[-0.1964, 0.1757] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.907e-06, welch-p=1.239e-11, boot95=[16.41, 21.3] (显著)
- vs Static: 胜率=60.00%, sign-p=5.034e-01, welch-p=1.073e-01, boot95=[-0.02554, 0.497] (不显著)

### 平均步奖励
- vs CP-EFT: 胜率=65.00%, sign-p=2.632e-01, welch-p=5.368e-01, boot95=[-0.005181, 0.01038] (不显著)
- vs EFT: 胜率=70.00%, sign-p=1.153e-01, welch-p=2.906e-02, boot95=[0.001263, 0.01318] (不显著)
- vs Greedy: 胜率=25.00%, sign-p=4.139e-02, welch-p=2.049e-02, boot95=[-0.01264, -0.001085] (显著)
- vs LB-Greedy: 胜率=70.00%, sign-p=1.153e-01, welch-p=2.713e-02, boot95=[0.001145, 0.01313] (不显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.907e-06, welch-p=1.027e-13, boot95=[0.04035, 0.0499] (显著)
- vs Oracle-Min: 胜率=70.00%, sign-p=1.153e-01, welch-p=3.217e-02, boot95=[0.001064, 0.01367] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.907e-06, welch-p=1.532e-13, boot95=[0.04111, 0.0508] (显著)
- vs Static: 胜率=65.00%, sign-p=2.632e-01, welch-p=2.282e-01, boot95=[-0.002178, 0.009844] (不显著)

### 任务成功率
- vs CP-EFT: 胜率=43.75%, sign-p=8.036e-01, welch-p=5.171e-01, boot95=[-9, 4] (不显著)
- vs EFT: 胜率=76.47%, sign-p=4.904e-02, welch-p=2.497e-02, boot95=[1.5, 15] (显著)
- vs Greedy: 胜率=31.25%, sign-p=2.101e-01, welch-p=8.535e-02, boot95=[-11.25, 0.25] (不显著)
- vs LB-Greedy: 胜率=76.47%, sign-p=4.904e-02, welch-p=2.269e-02, boot95=[1.5, 15.5] (显著)
- vs Local-Only: 胜率=100.00%, sign-p=1.907e-06, welch-p=2.427e-21, boot95=[54.75, 66.75] (显著)
- vs Oracle-Min: 胜率=65.00%, sign-p=2.632e-01, welch-p=4.713e-02, boot95=[0.75, 14.88] (不显著)
- vs Random: 胜率=100.00%, sign-p=1.907e-06, welch-p=8.267e-19, boot95=[72.75, 81.5] (显著)
- vs Static: 胜率=47.06%, sign-p=1.000e+00, welch-p=8.135e-01, boot95=[-5.25, 6.5] (不显著)

## On-Task 条件统计
- 若无显式 has_task 列，则使用 on_task_rate = 1 - no_task_rate 近似恢复。
- Greedy: on_task_rate=0.4445, P(local|on_task)=0.5655, P(rsu|on_task)=0.4318, P(v2v|on_task)=0.0027
- CP-EFT: on_task_rate=0.4270, P(local|on_task)=0.6561, P(rsu|on_task)=0.3118, P(v2v|on_task)=0.0321
- 当前算法(MAPPO): on_task_rate=0.4143, P(local|on_task)=0.0007, P(rsu|on_task)=0.9980, P(v2v|on_task)=0.0012
- Oracle-Min: on_task_rate=0.4089, P(local|on_task)=0.5991, P(rsu|on_task)=0.3820, P(v2v|on_task)=0.0189
- EFT: on_task_rate=0.4040, P(local|on_task)=0.6036, P(rsu|on_task)=0.3785, P(v2v|on_task)=0.0179
- LB-Greedy: on_task_rate=0.4029, P(local|on_task)=0.6046, P(rsu|on_task)=0.3774, P(v2v|on_task)=0.0180
- Static: on_task_rate=0.3802, P(local|on_task)=0.0238, P(rsu|on_task)=0.9762, P(v2v|on_task)=0.0000
- Local-Only: on_task_rate=0.2085, P(local|on_task)=1.0000, P(rsu|on_task)=0.0000, P(v2v|on_task)=0.0000
- Random: on_task_rate=0.0410, P(local|on_task)=0.1572, P(rsu|on_task)=0.1617, P(v2v|on_task)=0.6811
