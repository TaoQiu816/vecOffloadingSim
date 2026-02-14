# 严谨对比报告（中文）

## 方法说明
- 使用全程数据（非仅末段）进行趋势分析。
- 同时报告多窗口统计（全程/末500/末300/末100）评估结论稳健性。
- 采用逐 Episode 配对比较（Sign Test）给出胜率与显著性。
- 训练阶段划分: 前期[1-333], 中期[334-666], 后期[667-1000]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=-0.0053, rank=1
- Greedy: mean=-0.0167, rank=2
- Static: mean=-0.0951, rank=3
- Local-Only: mean=-0.3285, rank=4
- LB-Greedy: mean=-1.3656, rank=5
- EFT: mean=-2.2551, rank=6
- CP-EFT: mean=-2.4321, rank=7
- Random: mean=-6.4955, rank=8

### 任务成功率
- Static: mean=95.0000, rank=1
- Greedy: mean=85.0000, rank=2
- 当前算法(MAPPO): mean=82.0500, rank=3
- Local-Only: mean=65.0000, rank=4
- CP-EFT: mean=45.0000, rank=5
- EFT: mean=30.0000, rank=6
- LB-Greedy: mean=25.0000, rank=7
- Random: mean=0.0000, rank=8

### 超时失败率
- Static: mean=5.0000, rank=1
- Greedy: mean=15.0000, rank=2
- 当前算法(MAPPO): mean=17.9500, rank=3
- Local-Only: mean=35.0000, rank=4
- CP-EFT: mean=55.0000, rank=5
- EFT: mean=70.0000, rank=6
- LB-Greedy: mean=75.0000, rank=7
- Random: mean=100.0000, rank=8

### 平均完工时间估计
- 当前算法(MAPPO): mean=11.3359, rank=1
- Static: mean=11.6826, rank=2
- Greedy: mean=12.7556, rank=3
- Local-Only: mean=14.9738, rank=4
- CP-EFT: mean=21.8967, rank=5
- EFT: mean=22.3429, rank=6
- LB-Greedy: mean=22.4635, rank=7
- Random: mean=25.5625, rank=8

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=82.56%, p=9.929e-99 (显著)
- vs EFT: 胜率=87.68%, p=4.391e-137 (显著)
- vs Greedy: 胜率=52.22%, p=1.830e-01 (不显著)
- vs LB-Greedy: 胜率=88.95%, p=1.054e-149 (显著)
- vs Local-Only: 胜率=70.59%, p=1.671e-38 (显著)
- vs Random: 胜率=100.00%, p=1.642e-288 (显著)
- vs Static: 胜率=27.83%, p=2.031e-38 (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=86.50%, p=6.273e-131 (显著)
- vs EFT: 胜率=89.50%, p=6.802e-157 (显著)
- vs Greedy: 胜率=62.60%, p=1.472e-15 (显著)
- vs LB-Greedy: 胜率=90.50%, p=2.003e-166 (显著)
- vs Local-Only: 胜率=77.20%, p=1.124e-69 (显著)
- vs Random: 胜率=98.40%, p=8.038e-267 (显著)
- vs Static: 胜率=48.20%, p=2.684e-01 (不显著)

### 平均步奖励
- vs CP-EFT: 胜率=89.90%, p=1.196e-160 (显著)
- vs EFT: 胜率=89.00%, p=2.649e-152 (显著)
- vs Greedy: 胜率=52.00%, p=2.174e-01 (不显著)
- vs LB-Greedy: 胜率=87.90%, p=1.378e-142 (显著)
- vs Local-Only: 胜率=81.00%, p=1.142e-91 (显著)
- vs Random: 胜率=96.60%, p=3.715e-238 (显著)
- vs Static: 胜率=78.20%, p=4.227e-75 (显著)

### 任务成功率
- vs CP-EFT: 胜率=82.56%, p=9.929e-99 (显著)
- vs EFT: 胜率=87.68%, p=4.391e-137 (显著)
- vs Greedy: 胜率=52.22%, p=1.830e-01 (不显著)
- vs LB-Greedy: 胜率=88.95%, p=1.054e-149 (显著)
- vs Local-Only: 胜率=70.59%, p=1.671e-38 (显著)
- vs Random: 胜率=100.00%, p=1.642e-288 (显著)
- vs Static: 胜率=27.83%, p=2.031e-38 (显著)
