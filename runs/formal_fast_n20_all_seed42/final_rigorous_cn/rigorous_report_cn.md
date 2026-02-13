# 严谨对比报告（中文）

## 方法说明
- 使用全程数据（非仅末段）进行趋势分析。
- 同时报告多窗口统计（全程/末500/末300/末100）评估结论稳健性。
- 采用逐 Episode 配对比较（Sign Test）给出胜率与显著性。
- 训练阶段划分: 前期[1-333], 中期[334-666], 后期[667-1000]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=-1.5026, rank=1
- Greedy: mean=-1.6046, rank=2
- Static: mean=-1.7266, rank=3
- Local-Only: mean=-1.9745, rank=4
- CP-EFT: mean=-2.1238, rank=5
- EFT: mean=-2.1818, rank=6
- Random: mean=-2.4508, rank=7

### 任务成功率
- 当前算法(MAPPO): mean=90.0500, rank=1
- Greedy: mean=75.0000, rank=2
- Local-Only: mean=55.0000, rank=3
- EFT: mean=45.0000, rank=4
- Static: mean=30.0000, rank=5
- CP-EFT: mean=25.0000, rank=6
- Random: mean=0.0000, rank=7

### 超时失败率
- 当前算法(MAPPO): mean=9.9500, rank=1
- Greedy: mean=25.0000, rank=2
- Local-Only: mean=45.0000, rank=3
- EFT: mean=55.0000, rank=4
- Static: mean=70.0000, rank=5
- CP-EFT: mean=75.0000, rank=6
- Random: mean=100.0000, rank=7

### 平均完工时间估计
- 当前算法(MAPPO): mean=10.8807, rank=1
- Greedy: mean=12.0247, rank=2
- Local-Only: mean=16.1674, rank=3
- Static: mean=16.6129, rank=4
- CP-EFT: mean=23.8076, rank=5
- EFT: mean=24.2543, rank=6
- Random: mean=24.7617, rank=7

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=91.78%, p=8.771e-179 (显著)
- vs EFT: 胜率=89.51%, p=3.363e-154 (显著)
- vs Greedy: 胜率=73.73%, p=7.624e-50 (显著)
- vs Local-Only: 胜率=85.13%, p=3.028e-117 (显著)
- vs Random: 胜率=99.90%, p=9.871e-287 (显著)
- vs Static: 胜率=89.88%, p=1.542e-158 (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=94.30%, p=9.632e-208 (显著)
- vs EFT: 胜率=94.10%, p=2.506e-205 (显著)
- vs Greedy: 胜率=60.90%, p=5.620e-12 (显著)
- vs Local-Only: 胜率=88.50%, p=7.999e-148 (显著)
- vs Random: 胜率=98.90%, p=4.475e-276 (显著)
- vs Static: 胜率=88.80%, p=1.691e-150 (显著)

### 平均步奖励
- vs CP-EFT: 胜率=93.90%, p=6.072e-203 (显著)
- vs EFT: 胜率=93.90%, p=6.072e-203 (显著)
- vs Greedy: 胜率=84.70%, p=5.329e-117 (显著)
- vs Local-Only: 胜率=92.40%, p=5.790e-186 (显著)
- vs Random: 胜率=97.30%, p=1.237e-248 (显著)
- vs Static: 胜率=88.60%, p=1.037e-148 (显著)

### 任务成功率
- vs CP-EFT: 胜率=91.78%, p=8.771e-179 (显著)
- vs EFT: 胜率=89.51%, p=3.363e-154 (显著)
- vs Greedy: 胜率=73.73%, p=7.624e-50 (显著)
- vs Local-Only: 胜率=85.13%, p=3.028e-117 (显著)
- vs Random: 胜率=99.90%, p=9.871e-287 (显著)
- vs Static: 胜率=89.88%, p=1.542e-158 (显著)
