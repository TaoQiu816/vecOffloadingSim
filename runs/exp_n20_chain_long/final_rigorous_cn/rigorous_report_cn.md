# 严谨对比报告（中文）

## 方法说明
- 使用全程数据（非仅末段）进行趋势分析。
- 同时报告多窗口统计（全程/末500/末300/末100）评估结论稳健性。
- 采用逐 Episode 配对比较（Sign Test）给出胜率与显著性。
- 训练阶段划分: 前期[1-500], 中期[501-1000], 后期[1001-1500]

## 末100关键指标排名（均值）
### 平均步奖励
- 当前算法(MAPPO): mean=-1.4727, rank=1
- Greedy: mean=-1.7092, rank=2
- Static: mean=-1.7759, rank=3
- Local-Only: mean=-2.0415, rank=4
- EFT: mean=-2.7549, rank=5
- CP-EFT: mean=-2.7809, rank=6
- Random: mean=-4.2058, rank=7

### 任务成功率
- 当前算法(MAPPO): mean=90.9000, rank=1
- Local-Only: mean=70.3667, rank=2
- Static: mean=35.0333, rank=3
- Greedy: mean=27.1000, rank=4
- EFT: mean=18.8000, rank=5
- CP-EFT: mean=16.9667, rank=6
- Random: mean=0.1000, rank=7

### 超时失败率
- 当前算法(MAPPO): mean=9.1000, rank=1
- Local-Only: mean=29.6333, rank=2
- Static: mean=64.9667, rank=3
- Greedy: mean=72.9000, rank=4
- EFT: mean=81.2000, rank=5
- CP-EFT: mean=83.0333, rank=6
- Random: mean=99.9000, rank=7

### 平均完工时间估计
- 当前算法(MAPPO): mean=10.3463, rank=1
- Greedy: mean=15.3667, rank=2
- Static: mean=16.0775, rank=3
- Local-Only: mean=16.3019, rank=4
- CP-EFT: mean=23.9628, rank=5
- EFT: mean=23.9898, rank=6
- Random: mean=25.1948, rank=7

## 配对检验（当前算法 vs Baseline）
### 超时失败率
- vs CP-EFT: 胜率=96.66%, p=0.000e+00 (显著)
- vs EFT: 胜率=96.86%, p=0.000e+00 (显著)
- vs Greedy: 胜率=95.24%, p=0.000e+00 (显著)
- vs Local-Only: 胜率=82.71%, p=1.352e-151 (显著)
- vs Random: 胜率=100.00%, p=0.000e+00 (显著)
- vs Static: 胜率=93.97%, p=1.572e-303 (显著)

### 平均完工时间估计
- vs CP-EFT: 胜率=98.20%, p=0.000e+00 (显著)
- vs EFT: 胜率=98.20%, p=0.000e+00 (显著)
- vs Greedy: 胜率=92.80%, p=9.385e-285 (显著)
- vs Local-Only: 胜率=93.00%, p=4.233e-288 (显著)
- vs Random: 胜率=99.80%, p=0.000e+00 (显著)
- vs Static: 胜率=92.80%, p=9.385e-285 (显著)

### 平均步奖励
- vs CP-EFT: 胜率=98.13%, p=0.000e+00 (显著)
- vs EFT: 胜率=98.13%, p=0.000e+00 (显著)
- vs Greedy: 胜率=93.60%, p=2.218e-298 (显著)
- vs Local-Only: 胜率=96.40%, p=0.000e+00 (显著)
- vs Random: 胜率=99.60%, p=0.000e+00 (显著)
- vs Static: 胜率=93.93%, p=2.938e-304 (显著)

### 任务成功率
- vs CP-EFT: 胜率=96.66%, p=0.000e+00 (显著)
- vs EFT: 胜率=96.86%, p=0.000e+00 (显著)
- vs Greedy: 胜率=95.24%, p=0.000e+00 (显著)
- vs Local-Only: 胜率=82.71%, p=1.352e-151 (显著)
- vs Random: 胜率=100.00%, p=0.000e+00 (显著)
- vs Static: 胜率=93.97%, p=1.572e-303 (显著)
