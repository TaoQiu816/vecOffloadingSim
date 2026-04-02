# 论文实验结果生成系统 - 使用指南

## 概述

本系统为6组论文实验提供完整的评估和可视化流程，所有结果保存在 `runs/paper_final_results_20260327/`。

## 快速开始

### 一键生成所有结果

```bash
python scripts/paper/export_all_paper_results.py
```

此命令将：
1. 运行所有6组实验的评估脚本
2. 生成所有高质量论文图表（320 DPI）
3. 输出汇总报告

## 实验组说明

### Group 2: 综合性能对比
- **评估脚本**: `scripts/paper/eval_group2_comprehensive.py`
- **绘图脚本**: `scripts/paper/plot_group2_bars.py`
- **输出目录**: `runs/paper_final_results_20260327/group2_comprehensive_comparison/`
- **图表类型**: 柱状图（成功率、平均CFT、P95 CFT）
- **对比方法**: LO, NRO, EFT-H, IPPO-H, F-MAPPO, TERA-MAPPO

### Group 3: 消融实验
- **评估脚本**: `scripts/paper/eval_group3_ablation.py`
- **绘图脚本**: `scripts/paper/plot_group3_bars.py`
- **输出目录**: `runs/paper_final_results_20260327/group3_ablation_studies/`
- **图表类型**: 柱状图（性能对比）
- **对比变体**: TERA-MAPPO, w/o TDE, w/o CARE

### Group 4: 任务复杂度和截止期敏感性
- **评估脚本**: `scripts/paper/eval_group4_complexity.py`
- **绘图脚本**: `scripts/paper/plot_group4_lines.py`
- **输出目录**: `runs/paper_final_results_20260327/group4_complexity_sensitivity/`
- **图表类型**: 折线图
- **实验变量**: DAG规模 (3,5,7,10), 截止期因子 (1.2,1.5,2.0,3.0)

### Group 5: 系统负载和资源竞争
- **评估脚本**: `scripts/paper/eval_group5_system_load.py`
- **绘图脚本**: `scripts/paper/plot_group5_lines.py`
- **输出目录**: `runs/paper_final_results_20260327/group5_system_load/`
- **图表类型**: 折线图
- **实验变量**: 车辆数量 (3,5,7,10), RSU计算能力因子 (0.5,1.0,1.5,2.0)

### Group 6: 机制分析
- **评估脚本**: `scripts/paper/eval_group6_mechanism.py`
- **绘图脚本**: `scripts/paper/plot_group6_analysis.py`
- **输出目录**: `runs/paper_final_results_20260327/group6_mechanism_analysis/`
- **图表类型**: 饼图、堆叠柱状图、箱线图
- **分析内容**: 决策分布、延迟分解、资源利用率

## 单独运行实验组

### 运行单个评估
```bash
python scripts/paper/eval_group2_comprehensive.py
python scripts/paper/eval_group3_ablation.py
python scripts/paper/eval_group4_complexity.py
python scripts/paper/eval_group5_system_load.py
python scripts/paper/eval_group6_mechanism.py
```

### 生成所有图表
```bash
python scripts/paper/plot_all_figures.py
```

### 生成单组图表
```bash
python scripts/paper/plot_group2_bars.py
python scripts/paper/plot_group3_bars.py
python scripts/paper/plot_group4_lines.py
python scripts/paper/plot_group5_lines.py
python scripts/paper/plot_group6_analysis.py
```

## 输出结构

每个实验组的输出目录包含：

```
group{N}_*/
├── figures/              # 高质量图表 (320 DPI PNG)
│   ├── fig_group{N}_*.png
│   └── ...
├── tables/               # CSV格式数据表
│   ├── *_results.csv
│   └── *_summary.csv
└── metadata.json         # 实验元数据
```

## 图表规格

- **分辨率**: 320 DPI
- **格式**: PNG
- **字体**: Songti SC (中文), Arial (英文)
- **配色方案**: 
  - LO: #e74c3c (红色)
  - NRO: #3498db (蓝色)
  - EFT-H: #2ecc71 (绿色)
  - IPPO-H: #9b59b6 (紫色)
  - F-MAPPO: #e67e22 (橙色)
  - TERA-MAPPO: #1abc9c (青色)

## 依赖的训练数据

所有评估脚本依赖以下训练结果：
- TERA-MAPPO: `runs/run_1000ep_B_20260320/`
- F-MAPPO: `runs/rc1_default_fmappo_20260328_224844/fmappo_flat/`
- IPPO-H: `runs/run_1000ep_A_lrcritic_3e4_20260321/`
- 消融实验: `runs/rc1_ablation_*/`

详见 `runs/paper_final_results_20260327/DATA_INVENTORY.md`

## 故障排查

### 模型文件未找到
检查 `DATA_INVENTORY.md` 确认训练数据路径是否正确。

### 图表生成失败
确保已安装中文字体 Songti SC：
```bash
# macOS 系统自带，无需安装
# Linux 需要安装中文字体包
```

### 评估脚本超时
可以在评估脚本中调整 `num_episodes` 参数减少评估轮数。

## 技术支持

如有问题，请检查：
1. `runs/paper_final_results_20260327/IMPLEMENTATION_SUMMARY.md` - 实现细节
2. `runs/paper_final_results_20260327/DATA_INVENTORY.md` - 数据清单
3. `scripts/paper/plot_utils_paper.py` - 绘图工具函数

---

生成时间: 2026-03-29
版本: 1.0
