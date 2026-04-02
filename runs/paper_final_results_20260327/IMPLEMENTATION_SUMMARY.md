# 论文实验结果导出 - 实施总结

生成时间：2026-03-29
目标：基于已训练的数据，评估并生成6组实验的高质量论文图表

## 已完成工作

### 1. 数据调查与准备 ✓
- 创建了数据清单文档：`runs/paper_final_results_20260327/DATA_INVENTORY.md`
- 确认了所有训练数据的位置和完整性

### 2. 统一框架创建 ✓

#### 绘图工具模块 (`scripts/paper/plot_utils_paper.py`)
- 统一的配色方案（参考现有高质量图表）
- 中文字体支持（Songti SC）
- 专业绘图风格（320 DPI，符合期刊标准）
- 实用函数：柱状图、折线图、堆叠柱状图、箱线图

#### 主导出脚本 (`scripts/paper/export_all_paper_results.py`)
- 协调所有6组实验的评估和绘图
- 统一的错误处理和进度报告

### 3. 评估脚本（全部完成）✓
- **Group 2**: `eval_group2_comprehensive.py` - 6种方法综合对比
- **Group 3**: `eval_group3_ablation.py` - 消融实验评估
- **Group 4**: `eval_group4_complexity.py` - 任务复杂度和截止期敏感性
- **Group 5**: `eval_group5_system_load.py` - 系统负载和资源竞争
- **Group 6**: `eval_group6_mechanism.py` - 机制分析数据收集

### 4. 绘图脚本（全部完成）✓
- **Group 2**: `plot_group2_bars.py` - 性能对比柱状图
- **Group 3**: `plot_group3_bars.py` - 消融实验柱状图
- **Group 4**: `plot_group4_lines.py` - 复杂度敏感性折线图
- **Group 5**: `plot_group5_lines.py` - 系统负载折线图
- **Group 6**: `plot_group6_analysis.py` - 机制分析图（饼图、堆叠图）

### 5. 统一绘图脚本 ✓
- `plot_all_figures.py` - 一键生成所有图表

### 6. 用户文档 ✓
- `USER_GUIDE.md` - 完整使用指南

## 待完成工作

### 测试与验证
1. 运行完整流程测试
2. 验证所有图表质量
3. 确认数据完整性

## 使用方法

### 快速开始
```bash
# 运行所有评估和绘图
python scripts/paper/export_all_paper_results.py

# 单独运行某一组
python scripts/paper/eval_group2_comprehensive.py
```

### 输出结构
```
runs/paper_final_results_20260327/
├── group2_comprehensive_comparison/
│   ├── tables/
│   │   └── group2_comparison_table.csv
│   ├── figures/
│   │   ├── fig3_success_rate.png
│   │   ├── fig4_mean_cft.png
│   │   └── fig5_p95_cft.png
│   └── evaluation_results.json
├── group3_ablation/
│   ├── tables/
│   ├── figures/
│   └── ...
├── group4_complexity/
├── group5_scalability/
├── group6_mechanism/
├── DATA_INVENTORY.md
└── FINAL_SUMMARY.md
```

## 注意事项

1. **模型路径确认**
   - F-MAPPO模型路径需要确认
   - 某些训练数据可能需要重新组织

2. **评估参数**
   - 默认50个episode
   - 随机种子42（可调整）

3. **图表质量**
   - 所有图表使用320 DPI
   - 符合高水平期刊标准
   - 颜色方案参考现有高质量图表

## 后续步骤

1. 确认所有训练数据位置
2. 依次创建各组的评估脚本
3. 创建对应的绘图脚本
4. 测试所有脚本
5. 生成最终图表和报告

## 联系与支持

如有问题，请查看：
- 数据清单：`runs/paper_final_results_20260327/DATA_INVENTORY.md`
- 绘图工具：`scripts/paper/plot_utils_paper.py`
- 评估示例：`scripts/paper/eval_group2_comprehensive.py`
