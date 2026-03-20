# Baseline评估与对比分析指南

## 快速开始

### 1. 运行Baseline评估

在项目根目录执行：

```bash
cd runs/run_20260318_003251
python run_baseline_evaluation.py
```

### 2. 评估内容

脚本将自动评估以下4个baseline策略（每个100 episodes）：

1. **Local-Only**: 所有任务本地执行
2. **Random**: 随机选择执行位置
3. **Greedy**: 贪心选择最快完成的选项
4. **EFT** (Earliest Finish Time): 选择最早完成时间的选项

### 3. 预计运行时间

- 每个baseline约需 **15-30分钟**
- 总计约 **1-2小时**

### 4. 输出结果

评估完成后，将在 `baseline_results/` 目录下生成：

#### 数据文件
- `local_only_results.csv` - Local-Only策略详细结果
- `random_results.csv` - Random策略详细结果
- `greedy_results.csv` - Greedy策略详细结果
- `eft_results.csv` - EFT策略详细结果
- `baseline_summary.csv` - 所有baseline汇总

#### 对比图表
- `mappo_vs_baselines_comparison.png` - 4指标柱状图对比
  - 平均奖励
  - 任务成功率
  - 子任务成功率
  - 归一化能耗
  
- `mappo_vs_baselines_table.png` - 详细对比表格

#### 控制台输出
- 每个baseline的实时评估进度
- 统计结果摘要
- MAPPO相对于各baseline的性能提升百分比

## 示例输出

```
================================================================================
评估 Local-Only 策略...
================================================================================
Local-Only: 100%|████████████████████| 100/100 [15:23<00:00,  9.23s/it]

Local-Only 结果:
  平均奖励: 0.0145 ± 0.0123
  任务成功率: 0.8234
  子任务成功率: 0.9123

...

================================================================================
MAPPO相对于Baselines的性能提升
================================================================================

vs Local-Only:
  奖励提升: +35.9%
  任务成功率提升: +16.2%

vs Random:
  奖励提升: +89.3%
  任务成功率提升: +45.7%

...
```

## 故障排除

### 问题1: 环境创建失败
```
错误: 环境创建失败
```
**解决**: 检查config.json是否完整，确保所有必需参数都存在

### 问题2: Baseline策略导入失败
```
错误: No module named 'baselines'
```
**解决**: 确保在项目根目录运行，或检查Python路径设置

### 问题3: 内存不足
```
错误: MemoryError
```
**解决**: 减少num_eval_episodes（在脚本中修改，默认100）

## 自定义评估

### 修改评估episodes数
编辑 `run_baseline_evaluation.py` 第95行：
```python
num_eval_episodes = 50  # 改为50 episodes
```

### 添加其他baseline
在 `baselines` 字典中添加：
```python
baselines = {
    'Local-Only': LocalOnlyPolicy(),
    'Random': RandomPolicy(),
    'Greedy': GreedyPolicy(),
    'EFT': EFTPolicy(),
    'Your-Policy': YourPolicy(),  # 添加你的策略
}
```

### 修改对比指标
编辑 `plot_comparison` 函数，添加或修改指标

## 后续分析

评估完成后，可以：

1. 查看 `baseline_summary.csv` 了解各策略性能
2. 查看对比图表进行可视化分析
3. 将结果整合到论文或报告中
4. 基于对比结果调整MAPPO策略

## 注意事项

- 评估使用与训练相同的环境配置
- 所有baseline使用相同的随机种子(42)确保公平对比
- MAPPO数据取自最后100 episodes的平均值
- Baseline数据为100 episodes评估的平均值
