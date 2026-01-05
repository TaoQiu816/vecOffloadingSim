# 训练结果可视化指南

## 概述

训练结束后，系统会自动生成**20+**张分析图表，涵盖训练收敛、策略演化、性能对比、资源利用等多个维度。

## 图表列表

### 📊 基础训练指标 (7张)

1. **reward_curve_with_baselines.png**
   - 内容: MAPPO奖励曲线 + 基准策略对比
   - 用途: 评估训练收敛性和相对性能

2. **loss_curve.png**
   - 内容: Actor + Critic损失曲线
   - 用途: 诊断训练稳定性

3. **veh_success_rate_with_baselines.png**
   - 内容: 车辆DAG完成率（%）+ 基准对比
   - 用途: 评估任务成功率

4. **subtask_success_rate_with_baselines.png**
   - 内容: 子任务完成率（%）+ 基准对比
   - 用途: 细粒度成功率分析

5. **offloading_ratio.png**
   - 内容: Local/RSU/V2V决策分布（堆叠面积图）
   - 用途: 策略演化轨迹

6. **ma_collaboration_with_baselines.png**
   - 内容: V2V协作率（%）+ 基准对比
   - 用途: 评估多智能体协同

7. **agent_reward_boxplot.png**
   - 内容: 各车辆奖励分布箱线图（最后20 eps）
   - 用途: 个体差异分析

---

### 🔬 高级分析图表 (13张，新增)

#### 性能对比与权衡

8. **latency_energy_tradeoff.png** ⭐
   - 内容: 时延-能耗散点图（Pareto前沿）
   - 用途: 多目标优化效果分析
   - 特点: 颜色映射Episode进度，含Baseline对比

9. **performance_radar.png** ⭐
   - 内容: 多维性能雷达图（5个维度）
     - Task Success Rate
     - Subtask Success Rate
     - Normalized Reward
     - Resource Utilization
     - Queue Efficiency
   - 用途: 一图概览全局性能
   - 特点: 支持多策略对比

10. **success_rate_multilevel.png** ⭐
    - 内容: 任务/子任务/V2V三层成功率对比
    - 用途: 细粒度成功率演化
    - 特点: 含80%目标线

#### 资源与负载分析

11. **resource_utilization.png**
    - 内容: CPU利用率 + RSU队列 + Vehicle队列（3子图）
    - 用途: 资源利用效率时序分析

12. **queue_load_balance.png**
    - 内容: 队列负载均衡分析
    - 用途: 识别拥堵瓶颈
    - 特点: 含高/低负载阈值线

13. **cpu_efficiency.png**
    - 内容: 平均分配CPU算力（GHz）
    - 用途: 算力分配效率

#### 训练稳定性

14. **training_stability.png** ⭐
    - 内容: Reward标准差 + Success Rate标准差（滚动窗口）
    - 用途: 评估训练波动性
    - 特点: 低方差 = 稳定训练

15. **ma_fairness.png**
    - 内容: Jain公平性指数 [0-1]
    - 用途: 多智能体资源分配公平性

16. **ma_reward_gap.png**
    - 内容: 个体奖励差距（Max - Min）
    - 用途: 个体差异演化

#### Reward分解分析

17. **reward_decomposition.png** ⭐
    - 内容: Reward与4个关键因素的相关性（2x2子图）
      - Reward vs Success Rate
      - Reward vs RSU Usage
      - Reward vs Queue Congestion
      - Reward vs CPU Allocation
    - 用途: 理解Reward驱动因素
    - 特点: 散点图 + Episode颜色映射

#### Episode时长分析

18. **episode_duration.png**
    - 内容: Episode时长时序图 + 分布直方图
    - 用途: 评估训练效率
    - 特点: 含均值/中位数标记

#### 任务完成时间

19. **completion_time_cdf.png** ⭐
    - 内容: 任务完成时间累积分布函数（CDF）
    - 用途: 时延分布对比
    - 特点: MAPPO vs Baselines CDF曲线

#### 多智能体指标

20. **queue_len.png**
    - 内容: Vehicle队列 + RSU队列平均长度
    - 用途: 队列拥堵趋势

---

### 📈 plot_results.py 独立图表 (4张)

可通过命令行单独生成（用于论文）：

```bash
python plot_results.py --log-file runs/run_XXX/metrics/train_metrics.csv --output-dir plots/
```

21. **fig_convergence.png**
    - 内容: Reward + Task SR + Subtask SR（3子图）
    - 用途: 论文级收敛曲线

22. **fig_policy_evolution.png**
    - 内容: 堆叠面积图（Local/RSU/V2V比例）
    - 用途: 策略演化可视化

23. **fig_physics.png**
    - 内容: 时延（左轴） + 能耗（右轴）双轴图
    - 用途: 物理指标演化（需记录相关列）

24. **fig_training.png**
    - 内容: Actor Loss + Critic Loss + Entropy（3子图）
    - 用途: 训练诊断

---

## 使用方法

### 自动生成（训练结束后）

训练脚本会自动调用 `DataRecorder.auto_plot()`，生成所有图表到 `runs/run_XXX/plots/`。

### 手动生成（从已有日志）

```python
from utils.data_recorder import DataRecorder
import pandas as pd

recorder = DataRecorder(base_dir='runs/run_20260105_021203')
recorder.auto_plot()
```

### 只生成部分图表

修改 `data_recorder.py` 的 `auto_plot()` 方法，注释掉不需要的绘图调用。

---

## 图表质量设置

所有图表默认参数：
- **DPI**: 300（高分辨率，适合论文）
- **尺寸**: 10x6 ~ 14x12英寸
- **格式**: PNG（可改为PDF）
- **风格**: seaborn-whitegrid
- **字体**: 支持中文（SimHei）

修改示例（在 `data_recorder.py` 开头）：

```python
plt.rcParams['font.size'] = 14  # 增大字体
plt.rcParams['figure.dpi'] = 150  # 降低DPI加速预览
```

---

## 缺失列处理

部分图表依赖特定列（如 `avg_latency`, `avg_energy`）。如果训练时未记录，这些图表会自动跳过（不报错）。

**需要完整图表？** 确保训练时记录以下列：

```python
episode_data = {
    'avg_latency': ...,
    'avg_energy': ...,
    'avg_completion_time': ...,
    # ... 其他列
}
```

---

## 对比基准策略

训练时运行 `eval_baselines.py` 生成基准数据：

```bash
python eval_baselines.py --episodes 10 --output runs/run_XXX
```

基准数据会自动合并到 `episode_log.csv`，并在以下图表中显示：
- reward_curve_with_baselines.png
- veh_success_rate_with_baselines.png
- latency_energy_tradeoff.png
- performance_radar.png
- completion_time_cdf.png

---

## 常见问题

### Q1: 图表模糊？
A: 增加DPI: `plt.savefig(..., dpi=600)`

### Q2: 中文乱码？
A: 安装SimHei字体或使用英文标签

### Q3: 内存不足？
A: 减少数据点数量（取最后N个episode）

### Q4: 自定义图表？
A: 在 `DataRecorder` 类中添加新方法，仿照现有 `plot_xxx()` 格式

---

## 图表示例用途

### 论文撰写
- 收敛性: fig_convergence.png
- 策略分析: fig_policy_evolution.png, offloading_ratio.png
- 性能对比: performance_radar.png, veh_success_rate_with_baselines.png
- 权衡分析: latency_energy_tradeoff.png

### 调试训练
- 稳定性: training_stability.png, loss_curve.png
- 瓶颈: queue_load_balance.png, resource_utilization.png
- 个体差异: agent_reward_boxplot.png, ma_reward_gap.png

### 实验报告
- 综合对比: performance_radar.png
- 细节分解: reward_decomposition.png
- 分布分析: completion_time_cdf.png, episode_duration.png

---

## 新增图表设计理念

✅ **全面性**: 覆盖训练/性能/资源/稳定性多个维度  
✅ **对比性**: 所有关键图表支持Baseline对比  
✅ **可解释性**: Reward分解图帮助理解策略行为  
✅ **学术价值**: 高分辨率+标准化格式，直接用于论文  
✅ **健壮性**: 自动跳过缺失列，不影响其他图表生成  

---

**总计**: 20+ 张图表，覆盖训练全生命周期 🎉

