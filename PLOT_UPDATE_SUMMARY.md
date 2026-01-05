# 绘图功能更新总结

## 更新内容

### 新增图表 (9张)

1. **latency_energy_tradeoff.png** - 时延-能耗权衡散点图
2. **performance_radar.png** - 多维性能雷达图
3. **resource_utilization.png** - CPU+队列资源利用率（3子图）
4. **training_stability.png** - Reward/Success Rate波动性分析
5. **completion_time_cdf.png** - 任务完成时间累积分布
6. **queue_load_balance.png** - 队列负载均衡分析
7. **episode_duration.png** - Episode时长分布
8. **reward_decomposition.png** - Reward与关键因素相关性（4象限）
9. **success_rate_multilevel.png** - 任务/子任务/V2V三层成功率

### 原有图表 (11张)

- reward_curve_with_baselines.png
- loss_curve.png
- veh_success_rate_with_baselines.png
- subtask_success_rate_with_baselines.png
- offloading_ratio.png
- ma_fairness.png
- ma_collaboration_with_baselines.png
- ma_reward_gap.png
- queue_len.png
- cpu_efficiency.png
- agent_reward_boxplot.png

**总计: 20张图表**

---

## 修改文件

- `utils/data_recorder.py`: 新增9个绘图方法
- `PLOTTING_GUIDE.md`: 完整绘图指南（新建）
- `test_plotting.py`: 测试脚本（新建）

---

## 使用方法

### 自动生成（训练结束后）

训练脚本自动调用，无需额外操作。

### 手动测试

```bash
python test_plotting.py --run-dir runs/run_20260105_021203
```

### 查看文档

```bash
cat PLOTTING_GUIDE.md
```

---

## 特点

✅ **健壮性**: 自动处理缺失列，不影响其他图表  
✅ **对比性**: 支持Baseline对比（Random/Local-Only/Greedy）  
✅ **高质量**: 300 DPI，适合论文  
✅ **多维度**: 训练/性能/资源/稳定性全覆盖  
✅ **可解释性**: Reward分解帮助理解策略  

---

## 配置参数已优化

`configs/config.py`:
- BW_V2I: 20Mbps → 50Mbps
- MIN_COMP: 0.3e9 → 1.0e8 cycles
- MAX_COMP: 1.2e9 → 1.0e9 cycles
- MIN_DATA: 1.0e6 → 4.0e5 bits
- MAX_DATA: 4.0e6 → 2.0e6 bits
- MIN_EDGE_DATA: 0.2e6 → 8.0e5 bits
- MAX_EDGE_DATA: 0.6e6 → 4.0e6 bits

**预期效果:**
- 本地执行有压力（2.75s）
- RSU卸载有收益（比本地快75%）
- 策略可学习协同卸载

---

## 下一步

1. 运行新训练（使用优化参数）
2. 查看生成的20张图表
3. 对比修改前后的性能提升

