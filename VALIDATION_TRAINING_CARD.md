# 🚀 短期验证训练 - 快速参考卡片

## 一键启动
```bash
cd /Users/qiutao/研/毕设/毕设/vecOffloadingSim
python train.py
```

---

## 📊 训练配置

| 项目 | 值 | 说明 |
|------|-----|------|
| **Episodes** | 300 | 短期验证（1-2小时） |
| **Entropy Coef** | 0.005 | 控制探索强度 |
| **Batch Size** | 256 | 提升稳定性 |
| **Bias RSU** | 2.0 → 0.5 | 避免100% RSU |
| **Bias Local** | 1.5 → 0.5 | 平衡决策 |

---

## ✅ 验证通过标准（至少满足2项）

1. **Task Success Rate > 20%** (100 episode后)
2. **Decision分布平衡**: Local>15%, RSU<80%, V2V>5%
3. **Entropy < 1.5** (200 episode后)
4. **Subtask Success > 50%**

---

## 🔍 关键监控指标

### Episode 100检查
```bash
python -c "
import pandas as pd, glob
df = pd.read_csv(sorted(glob.glob('runs/run_*'))[-1]+'/episode_log.csv')
r = df.tail(10)
print(f'Task SR: {r[\"task_success_rate\"].mean()*100:.1f}%')
print(f'Local: {r[\"decision_frac_local\"].mean()*100:.1f}% | RSU: {r[\"decision_frac_rsu\"].mean()*100:.1f}% | V2V: {r[\"decision_frac_v2v\"].mean()*100:.1f}%')
"
```

---

## 📈 预期结果对比

| 指标 | 修改前 | 预期修改后 |
|------|--------|-----------|
| Task Success | 0% | >20% |
| Local决策 | 0-5% | 20-40% |
| RSU决策 | 95-100% | 40-60% |
| Entropy | 持续增长 | 收敛<1.5 |

---

## 🎯 训练结束后

### 生成图表
```bash
python test_plotting.py --run-dir runs/run_YYYYMMDD_HHMMSS
```

### 必看图表
1. `veh_success_rate_with_baselines.png` - 成功率
2. `offloading_ratio.png` - 决策演化
3. `training_stability.png` - 稳定性

---

## 🔧 如需微调

### Entropy仍增长（>1.5）
```python
# train_config.py
ENTROPY_COEF = 0.003  # 降低
```

### 仍100% RSU
```python
LOGIT_BIAS_RSU = 1.5  # 降低
```

### Task Success仍为0
→ 查看 CONFIG_UPDATE_SUMMARY.md 第三章

---

## ⏱️ 时间估算

- **CPU**: 3-4小时
- **GPU (CUDA)**: 1小时
- **Apple Silicon**: 2小时

---

## 📚 详细文档

- **完整配置说明**: `CONFIG_UPDATE_SUMMARY.md`
- **验证指南**: `QUICK_VALIDATION_GUIDE.md`
- **绘图指南**: `PLOTTING_GUIDE.md` (20张图表)

---

**核心目标**: 用300 episodes验证配置有效性，确认任务可完成且策略多样化。
