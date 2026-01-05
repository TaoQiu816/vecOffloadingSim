# 快速启动指南 - 短期验证训练

## 🚀 一键启动

```bash
cd /Users/qiutao/研/毕设/毕设/vecOffloadingSim
python train.py
```

**预计时间**: 50-75分钟  
**训练规模**: 300 episodes（短期验证）

---

## 📋 当前状态确认

```bash
python -c "
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TrnCfg
print(f'✓ MAX_EPISODES: {TrnCfg.MAX_EPISODES}')
print(f'✓ BW_V2I: {Cfg.BW_V2I/1e6:.0f}Mbps')
print(f'✓ ENTROPY_COEF: {TrnCfg.ENTROPY_COEF}')
print(f'✓ LOGIT_BIAS_RSU: {TrnCfg.LOGIT_BIAS_RSU}')
print('✓ 配置已就绪 - 可以开始训练')
"
```

---

## 🎯 验证目标

| 指标 | 之前 | 目标 |
|------|------|------|
| Task Success Rate | 0% | **> 10%** |
| Decision RSU | 100% | **30-50%** |
| Decision Local | 0% | **20-40%** |
| Entropy | 持续增长 | **收敛** |

---

## 📊 实时监控

训练过程中会每10个episodes打印一次日志：

```
Episode 10/300 | Reward: -3.45 | Task SR: 5.2%
Episode 50/300 | Reward: -2.12 | Task SR: 12.8%
Episode 100/300 | Reward: -1.45 | Task SR: 18.5%
...
```

---

## 📈 查看结果

训练完成后（或随时）：

```bash
# 查看最新运行
ls -lt runs/

# 生成图表
python test_plotting.py --run-dir runs/run_XXXXXX_XXXXXX

# 查看20+张图表
ls runs/run_XXXXXX_XXXXXX/plots/
```

**重点图表:**
- `reward_curve_with_baselines.png` - 奖励曲线
- `veh_success_rate_with_baselines.png` - 成功率
- `offloading_ratio.png` - 决策分布
- `performance_radar.png` - 综合性能

---

## ✅ 成功标准

**最低标准（继续完整训练）:**
- Task Success Rate > 5%
- Decision不是100%单一
- Entropy不再持续增长
- Reward有增长趋势

**理想标准（参数已优化）:**
- Task Success Rate > 10%
- Decision分布平衡
- Entropy收敛到0.5-1.0
- Reward稳定增长

---

## 🔧 如果出现问题

### Task SR仍为0%
```bash
# 检查环境
python debug_rsu_simple.py
```
可能需要：增加BW_V2I或降低数据量

### Decision仍100% RSU
修改 `configs/train_config.py`:
```python
LOGIT_BIAS_RSU = 1.5  # 降低
```

### Entropy持续增长
修改 `configs/train_config.py`:
```python
ENTROPY_COEF = 0.003  # 降低
```

---

## 📚 详细文档

- **VALIDATION_TRAINING_GUIDE.md** - 完整验证训练指南
- **SHORT_VS_FULL_TRAINING.md** - 短期vs完整训练对比
- **CONFIG_UPDATE_SUMMARY.md** - 参数调整总结
- **PLOTTING_GUIDE.md** - 20张图表详解

---

## 🔄 下一步

### 如果验证成功 → 切换到完整训练

```bash
# 修改configs/train_config.py
MAX_EPISODES = 1500
LR_DECAY_STEPS = 200
BIAS_DECAY_EVERY_EP = 200
SAVE_INTERVAL = 200

# 从checkpoint继续（推荐）
python train.py --load runs/run_XXXXXX/models/latest_model.pth --start-episode 300

# 或从头开始
python train.py
```

### 如果验证失败 → 调整参数

参考 `VALIDATION_TRAINING_GUIDE.md` 第五节"问题诊断"

---

## ⚡ 重要提醒

1. **GPU必需**: 训练需要GPU，确保CUDA可用
2. **监控显存**: 如果OOM，降低MINI_BATCH_SIZE
3. **定期查看**: 每50-100 episodes查看日志，及早发现问题
4. **保存日志**: 终端日志可用于后续分析
5. **备份checkpoint**: 每100 episodes自动保存

---

**开始训练:**
```bash
python train.py
```

**祝训练顺利！** 🎉
