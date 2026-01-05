# 短期验证训练指南

## 概述

当前配置适用于**短期验证性训练**，用于快速验证参数调整的有效性。

**训练规模**: 300 episodes（约50-75分钟）  
**目标**: 验证配置参数是否解决了之前的训练问题

---

## 一、当前配置总结

### 1.1 训练规模参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_EPISODES` | 300 | 短期验证训练 |
| `MAX_STEPS` | 200 | 每episode步数 |
| **预计时间** | **50-75分钟** | 取决于硬件 |

### 1.2 核心训练参数

| 参数 | 值 | 目的 |
|------|-----|------|
| `ENTROPY_COEF` | 0.005 | 降低探索，避免entropy发散 |
| `MINI_BATCH_SIZE` | 256 | 增加稳定性 |
| `LOGIT_BIAS_RSU` | 2.0 | 适度RSU偏置（之前5.0太高） |
| `LOGIT_BIAS_LOCAL` | 1.5 | 适度Local偏置 |
| `BIAS_MIN_RSU` | 0.5 | 保持最小RSU探索 |
| `BIAS_MIN_LOCAL` | 0.5 | 保持最小Local探索 |

### 1.3 衰减策略（适配短期）

| 参数 | 值 | 频率 |
|------|-----|------|
| `LR_DECAY_STEPS` | 100 episodes | 共衰减3次（ep 100, 200, 300） |
| `BIAS_DECAY_EVERY_EP` | 100 episodes | 共衰减3次 |
| `BIAS_DECAY_RSU` | 0.5 | 每次减0.5 |
| `BIAS_DECAY_LOCAL` | 0.3 | 每次减0.3 |

**Bias演化轨迹（预估）:**
- Episode 0-99: RSU=2.0, Local=1.5
- Episode 100-199: RSU=1.5, Local=1.2
- Episode 200-299: RSU=1.0, Local=0.9
- Episode 300: RSU=0.5(min), Local=0.5(min)

### 1.4 监控频率

| 参数 | 值 | 说明 |
|------|-----|------|
| `LOG_INTERVAL` | 10 | 每10个episodes打印日志 |
| `SAVE_INTERVAL` | 100 | 每100个episodes保存checkpoint（共4个） |
| `EVAL_INTERVAL` | 25 | 每25个episodes评估（共12次） |

---

## 二、环境参数（已优化）

### 2.1 网络带宽

- `BW_V2I`: **50Mbps**（之前20Mbps）
- `BW_V2V`: 10Mbps（保持）

### 2.2 任务负载

| 参数 | 值 | 说明 |
|------|-----|------|
| `MIN_COMP` | 1.0e8 cycles (0.1G) | 参考文献10倍 |
| `MAX_COMP` | 1.0e9 cycles (1.0G) | 参考文献10倍 |
| `MIN_DATA` | 4.0e5 bits (50KB) | 参考文献范围 |
| `MAX_DATA` | 2.0e6 bits (250KB) | 参考文献中值 |
| `MIN_EDGE_DATA` | 8.0e5 bits (100KB) | 参考文献 |
| `MAX_EDGE_DATA` | 4.0e6 bits (500KB) | 参考文献 |

### 2.3 性能预估

**单DAG平均（10节点）:**
- **本地执行**: 2.75s (55步) - 有队列压力 ✓
- **RSU单车**: 0.70s (14步) - 比本地快75% ✓
- **RSU 12车**: 2.88s传输 (58步) - 可完成 ✓

---

## 三、验证目标

### 3.1 核心指标（300 episodes结束时）

| 指标 | 之前（200ep） | 目标（300ep） | 验证标准 |
|------|--------------|--------------|---------|
| **Task Success Rate** | 0% | **> 10%** | 任务开始完成 |
| **Subtask Success Rate** | ~10% | **> 50%** | 子任务完成率提升 |
| **Decision Local** | 0-5% | **20-40%** | 平衡策略 |
| **Decision RSU** | 95-100% | **30-50%** | 不再单一 |
| **Decision V2V** | 0-5% | **10-30%** | 开始协同 |
| **Entropy** | 持续增长(0.5→2.3) | **收敛到0.5-1.0** | 策略稳定 |
| **Reward** | 无增长(-4.xx) | **稳定增长或收敛** | 学习有效 |

### 3.2 稳定性指标

✅ **无训练崩溃**: 不出现episode 92-99式reward暴跌  
✅ **Critic Loss稳定**: 保持在 < 10  
✅ **Entropy收敛**: 不再持续增长  
✅ **无单一策略**: 策略不收敛到100% RSU

---

## 四、运行与监控

### 4.1 启动训练

```bash
cd /Users/qiutao/研/毕设/毕设/vecOffloadingSim
python train.py
```

**预计时间**: 50-75分钟（取决于硬件）

### 4.2 实时监控

**终端日志（每10个episodes）:**
```
Episode 10: Reward=-3.45, Task SR=5.2%, RSU=45%, Local=35%, V2V=20%
Episode 20: Reward=-2.89, Task SR=8.7%, RSU=42%, Local=38%, V2V=20%
...
```

**关键观察点:**
- **Episode 0-50**: 初始探索，Decision应该开始平衡
- **Episode 50-100**: Reward应该开始增长
- **Episode 100**: 第1次衰减，观察策略调整
- **Episode 100-200**: Task SR应该> 0%
- **Episode 200**: 第2次衰减
- **Episode 200-300**: 指标稳定或收敛

### 4.3 查看结果

训练结束后自动生成20+张图表：

```bash
# 查看图表
ls runs/run_*/plots/

# 或手动生成
python test_plotting.py --run-dir runs/run_XXXXXX_XXXXXX
```

**重点图表:**
- `reward_curve_with_baselines.png` - 奖励收敛
- `veh_success_rate_with_baselines.png` - 任务成功率
- `offloading_ratio.png` - 决策分布演化
- `training_stability.png` - 训练稳定性
- `performance_radar.png` - 综合性能对比

---

## 五、问题诊断

### 5.1 如果Task Success Rate仍为0%

**可能原因:**
1. Deadline设置过紧 → 检查 `DEADLINE_TIGHTENING_MIN/MAX`
2. RSU队列拥堵 → 检查 `queue_load_balance.png`
3. 传输仍超时 → 增加 `BW_V2I` 到 100Mbps

**快速测试:**
```bash
python debug_rsu_simple.py  # 检查RSU是否正常工作
```

### 5.2 如果Decision仍然100% RSU

**可能原因:**
1. LOGIT_BIAS_RSU仍然过高 → 降至1.5
2. Bias衰减太慢 → 减少 `BIAS_DECAY_EVERY_EP` 到50
3. 本地CPU太慢 → 降低 `MIN_COMP`

**调整建议:**
```python
# configs/train_config.py
LOGIT_BIAS_RSU = 1.5  # 降低
LOGIT_BIAS_LOCAL = 1.5  # 保持相同
```

### 5.3 如果Entropy持续增长

**可能原因:**
1. ENTROPY_COEF仍然过高 → 降至0.003
2. 策略网络容量不足 → 增加 `EMBED_DIM` 到256
3. Reward信号不清晰 → 检查reward计算逻辑

**调整建议:**
```python
ENTROPY_COEF = 0.003  # 进一步降低
```

### 5.4 如果训练崩溃（NaN/Inf）

**检查项:**
```bash
# 检查梯度
python -c "from agents.mappo_agent import MAPPOAgent; print('Agent OK')"

# 检查环境
python debug_short_episode.py
```

**常见原因:**
- 梯度爆炸 → 降低 `LR_ACTOR` 到1e-4
- Reward异常 → 检查 `reward_engine.py`
- 数据异常 → 检查观测归一化

---

## 六、成功标准

### 6.1 最低成功标准（继续训练）

- ✅ Task Success Rate > 5%
- ✅ Decision不是100%单一选择
- ✅ Entropy不再持续增长
- ✅ Reward有增长趋势

### 6.2 理想成功标准（参数已优化）

- ✅ Task Success Rate > 10%
- ✅ Decision分布: Local 25-40%, RSU 35-50%, V2V 15-30%
- ✅ Entropy收敛到0.5-1.0
- ✅ Reward稳定增长

### 6.3 下一步行动

**如果达到最低标准:**
```bash
# 继续训练到1500 episodes
# 修改 configs/train_config.py
MAX_EPISODES = 1500
LR_DECAY_STEPS = 200
BIAS_DECAY_EVERY_EP = 200
SAVE_INTERVAL = 200

python train.py --load runs/run_XXXXXX/models/latest_model.pth
```

**如果未达到标准:**
- 重新审视参数（参考第五节诊断）
- 运行debug脚本检查环境
- 查看 `CONFIG_UPDATE_SUMMARY.md` 理解参数设计

---

## 七、快速参考

### 7.1 关键文件

```
configs/
  ├── config.py              # 环境参数（已优化）
  └── train_config.py        # 训练参数（短期验证）

runs/run_XXXXXX/
  ├── episode_log.csv        # Episode级数据
  ├── plots/                 # 20+张图表
  ├── models/                # Checkpoints
  └── config.json            # 训练时配置快照

文档/
  ├── VALIDATION_TRAINING_GUIDE.md   # 本文档
  ├── CONFIG_UPDATE_SUMMARY.md       # 参数调整总结
  ├── PLOTTING_GUIDE.md              # 绘图指南
  └── PLOT_UPDATE_SUMMARY.md         # 绘图更新
```

### 7.2 常用命令

```bash
# 启动训练
python train.py

# 测试绘图
python test_plotting.py --run-dir runs/run_XXXXXX

# 对比baseline
python eval_baselines.py --episodes 10

# Debug环境
python debug_rsu_simple.py
python debug_channel_detail.py

# 查看配置
python -c "from configs.train_config import TrainConfig; print(TrainConfig.MAX_EPISODES)"
```

---

## 八、预期输出示例

### Episode日志示例（目标）

```
Episode 10/300 | Reward: -3.45 | Task SR: 5.2% | Subtask SR: 42.3%
  Decisions: Local 35%, RSU 45%, V2V 20%
  Loss: 2.34 | Entropy: 1.82 | Duration: 8.5s

Episode 50/300 | Reward: -2.12 | Task SR: 12.8% | Subtask SR: 58.7%
  Decisions: Local 38%, RSU 42%, V2V 20%
  Loss: 1.87 | Entropy: 1.45 | Duration: 8.2s

Episode 100/300 | Reward: -1.45 | Task SR: 18.5% | Subtask SR: 65.3%
  Decisions: Local 35%, RSU 40%, V2V 25%
  Loss: 1.52 | Entropy: 1.12 | Duration: 8.0s
  ✓ LR Decay: 3e-4 → 2.76e-4
  ✓ Bias Decay: RSU 2.0→1.5, Local 1.5→1.2

Episode 200/300 | Reward: -0.87 | Task SR: 24.7% | Subtask SR: 72.1%
  Decisions: Local 32%, RSU 38%, V2V 30%
  Loss: 1.23 | Entropy: 0.95 | Duration: 7.8s
  ✓ LR Decay: 2.76e-4 → 2.54e-4
  ✓ Bias Decay: RSU 1.5→1.0, Local 1.2→0.9

Episode 300/300 | Reward: -0.65 | Task SR: 28.3% | Subtask SR: 76.4%
  Decisions: Local 30%, RSU 35%, V2V 35%
  Loss: 1.08 | Entropy: 0.82 | Duration: 7.5s
  ✅ Training Complete!
```

---

## 九、总结

**短期验证训练（300 episodes）的目的:**
1. 快速验证参数调整是否有效
2. 观察Task Success Rate是否> 0%
3. 确认Decision分布是否平衡
4. 检查训练稳定性（无崩溃）

**如果验证成功 → 继续训练到1500+ episodes**  
**如果验证失败 → 参考问题诊断调整参数**

---

**预祝训练成功！** 🚀

