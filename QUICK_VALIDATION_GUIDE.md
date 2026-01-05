# 短期验证性训练指南

## 训练配置（已优化）

### 训练规模
- **MAX_EPISODES**: 300（短期验证）
- **预计时长**: 约1-2小时（取决于硬件）
- **目标**: 快速验证配置有效性，观察关键指标趋势

### 验证目标

#### ✅ 必须达成（核心指标）
1. **Task Success Rate > 0%**
   - 当前问题: 0%（无任务完成）
   - 目标: > 20%（说明任务可完成）
   - 观察: 前100个episode应出现成功

2. **Decision分布合理**
   - 当前问题: 100% RSU（单一策略）
   - 目标分布:
     - Local: 20-40%
     - RSU: 40-60%
     - V2V: 5-20%
   - 观察: 100 episode后应出现平衡

3. **Entropy收敛**
   - 当前问题: 持续增长（0.5→2.3）
   - 目标: 稳定在0.5-1.2范围内
   - 观察: 不应持续单调增长

#### ⭐ 期望达成（性能指标）
4. **Subtask Success Rate > 50%**
   - 当前: ~10%
   - 目标: > 50%

5. **Reward增长**
   - 当前: 无明显增长
   - 目标: 总体上升趋势

6. **训练稳定性**
   - 无Episode 92-99式崩溃
   - Critic loss < 20
   - 无异常值（NaN/Inf）

---

## 快速启动

### 1. 启动训练
```bash
cd /Users/qiutao/研/毕设/毕设/vecOffloadingSim
python train.py --episodes 300
```

### 2. 实时监控（另开终端）
```bash
# 查看最新日志
tail -f runs/run_*/episode_log.csv

# 或使用watch命令每2秒刷新
watch -n 2 'tail -20 runs/run_*/episode_log.csv'
```

### 3. 中期检查点（Episode 100）
```bash
python -c "
import pandas as pd
import glob

# 找到最新运行
runs = sorted(glob.glob('runs/run_*'))
latest = runs[-1]
df = pd.read_csv(f'{latest}/episode_log.csv')

print('='*60)
print(f'Episode 100检查点 ({latest})')
print('='*60)

# 取最近10个episode的平均值
recent = df.tail(10)

print(f'\n【核心指标】')
print(f'  Task Success Rate:    {recent[\"task_success_rate\"].mean()*100:.1f}%')
print(f'  Subtask Success Rate: {recent[\"subtask_success_rate\"].mean()*100:.1f}%')
print(f'  Avg Reward:           {recent[\"total_reward\"].mean():.2f}')
print(f'  Entropy:              {recent.get(\"entropy\", [0]).mean():.2f}')

print(f'\n【Decision分布】')
print(f'  Local: {recent[\"decision_frac_local\"].mean()*100:.1f}%')
print(f'  RSU:   {recent[\"decision_frac_rsu\"].mean()*100:.1f}%')
print(f'  V2V:   {recent[\"decision_frac_v2v\"].mean()*100:.1f}%')

print(f'\n【验证结论】')
task_sr = recent['task_success_rate'].mean()
if task_sr > 0:
    print('  ✅ 任务可完成（Task Success Rate > 0%）')
else:
    print('  ❌ 任务仍无法完成，需检查参数')

local_ratio = recent['decision_frac_local'].mean()
rsu_ratio = recent['decision_frac_rsu'].mean()
if rsu_ratio < 0.9 and local_ratio > 0.1:
    print('  ✅ Decision分布合理（非100% RSU）')
else:
    print('  ⚠️  Decision分布仍不平衡')
"
```

---

## 验证通过标准

### ✅ 配置有效（可进行完整训练）
满足以下**至少2项**:
1. Task Success Rate > 10%（100 episode后）
2. Decision分布: Local > 15%, V2V > 5%
3. Entropy < 1.5（200 episode后）
4. Subtask Success Rate > 40%

### ⚠️ 需微调
满足1项但不满足2项:
- 可尝试微调ENTROPY_COEF（0.003-0.01）
- 或调整LOGIT_BIAS（±0.5）

### ❌ 配置无效（需重新检查）
0项满足:
- 检查环境参数（BW_V2I, COMP, DATA）
- 检查仿真逻辑（队列、传输、计算）
- 查看CONFIG_UPDATE_SUMMARY.md

---

## 训练结束后分析

### 生成完整图表
```bash
python test_plotting.py --run-dir runs/run_YYYYMMDD_HHMMSS
```
将生成20张图表到 `runs/.../plots/`

### 关键图表查看优先级

**1. 核心验证（必看）**
- `veh_success_rate_with_baselines.png` - 任务成功率
- `offloading_ratio.png` - 决策分布演化
- `reward_curve_with_baselines.png` - 奖励趋势

**2. 问题诊断**
- `training_stability.png` - 训练波动性
- `success_rate_multilevel.png` - 多层成功率
- `resource_utilization.png` - 资源利用

**3. 深入分析**
- `performance_radar.png` - 综合性能对比
- `reward_decomposition.png` - 奖励组成分析

---

## 训练时间估算

### 硬件依赖
- **CPU (无GPU)**: ~3-4小时（300 episodes）
- **GPU (CUDA)**: ~1-1.5小时
- **Apple Silicon (MPS)**: ~2-3小时

### Episode时长
- 单个Episode: ~20-40秒
- 300 Episodes: 100-200分钟（纯计算时间）
- 加上日志/保存: +10-20%

### 优化建议
如果训练太慢：
```python
# configs/train_config.py
MAX_EPISODES = 200      # 减少到200
LOG_INTERVAL = 20       # 减少日志频率
SAVE_INTERVAL = 200     # 只保存最后一次
```

---

## 常见问题

### Q1: 前50个episode全部超时？
A: 正常现象。Bias初始值较高，策略需要学习。
   观察100-150 episode区间是否改善。

### Q2: Entropy持续增长？
A: 如果超过1.5仍增长，说明ENTROPY_COEF仍偏高。
   可尝试降至0.003。

### Q3: Task Success仍为0%？
A: 检查：
   1. `avg_queue_len` 是否爆炸（>20）
   2. Episode是否提前终止（duration < 10s）
   3. 查看`debug_tx_rate.py`检查传输速率

### Q4: 100% RSU未改善？
A: 降低`LOGIT_BIAS_RSU`至1.5，或检查Bias退火是否生效。

### Q5: 训练中断如何恢复？
A: 自动保存latest_model.pth，重新运行train.py即可继续。

---

## 下一步

### ✅ 验证通过
→ 进行完整训练（1000-2000 episodes）
```bash
# 修改train_config.py
MAX_EPISODES = 1500
python train.py
```

### ⚠️ 需微调
→ 根据验证结果调整1-2个参数，重新验证

### ❌ 验证失败
→ 查看CONFIG_UPDATE_SUMMARY.md第六章"常见问题"
→ 或创建issue附带episode_log.csv

---

**验证训练目标**: 用最短时间确认配置方向正确，而非追求最优性能。
