# 🚨 紧急修复方案 - 训练数据分析

## 问题诊断（Episode 1-20）

### ❌ 关键问题

1. **Task Success Rate = 0%** (20个episodes全部失败)
2. **策略快速收敛到100% RSU** (从Episode 13开始)
3. **Subtask SR仅10%** (大部分子任务超时)
4. **Reward信号极弱** (-0.04 ~ 0.02，无法指导学习)

---

## 根本原因

### 时间预算不足（最坏情况）

```
最大计算量DAG (12节点 × 1e9 cycles):
  总计算量: 12G cycles
  最慢车辆(1GHz): 需要 12秒
  Episode总时长: 10秒
  
  → 本地执行必然超时！ ❌
```

### Deadline过紧

```
平均本地时间: 2.75秒
Deadline (γ=1.5): 4.12秒
最坏情况需要: 12秒

→ 大部分任务会因deadline失败！
```

---

## 🔧 紧急修复方案

### 方案A: 大幅降低计算量（推荐）

```python
# configs/config.py

# 从当前值
MIN_COMP = 1.0e8  # 0.1G cycles
MAX_COMP = 1.0e9  # 1.0G cycles

# 修改为（降低80%）
MIN_COMP = 2.0e7  # 0.02G cycles (参考文献2倍)
MAX_COMP = 2.0e8  # 0.2G cycles (参考文献2倍)
```

**效果预估:**
- 平均本地时间: 2.75s → 0.55s ✓
- 最坏本地时间: 12s → 2.4s ✓
- Deadline: 4.12s → 0.82s（需要同步调整）

---

### 方案B: 放宽Deadline（配合方案A）

```python
# configs/config.py

# 从当前值
DEADLINE_TIGHTENING_MIN = 1.0
DEADLINE_TIGHTENING_MAX = 1.5

# 修改为
DEADLINE_TIGHTENING_MIN = 2.0
DEADLINE_TIGHTENING_MAX = 3.0
```

**效果预估:**
- Deadline: 0.82s → 1.65s (方案A) 或 4.12s → 8.24s (当前计算量)
- 给RSU卸载留出充足时间

---

### 方案C: 进一步降低数据量

```python
# configs/config.py

# 从当前值
MIN_DATA = 4.0e5  # 50KB
MAX_DATA = 2.0e6  # 250KB

# 修改为（降低50%）
MIN_DATA = 2.0e5  # 25KB
MAX_DATA = 1.0e6  # 125KB
```

**效果预估:**
- RSU传输时间: 0.48s → 0.24s
- 12车共享: 5.76s → 2.88s
- 减少传输瓶颈

---

## 📊 完整修复配置（推荐）

### configs/config.py

```python
# 1. 计算量（降至参考文献2倍）
MIN_COMP = 2.0e7    # 0.02G cycles
MAX_COMP = 2.0e8    # 0.2G cycles

# 2. 数据量（降低50%）
MIN_DATA = 2.0e5    # 25KB
MAX_DATA = 1.0e6    # 125KB

# 3. 边数据（保持参考文献）
MIN_EDGE_DATA = 8.0e5   # 100KB
MAX_EDGE_DATA = 4.0e6   # 500KB

# 4. Deadline（放宽到2-3倍本地时间）
DEADLINE_TIGHTENING_MIN = 2.0
DEADLINE_TIGHTENING_MAX = 3.0

# 5. 带宽（保持50Mbps）
BW_V2I = 50e6
```

---

## 验证计算

### 修复后性能预估

```
平均DAG (10节点):
  计算量: 1.1G cycles (之前5.5G)
  数据量: 6MB (之前12MB)

本地执行:
  平均车辆(2GHz): 0.55s (11步) ✓
  最慢车辆(1GHz): 1.1s (22步) ✓
  
RSU执行:
  传输(单车): 0.12s (2步)
  计算: 0.09s (2步)
  总时间: 0.21s (4步) ✓ 比本地快62%
  
RSU执行(12车):
  传输: 1.44s (29步) ✓

Deadline:
  平均: 0.55s × 3.0 = 1.65s (33步) ✓
  最坏: 1.1s × 3.0 = 3.3s (66步) ✓
  
→ 全部可完成！
```

---

## 🔄 执行步骤

### 1. 停止当前训练

```bash
Ctrl+C  # 停止当前训练
```

### 2. 修改配置

```bash
vim configs/config.py
# 按照上述方案修改5个参数
```

### 3. 重新启动

```bash
python train.py
```

### 4. 验证前10个episodes

**期望看到:**
- Task SR > 0% (开始有任务完成)
- Subtask SR > 50% (显著提升)
- Decision保持平衡 (不会100% RSU)
- Reward开始有明显波动 (信号增强)

---

## ⚠️ 重要说明

### 为什么需要如此大幅调整？

1. **参考文献计算量**: 1e7-1e8 cycles
2. **之前使用**: 1e8-1e9 (参考文献10倍)
3. **现在调整**: 2e7-2e8 (参考文献2倍)

**理由:**
- 之前10倍是假设完全卸载场景
- 但实际需要支持**本地执行**作为baseline
- 所以计算量必须让本地执行可行
- 2倍参考文献值在保持挑战性的同时确保可训练性

### 这会不会让任务太简单？

**不会！** 因为：
1. 仍有12车竞争带宽
2. 仍有队列等待
3. 仍有动态拓扑
4. 重点是学习**何时卸载**，而非**必须卸载**

---

## 📈 预期训练曲线（修复后）

```
Episode 1-20:
  Task SR: 0% → 20-40%
  Subtask SR: 10% → 60-80%
  Decision: 开始平衡（不再100% RSU）
  Reward: -0.04~0.02 → -2~0 (信号增强100倍)

Episode 20-50:
  Task SR: 40% → 60%
  Decision趋于最优分布
  
Episode 50-100:
  Task SR: 60% → 70%+
  策略收敛
```

---

## 立即行动

**1. 停止训练**  
**2. 修改5个参数**  
**3. 重新启动**  
**4. 观察前10个episodes**

如果前10个episodes看到Task SR > 0%，说明修复成功！

