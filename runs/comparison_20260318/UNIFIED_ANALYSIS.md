# 基于UNIFIED奖励方案的客观分析与修改建议

## 一、奖励函数方案确认

### 1.1 当前奖励方案: UNIFIED

**组成**: `r_total = r_prog + r_term + r_illegal`

**1. Progress Reward (r_prog)** - 步进奖励
```python
r_prog = (phi_prev - phi_curr) / REWARD_PROGRESS_TNORM
r_prog = clip(r_prog, -REWARD_PROGRESS_RMAX, REWARD_PROGRESS_RMAX)

参数:
- REWARD_PROGRESS_TNORM = 2.5  # 归一化时间尺度
- REWARD_PROGRESS_RMAX = 1.0   # 最大奖励值
```
- `phi`: 关键路径剩余时间 (Critical Path Remaining Time)
- 含义: 关键路径剩余时间减少 → 正奖励（任务进展）

**2. Terminal Reward (r_term)** - 终止奖励
```python
# 成功且按时完成
if success and finish_time <= deadline:
    early_ratio = (deadline - finish_time) / deadline
    r_term = R_SUCCESS_ANCHOR + ALPHA_SUCCESS * early_ratio
    # 范围: [3.0, 5.0]

# 超时
elif finish_time > deadline:
    miss_ratio = (finish_time - deadline) / deadline
    r_term = -(R_FAIL_ANCHOR + ALPHA_MISS * miss_ratio)
    # 范围: [-3.0, -5.0]

# 失败
else:
    r_term = -(R_FAIL_ANCHOR + ALPHA_FAIL * severity)
    # 范围: [-3.0, -5.0]

参数:
- R_SUCCESS_ANCHOR = 3.0
- ALPHA_SUCCESS = 2.0
- R_FAIL_ANCHOR = 3.0
- ALPHA_FAIL = 2.0
- ALPHA_MISS = 2.0
- MISS_CAP = 1.0
```

**3. Illegal Penalty (r_illegal)** - 非法动作惩罚
```python
r_illegal = -W_ILLEGAL if illegal else 0.0

参数:
- W_ILLEGAL = 30.0
```

**4. Trust/Risk (保留，后续调整)**
- 当前启用: TRUST_ENABLED = True
- 失败范围: v2v_only (仅V2V可能失败)
- 延迟: TRUST_DELAY_STEPS = 3

---

## 二、两次训练的奖励分析

### 2.1 奖励组成对比（最后100集）

| 指标 | Run 1 (003251) | Run 2 (145633) | 差异 |
|------|----------------|----------------|------|
| **r_prog (步进)** | 0.0177 | 0.0132 | -25.4% |
| **r_term (终止)** | 0.1790 | 0.1053 | -41.2% |
| **episode_reward** | 0.139 | -0.028 | -120% |
| **任务成功率** | 95.8% | 89.9% | -5.9% |
| **完成时间** | 1.346s | 1.567s | +16.4% |
| **超时率** | 3.6% | 9.8% | +172% |

### 2.2 奖励比例分析

**Run 1**:
- r_prog占比: 9.0%
- r_term占比: 91.0%
- **Terminal奖励主导**

**Run 2**:
- r_prog占比: 11.1%
- r_term占比: 88.9%
- **Terminal奖励主导**

**关键发现**:
1. **Terminal奖励是主要信号** (占90%+)
2. Run 1的r_term更高 (0.179 vs 0.105)，说明：
   - 更多任务成功完成
   - 完成时间更早（early_ratio更高）
3. Run 2的r_term低41%，主要因为：
   - 成功率低5.9% → 更多负奖励
   - 完成时间长16% → early_ratio更低
   - 超时率高172% → 更多超时惩罚

---

## 三、核心问题诊断

### 3.1 Run 2为什么性能更差？

**问题1: 任务成功率下降5.9%**

| 指标 | Run 1 | Run 2 | 影响 |
|------|-------|-------|------|
| 成功率 | 95.8% | 89.9% | -5.9% |
| 超时率 | 3.6% | 9.8% | +6.2% |

**原因分析**:

99.9% RSU策略导致：
1. **通信瓶颈**: 所有任务都需要V2I通信
   - 简单任务也卸载，通信时间占比高
   - 距离远的车辆通信延迟大
   - 通信失败/重传增加延迟

2. **缺乏灵活性**: 无法根据任务特征调整
   - 简单任务Local执行可能更快
   - 紧急任务（deadline紧）应优先Local
   - 距离远的车辆应考虑Local

3. **RSU排队延迟**: 虽然队列利用率仅17%，但：
   - 瞬时峰值可能更高
   - 多个任务同时到达时排队
   - 排队时间不可预测

**问题2: 完成时间增加16.4%**

| 指标 | Run 1 | Run 2 | 影响 |
|------|-------|-------|------|
| 完成时间 | 1.346s | 1.567s | +0.221s |

**时间分解**:
```
总时间 = 通信时间 + 计算时间 + 排队时间

Run 1 (77% RSU + 21% Local):
- 77%任务: T_comm + T_comp(RSU) + T_queue
- 21%任务: T_comp(Local)
- 加权平均: 1.346s

Run 2 (99.9% RSU):
- 99.9%任务: T_comm + T_comp(RSU) + T_queue
- 加权平均: 1.567s
```

**增加的0.221s来自**:
1. 21%简单任务从Local改为RSU
2. 增加的通信时间 > 节省的计算时间
3. 证明: **通信是瓶颈，不是计算**

**问题3: Terminal奖励下降41%**

```
r_term = R_SUCCESS + ALPHA_SUCCESS * early_ratio  (成功)
r_term = -(R_FAIL + ALPHA_MISS * miss_ratio)      (超时)

Run 1: r_term = 0.179
- 95.8%成功 × (3.0 + 2.0 × early_ratio)
- 3.6%超时 × -(3.0 + 2.0 × miss_ratio)
- 0.6%失败 × -(3.0 + 2.0 × severity)

Run 2: r_term = 0.105
- 89.9%成功 × (3.0 + 2.0 × early_ratio)
- 9.8%超时 × -(3.0 + 2.0 × miss_ratio)
- 0.3%失败 × -(3.0 + 2.0 × severity)
```

**下降原因**:
1. 成功率降低 → 正奖励减少
2. 超时率增加 → 负奖励增加
3. 完成时间增加 → early_ratio降低

### 3.2 Run 1为什么更优？

**优势1: 任务感知的自适应策略**

77% RSU + 21% Local的混合策略说明智能体学会了：
1. **任务分类**: 区分简单/复杂任务
2. **时间权衡**: 通信时间 vs 计算时间
3. **资源优化**: 利用车辆空闲算力

**优势2: 更高的Terminal奖励**

r_term = 0.179 vs 0.105 (+70%)
- 更多任务成功 → 更多正奖励
- 更少超时 → 更少负奖励
- 更早完成 → 更高early_ratio

**优势3: 更好的Progress奖励**

r_prog = 0.0177 vs 0.0132 (+34%)
- 关键路径剩余时间减少更快
- 说明任务进展更顺利

---

## 四、参数影响的重新分析

### 4.1 基于UNIFIED奖励的理解

**核心目标**: 任务完成率 + 降低时延

**奖励信号**:
1. **r_prog**: 引导智能体减少关键路径剩余时间（降低时延）
2. **r_term**: 强化任务完成（成功率）和提前完成（降低时延）
3. **r_illegal**: 惩罚非法动作

**Terminal奖励占主导** (90%+):
- 任务成功/失败是最强信号
- 完成时间（early_ratio）是次要信号
- Progress奖励是辅助引导

### 4.2 Critic学习率的影响

**降低Critic学习率 (5e-4 → 2e-4) 的效果**:

在UNIFIED奖励下：
1. **Terminal信号主导** (90%)
2. 低学习率 → Critic快速学到"RSU成功率高"
3. 但忽略了"RSU完成时间长" → early_ratio低
4. 导致过度依赖RSU

**Run 1 (高学习率5e-4)**:
- Critic持续调整价值估计
- 学到"Local对简单任务更优"（通信时间<计算时间差）
- 找到77% RSU + 21% Local的平衡

**Run 2 (低学习率2e-4)**:
- Critic快速固化"RSU好"的估计
- 忽略了任务特征差异
- 收敛到99.9% RSU

### 4.3 熵系数的影响

**提升熵系数 (0.012 → 0.02) 的效果**:

在Terminal奖励主导下：
1. 熵正则化对Terminal信号影响有限
2. 0.02的熵系数 vs 3-5的Terminal奖励 → 比例仅0.4-0.7%
3. 无法对抗"RSU成功率高"的强信号

**关键问题**: 熵系数太小，相对Terminal奖励可忽略

---

## 五、修改建议

### 5.1 立即执行（保持当前奖励方案）

**建议1: 采用Run 1的超参数**

```python
LR_CRITIC = 5e-4      # 保持较高学习率
ENTROPY_COEF = 0.012  # 适中熵系数
MAX_EPISODES = 2000   # 延长训练
```

**理由**:
- Run 1在UNIFIED奖励下取得更好性能
- 混合策略(77% RSU + 21% Local)更优
- 所有指标均优于Run 2

**建议2: 增强熵系数（如果需要更多探索）**

```python
ENTROPY_COEF = 0.03   # 从0.012提升到0.03
```

**理由**:
- 当前0.012相对Terminal奖励(3-5)太小
- 提升到0.03可增强探索
- 但需配合其他机制

**建议3: 增强Logit Bias**

```python
LOGIT_BIAS_LOCAL = 0.3   # 从0.1提升到0.3
LOGIT_BIAS_RSU = -0.2    # 从0.0降到-0.2
LOGIT_BIAS_V2V = 0.2     # 从0.1提升到0.2
```

**理由**:
- 当前bias (0.1)相对Terminal奖励太小
- 需要更强的先验引导
- 抑制RSU过度使用

### 5.2 调整奖励参数（优化时延信号）

**问题**: Terminal奖励主导(90%)，但early_ratio信号较弱

**建议1: 增强early_ratio的权重**

```python
# 当前
R_SUCCESS_ANCHOR = 3.0
ALPHA_SUCCESS = 2.0
# 成功奖励范围: [3.0, 5.0]

# 建议
R_SUCCESS_ANCHOR = 2.0    # 降低基础奖励
ALPHA_SUCCESS = 4.0       # 提升early_ratio权重
# 成功奖励范围: [2.0, 6.0]
```

**效果**:
- 提前完成的奖励更高（6.0 vs 5.0）
- 刚好完成的奖励更低（2.0 vs 3.0）
- **强化"降低时延"的目标**

**建议2: 增强Progress奖励的权重**

```python
# 当前
REWARD_PROGRESS_TNORM = 2.5
REWARD_PROGRESS_RMAX = 1.0
# r_prog范围: [-1.0, 1.0]

# 建议
REWARD_PROGRESS_TNORM = 2.0   # 降低归一化尺度
REWARD_PROGRESS_RMAX = 2.0    # 提升最大值
# r_prog范围: [-2.0, 2.0]
```

**效果**:
- Progress奖励从[-1, 1]扩大到[-2, 2]
- 占比从9%提升到约15-20%
- **增强步进引导信号**

**建议3: 调整超时惩罚**

```python
# 当前
R_FAIL_ANCHOR = 3.0
ALPHA_MISS = 2.0
MISS_CAP = 1.0
# 超时惩罚范围: [-3.0, -5.0]

# 建议
R_FAIL_ANCHOR = 2.0
ALPHA_MISS = 4.0      # 提升超时惩罚
MISS_CAP = 1.5        # 放宽上限
# 超时惩罚范围: [-2.0, -8.0]
```

**效果**:
- 轻微超时惩罚降低（-2.0 vs -3.0）
- 严重超时惩罚增加（-8.0 vs -5.0）
- **更强的时延敏感性**

### 5.3 引入任务特征引导

**问题**: 智能体未充分利用任务特征

**建议1: 在观测中增强任务复杂度信号**

```python
# 在任务特征中增加
task_complexity = comp / NORM_MAX_COMP      # 计算量归一化
task_urgency = t_remaining / deadline       # 紧急度
comm_cost_estimate = data / rate_estimate   # 通信成本估计
```

**效果**: 帮助智能体学习任务感知策略

**建议2: 基于任务特征的Logit Bias**

```python
# 动态调整bias
if task_complexity < 0.3:  # 简单任务
    LOGIT_BIAS_LOCAL += 0.2
elif task_complexity > 0.7:  # 复杂任务
    LOGIT_BIAS_RSU += 0.1
```

**效果**: 引导简单任务Local，复杂任务RSU

### 5.4 信誉风险权重调整（后续）

**当前配置**:
```python
TRUST_ENABLED = True
TRUST_DELAY_STEPS = 3
TRUST_FAIL_SCOPE = "v2v_only"
```

**建议**: 暂时保持，后续根据V2V使用情况调整

如果V2V占比提升（如>10%），考虑：
```python
# 调整信誉失败的惩罚
TRUST_FAIL_PENALTY = -5.0  # 当前隐含在r_term中
# 或增加信誉成本项
TRUST_RISK_WEIGHT = 0.1
```

---

## 六、实验计划

### 6.1 基线实验（验证Run 1）

**实验1: 复现Run 1**
```python
LR_CRITIC = 5e-4
ENTROPY_COEF = 0.012
MAX_EPISODES = 2000
```

**目标**:
- 验证结果可复现
- Local占比保持20%左右
- 任务成功率>95%

### 6.2 超参数优化实验

**实验2: 增强熵系数**
```python
LR_CRITIC = 5e-4
ENTROPY_COEF = 0.03      # 从0.012提升
MAX_EPISODES = 2000
```

**预期**: Local占比提升至25-30%

**实验3: 增强Logit Bias**
```python
LR_CRITIC = 5e-4
ENTROPY_COEF = 0.012
LOGIT_BIAS_LOCAL = 0.3   # 从0.1提升
LOGIT_BIAS_RSU = -0.2    # 从0.0降低
MAX_EPISODES = 2000
```

**预期**: Local占比提升至30-35%

**实验4: 组合优化**
```python
LR_CRITIC = 5e-4
ENTROPY_COEF = 0.03
LOGIT_BIAS_LOCAL = 0.3
LOGIT_BIAS_RSU = -0.2
MAX_EPISODES = 2000
```

**预期**: Local占比提升至35-40%

### 6.3 奖励参数优化实验

**实验5: 增强early_ratio权重**
```python
# 超参数同Run 1
R_SUCCESS_ANCHOR = 2.0    # 从3.0降低
ALPHA_SUCCESS = 4.0       # 从2.0提升
```

**预期**:
- 完成时间进一步降低
- early_ratio信号更强

**实验6: 增强Progress奖励**
```python
# 超参数同Run 1
REWARD_PROGRESS_TNORM = 2.0   # 从2.5降低
REWARD_PROGRESS_RMAX = 2.0    # 从1.0提升
```

**预期**:
- Progress信号占比提升至15-20%
- 步进引导更强

**实验7: 综合优化**
```python
# 超参数
LR_CRITIC = 5e-4
ENTROPY_COEF = 0.03
LOGIT_BIAS_LOCAL = 0.3
LOGIT_BIAS_RSU = -0.2

# 奖励参数
R_SUCCESS_ANCHOR = 2.0
ALPHA_SUCCESS = 4.0
REWARD_PROGRESS_RMAX = 2.0
```

**预期**: 最优性能

### 6.4 评估指标

每个实验记录：
1. **核心指标**:
   - 任务成功率（目标≥95%）
   - 平均完成时间（目标<1.3s）
   - 超时率（目标<3%）

2. **动作分布**:
   - RSU/Local/V2V占比
   - 策略熵（目标>0.35）

3. **奖励组成**:
   - r_prog平均值
   - r_term平均值
   - episode_reward

4. **训练稳定性**:
   - 成功率标准差
   - 奖励标准差

---

## 七、总结

### 7.1 核心发现

**1. UNIFIED奖励方案下的特点**:
- Terminal奖励主导(90%)
- 任务成功/失败是最强信号
- early_ratio信号相对较弱

**2. Run 1优于Run 2的原因**:
- 混合策略(77% RSU + 21% Local)更优
- 任务成功率高5.9%
- 完成时间快16.4%
- Terminal奖励高70%

**3. Run 2性能劣化的原因**:
- 99.9% RSU策略缺乏灵活性
- 通信瓶颈导致时延增加
- 超时率增加172%

**4. 参数影响**:
- 降低Critic学习率 → 价值估计快速固化 → 过早收敛
- 提升熵系数效果有限 → 相对Terminal奖励太小

### 7.2 关键建议

**立即执行**:
1. 采用Run 1超参数 (LR_critic=5e-4, entropy=0.012)
2. 增强Logit Bias (Local=0.3, RSU=-0.2)
3. 训练2000 episodes

**后续优化**:
1. 增强early_ratio权重 (R_SUCCESS_ANCHOR=2.0, ALPHA_SUCCESS=4.0)
2. 增强Progress奖励 (REWARD_PROGRESS_RMAX=2.0)
3. 调整超时惩罚 (ALPHA_MISS=4.0)

**不建议**:
- ❌ 继续降低Critic学习率
- ❌ 大幅修改环境配置
- ❌ 引入其他奖励项（保持UNIFIED方案纯粹性）

### 7.3 理论启示

**1. Terminal奖励主导的训练特点**:
- 成功/失败信号最强
- 需要充分探索才能学到细粒度策略
- 高Critic学习率有助于持续调整

**2. 混合策略的价值**:
- 任务感知的自适应决策
- 通信 vs 计算的权衡
- 鲁棒性和性能的平衡

**3. 时延优化的关键**:
- 增强early_ratio权重
- 增强Progress奖励
- 引导简单任务Local执行

---

**报告生成时间**: 2026-03-18
**奖励方案**: UNIFIED (r_prog + r_term + r_illegal)
**核心目标**: 任务完成率 + 降低时延
**关键结论**: Run 1配置更优，需增强时延信号和Local引导
