# 审计系统使用指南

## 快速开始

### 1. 运行完整审计（20 episodes）

```bash
python scripts/debug_rollout_audit.py --episodes 20 --seed 42
```

**输出文件**：
- `logs/audit_summary_<timestamp>.jsonl` - 每个episode的指标摘要
- `logs/audit_events_<timestamp>.jsonl` - 异常事件记录
- `logs/audit_report_<timestamp>.txt` - 人类可读报告

### 2. 检查环境初始状态

```bash
python scripts/debug_env_state.py
```

检查：RSU位置、车辆分布、距离关系、队列状态

### 3. 应用审计建议的修复

```bash
python scripts/fix_deadline_config.py
```

自动修改配置并运行快速验证测试。

---

## 审计指标详解

### 核心12项指标

#### (1) RSU Mask可用性
- **定义**：每个决策点RSU是否可用的比例
- **正常范围**：20-50%（取决于RSU部署密度）
- **异常标志**：长期<5% → RSU mask逻辑错误或队列永久满载

#### (2) V2V可选邻居数
- **定义**：每个决策点可选V2V目标的平均数量
- **正常范围**：1-5个（取决于车流密度和V2V_RANGE）
- **异常标志**：>10个 → 类别不平衡，V2V概率淹没RSU/Local

#### (3) Illegal动作统计
- **定义**：Agent选择了mask=False的动作的次数
- **正常值**：<0.1%（训练中期）；0%（训练后期）
- **CRITICAL**：如果illegal动作的mask=True → 环境与mask不一致（致命bug）

#### (4) RSU队列长度
- **定义**：RSU上active tasks的平均数量
- **正常范围**：0-8（取决于任务负载和RSU算力）
- **异常标志**：长期卡在上限且不降 → 任务完成回调失效

#### (5) V2V生命周期守恒（6连）
- **定义**：V2V任务从发起到完成的6个阶段计数必须守恒
  1. TX Started（传输开始）
  2. TX Done（传输完成）
  3. Received（接收端接收）
  4. Added to Active（加入执行队列）
  5. CPU Finished（计算完成）
  6. DAG Completed（DAG状态更新）
- **守恒条件**：6个计数差值<1%
- **CRITICAL**：任何断层 → V2V执行链断裂

#### (6) Active准入断言失败
- **定义**：`add_active_task()`中硬断言触发次数
- **正常值**：0（生产运行）
- **触发原因**：
  - `is_dag_ready=False` → DAG依赖未满足
  - `is_data_ready=False` → V2V/V2I数据未传输完成
  - `task_status=PENDING` → 状态错误

#### (7) Done Reason唯一性
- **定义**：每个episode只能有一个终止原因
- **可选值**：`SUCCESS` | `FAIL` | `TRUNCATED`
- **异常**：同时出现多个原因 → 终止逻辑混乱

#### (8) Reward分解
- **定义**：按动作类型（Local/RSU/V2V）分组统计reward组成
- **检查点**：
  - V2V任务未完成但拿到正delta_cft → reward计算bug
  - Energy项为0 → 传输能耗未计入

#### (9) Terminated触发源
- **定义**：记录是什么条件触发了`done=True`
- **Force Continuation后**：应只有`time_limit`触发

#### (10) Latency指标
- **定义**：`wall_clock_latency = finish_time - arrival_time`
- **CRITICAL**：如果出现负值 → metric命名bug或计算错误

#### (11) 任务状态冲突
- **定义**：同一任务同时处于多个互斥状态
- **检测**：通过`_audit_task_registry`追踪
- **异常**：同一任务同时在waiting和active → 双重推进

#### (12) 双重推进检测
- **定义**：同一个rem_comp在一个step被扣两次
- **触发场景**：旧引擎和新引擎同时推进
- **修复**：确保只用一种推进方式

---

## 通过标准

### 生产就绪标准

| 指标 | 阈值 | 说明 |
|------|------|------|
| Illegal率 | <0.1% | 策略基本合法 |
| Mask一致性 | 100% | illegal只发生在mask=False |
| V2V守恒 | 差值<1% | 生命周期无断层 |
| 状态冲突 | 0次 | 无双重推进 |
| Latency非负 | 100% | metric正确 |
| RSU mask | >10% | RSU不被长期屏蔽 |

### 调试标准

如果未通过上述标准，审计报告会标注`❌ FAIL`并提供：
1. 失败原因定位（如"V2V: received>added_to_active差距大"）
2. 示例任务key（owner_id, subtask_id）
3. 相关代码位置（如"检查step_inter_task_transfers中的handover逻辑"）

---

## 审计策略选择

### Greedy策略（默认）
- **用途**：测试环境物理正确性
- **特点**：确定性、无探索、RSU优先
- **局限**：不测试V2V路径

### Random策略
```python
# 在audit脚本中替换策略部分
target = np.random.randint(0, Cfg.MAX_TARGETS)
```
- **用途**：压力测试illegal逻辑
- **特点**：大量非法动作，测试mask鲁棒性

### Trained MAPPO策略
```python
# 加载训练好的模型
agent = MAPPOAgent(...)
agent.load_model('models/best_model.pth')

# 在audit循环中
actions = agent.get_actions(obs, deterministic=True)
```
- **用途**：测试实际策略性能和V2V路径
- **特点**：真实训练行为，可验证V2V守恒

---

## 常见问题排查

### Q1: RSU长期被mask（可用率<5%）

**排查步骤**：
1. 运行`scripts/debug_env_state.py`检查RSU位置和覆盖
2. 检查`_select_best_rsu()`返回值
3. 检查`is_queue_full()`逻辑
4. 检查队列释放回调是否触发

**常见原因**：
- RSU队列满且不释放
- `is_in_coverage()`判定错误
- Mask数据提取时机错误

### Q2: V2V生命周期断层

**定位断点**：
```python
# 查看audit_summary.jsonl中的具体计数
tx_started:      100
tx_done:         100
received:        100
added_to_active:  85  # ← 断点在这里
cpu_finished:     85
dag_completed:    85
```

**对应修复点**：
- `tx_started` → `tx_done`：检查`step_inter_task_transfers`中的传输推进
- `tx_done` → `received`：检查数据到达判定逻辑
- `received` → `added_to_active`：检查handover逻辑中的`add_active_task`调用
- `added_to_active` → `cpu_finished`：检查`vehicle.step(dt)`或`rsu.step(dt)`
- `cpu_finished` → `dag_completed`：检查`_handle_task_completion`中的DAG更新

### Q3: 成功率为0%

**原因诊断**：
1. Deadline过紧 → 放宽`DEADLINE_TIGHTENING`
2. 任务过重 → 降低`MIN/MAX_COMP`
3. 策略未训练 → 训练更多episodes
4. 物理bug → 运行审计定位具体问题

---

## 集成到训练流程

### 方法1：训练后审计

```bash
# 1. 训练
python train.py --max-episodes 200 --run-id exp_v1

# 2. 审计
python scripts/debug_rollout_audit.py --episodes 20 --model models/exp_v1_best.pth

# 3. 对比
python scripts/compare_audit_reports.py logs/audit_report_*.txt
```

### 方法2：CI/CD自动审计

```yaml
# .github/workflows/audit.yml
on: [push]
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - name: Run Audit
        run: python scripts/debug_rollout_audit.py --episodes 20
      - name: Check Thresholds
        run: |
          illegal_rate=$(jq '.illegal_rate' logs/audit_summary_latest.jsonl | head -1)
          if (( $(echo "$illegal_rate > 0.01" | bc -l) )); then
            echo "❌ Illegal rate too high: $illegal_rate"
            exit 1
          fi
```

---

## 审计报告解读

### 示例报告片段

```
【3. Illegal动作】
  总Illegal: 42
  Illegal率: 0.40%
  Local: 0
  RSU: 0
  V2V: 42
  ❌ CRITICAL: 3 次mask=True但illegal（mask与env不一致）
```

**解读**：
- 有42个V2V illegal动作
- 其中3次是mask说可以但env拒绝 → **致命bug**
- 需检查：V2V邻居mask生成与`_get_vehicle_by_id()`的一致性

### 修复优先级

| 标签 | 优先级 | 说明 |
|------|--------|------|
| `❌ CRITICAL` | P0 | 立即修复，破坏核心逻辑 |
| `⚠️ WARNING` | P1 | 短期修复，影响性能 |
| `ℹ️ INFO` | P2 | 长期优化，不影响功能 |

---

## 扩展审计系统

### 添加新指标

1. 在`AuditCollector.reset_episode()`中初始化计数器
2. 在`_collect_audit_step_info()`或相关位置收集数据
3. 在`finalize_episode()`中计算统计量
4. 在`generate_report()`中添加展示逻辑

### 示例：添加"RSU负载不均衡"指标

```python
# 在_collect_audit_step_info中
rsu_loads = [rsu.get_num_active_tasks() for rsu in self.rsus]
audit_info['rsu_load_std'] = np.std(rsu_loads)

# 在generate_report中
avg_std = np.mean([s['rsu_load_std'] for s in summaries])
f.write(f"RSU负载标准差: {avg_std:.2f}\n")
if avg_std > 3.0:
    f.write("  ⚠️ RSU负载不均，考虑优化调度策略\n")
```

---

## 总结

审计系统是**质量保障的最后一道防线**。通过12项核心指标的持续监控，能在问题扩散前及时发现并定位bug，确保训练过程稳定和结果可靠。

**最佳实践**：
1. 每次重大改动后运行审计
2. 训练前/中/后各审计一次
3. 保存审计报告作为实验记录
4. 将审计阈值纳入回归测试

