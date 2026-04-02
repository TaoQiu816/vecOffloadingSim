# 论文实验数据清单

生成时间：2026-03-29

## 第二组：综合性能对比实验

### Baseline方法（需要评估）
- **LO (Local-Only)**: 需要运行评估
- **NRO (Nearest-RSU)**: 需要运行评估  
- **EFT-H**: 需要运行评估

### RL方法（已有训练数据）
- **IPPO-H**: `runs/rc1_batch1_part1_topology_20260323_182712/ippo_main/`
  - 训练数据: `logs/training_stats.csv` ✓
  - 配置: `logs/config_snapshot.json` ✓
  
- **F-MAPPO**: 需要确认位置（可能在 rc1_default_fmappo_20260328_224844）
  
- **TERA-MAPPO**: `runs/rc1_ablation_1500ep_20260322_180707/full/`
  - 训练数据: `logs/training_stats.csv` ✓
  - 配置: `logs/config_snapshot.json` ✓

## 第三组：消融实验

- **TERA-MAPPO (完整版)**: `runs/rc1_ablation_1500ep_20260322_180707/full/` ✓
- **w/o TDE**: `runs/rc1_ablation_1500ep_20260322_180707/wo_dag/` ✓
- **w/o CARE**: `runs/rc1_ablation_1500ep_20260322_180707/wo_resource/` ✓

## 第四组：任务复杂度与截止期敏感性

### 4A: DAG规模变化
- **Balanced**: `runs/rc1_batch1_part1_topology_20260323_182712/topology_balanced/full/` ✓
- **Deep**: `runs/rc1_batch1_part1_topology_20260323_182712/topology_deep/full/` ✓
- **Parallel**: `runs/rc1_batch1_part1_topology_20260323_182712/topology_parallel/full/` ✓

### 4B: 截止期因子变化
- 需要确认是否有专门的截止期扫描实验数据

## 第五组：系统负载与资源竞争

### 5A: 车辆数量变化
- **10辆**: `runs/rc1_batch2_vehicle_20260324_181254/vehicle_10/` ✓
- **20辆**: `runs/rc1_batch2_vehicle_20260324_181254/vehicle_20/` ✓
- **30辆**: `runs/rc1_batch2_vehicle_20260324_181254/vehicle_30/` ✓

### 5B: RSU算力变化
- **4 RSUs**: `runs/rc1_batch3_frsu_20260325_163701/frsu_4/` ✓
- **6 RSUs**: `runs/rc1_batch3_frsu_20260325_163701/frsu_6/` ✓
- **8 RSUs**: `runs/rc1_batch3_frsu_20260325_163701/frsu_8/` ✓

## 第六组：机制分析

需要从训练数据中提取：
- 执行模式占比（ratio_local, ratio_rsu, ratio_v2v）
- 时延分解（需要从episode_log或详细指标中提取）
- 功率分布（avg_power, power_ratio等）

## 数据完整性检查

### 已确认 ✓
- 消融实验数据完整
- DAG拓扑变化数据完整
- 车辆数量变化数据完整
- RSU数量变化数据完整
- IPPO-H训练数据完整

### 需要确认 ⚠️
- F-MAPPO训练数据位置
- 截止期因子扫描实验数据
- Baseline方法评估结果

### 需要生成 ⭕
- LO/NRO/EFT-H的评估结果
- 可能需要补充的截止期敏感性实验
