#!/bin/bash
# 黄金场景实验：设计能体现方法优越性的实验矩阵
# 核心思想：通过场景梯度 + 多baseline对比 + 针对性消融

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/golden_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "======================================"
echo "黄金场景实验 - 体现方法优越性"
echo "时间戳: $TIMESTAMP"
echo "======================================"

# =============================================================================
# 1. 严格挑战场景 - Full Model + 消融
# 目标：MAPPO 85%, Ablation 75%, Random ~70%
# 参数说明：
# - DAG节点数 25-35（比基线18-24更大，体现结构编码能力）
# - Deadline收紧到0.5-0.65（严格时间压力，经验证Random降至70%）
# - 车辆数50（激烈资源竞争）
# - RSU队列限制100G（容易拥塞）
# =============================================================================
echo ""
echo "=== 阶段1：严格挑战场景 ==="

# Full Model
echo "[1/9] Full Model on Strict-Challenge"
MIN_NODES=25 MAX_NODES=35 \
    DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.65 \
    NUM_VEHICLES=50 \
    MAX_COMP=4.0e9 NORM_MAX_COMP=5.0e9 \
    RSU_QUEUE_CYCLES_LIMIT=100.0e9 \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_strict_full" \
    > $LOG_DIR/strict_full.log 2>&1 &
PID1=$!

# w/o Transformer (NUM_LAYERS=0)
echo "[2/9] w/o Transformer on Strict-Challenge"
MIN_NODES=25 MAX_NODES=35 \
    DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.65 \
    NUM_VEHICLES=50 \
    MAX_COMP=4.0e9 NORM_MAX_COMP=5.0e9 \
    RSU_QUEUE_CYCLES_LIMIT=100.0e9 \
    NUM_LAYERS=0 \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_strict_no_transformer" \
    > $LOG_DIR/strict_no_transformer.log 2>&1 &
PID2=$!

# w/o Edge+Spatial Bias
echo "[3/9] w/o Edge/Spatial Bias on Strict-Challenge"
MIN_NODES=25 MAX_NODES=35 \
    DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.65 \
    NUM_VEHICLES=50 \
    MAX_COMP=4.0e9 NORM_MAX_COMP=5.0e9 \
    RSU_QUEUE_CYCLES_LIMIT=100.0e9 \
    USE_EDGE_BIAS=false USE_SPATIAL_BIAS=false \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_strict_no_edge_bias" \
    > $LOG_DIR/strict_no_edge_bias.log 2>&1 &
PID3=$!

# w/o Physics Bias
echo "[4/9] w/o Physics Bias on Strict-Challenge"
MIN_NODES=25 MAX_NODES=35 \
    DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.65 \
    NUM_VEHICLES=50 \
    MAX_COMP=4.0e9 NORM_MAX_COMP=5.0e9 \
    RSU_QUEUE_CYCLES_LIMIT=100.0e9 \
    USE_PHYSICS_BIAS=false \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_strict_no_physics_bias" \
    > $LOG_DIR/strict_no_physics_bias.log 2>&1 &
PID4=$!

# =============================================================================
# 2. 大规模DAG场景 - 对比Transformer的价值
# 参数说明：
# - DAG节点数 35-50（大规模DAG，长依赖链）
# - Deadline适度放宽到0.75-0.9（让任务可完成但有挑战）
# - 计算量降低（3.0G），让DAG结构成为主要挑战
# =============================================================================
echo ""
echo "=== 阶段2：大规模DAG场景（体现Transformer优势）==="

# Full Model
echo "[5/9] Full Model on Large-DAG"
MIN_NODES=35 MAX_NODES=50 \
    DEADLINE_TIGHTENING_MIN=0.75 DEADLINE_TIGHTENING_MAX=0.9 \
    NUM_VEHICLES=30 \
    MAX_COMP=3.0e9 NORM_MAX_COMP=4.0e9 \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_largedag_full" \
    > $LOG_DIR/largedag_full.log 2>&1 &
PID5=$!

# w/o Transformer
echo "[6/9] w/o Transformer on Large-DAG"
MIN_NODES=35 MAX_NODES=50 \
    DEADLINE_TIGHTENING_MIN=0.75 DEADLINE_TIGHTENING_MAX=0.9 \
    NUM_VEHICLES=30 \
    MAX_COMP=3.0e9 NORM_MAX_COMP=4.0e9 \
    NUM_LAYERS=0 \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_largedag_no_transformer" \
    > $LOG_DIR/largedag_no_transformer.log 2>&1 &
PID6=$!

# =============================================================================
# 3. 资源竞争场景 - 对比Physics Bias的价值
# 参数说明：
# - 车辆数50（高密度，激烈竞争）
# - RSU队列限制80G（容易拥塞）
# - V2V候选数7（更多协作选择）
# - Deadline收紧0.65-0.8（必须做出好的资源选择）
# =============================================================================
echo ""
echo "=== 阶段3：资源竞争场景（体现Physics Bias优势）==="

# Full Model
echo "[7/9] Full Model on Resource-Contention"
MIN_NODES=20 MAX_NODES=28 \
    DEADLINE_TIGHTENING_MIN=0.65 DEADLINE_TIGHTENING_MAX=0.8 \
    NUM_VEHICLES=50 \
    RSU_QUEUE_CYCLES_LIMIT=80.0e9 \
    V2V_TOP_K=7 \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_resource_full" \
    > $LOG_DIR/resource_full.log 2>&1 &
PID7=$!

# w/o Physics Bias
echo "[8/9] w/o Physics Bias on Resource-Contention"
MIN_NODES=20 MAX_NODES=28 \
    DEADLINE_TIGHTENING_MIN=0.65 DEADLINE_TIGHTENING_MAX=0.8 \
    NUM_VEHICLES=50 \
    RSU_QUEUE_CYCLES_LIMIT=80.0e9 \
    V2V_TOP_K=7 \
    USE_PHYSICS_BIAS=false \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_resource_no_physics_bias" \
    > $LOG_DIR/resource_no_physics_bias.log 2>&1 &
PID8=$!

# w/o Edge Bias (在资源竞争场景下也测试)
echo "[9/9] w/o Edge Bias on Resource-Contention"
MIN_NODES=20 MAX_NODES=28 \
    DEADLINE_TIGHTENING_MIN=0.65 DEADLINE_TIGHTENING_MAX=0.8 \
    NUM_VEHICLES=50 \
    RSU_QUEUE_CYCLES_LIMIT=80.0e9 \
    V2V_TOP_K=7 \
    USE_EDGE_BIAS=false USE_SPATIAL_BIAS=false \
    python train.py --seed 42 --max-episodes 1000 --run-id "golden_resource_no_edge_bias" \
    > $LOG_DIR/resource_no_edge_bias.log 2>&1 &
PID9=$!

echo ""
echo "所有9个实验已启动，等待完成..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9

# =============================================================================
# 4. 生成对比图
# =============================================================================
echo ""
echo "=== 生成对比分析图 ==="
python scripts/plot_golden_comparison.py --timestamp $TIMESTAMP

echo ""
echo "======================================"
echo "实验完成！"
echo "结果目录: runs/golden_study_${TIMESTAMP}"
echo "日志目录: $LOG_DIR"
echo "======================================"
