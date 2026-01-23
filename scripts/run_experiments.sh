#!/bin/bash
# =============================================================================
# 三组对比实验训练脚本
# Run Three Experimental Configurations (1500 episodes each)
# =============================================================================
# 使用方法:
#   bash scripts/run_experiments.sh        # 并行运行三组实验
#   bash scripts/run_experiments.sh A      # 只运行方案A
#   bash scripts/run_experiments.sh B      # 只运行方案B
#   bash scripts/run_experiments.sh C      # 只运行方案C
# =============================================================================

set -e

# 设置基础路径
BASE_DIR="/Users/qiutao/研/毕设/毕设/vecOffloadingSim"
cd "$BASE_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p logs

# 选择运行的实验
RUN_TARGETS=${1:-"ABC"}

# =============================================================================
# 方案A: 基准延长训练 (Baseline Extended)
# 目标: 验证当前方案在1500ep下的收敛极限
# =============================================================================
run_exp_A() {
    echo "=========================================="
    echo "[$(date)] Starting Experiment A: Baseline Extended (1500ep)"
    echo "=========================================="
    
    LR_DECAY_STEPS=200 \
    LR_DECAY_RATE=0.95 \
    SAVE_INTERVAL=200 \
    nohup python train.py \
        --seed 42 \
        --max-episodes 1500 \
        --run-id "exp_A_baseline_${TIMESTAMP}" \
        > "logs/exp_A_${TIMESTAMP}.log" 2>&1 &
    
    PID_A=$!
    echo "Experiment A started with PID: $PID_A"
    echo $PID_A > "logs/exp_A_${TIMESTAMP}.pid"
}

# =============================================================================
# 方案B: 高压力场景 (Tight Deadline + Dense Task)
# 目标: 测试算法在更紧迫deadline和更大任务规模下的表现
# =============================================================================
run_exp_B() {
    echo "=========================================="
    echo "[$(date)] Starting Experiment B: High Pressure (1500ep)"
    echo "=========================================="
    
    DEADLINE_TIGHTENING_MIN=0.65 \
    DEADLINE_TIGHTENING_MAX=0.85 \
    MIN_NODES=22 \
    MAX_NODES=30 \
    MAX_COMP=4.0e9 \
    NORM_MAX_COMP=5.0e9 \
    RSU_QUEUE_CYCLES_LIMIT=180.0e9 \
    LR_DECAY_STEPS=200 \
    LR_DECAY_RATE=0.95 \
    SAVE_INTERVAL=200 \
    nohup python train.py \
        --seed 42 \
        --max-episodes 1500 \
        --run-id "exp_B_highpressure_${TIMESTAMP}" \
        > "logs/exp_B_${TIMESTAMP}.log" 2>&1 &
    
    PID_B=$!
    echo "Experiment B started with PID: $PID_B"
    echo $PID_B > "logs/exp_B_${TIMESTAMP}.pid"
}

# =============================================================================
# 方案C: 大规模车辆场景 (Scalability Test)
# 目标: 测试算法在更多车辆、更大场景下的可扩展性
# =============================================================================
run_exp_C() {
    echo "=========================================="
    echo "[$(date)] Starting Experiment C: Large Scale (1500ep)"
    echo "=========================================="
    
    NUM_VEHICLES=50 \
    V2V_TOP_K=7 \
    MAP_SIZE=1500.0 \
    NUM_RSU=4 \
    VEHICLE_SPAWN_X_MAX=0.85 \
    RSU_QUEUE_CYCLES_LIMIT=200.0e9 \
    VEHICLE_QUEUE_CYCLES_LIMIT=15.0e9 \
    ENTROPY_COEF=0.003 \
    LR_DECAY_STEPS=200 \
    LR_DECAY_RATE=0.95 \
    SAVE_INTERVAL=200 \
    nohup python train.py \
        --seed 42 \
        --max-episodes 1500 \
        --run-id "exp_C_largescale_${TIMESTAMP}" \
        > "logs/exp_C_${TIMESTAMP}.log" 2>&1 &
    
    PID_C=$!
    echo "Experiment C started with PID: $PID_C"
    echo $PID_C > "logs/exp_C_${TIMESTAMP}.pid"
}

# =============================================================================
# 根据参数运行实验
# =============================================================================
echo "=============================================="
echo "VEC DAG Offloading MAPPO - Ablation Study"
echo "Timestamp: ${TIMESTAMP}"
echo "=============================================="

if [[ "$RUN_TARGETS" == *"A"* ]]; then
    run_exp_A
fi

if [[ "$RUN_TARGETS" == *"B"* ]]; then
    run_exp_B
fi

if [[ "$RUN_TARGETS" == *"C"* ]]; then
    run_exp_C
fi

echo ""
echo "=============================================="
echo "实验已启动 / Experiments Started"
echo "=============================================="
echo ""
echo "监控日志 / Monitor logs:"
if [[ "$RUN_TARGETS" == *"A"* ]]; then
    echo "  tail -f logs/exp_A_${TIMESTAMP}.log"
fi
if [[ "$RUN_TARGETS" == *"B"* ]]; then
    echo "  tail -f logs/exp_B_${TIMESTAMP}.log"
fi
if [[ "$RUN_TARGETS" == *"C"* ]]; then
    echo "  tail -f logs/exp_C_${TIMESTAMP}.log"
fi
echo ""
echo "查看GPU使用 / Check GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止所有实验 / Stop all:"
echo "  pkill -f 'python train.py'"
echo ""
