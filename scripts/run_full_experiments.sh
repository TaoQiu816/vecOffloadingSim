#!/bin/bash
# =============================================================================
# 完整实验套件 - Full Experiment Suite
# =============================================================================
# 
# 包含：
# 1. 核心消融实验 (4组)
# 2. 难度测试 - Hard Mode (1组)
# 3. 之前未完成的B实验 (1组)
# 4. 自动生成详细对比图
#
# 总计：6组实验并行运行
# =============================================================================

set -e

cd ~/vecOffloadingSim
EPISODES=1500
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

echo "=============================================="
echo "Full Experiment Suite - $TIMESTAMP"
echo "Episodes: $EPISODES, Seed: $SEED"
echo "Total: 6 experiments in parallel"
echo "=============================================="

# =============================================================================
# 1. 基准模型 (Full Model)
# =============================================================================
echo "[1/6] Starting Baseline (Full Model)..."
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_full_model_${TIMESTAMP}" \
    > logs/exp_full_model_${TIMESTAMP}.log 2>&1 &
PID_FULL=$!
echo "  PID: $PID_FULL"

# =============================================================================
# 2. 消融: w/o Transformer (NUM_LAYERS=0)
# =============================================================================
echo "[2/6] Starting w/o Transformer..."
NUM_LAYERS=0 python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_no_transformer_${TIMESTAMP}" \
    > logs/exp_no_transformer_${TIMESTAMP}.log 2>&1 &
PID_NO_TF=$!
echo "  PID: $PID_NO_TF"

# =============================================================================
# 3. 消融: w/o Edge & Spatial Bias
# =============================================================================
echo "[3/6] Starting w/o Edge Bias..."
USE_EDGE_BIAS=false USE_SPATIAL_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_no_edge_bias_${TIMESTAMP}" \
    > logs/exp_no_edge_bias_${TIMESTAMP}.log 2>&1 &
PID_NO_EDGE=$!
echo "  PID: $PID_NO_EDGE"

# =============================================================================
# 4. 消融: w/o Physics Bias
# =============================================================================
echo "[4/6] Starting w/o Physics Bias..."
USE_PHYSICS_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_no_physics_bias_${TIMESTAMP}" \
    > logs/exp_no_physics_bias_${TIMESTAMP}.log 2>&1 &
PID_NO_PHY=$!
echo "  PID: $PID_NO_PHY"

# =============================================================================
# 5. Hard Mode (紧迫Deadline + 大规模DAG)
# =============================================================================
echo "[5/6] Starting Hard Mode..."
DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.7 \
MIN_NODES=25 MAX_NODES=35 \
RSU_NUM_PROCESSORS=3 \
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_hard_mode_${TIMESTAMP}" \
    > logs/exp_hard_mode_${TIMESTAMP}.log 2>&1 &
PID_HARD=$!
echo "  PID: $PID_HARD"

# =============================================================================
# 6. B实验重跑 (High Pressure - 之前未完成)
# =============================================================================
echo "[6/6] Starting High Pressure (B experiment)..."
DEADLINE_TIGHTENING_MIN=0.65 DEADLINE_TIGHTENING_MAX=0.85 \
MIN_NODES=22 MAX_NODES=30 MAX_COMP=4.0e9 NORM_MAX_COMP=5.0e9 \
RSU_QUEUE_CYCLES_LIMIT=180.0e9 \
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_high_pressure_${TIMESTAMP}" \
    > logs/exp_high_pressure_${TIMESTAMP}.log 2>&1 &
PID_HP=$!
echo "  PID: $PID_HP"

# =============================================================================
# 保存PID信息
# =============================================================================
echo "PID_FULL=$PID_FULL" > logs/experiment_pids_${TIMESTAMP}.txt
echo "PID_NO_TF=$PID_NO_TF" >> logs/experiment_pids_${TIMESTAMP}.txt
echo "PID_NO_EDGE=$PID_NO_EDGE" >> logs/experiment_pids_${TIMESTAMP}.txt
echo "PID_NO_PHY=$PID_NO_PHY" >> logs/experiment_pids_${TIMESTAMP}.txt
echo "PID_HARD=$PID_HARD" >> logs/experiment_pids_${TIMESTAMP}.txt
echo "PID_HP=$PID_HP" >> logs/experiment_pids_${TIMESTAMP}.txt

echo ""
echo "=============================================="
echo "All 6 experiments started!"
echo "=============================================="
echo ""
echo "Monitor commands:"
echo "  tail -f logs/exp_full_model_${TIMESTAMP}.log"
echo "  tail -f logs/exp_no_transformer_${TIMESTAMP}.log"
echo "  tail -f logs/exp_no_edge_bias_${TIMESTAMP}.log"
echo "  tail -f logs/exp_no_physics_bias_${TIMESTAMP}.log"
echo "  tail -f logs/exp_hard_mode_${TIMESTAMP}.log"
echo "  tail -f logs/exp_high_pressure_${TIMESTAMP}.log"
echo ""
echo "Check GPU: nvidia-smi"
echo "Kill all: pkill -f 'python train.py'"
echo ""

# =============================================================================
# 等待所有实验完成
# =============================================================================
echo "Waiting for all experiments to complete..."
wait $PID_FULL $PID_NO_TF $PID_NO_EDGE $PID_NO_PHY $PID_HARD $PID_HP

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="

# =============================================================================
# 生成详细对比图
# =============================================================================
echo "Generating detailed comparison plots..."
python scripts/plot_ablation_comparison.py --timestamp $TIMESTAMP --output runs/ablation_results_${TIMESTAMP}

echo ""
echo "=============================================="
echo "All done! Results saved to:"
echo "  runs/ablation_results_${TIMESTAMP}/"
echo "=============================================="
