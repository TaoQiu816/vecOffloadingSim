#!/bin/bash
# =============================================================================
# 完整消融实验 + B实验补完 + 自动绘图
# Full Ablation Study + Exp B Completion + Auto Plotting
# =============================================================================
#
# 包含：
# 1. 核心消融实验（4组）
# 2. Hard Mode难度测试
# 3. 补完之前未完成的B实验
# 4. 自动生成详细对比图
#
# =============================================================================

set -e
cd ~/vecOffloadingSim

EPISODES=1500
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/ablation_study_${TIMESTAMP}"

mkdir -p logs
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Full Ablation Study - $TIMESTAMP"
echo "Episodes: $EPISODES, Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# =============================================================================
# A. 核心消融实验 (4组)
# =============================================================================

# A0. 基准模型 (Full Model)
echo "[A0] Starting Full Model (Baseline)..."
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_full_${TIMESTAMP}" \
    > logs/ablation_full_${TIMESTAMP}.log 2>&1 &
PID_FULL=$!
echo "  PID: $PID_FULL"

sleep 3

# A1. 消融Transformer层 (NUM_LAYERS=0, 只用Embedding+MLP)
echo "[A1] Starting w/o Transformer..."
NUM_LAYERS=0 python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_no_transformer_${TIMESTAMP}" \
    > logs/ablation_no_transformer_${TIMESTAMP}.log 2>&1 &
PID_NO_TF=$!
echo "  PID: $PID_NO_TF"

sleep 3

# A2. 消融Edge/Spatial Bias (标准Transformer)
echo "[A2] Starting w/o Edge & Spatial Bias..."
USE_EDGE_BIAS=false USE_SPATIAL_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_no_edge_bias_${TIMESTAMP}" \
    > logs/ablation_no_edge_bias_${TIMESTAMP}.log 2>&1 &
PID_NO_EDGE=$!
echo "  PID: $PID_NO_EDGE"

sleep 3

# A3. 消融Physics Bias (纯Cross-Attention)
echo "[A3] Starting w/o Physics Bias..."
USE_PHYSICS_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_no_physics_bias_${TIMESTAMP}" \
    > logs/ablation_no_physics_bias_${TIMESTAMP}.log 2>&1 &
PID_NO_PHY=$!
echo "  PID: $PID_NO_PHY"

# =============================================================================
# B. 补完之前未完成的B实验 (High Pressure场景)
# =============================================================================

sleep 3

echo "[B] Starting High Pressure Scenario (Exp B)..."
DEADLINE_TIGHTENING_MIN=0.65 DEADLINE_TIGHTENING_MAX=0.85 \
MIN_NODES=22 MAX_NODES=30 \
MAX_COMP=4.0e9 NORM_MAX_COMP=5.0e9 \
RSU_QUEUE_CYCLES_LIMIT=180.0e9 \
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "exp_B_highpressure_${TIMESTAMP}" \
    > logs/exp_B_highpressure_${TIMESTAMP}.log 2>&1 &
PID_B=$!
echo "  PID: $PID_B"

# =============================================================================
# C. Hard Mode难度测试
# =============================================================================

sleep 3

echo "[C] Starting Hard Mode..."
DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.7 \
MIN_NODES=25 MAX_NODES=35 \
RSU_NUM_PROCESSORS=3 \
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_hard_mode_${TIMESTAMP}" \
    > logs/ablation_hard_mode_${TIMESTAMP}.log 2>&1 &
PID_HARD=$!
echo "  PID: $PID_HARD"

# =============================================================================
# 监控和等待
# =============================================================================

echo ""
echo "=============================================="
echo "All 6 experiments started!"
echo "PIDs: Full=$PID_FULL, NoTF=$PID_NO_TF, NoEdge=$PID_NO_EDGE, NoPhy=$PID_NO_PHY, ExpB=$PID_B, Hard=$PID_HARD"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/ablation_*_${TIMESTAMP}.log"
echo "  tail -f logs/exp_B_*_${TIMESTAMP}.log"
echo ""
echo "Check GPU:"
echo "  watch -n 5 nvidia-smi"
echo "=============================================="

# 等待所有实验完成
wait $PID_FULL $PID_NO_TF $PID_NO_EDGE $PID_NO_PHY $PID_B $PID_HARD

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Generating comprehensive comparison plots..."
echo "=============================================="

# =============================================================================
# D. 自动生成详细对比图
# =============================================================================

python scripts/plot_ablation_comparison.py \
    --full "runs/ablation_full_${TIMESTAMP}" \
    --no-transformer "runs/ablation_no_transformer_${TIMESTAMP}" \
    --no-edge "runs/ablation_no_edge_bias_${TIMESTAMP}" \
    --no-physics "runs/ablation_no_physics_bias_${TIMESTAMP}" \
    --exp-b "runs/exp_B_highpressure_${TIMESTAMP}" \
    --hard "runs/ablation_hard_mode_${TIMESTAMP}" \
    --output "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Done! All results saved to: $OUTPUT_DIR"
echo "=============================================="
