#!/bin/bash
# =============================================================================
# 核心消融实验 - Core Ablation Study (精简版)
# =============================================================================
# 
# 只包含论文必需的消融对比：
# 1. DAG编码器消融 (Transformer vs MLP, Edge Bias vs None)
# 2. 资源感知消融 (Physics Bias vs None)  
# 3. 基线对比 (Random, Greedy, Local-Only)
# 4. 难度测试 (Hard Mode)
#
# =============================================================================

cd ~/vecOffloadingSim
EPISODES=1000
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

echo "=============================================="
echo "Core Ablation Study - $TIMESTAMP"
echo "Episodes: $EPISODES"
echo "=============================================="

# =============================================================================
# A. 核心算法消融 (3组)
# =============================================================================

# A0. 基准模型
echo "[A0] Baseline (Full Model)..."
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_full_${TIMESTAMP}" \
    > logs/ablation_full_${TIMESTAMP}.log 2>&1 &
PID_FULL=$!

# A1. 消融Transformer层 (只用Embedding+MLP)
echo "[A1] w/o Transformer (MLP baseline)..."
NUM_LAYERS=0 python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_no_transformer_${TIMESTAMP}" \
    > logs/ablation_no_transformer_${TIMESTAMP}.log 2>&1 &
PID_NO_TF=$!

# A2. 消融Edge Bias (标准Transformer)
echo "[A2] w/o Edge Bias (Standard Transformer)..."
USE_EDGE_BIAS=false USE_SPATIAL_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_no_edge_bias_${TIMESTAMP}" \
    > logs/ablation_no_edge_bias_${TIMESTAMP}.log 2>&1 &
PID_NO_EDGE=$!

# A3. 消融Physics Bias (纯Cross-Attention)
echo "[A3] w/o Physics Bias (Pure Cross-Attention)..."
USE_PHYSICS_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_no_physics_bias_${TIMESTAMP}" \
    > logs/ablation_no_physics_bias_${TIMESTAMP}.log 2>&1 &
PID_NO_PHY=$!

echo ""
echo "Core ablation experiments started: $PID_FULL $PID_NO_TF $PID_NO_EDGE $PID_NO_PHY"

# =============================================================================
# B. 难度测试 - Hard Mode
# =============================================================================

echo "[B] Hard Mode (Tight Deadline + Dense DAG)..."
DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.7 \
MIN_NODES=25 MAX_NODES=35 \
RSU_NUM_PROCESSORS=3 \
python train.py --seed $SEED --max-episodes $EPISODES \
    --run-id "ablation_hard_mode_${TIMESTAMP}" \
    > logs/ablation_hard_mode_${TIMESTAMP}.log 2>&1 &
PID_HARD=$!

echo "Hard mode started: $PID_HARD"

# =============================================================================
# 等待完成
# =============================================================================
echo ""
echo "=============================================="
echo "Waiting for all experiments..."
echo "Monitor: tail -f logs/ablation_*_${TIMESTAMP}.log"
echo "=============================================="

wait $PID_FULL $PID_NO_TF $PID_NO_EDGE $PID_NO_PHY $PID_HARD

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Generating comparison plots..."
echo "=============================================="

python scripts/plot_experiment_comparison.py --output runs/core_ablation_${TIMESTAMP}

echo "Done! Results in runs/core_ablation_${TIMESTAMP}/"
