#!/bin/bash
# =============================================================================
# 算法组件消融实验 - Algorithm Component Ablation Study
# =============================================================================
# 
# 消融维度：
# 1. DAG编码器: Transformer vs MLP
# 2. 结构偏置: Edge/Spatial Bias vs None
# 3. 优先级先验: Rank Bias (GA-DRL) vs None
# 4. 资源融合: Cross-Attention vs Concatenation
# 5. 功率控制: Beta分布 vs 固定功率
# 6. 奖励塑形: PBRS vs None
#
# =============================================================================

cd ~/vecOffloadingSim
EPISODES=1000
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs

echo "=============================================="
echo "Algorithm Ablation Study - $TIMESTAMP"
echo "Episodes: $EPISODES, Seed: $SEED"
echo "=============================================="

# =============================================================================
# 0. 基准模型 (Full Model)
# =============================================================================
run_baseline() {
    echo "[0/7] Running Baseline (Full Model)..."
    python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_baseline_${TIMESTAMP}" \
        > logs/ablation_baseline_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 1. 消融: DAG编码器 - Transformer vs MLP
# 使用 NUM_LAYERS=0 禁用Transformer层，只保留嵌入
# =============================================================================
run_no_transformer() {
    echo "[1/7] Running w/o Transformer (MLP only)..."
    NUM_LAYERS=0 python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_no_transformer_${TIMESTAMP}" \
        > logs/ablation_no_transformer_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 2. 消融: Edge/Spatial Bias
# 禁用边特征偏置和空间距离偏置
# =============================================================================
run_no_edge_bias() {
    echo "[2/7] Running w/o Edge & Spatial Bias..."
    USE_EDGE_BIAS=false USE_SPATIAL_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_no_edge_bias_${TIMESTAMP}" \
        > logs/ablation_no_edge_bias_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 3. 消融: Rank Bias (GA-DRL启发的优先级先验)
# =============================================================================
run_no_rank_bias() {
    echo "[3/7] Running w/o Rank Bias (GA-DRL)..."
    USE_RANK_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_no_rank_bias_${TIMESTAMP}" \
        > logs/ablation_no_rank_bias_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 4. 消融: Cross-Attention Physics Bias
# 禁用物理偏置，只用纯Cross-Attention
# =============================================================================
run_no_physics_bias() {
    echo "[4/7] Running w/o Physics Bias in Cross-Attention..."
    USE_PHYSICS_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_no_physics_bias_${TIMESTAMP}" \
        > logs/ablation_no_physics_bias_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 5. 消融: Beta分布功率控制 → 固定功率
# =============================================================================
run_fixed_power() {
    echo "[5/7] Running with Fixed Power (no Beta)..."
    USE_FIXED_POWER=true python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_fixed_power_${TIMESTAMP}" \
        > logs/ablation_fixed_power_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 6. 消融: PBRS奖励塑形
# =============================================================================
run_no_pbrs() {
    echo "[6/7] Running w/o PBRS (REWARD_BETA=0)..."
    REWARD_BETA=0 python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_no_pbrs_${TIMESTAMP}" \
        > logs/ablation_no_pbrs_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 7. 消融: Logit Bias (动作平衡)
# =============================================================================
run_no_logit_bias() {
    echo "[7/7] Running w/o Logit Bias..."
    USE_LOGIT_BIAS=false python train.py --seed $SEED --max-episodes $EPISODES \
        --run-id "ablation_no_logit_bias_${TIMESTAMP}" \
        > logs/ablation_no_logit_bias_${TIMESTAMP}.log 2>&1 &
    echo "  PID: $!"
}

# =============================================================================
# 主入口
# =============================================================================
case "${1:-all}" in
    baseline)
        run_baseline
        ;;
    transformer)
        run_no_transformer
        ;;
    edge)
        run_no_edge_bias
        ;;
    rank)
        run_no_rank_bias
        ;;
    physics)
        run_no_physics_bias
        ;;
    power)
        run_fixed_power
        ;;
    pbrs)
        run_no_pbrs
        ;;
    logit)
        run_no_logit_bias
        ;;
    all)
        run_baseline
        sleep 2
        run_no_transformer
        sleep 2
        run_no_edge_bias
        sleep 2
        run_no_rank_bias
        sleep 2
        run_no_physics_bias
        sleep 2
        run_no_pbrs
        sleep 2
        run_no_logit_bias
        ;;
    *)
        echo "Usage: $0 {baseline|transformer|edge|rank|physics|power|pbrs|logit|all}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Experiments started. Monitor with:"
echo "  tail -f logs/ablation_*_${TIMESTAMP}.log"
echo "=============================================="

# 等待所有实验完成（如果是all模式）
if [ "${1:-all}" = "all" ]; then
    echo "Waiting for all experiments to complete..."
    wait
    echo "All experiments completed!"
    echo "Generating comparison plots..."
    python scripts/plot_experiment_comparison.py --output runs/ablation_comparison_${TIMESTAMP}
fi
