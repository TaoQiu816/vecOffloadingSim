#!/bin/bash
# =============================================================================
# 消融实验脚本 - Ablation Study
# =============================================================================
# 使用方法:
#   bash scripts/run_ablation.sh         # 运行所有消融
#   bash scripts/run_ablation.sh reward  # 只运行奖励消融
#   bash scripts/run_ablation.sh network # 只运行网络消融
# =============================================================================

cd ~/vecOffloadingSim
EPISODES=1000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# 奖励函数消融
# =============================================================================
run_reward_ablation() {
    echo "=== Reward Ablation ==="
    
    # 基准
    python train.py --seed 42 --max-episodes $EPISODES --run-id "ablation_baseline_${TIMESTAMP}" &
    PID_BASE=$!
    
    # w/o PBRS
    REWARD_BETA=0 python train.py --seed 42 --max-episodes $EPISODES --run-id "ablation_no_pbrs_${TIMESTAMP}" &
    PID_PBRS=$!
    
    # w/o Queue Penalty
    TIME_QUEUE_PENALTY_WEIGHT=0 python train.py --seed 42 --max-episodes $EPISODES --run-id "ablation_no_queue_penalty_${TIMESTAMP}" &
    PID_QUEUE=$!
    
    wait $PID_BASE $PID_PBRS $PID_QUEUE
    echo "Reward ablation completed"
}

# =============================================================================
# 网络结构消融
# =============================================================================
run_network_ablation() {
    echo "=== Network Ablation ==="
    
    # w/o Rank Bias
    USE_RANK_BIAS=false python train.py --seed 42 --max-episodes $EPISODES --run-id "ablation_no_rank_bias_${TIMESTAMP}" &
    PID_RANK=$!
    
    # w/o Logit Bias
    USE_LOGIT_BIAS=false python train.py --seed 42 --max-episodes $EPISODES --run-id "ablation_no_logit_bias_${TIMESTAMP}" &
    PID_LOGIT=$!
    
    wait $PID_RANK $PID_LOGIT
    echo "Network ablation completed"
}

# =============================================================================
# 难度提升测试
# =============================================================================
run_hard_mode() {
    echo "=== Hard Mode Test ==="
    
    DEADLINE_TIGHTENING_MIN=0.5 DEADLINE_TIGHTENING_MAX=0.7 \
    MIN_NODES=25 MAX_NODES=35 \
    RSU_NUM_PROCESSORS=3 \
    python train.py --seed 42 --max-episodes $EPISODES --run-id "ablation_hard_mode_${TIMESTAMP}" &
    
    wait
    echo "Hard mode test completed"
}

# =============================================================================
# 主入口
# =============================================================================
case "${1:-all}" in
    reward)
        run_reward_ablation
        ;;
    network)
        run_network_ablation
        ;;
    hard)
        run_hard_mode
        ;;
    all)
        run_reward_ablation
        run_network_ablation
        run_hard_mode
        ;;
    *)
        echo "Usage: $0 {reward|network|hard|all}"
        exit 1
        ;;
esac

echo "=== All ablation experiments completed ==="
python scripts/plot_experiment_comparison.py --output runs/ablation_comparison_${TIMESTAMP}
