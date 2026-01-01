#!/usr/bin/env bash
set -euo pipefail

# Recommended stable training preset for train_ready_v4.

export CFG_PROFILE=${CFG_PROFILE:-train_ready_v4}
export SEED=${SEED:-7}
export MAX_STEPS=${MAX_STEPS:-300}
export MAX_EPISODES=${MAX_EPISODES:-2000}
export LOG_INTERVAL=${LOG_INTERVAL:-10}
export DEVICE=${DEVICE:-cuda}

# Time-limit penalty (fixed by default)
export TIME_LIMIT_PENALTY_MODE=${TIME_LIMIT_PENALTY_MODE:-fixed}
export TIME_LIMIT_PENALTY=${TIME_LIMIT_PENALTY:--1}

# PPO/training knobs
export GAMMA=${GAMMA:-0.995}
export CLIP_PARAM=${CLIP_PARAM:-0.10}
export ENTROPY_COEF=${ENTROPY_COEF:-0.005}
export LR_ACTOR=${LR_ACTOR:-2e-4}
export LR_CRITIC=${LR_CRITIC:-3e-4}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
export USE_LOGIT_BIAS=${USE_LOGIT_BIAS:-0}

# Environment knobs
export VEHICLE_ARRIVAL_RATE=${VEHICLE_ARRIVAL_RATE:-0.0}
export RSU_NUM_PROCESSORS=${RSU_NUM_PROCESSORS:-2}
export BW_V2V=${BW_V2V:-80000000}
export MIN_CPU=${MIN_CPU:-1000000000}
export MAX_CPU=${MAX_CPU:-4000000000}

python train.py \
  --cfg-profile "$CFG_PROFILE" \
  --seed "$SEED" \
  --max-episodes "$MAX_EPISODES" \
  --max-steps "$MAX_STEPS" \
  --log-interval "$LOG_INTERVAL" \
  --device "$DEVICE" \
  --run-id "v4_stable_${SEED}_$(date +%Y%m%d_%H%M%S)"
