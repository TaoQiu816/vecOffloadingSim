#!/usr/bin/env bash
set -euo pipefail

# Train-ready starter with reproducible outputs.
# Usage: RUN_ID=myrun MAX_EPISODES=200 bash scripts/run_train_ready.sh

export CFG_PROFILE="${CFG_PROFILE:-train_ready_v1}"
export REWARD_MODE="${REWARD_MODE:-delta_cft}"
export BONUS_MODE="${BONUS_MODE:-none}"
export DEVICE_NAME="${DEVICE_NAME:-cuda}"
export SEED="${SEED:-7}"
START_TS="$(date +%Y%m%d_%H%M%S)"
export RUN_ID="${RUN_ID:-train_ready_v1_seed7}"
if [[ ! "${RUN_ID}" =~ [0-9]{8}_[0-9]{6}$ ]]; then
  RUN_ID="${RUN_ID}_${START_TS}"
fi
export RUN_DIR="${RUN_DIR:-runs/${RUN_ID}}"
if [[ ! "$(basename "${RUN_DIR}")" =~ [0-9]{8}_[0-9]{6}$ ]]; then
  RUN_DIR="${RUN_DIR}_${START_TS}"
  RUN_ID="$(basename "${RUN_DIR}")"
fi
export EPISODE_JSONL_STDOUT="${EPISODE_JSONL_STDOUT:-0}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TB_LOG_OBS="${TB_LOG_OBS:-1}"

# Optional overrides
export MAX_EPISODES="${MAX_EPISODES:-200}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

# Avoid slow baselines/auto-plot during training; use plot script after training.
export DISABLE_BASELINE_EVAL="${DISABLE_BASELINE_EVAL:-1}"
export DISABLE_AUTO_PLOT="${DISABLE_AUTO_PLOT:-1}"

mkdir -p "${RUN_DIR}/logs"

echo "[INFO] CFG_PROFILE=${CFG_PROFILE}"
echo "[INFO] REWARD_MODE=${REWARD_MODE}"
echo "[INFO] RUN_DIR=${RUN_DIR}"

echo "[INFO] start training..."
python -u train.py | tee "${RUN_DIR}/logs/train.log"

echo "[INFO] plotting metrics..."
python scripts/plot_training_metrics.py --run_dir "${RUN_DIR}"
