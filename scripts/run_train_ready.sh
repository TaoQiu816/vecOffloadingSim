#!/usr/bin/env bash
set -euo pipefail

# Purpose: launch train.py with reproducible run dirs and long-run defaults.
# Inputs: env vars (MAX_EPISODES/MAX_STEPS/CFG_PROFILE/SEED/DEVICE_NAME) + optional flags.
# Outputs: runs/<RUN_ID> with logs/metrics/plots/models.
# Example: MAX_EPISODES=5000 MAX_STEPS=300 CFG_PROFILE=train_ready_v1 SEED=7 bash scripts/run_train_ready.sh

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      export SEED="$2"
      shift 2
      ;;
    --device)
      export DEVICE_NAME="$2"
      shift 2
      ;;
    --episodes)
      export MAX_EPISODES="$2"
      shift 2
      ;;
    --steps)
      export MAX_STEPS="$2"
      shift 2
      ;;
    --run-id)
      export RUN_ID="$2"
      shift 2
      ;;
    --run-dir)
      export RUN_DIR="$2"
      shift 2
      ;;
    --log-interval)
      export LOG_INTERVAL="$2"
      shift 2
      ;;
    --eval-interval)
      export EVAL_INTERVAL="$2"
      shift 2
      ;;
    --save-interval)
      export SAVE_INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

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
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TB_LOG_OBS="${TB_LOG_OBS:-1}"

export MAX_EPISODES="${MAX_EPISODES:-5000}"
export MAX_STEPS="${MAX_STEPS:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

# Avoid slow baselines/auto-plot during training; use plot script after training.
export DISABLE_BASELINE_EVAL="${DISABLE_BASELINE_EVAL:-1}"
export DISABLE_AUTO_PLOT="${DISABLE_AUTO_PLOT:-1}"

mkdir -p "${RUN_DIR}/logs"

python -u train.py | tee "${RUN_DIR}/logs/train.log"
python scripts/plot_training_metrics.py --run_dir "${RUN_DIR}"
