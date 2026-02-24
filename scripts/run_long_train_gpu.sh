#!/usr/bin/env bash
set -euo pipefail

# Canonical long-train entrypoint (GPU server).
# Writes everything under runs/<run_dir>.
#
# Usage:
#   ./scripts/run_long_train_gpu.sh runs/exp_n20_chain 42 1500
#
# Notes:
# - Uses --exact-run-dir so follow-up scripts can locate the run directory.
# - RC1 default: does NOT force chain/trust flags. Set CHAIN_ENABLED=1 explicitly if needed.

RUN_DIR="${1:-runs/exp_n20_chain}"
SEED="${2:-42}"
EPISODES="${3:-1500}"

cd "$(dirname "$0")/.."

export NUM_VEHICLES="${NUM_VEHICLES:-20}"
export CANDIDATE_MODE="${CANDIDATE_MODE:-ALL}"
export DEVICE_NAME="${DEVICE_NAME:-cuda}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Stability tweak (recommended for long training based on observed late-stage regression).
# Default is also set in configs/train_config.py; keep env var for explicitness on servers.
export ENTROPY_COEF="${ENTROPY_COEF:-0.001}"

# Optional chain/trust settings (only applied if caller explicitly enables chain)
export CHAIN_ENABLED="${CHAIN_ENABLED:-0}"
if [[ "${CHAIN_ENABLED}" == "1" ]]; then
  export CHAIN_MODE="${CHAIN_MODE:-SWITCH}"
  export CHAIN_SWITCH_PERIOD_STEPS="${CHAIN_SWITCH_PERIOD_STEPS:-200}"
  export CHAIN_P95_LOW="${CHAIN_P95_LOW:-0.05}"
  export CHAIN_P95_HIGH="${CHAIN_P95_HIGH:-0.40}"
  export CHAIN_PFAIL_LOW="${CHAIN_PFAIL_LOW:-0.01}"
  export CHAIN_PFAIL_HIGH="${CHAIN_PFAIL_HIGH:-0.08}"
  export CHAIN_NOISE_STD="${CHAIN_NOISE_STD:-0.0}"
  export CHAIN_TRUST_DELAY_COUPLED="${CHAIN_TRUST_DELAY_COUPLED:-1}"
  export CHAIN_TRUST_DELAY_MIN_STEPS="${CHAIN_TRUST_DELAY_MIN_STEPS:-1}"
  export CHAIN_TRUST_DELAY_MAX_STEPS="${CHAIN_TRUST_DELAY_MAX_STEPS:-50}"
fi

python -u train.py \
  --seed "${SEED}" \
  --max-episodes "${EPISODES}" \
  --run-dir "${RUN_DIR}" \
  --exact-run-dir \
  --disable-baseline-eval

# Ensure a "full" metrics CSV exists even if some columns are only available via env_reward.jsonl.
python -u scripts/postprocess_run.py --run-dir "${RUN_DIR}" --overwrite || true

# Produce final plots folder (baselines may be missing at this point; plotting script handles that).
PLOT_CSV="${RUN_DIR}/metrics/train_metrics_full.csv"
if [[ ! -f "${PLOT_CSV}" ]]; then
  PLOT_CSV="${RUN_DIR}/metrics/train_metrics.csv"
fi
python -u scripts/plot_results.py --log-file "${PLOT_CSV}" --output-dir "${RUN_DIR}/plots" || true
