#!/usr/bin/env bash
set -euo pipefail

# Canonical baseline entrypoint (CPU server).
# Runs baselines in parallel by policy and merges parts.
#
# Usage:
#   ./scripts/run_baselines_cpu_parallel.sh runs/exp_n20_chain 42 500 200
#
# Args:
#   run_dir seed num_episodes max_steps

RUN_DIR="${1:?run_dir required}"
SEED="${2:-42}"
NUM_EP="${3:-}"
MAX_STEPS="${4:-200}"

cd "$(dirname "$0")/.."

PARTS_DIR="${RUN_DIR}/logs/baseline_parts"
mkdir -p "${PARTS_DIR}"

POLICIES=("Random" "Local-Only" "Greedy" "EFT" "CP-EFT" "Static")

if [[ -z "${NUM_EP}" ]]; then
  # Default: match RL training episode count for fair, same-length comparison curves.
  # Prefer train_metrics_full.csv if present.
  if [[ -f "${RUN_DIR}/metrics/train_metrics_full.csv" ]]; then
    NUM_EP="$(python - <<PY\nimport pandas as pd\np='${RUN_DIR}/metrics/train_metrics_full.csv'\ndf=pd.read_csv(p)\nprint(int(df['episode'].max()))\nPY\n)"
  elif [[ -f "${RUN_DIR}/metrics/train_metrics.csv" ]]; then
    NUM_EP="$(python - <<PY\nimport pandas as pd\np='${RUN_DIR}/metrics/train_metrics.csv'\ndf=pd.read_csv(p)\nprint(int(df['episode'].max()))\nPY\n)"
  else
    NUM_EP="500"
  fi
fi

echo "[BaselinesParallel] run_dir=${RUN_DIR} seed=${SEED} episodes=${NUM_EP} max_steps=${MAX_STEPS}"

pids=()
for p in "${POLICIES[@]}"; do
  out="${PARTS_DIR}/${p}.csv"
  echo "[BaselinesParallel] start policy=${p} -> ${out}"
  python -u scripts/run_baselines.py \
    --run-dir "${RUN_DIR}" \
    --num-episodes "${NUM_EP}" \
    --seed "${SEED}" \
    --max-steps "${MAX_STEPS}" \
    --policies "${p}" \
    --output-csv "${out}" \
    >/dev/null 2>"${PARTS_DIR}/${p}.err" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

python -u scripts/merge_baseline_parts.py --run-dir "${RUN_DIR}" --parts "logs/baseline_parts/*.csv"

if [[ "${fail}" -ne 0 ]]; then
  echo "[BaselinesParallel] some policies failed; merged partial results anyway."
fi
