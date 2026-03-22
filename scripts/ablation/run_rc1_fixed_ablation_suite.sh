#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUN_DIR="${ROOT_DIR}/runs/run_1000ep_A_20260320"
BASE_SNAPSHOT="${BASE_RUN_DIR}/logs/config_snapshot.json"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-1000}"

if [[ ! -f "${BASE_SNAPSHOT}" ]]; then
  echo "missing snapshot: ${BASE_SNAPSHOT}" >&2
  exit 1
fi

run_one() {
  local mode="$1"
  local run_id="$2"
  echo "[RC1 Ablation] mode=${mode} run_id=${run_id}"
  ABLATION_MODE="${mode}" \
  python "${ROOT_DIR}/train.py" \
    --config-snapshot "${BASE_SNAPSHOT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --max-episodes "${EPISODES}" \
    --run-id "${run_id}"
}

run_one full "rc1_ablation_full"
run_one no_dag "rc1_ablation_wo_dag"
run_one no_resource "rc1_ablation_wo_resource"
run_one no_dag_resource "rc1_ablation_wo_dag_resource"
