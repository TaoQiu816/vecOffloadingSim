#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <run_dir> <config_snapshot> <ablation_mode> <episodes> [device]" >&2
  exit 2
fi

RUN_DIR="$1"
CONFIG_SNAPSHOT="$2"
ABLATION_MODE_ARG="$3"
EPISODES="$4"
DEVICE_NAME_ARG="${5:-cuda}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STATUS_DIR="${RUN_DIR}/launcher_status"
TRAIN_LOG="${RUN_DIR}/launcher_train.log"

mkdir -p "${RUN_DIR}" "${STATUS_DIR}"
rm -f "${STATUS_DIR}/success" "${STATUS_DIR}/failed"
touch "${STATUS_DIR}/started"

cd "${ROOT_DIR}"

export ABLATION_MODE="${ABLATION_MODE_ARG}"
export DEVICE_NAME="${DEVICE_NAME_ARG}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

{
  echo "[Launcher] root=${ROOT_DIR}"
  echo "[Launcher] run_dir=${RUN_DIR}"
  echo "[Launcher] config_snapshot=${CONFIG_SNAPSHOT}"
  echo "[Launcher] ABLATION_MODE=${ABLATION_MODE}"
  echo "[Launcher] EPISODES=${EPISODES}"
  echo "[Launcher] DEVICE_NAME=${DEVICE_NAME}"
  echo "[Launcher] started_at=$(date '+%F %T %Z')"
  echo
} | tee "${TRAIN_LOG}"

set +e
python -u train.py \
  --config-snapshot "${CONFIG_SNAPSHOT}" \
  --run-dir "${RUN_DIR}" \
  --exact-run-dir \
  --device "${DEVICE_NAME}" \
  --max-episodes "${EPISODES}" \
  --disable-baseline-eval \
  2>&1 | tee -a "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

if [[ ${TRAIN_EXIT} -ne 0 ]]; then
  echo "[Launcher] train failed exit=${TRAIN_EXIT}" | tee -a "${TRAIN_LOG}"
  echo "${TRAIN_EXIT}" > "${STATUS_DIR}/failed"
  exit "${TRAIN_EXIT}"
fi

set +e
python -u scripts/postprocess_run.py --run-dir "${RUN_DIR}" --overwrite 2>&1 | tee -a "${TRAIN_LOG}"
POST_EXIT=${PIPESTATUS[0]}
set -e

if [[ ${POST_EXIT} -ne 0 ]]; then
  echo "[Launcher] postprocess warning exit=${POST_EXIT}" | tee -a "${TRAIN_LOG}"
fi

{
  echo
  echo "[Launcher] finished_at=$(date '+%F %T %Z')"
  echo "[Launcher] success"
} | tee -a "${TRAIN_LOG}"

touch "${STATUS_DIR}/success"
