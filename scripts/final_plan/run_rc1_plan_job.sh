#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

JOB_KIND="${JOB_KIND:-train}"                  # train | baseline
RUN_DIR="${RUN_DIR:-}"
RUN_LABEL="${RUN_LABEL:-unnamed}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-${ROOT_DIR}/runs/run_1000ep_A_20260320/logs/config_snapshot.json}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
EPISODES="${EPISODES:-1500}"
BASELINE_EPISODES="${BASELINE_EPISODES:-50}"
POLICIES="${POLICIES:-}"
ABLATION_MODE_ARG="${ABLATION_MODE:-full}"
SNAPSHOT_OVERRIDES="${SNAPSHOT_OVERRIDES:-}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-RC1 ${JOB_KIND} ${RUN_LABEL}}"
MPLBACKEND="${MPLBACKEND:-Agg}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR is required" >&2
  exit 2
fi
if [[ ! -f "${BASE_SNAPSHOT}" ]]; then
  echo "Missing BASE_SNAPSHOT: ${BASE_SNAPSHOT}" >&2
  exit 2
fi

STATUS_DIR="${RUN_DIR}/launcher_status"
TRAIN_LOG="${RUN_DIR}/launcher_train.log"
SNAPSHOT_OUT="${RUN_DIR}/logs/config_snapshot.json"

mkdir -p "${RUN_DIR}" "${RUN_DIR}/logs" "${STATUS_DIR}"
rm -f "${STATUS_DIR}/success" "${STATUS_DIR}/failed"
touch "${STATUS_DIR}/started"

{
  echo "[Launcher] root=${ROOT_DIR}"
  echo "[Launcher] job_kind=${JOB_KIND}"
  echo "[Launcher] run_label=${RUN_LABEL}"
  echo "[Launcher] run_dir=${RUN_DIR}"
  echo "[Launcher] base_snapshot=${BASE_SNAPSHOT}"
  echo "[Launcher] snapshot_out=${SNAPSHOT_OUT}"
  echo "[Launcher] device=${DEVICE_NAME}"
  echo "[Launcher] episodes=${EPISODES}"
  echo "[Launcher] baseline_episodes=${BASELINE_EPISODES}"
  echo "[Launcher] policies=${POLICIES}"
  echo "[Launcher] ablation_mode=${ABLATION_MODE_ARG}"
  echo "[Launcher] snapshot_overrides=${SNAPSHOT_OVERRIDES}"
  echo "[Launcher] started_at=$(date '+%F %T %Z')"
  echo
} | tee "${TRAIN_LOG}"

SNAP_ARGS=()
if [[ -n "${SNAPSHOT_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  KV_PAIRS=( ${SNAPSHOT_OVERRIDES} )
  for kv in "${KV_PAIRS[@]}"; do
    SNAP_ARGS+=( --set "${kv}" )
  done
fi

python -u scripts/final_plan/make_snapshot_variant.py \
  --base-snapshot "${BASE_SNAPSHOT}" \
  --out-snapshot "${SNAPSHOT_OUT}" \
  "${SNAP_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"

set +e
if [[ "${JOB_KIND}" == "train" ]]; then
  export ABLATION_MODE="${ABLATION_MODE_ARG}"
  export DEVICE_NAME="${DEVICE_NAME}"
  export MPLBACKEND
  python -u train.py \
    --config-snapshot "${SNAPSHOT_OUT}" \
    --run-dir "${RUN_DIR}" \
    --exact-run-dir \
    --device "${DEVICE_NAME}" \
    --max-episodes "${EPISODES}" \
    --disable-baseline-eval \
    2>&1 | tee -a "${TRAIN_LOG}"
  JOB_EXIT=${PIPESTATUS[0]}
  if [[ ${JOB_EXIT} -eq 0 ]]; then
    python -u scripts/postprocess_run.py --run-dir "${RUN_DIR}" --overwrite 2>&1 | tee -a "${TRAIN_LOG}"
  fi
elif [[ "${JOB_KIND}" == "baseline" ]]; then
  if [[ -z "${POLICIES}" ]]; then
    echo "[Launcher] POLICIES is required for baseline jobs" | tee -a "${TRAIN_LOG}"
    JOB_EXIT=2
  else
    python -u scripts/run_baselines.py \
      --run-dir "${RUN_DIR}" \
      --num-episodes "${BASELINE_EPISODES}" \
      --policies "${POLICIES}" \
      2>&1 | tee -a "${TRAIN_LOG}"
    JOB_EXIT=${PIPESTATUS[0]}
  fi
else
  echo "[Launcher] unsupported JOB_KIND=${JOB_KIND}" | tee -a "${TRAIN_LOG}"
  JOB_EXIT=2
fi
set -e

if [[ ${JOB_EXIT} -ne 0 ]]; then
  echo "[Launcher] job failed exit=${JOB_EXIT}" | tee -a "${TRAIN_LOG}"
  echo "${JOB_EXIT}" > "${STATUS_DIR}/failed"
  exit "${JOB_EXIT}"
fi

scripts/final_plan/git_sync_run.sh "${RUN_DIR}" "${COMMIT_MESSAGE}" 2>&1 | tee -a "${TRAIN_LOG}"

{
  echo
  echo "[Launcher] finished_at=$(date '+%F %T %Z')"
  echo "[Launcher] success"
} | tee -a "${TRAIN_LOG}"

touch "${STATUS_DIR}/success"
