#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_SNAPSHOT="${ROOT_DIR}/runs/run_1000ep_A_20260320/logs/config_snapshot.json"
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
EPISODES="${EPISODES:-1500}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"

RUN_ROOT="${ROOT_DIR}/runs/rc1_ablation_1500ep_${TIMESTAMP}"
RUN_FULL="${RUN_ROOT}/full"
RUN_WO_DAG="${RUN_ROOT}/wo_dag"
RUN_WO_RESOURCE="${RUN_ROOT}/wo_resource"
RUN_WO_DAG_RESOURCE="${RUN_ROOT}/wo_dag_resource"

SESSION_FULL="rc1_full_${TIMESTAMP}"
SESSION_WO_DAG="rc1_wo_dag_${TIMESTAMP}"
SESSION_WO_RESOURCE="rc1_wo_res_${TIMESTAMP}"
SESSION_WO_DAG_RESOURCE="rc1_wo_both_${TIMESTAMP}"

REMOTE_NAME="${GIT_REMOTE_NAME:-origin}"
BRANCH_NAME="${GIT_BRANCH_NAME:-$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)}"
COMMIT_MESSAGE="${GIT_COMMIT_MESSAGE:-Add RC1 fixed ablation runs (1500ep)}"
SHUTDOWN_DELAY_SECONDS="${SHUTDOWN_DELAY_SECONDS:-120}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-shutdown -h now}"
FINAL_LOG_PATH="${FINAL_LOG_PATH:-${ROOT_DIR}/runs/rc1_ablation_finalize_${TIMESTAMP}.log}"
NOHUP_LOG="${NOHUP_LOG:-${ROOT_DIR}/runs/rc1_ablation_finalize_${TIMESTAMP}.nohup.out}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${BASE_SNAPSHOT}" ]]; then
  echo "Missing base snapshot: ${BASE_SNAPSHOT}" >&2
  exit 1
fi

launch_job() {
  local session_name="$1"
  local run_dir="$2"
  local mode="$3"
  mkdir -p "${run_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DryRun] screen -dmS ${session_name} bash -lc 'cd ${ROOT_DIR} && exec scripts/ablation/run_rc1_ablation_screen_job.sh ${run_dir} ${BASE_SNAPSHOT} ${mode} ${EPISODES} ${DEVICE_NAME}'"
    return 0
  fi
  screen -dmS "${session_name}" bash -lc \
    "cd ${ROOT_DIR} && exec scripts/ablation/run_rc1_ablation_screen_job.sh ${run_dir} ${BASE_SNAPSHOT} ${mode} ${EPISODES} ${DEVICE_NAME}"
}

TO_WATCH=("${RUN_FULL}" "${RUN_WO_DAG}" "${RUN_WO_RESOURCE}" "${RUN_WO_DAG_RESOURCE}")

echo "[RC1 Ablation] run_root=${RUN_ROOT}"
echo "[RC1 Ablation] snapshot=${BASE_SNAPSHOT}"
echo "[RC1 Ablation] episodes=${EPISODES} device=${DEVICE_NAME}"

launch_job "${SESSION_FULL}" "${RUN_FULL}" "full"
launch_job "${SESSION_WO_DAG}" "${RUN_WO_DAG}" "no_dag"
launch_job "${SESSION_WO_RESOURCE}" "${RUN_WO_RESOURCE}" "no_resource"
launch_job "${SESSION_WO_DAG_RESOURCE}" "${RUN_WO_DAG_RESOURCE}" "no_dag_resource"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DryRun] nohup finalize watcher for: ${TO_WATCH[*]}"
  exit 0
fi

nohup bash -lc \
  "cd ${ROOT_DIR} && \
   GIT_REMOTE_NAME='${REMOTE_NAME}' \
   GIT_BRANCH_NAME='${BRANCH_NAME}' \
   GIT_COMMIT_MESSAGE='${COMMIT_MESSAGE}' \
   SHUTDOWN_DELAY_SECONDS='${SHUTDOWN_DELAY_SECONDS}' \
   SHUTDOWN_CMD='${SHUTDOWN_CMD}' \
   FINAL_LOG_PATH='${FINAL_LOG_PATH}' \
   exec scripts/ablation/finalize_rc1_ablation_runs.sh ${TO_WATCH[*]}" \
  > "${NOHUP_LOG}" 2>&1 &

echo
echo "[RC1 Ablation] launched screens:"
screen -ls | sed 's/^/  /'
echo
echo "[RC1 Ablation] attach commands:"
echo "  screen -r ${SESSION_FULL}"
echo "  screen -r ${SESSION_WO_DAG}"
echo "  screen -r ${SESSION_WO_RESOURCE}"
echo "  screen -r ${SESSION_WO_DAG_RESOURCE}"
echo
echo "[RC1 Ablation] run root: ${RUN_ROOT}"
echo "[RC1 Ablation] finalize log: ${FINAL_LOG_PATH}"
echo "[RC1 Ablation] nohup log: ${NOHUP_LOG}"
