#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_SNAPSHOT="${ROOT_DIR}/runs/run_1000ep_A_20260320/logs/config_snapshot.json"
BASE_RUN_EXP1="${ROOT_DIR}/runs/run_1000ep_A_20260320"

RUN_EXP2="${ROOT_DIR}/runs/run_1000ep_A_lrcritic_3e4_20260321"
RUN_EXP3="${ROOT_DIR}/runs/run_1000ep_A_lrcritic_5e4_20260321"

SESSION_EXP2="lr_exp2_c3e4"
SESSION_EXP3="lr_exp3_c5e4"

REMOTE_NAME="${GIT_REMOTE_NAME:-origin}"
BRANCH_NAME="${GIT_BRANCH_NAME:-$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)}"
COMMIT_MESSAGE="${GIT_COMMIT_MESSAGE:-Add A-based LR critic sweep runs (3e-4, 5e-4)}"
SHUTDOWN_DELAY_SECONDS="${SHUTDOWN_DELAY_SECONDS:-120}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-shutdown -h now}"
FINAL_LOG_PATH="${FINAL_LOG_PATH:-${ROOT_DIR}/runs/lr_critic_sweep_finalize_20260321.log}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${BASE_SNAPSHOT}" ]]; then
  echo "Missing base snapshot: ${BASE_SNAPSHOT}" >&2
  exit 1
fi

is_completed_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/logs/training_stats.csv" ]] && [[ -s "${run_dir}/logs/config_snapshot.json" ]]
}

launch_job() {
  local session_name="$1"
  local run_dir="$2"
  local actor_lr="$3"
  local critic_lr="$4"
  local device_name="${5:-cuda}"
  mkdir -p "${run_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DryRun] screen -dmS ${session_name} bash -lc 'cd ${ROOT_DIR} && exec scripts/sweep/run_lr_screen_job.sh ${run_dir} ${BASE_SNAPSHOT} ${actor_lr} ${critic_lr} ${device_name}'"
    return 0
  fi
  screen -dmS "${session_name}" bash -lc \
    "cd ${ROOT_DIR} && exec scripts/sweep/run_lr_screen_job.sh ${run_dir} ${BASE_SNAPSHOT} ${actor_lr} ${critic_lr} ${device_name}"
}

echo "[Sweep] exp1 existing baseline run: ${BASE_RUN_EXP1}"
if is_completed_run "${BASE_RUN_EXP1}"; then
  echo "[Sweep] exp1 already completed, skip rerun"
else
  echo "[Sweep][WARN] exp1 base run not found complete: ${BASE_RUN_EXP1}"
fi

TO_WATCH=()

if is_completed_run "${RUN_EXP2}"; then
  echo "[Sweep] exp2 already completed, skip: ${RUN_EXP2}"
else
  echo "[Sweep] launching exp2 in screen ${SESSION_EXP2}"
  TO_WATCH+=("${RUN_EXP2}")
  launch_job "${SESSION_EXP2}" "${RUN_EXP2}" "2e-4" "3e-4" "cuda"
fi

if is_completed_run "${RUN_EXP3}"; then
  echo "[Sweep] exp3 already completed, skip: ${RUN_EXP3}"
else
  echo "[Sweep] launching exp3 in screen ${SESSION_EXP3}"
  TO_WATCH+=("${RUN_EXP3}")
  launch_job "${SESSION_EXP3}" "${RUN_EXP3}" "2e-4" "5e-4" "cuda"
fi

if [[ ${#TO_WATCH[@]} -eq 0 ]]; then
  echo "[Sweep] no pending runs, nothing launched"
  exit 0
fi

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
   exec scripts/sweep/finalize_lr_screen_runs.sh ${TO_WATCH[*]}" \
  > "${ROOT_DIR}/runs/lr_critic_sweep_finalize_nohup.out" 2>&1 &

echo
echo "[Sweep] launched screens:"
screen -ls | sed 's/^/  /'
echo
echo "[Sweep] attach commands:"
if [[ " ${TO_WATCH[*]} " == *" ${RUN_EXP2} "* ]]; then
  echo "  screen -r ${SESSION_EXP2}"
fi
if [[ " ${TO_WATCH[*]} " == *" ${RUN_EXP3} "* ]]; then
  echo "  screen -r ${SESSION_EXP3}"
fi
echo
echo "[Sweep] finalize log: ${FINAL_LOG_PATH}"
echo "[Sweep] nohup log: ${ROOT_DIR}/runs/lr_critic_sweep_finalize_nohup.out"
