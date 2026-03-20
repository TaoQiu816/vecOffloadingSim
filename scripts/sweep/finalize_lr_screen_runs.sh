#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir1> [run_dir2 ...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_NAME="${GIT_REMOTE_NAME:-origin}"
BRANCH_NAME="${GIT_BRANCH_NAME:-$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)}"
COMMIT_MESSAGE="${GIT_COMMIT_MESSAGE:-Add LR critic sweep runs from A snapshot}"
SHUTDOWN_DELAY_SECONDS="${SHUTDOWN_DELAY_SECONDS:-120}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-shutdown -h now}"
FINAL_LOG="${FINAL_LOG_PATH:-${ROOT_DIR}/runs/lr_screen_finalize_$(date '+%Y%m%d_%H%M%S').log}"

mkdir -p "$(dirname "${FINAL_LOG}")"
touch "${FINAL_LOG}"

log() {
  echo "$1" | tee -a "${FINAL_LOG}"
}

cd "${ROOT_DIR}"
log "[Finalize] root=${ROOT_DIR}"
log "[Finalize] branch=${BRANCH_NAME} remote=${REMOTE_NAME}"
log "[Finalize] waiting for runs: $*"

RUN_DIRS=("$@")

while true; do
  all_done=1
  for run_dir in "${RUN_DIRS[@]}"; do
    status_dir="${run_dir}/launcher_status"
    if [[ -f "${status_dir}/failed" ]]; then
      code="$(cat "${status_dir}/failed" 2>/dev/null || echo 1)"
      log "[Finalize] detected failure: ${run_dir} exit=${code}"
      exit 1
    fi
    if [[ ! -f "${status_dir}/success" ]]; then
      all_done=0
    fi
  done
  if [[ ${all_done} -eq 1 ]]; then
    break
  fi
  sleep 30
done

log "[Finalize] all runs finished successfully"

SCRIPT_PATHS=(
  "scripts/sweep/run_lr_screen_job.sh"
  "scripts/sweep/finalize_lr_screen_runs.sh"
  "scripts/sweep/launch_lr_critic_from_A_screens.sh"
)

GIT_ADD_ARGS=()
for run_dir in "${RUN_DIRS[@]}"; do
  rel_run_dir="${run_dir#${ROOT_DIR}/}"
  GIT_ADD_ARGS+=("${rel_run_dir}")
done
for path in "${SCRIPT_PATHS[@]}"; do
  GIT_ADD_ARGS+=("${path}")
done

git add -A -- "${GIT_ADD_ARGS[@]}"

if git diff --cached --quiet -- "${GIT_ADD_ARGS[@]}"; then
  log "[Finalize] nothing staged in run/script scope, skip commit/push"
else
  git commit -m "${COMMIT_MESSAGE}" | tee -a "${FINAL_LOG}"
  git push "${REMOTE_NAME}" "${BRANCH_NAME}" | tee -a "${FINAL_LOG}"
  log "[Finalize] git push done"
fi

log "[Finalize] sleeping ${SHUTDOWN_DELAY_SECONDS}s before shutdown"
sleep "${SHUTDOWN_DELAY_SECONDS}"
log "[Finalize] executing shutdown: ${SHUTDOWN_CMD}"
eval "${SHUTDOWN_CMD}"
