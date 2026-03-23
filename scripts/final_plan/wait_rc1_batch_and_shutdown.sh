#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir1> [run_dir2 ...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FINAL_LOG="${FINAL_LOG_PATH:-${ROOT_DIR}/runs/rc1_batch_wait_$(date '+%Y%m%d_%H%M%S').log}"
SHUTDOWN_DELAY_SECONDS="${SHUTDOWN_DELAY_SECONDS:-120}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-shutdown -h now}"
DO_SHUTDOWN="${DO_SHUTDOWN:-1}"

mkdir -p "$(dirname "${FINAL_LOG}")"
touch "${FINAL_LOG}"

log() {
  echo "$1" | tee -a "${FINAL_LOG}"
}

RUN_DIRS=("$@")
log "[Waiter] watching: ${RUN_DIRS[*]}"

while true; do
  all_done=1
  for run_dir in "${RUN_DIRS[@]}"; do
    status_dir="${run_dir}/launcher_status"
    if [[ -f "${status_dir}/failed" ]]; then
      code="$(cat "${status_dir}/failed" 2>/dev/null || echo 1)"
      log "[Waiter] detected failure: ${run_dir} exit=${code}"
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

log "[Waiter] all runs finished successfully"
if [[ "${DO_SHUTDOWN}" != "1" ]]; then
  log "[Waiter] DO_SHUTDOWN=0, exit without shutdown"
  exit 0
fi

log "[Waiter] sleeping ${SHUTDOWN_DELAY_SECONDS}s before shutdown"
sleep "${SHUTDOWN_DELAY_SECONDS}"
log "[Waiter] executing shutdown: ${SHUTDOWN_CMD}"
eval "${SHUTDOWN_CMD}"
