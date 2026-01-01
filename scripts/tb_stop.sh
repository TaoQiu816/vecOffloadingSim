#!/usr/bin/env bash
set -euo pipefail

# Purpose: stop TensorBoard launched via tb_autodl.sh (uses pid in logs/tb).
# Usage: bash scripts/tb_stop.sh [--run_dir PATH]

RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

_abs_path() {
  python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

_latest_run_dir() {
  local latest
  latest=$(ls -dt runs/* 2>/dev/null | head -n 1 || true)
  [[ -n "${latest}" ]] && echo "${latest}" || return 1
}

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR=$(_latest_run_dir) || { echo "[TB] 未找到 runs/*；请用 --run_dir 指定" >&2; exit 1; }
fi
RUN_DIR=$(_abs_path "${RUN_DIR}")
PID_FILE="${RUN_DIR}/logs/tb/tensorboard.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "[TB] pid 文件不存在: ${PID_FILE}" >&2
  exit 0
fi
PID=$(cat "${PID_FILE}")
if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "[TB] stopped pid=${PID}"
else
  rm -f "${PID_FILE}"
  echo "[TB] not running"
fi
