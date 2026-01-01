#!/usr/bin/env bash
set -euo pipefail

# Purpose: AutoDL-friendly TensorBoard launcher with sensible defaults.
# Inputs: --run_dir (optional), --port (default 6006 auto-increment to 6100), --host (default 0.0.0.0)
# Env: TB_DAEMON=1 to run in background (nohup, pid/log in <run_dir>/logs/tb)
# Example: bash scripts/tb_autodl.sh --run_dir runs/my_run

RUN_DIR=""
PORT=6006
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --host)
      HOST="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

_latest_run_dir() {
  local latest
  latest=$(ls -dt runs/* 2>/dev/null | head -n 1 || true)
  [[ -n "${latest}" ]] && echo "${latest}" || return 1
}

_abs_path() {
  python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

_pick_logdir() {
  local rd="$1"
  if [[ -z "${rd}" ]]; then
    rd=$(_latest_run_dir) || {
      echo "[TB] 未找到 runs/*；请先训练或指定 --run_dir" >&2
      exit 1
    }
  fi
  rd=$(_abs_path "${rd}")
  local ld="${rd}/logs/tb"
  if [[ ! -d "${ld}" ]]; then
    echo "[TB] logdir 不存在: ${ld}；请确认 run_dir 正确" >&2
    exit 1
  fi
  echo "${rd}|${ld}"
}

_port_available() {
  python - "$HOST" "$1" <<'PY'
import socket, sys
host = sys.argv[1]; port = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((host, port))
    ok = True
except OSError:
    ok = False
finally:
    s.close()
sys.exit(0 if ok else 1)
PY
}

_find_port() {
  local base=$1
  for p in $(seq ${base} 6100); do
    if _port_available "$p"; then
      echo "$p"; return 0
    fi
  done
  echo "[TB] no free port in [${base},6100]" >&2
  exit 1
}

START_TS=$(date +%Y%m%d_%H%M%S)
IFS='|' read -r RUN_DIR LOGDIR <<< "$(_pick_logdir "${RUN_DIR}")"
PORT=$(_find_port ${PORT})
mkdir -p "${LOGDIR}"
PID_FILE="${LOGDIR}/tensorboard.pid"
LOG_FILE="${LOGDIR}/tensorboard.log"

if [[ "${TB_DAEMON:-0}" == "1" ]]; then
  nohup tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}" --reload_interval 10 > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
else
  tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}" --reload_interval 10 &> "${LOG_FILE}" &
  echo $! > "${PID_FILE}"
fi

echo "[TB] LOGDIR=${LOGDIR}"
echo "[TB] HOST=${HOST} PORT=${PORT}"
echo "[TB] pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
echo "[TB] url_local=http://127.0.0.1:${PORT}"
echo "[TB] url_remote=http://${HOST}:${PORT} (AutoDL: 映射/转发该端口)"
