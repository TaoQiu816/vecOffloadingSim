#!/usr/bin/env bash
set -euo pipefail

# Purpose: manage TensorBoard for the latest run or a specified run dir.
# Inputs: start|stop|status|restart with optional --run_dir/--logdir/--port/--host.
# Outputs: tensorboard.log and tensorboard.pid under <run_dir>/logs/tb.
# Example: bash scripts/tensorboard_ctl.sh start --run_dir runs/my_run --port 6006

CMD="${1:-}"
if [[ -z "${CMD}" ]]; then
  echo "Usage: $0 {start|stop|status|restart} [--run_dir PATH] [--logdir PATH] [--port N] [--host HOST]" >&2
  exit 1
fi
shift || true

RUN_DIR=""
LOGDIR=""
PORT="6006"
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --logdir)
      LOGDIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

_latest_run_dir() {
  local latest
  latest=$(ls -dt runs/* 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  echo "${latest}"
}

_resolve_paths() {
  if [[ -n "${LOGDIR}" ]]; then
    if [[ -z "${RUN_DIR}" ]]; then
      if [[ "${LOGDIR}" == */logs/tb ]]; then
        RUN_DIR="${LOGDIR%/logs/tb}"
      else
        RUN_DIR="$(dirname "${LOGDIR}")"
      fi
    fi
  else
    if [[ -z "${RUN_DIR}" ]]; then
      RUN_DIR="$(_latest_run_dir)" || {
        echo "[TB] no runs/* found; please pass --run_dir or --logdir" >&2
        exit 1
      }
    fi
    LOGDIR="${RUN_DIR}/logs/tb"
  fi

  RUN_DIR="$(python - "${RUN_DIR}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
  LOGDIR="$(python - "${LOGDIR}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
}

_port_available() {
  local host="$1"
  local port="$2"
  python - <<PY
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind((host, port))
    except OSError:
        sys.exit(1)
PY
}

_pick_port() {
  local base_port="$1"
  local host="$2"
  local cand
  for i in $(seq 0 19); do
    cand=$((base_port + i))
    if _port_available "${host}" "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

_pid_file() {
  echo "${RUN_DIR}/logs/tb/tensorboard.pid"
}

_log_file() {
  echo "${RUN_DIR}/logs/tb/tensorboard.log"
}

_cmdline_for_pid() {
  local pid="$1"
  ps -p "${pid}" -o command= 2>/dev/null || true
}

_print_status() {
  local pid_file
  pid_file="$(_pid_file)"
  if [[ ! -f "${pid_file}" ]]; then
    echo "[TB] not running"
    return 0
  fi
  local pid
  pid=$(cat "${pid_file}")
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "[TB] not running"
    return 0
  fi
  local cmd logdir port
  cmd="$(_cmdline_for_pid "${pid}")"
  logdir=$(echo "${cmd}" | sed -n 's/.*--logdir[= ]\([^ ]*\).*/\1/p')
  port=$(echo "${cmd}" | sed -n 's/.*--port[= ]\([0-9]*\).*/\1/p')
  logdir=${logdir:-${LOGDIR}}
  port=${port:-${PORT}}
  echo "[TB] pid=${pid} port=${port} logdir=${logdir}"
  echo "[TB] url_local=http://127.0.0.1:${port}"
  echo "[TB] url_remote=http://0.0.0.0:${port}"
}

_resolve_paths
mkdir -p "${RUN_DIR}/logs/tb"

case "${CMD}" in
  start)
    if ! command -v tensorboard >/dev/null 2>&1; then
      echo "[TB] tensorboard not found in PATH" >&2
      exit 1
    fi
    local_pid_file="$(_pid_file)"
    if [[ -f "${local_pid_file}" ]]; then
      pid=$(cat "${local_pid_file}")
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        _print_status
        exit 0
      fi
    fi
    chosen_port="$(_pick_port "${PORT}" "${HOST}")" || {
      echo "[TB] no free port found starting at ${PORT}" >&2
      exit 1
    }
    log_file="$(_log_file)"
    nohup tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${chosen_port}" > "${log_file}" 2>&1 &
    echo $! > "${local_pid_file}"
    echo "[TB] pid=$! port=${chosen_port} logdir=${LOGDIR}"
    echo "[TB] url_local=http://127.0.0.1:${chosen_port}"
    echo "[TB] url_remote=http://0.0.0.0:${chosen_port}"
    ;;
  stop)
    pid_file="$(_pid_file)"
    if [[ ! -f "${pid_file}" ]]; then
      echo "[TB] not running"
      exit 0
    fi
    pid=$(cat "${pid_file}")
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      rm -f "${pid_file}"
      echo "[TB] stopped pid=${pid}"
    else
      rm -f "${pid_file}"
      echo "[TB] not running"
    fi
    ;;
  status)
    _print_status
    ;;
  restart)
    "$0" stop --run_dir "${RUN_DIR}" --logdir "${LOGDIR}" --port "${PORT}" --host "${HOST}"
    "$0" start --run_dir "${RUN_DIR}" --logdir "${LOGDIR}" --port "${PORT}" --host "${HOST}"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|restart} [--run_dir PATH] [--logdir PATH] [--port N] [--host HOST]" >&2
    exit 1
    ;;
esac
