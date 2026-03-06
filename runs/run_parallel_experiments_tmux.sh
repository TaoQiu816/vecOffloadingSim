#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_NAME="${1:-vec-parallel-exp}"
cd "${ROOT_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

GPU0="${GPU0:-0}"
GPU1="${GPU1:-${GPU0}}"
GPU2="${GPU2:-${GPU0}}"
GPU3="${GPU3:-${GPU0}}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-1000}"

DONE_DIR="${ROOT_DIR}/runs/_parallel_status/${SESSION_NAME}"
mkdir -p "${DONE_DIR}"

run_window() {
  local window_name="$1"
  local gpu_id="$2"
  local run_id="$3"
  local run_dir="$4"
  local extra_env="$5"

  mkdir -p "${run_dir}"

  local cmd
  cmd=$(cat <<EOF
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES='${gpu_id}'
export DEVICE_NAME='${DEVICE_NAME}'
export SEED='${SEED}'
export MAX_EPISODES='${EPISODES}'
export RUN_ID='${run_id}'
export RUN_DIR='${run_dir}'
${extra_env}
python -u train.py --exact-run-dir --run-dir '${run_dir}' 2>&1 | tee '${run_dir}/console.log'
python -u scripts/postprocess_run.py --run-dir '${run_dir}' --overwrite 2>&1 | tee -a '${run_dir}/console.log'
PLOT_CSV='${run_dir}/metrics/train_metrics_full.csv'
if [[ ! -f "\${PLOT_CSV}" ]]; then
  PLOT_CSV='${run_dir}/metrics/train_metrics.csv'
fi
python -u scripts/plot_results.py --log-file "\${PLOT_CSV}" --output-dir '${run_dir}/plots' 2>&1 | tee -a '${run_dir}/console.log'
touch '${DONE_DIR}/${window_name}.done'
echo '[DONE] ${window_name}'
exec "\$SHELL"
EOF
)

  tmux new-window -t "${SESSION_NAME}" -n "${window_name}" "${cmd}"
}

sync_cmd=$(cat <<EOF
cd '${ROOT_DIR}'
echo '[SYNC] waiting for 4 experiments to finish...'
while true; do
  done_count=\$(find '${DONE_DIR}' -name '*.done' | wc -l | tr -d ' ')
  date
  echo "[SYNC] completed: \${done_count}/4"
  if [[ "\${done_count}" == "4" ]]; then
    break
  fi
  sleep 30
done
git add \
  configs/config.py \
  configs/train_config.py \
  train.py \
  envs/vec_offloading_env.py \
  runs/run_parallel_experiments_tmux.sh \
  runs/e0_head_control \
  runs/e1_schemeA \
  runs/e2_schemeA_balanced \
  runs/e3_schemeA_balanced_critic0
git commit -m 'experiments: parallel single-seed schemeA ablations' || true
git push || true
echo '[SYNC] git sync complete (or skipped if nothing to commit / push failed).'
echo '[SYNC] scheduling shutdown in 2 minutes; sudo may prompt for password.'
sudo shutdown -h +2
exec "\$SHELL"
EOF
)

tmux new-session -d -s "${SESSION_NAME}" -n "e0" "bash"
tmux send-keys -t "${SESSION_NAME}:e0" "clear" C-m
tmux send-keys -t "${SESSION_NAME}:e0" "echo '[INFO] launching e0 control'" C-m
tmux send-keys -t "${SESSION_NAME}:e0" "cd '${ROOT_DIR}'" C-m
tmux send-keys -t "${SESSION_NAME}:e0" "export CUDA_VISIBLE_DEVICES='${GPU0}' DEVICE_NAME='${DEVICE_NAME}' SEED='${SEED}' MAX_EPISODES='${EPISODES}' RUN_ID='e0_head_control' RUN_DIR='runs/e0_head_control'" C-m
tmux send-keys -t "${SESSION_NAME}:e0" "python -u train.py --exact-run-dir --run-dir 'runs/e0_head_control' 2>&1 | tee 'runs/e0_head_control/console.log'; python -u scripts/postprocess_run.py --run-dir 'runs/e0_head_control' --overwrite 2>&1 | tee -a 'runs/e0_head_control/console.log'; PLOT_CSV='runs/e0_head_control/metrics/train_metrics_full.csv'; if [[ ! -f \"\${PLOT_CSV}\" ]]; then PLOT_CSV='runs/e0_head_control/metrics/train_metrics.csv'; fi; python -u scripts/plot_results.py --log-file \"\${PLOT_CSV}\" --output-dir 'runs/e0_head_control/plots' 2>&1 | tee -a 'runs/e0_head_control/console.log'; touch '${DONE_DIR}/e0.done'; echo '[DONE] e0'; exec \"\$SHELL\"" C-m

run_window "e1" "${GPU1}" "e1_schemeA" "runs/e1_schemeA" \
"export UNIFIED_MAIN_REWARD_MODE='time_margin_term_illegal_interf'
export W_PROGRESS='0.0'"

run_window "e2" "${GPU2}" "e2_schemeA_balanced" "runs/e2_schemeA_balanced" \
"export UNIFIED_MAIN_REWARD_MODE='time_margin_term_illegal_interf'
export W_PROGRESS='0.0'
export RSU_RANGE='320'
export RSU_QUEUE_CYCLES_LIMIT='80000000000'"

run_window "e3" "${GPU3}" "e3_schemeA_balanced_critic0" "runs/e3_schemeA_balanced_critic0" \
"export UNIFIED_MAIN_REWARD_MODE='time_margin_term_illegal_interf'
export W_PROGRESS='0.0'
export RSU_RANGE='320'
export RSU_QUEUE_CYCLES_LIMIT='80000000000'
export CRITIC_INACTIVE_SAMPLE_WEIGHT='0.0'"

tmux new-window -t "${SESSION_NAME}" -n "sync" "${sync_cmd}"

echo "tmux session created: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "GPU mapping: e0=${GPU0}, e1=${GPU1}, e2=${GPU2}, e3=${GPU3}"
tmux attach -t "${SESSION_NAME}"
