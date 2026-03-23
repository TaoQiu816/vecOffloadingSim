#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [commit_message]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="$1"
COMMIT_MESSAGE="${2:-Update RC1 run: $(basename "$RUN_DIR")}"
REMOTE_NAME="${GIT_REMOTE_NAME:-origin}"
BRANCH_NAME="${GIT_BRANCH_NAME:-$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)}"
LOCK_DIR="${ROOT_DIR}/.git/rc1_sync.lock.d"
RUN_REL="${RUN_DIR#${ROOT_DIR}/}"
KEEP_CHECKPOINTS_CSV="${KEEP_CHECKPOINTS_CSV:-ep0200,ep0500,ep0800,ep1000}"
SYNC_PLOTS="${SYNC_PLOTS:-1}"
SYNC_JSONL="${SYNC_JSONL:-0}"

acquire_lock() {
  while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    sleep 2
  done
}

release_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

trap release_lock EXIT
acquire_lock

cd "${ROOT_DIR}"

stage_if_exists() {
  local rel="$1"
  [[ -e "${rel}" ]] || return 0
  git add -A -- "${rel}"
}

stage_glob_if_exists() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( ${pattern} )
  shopt -u nullglob
  local rel
  for rel in "${matches[@]}"; do
    git add -A -- "${rel}"
  done
}

stage_if_exists "scripts/final_plan"
stage_if_exists "diagnostics/final_plan"

# Core run artifacts needed for final quantitative evaluation / reproduction.
stage_if_exists "${RUN_REL}/run_meta.json"
stage_if_exists "${RUN_REL}/config.json"
stage_if_exists "${RUN_REL}/config_dump.json"
stage_if_exists "${RUN_REL}/episode_log.csv"
stage_if_exists "${RUN_REL}/baseline_run_meta.json"

stage_if_exists "${RUN_REL}/models/best_model.pth"
stage_if_exists "${RUN_REL}/models/last_model.pth"

IFS=',' read -r -a KEEP_CKPTS <<< "${KEEP_CHECKPOINTS_CSV}"
for ckpt_name in "${KEEP_CKPTS[@]}"; do
  ckpt_name="$(echo "${ckpt_name}" | xargs)"
  [[ -n "${ckpt_name}" ]] || continue
  stage_if_exists "${RUN_REL}/models/checkpoints/${ckpt_name}.pth"
done

stage_if_exists "${RUN_REL}/logs/config_snapshot.json"
stage_if_exists "${RUN_REL}/logs/training_stats.csv"
stage_if_exists "${RUN_REL}/logs/metrics.csv"
stage_if_exists "${RUN_REL}/logs/baseline_stats.csv"
stage_if_exists "${RUN_REL}/logs/baseline_eval_core_summary.csv"

stage_if_exists "${RUN_REL}/authoritative_eval"
stage_if_exists "${RUN_REL}/formal_eval"
stage_if_exists "${RUN_REL}/diagnostics"
stage_if_exists "${RUN_REL}/ablation_compare"
stage_if_exists "${RUN_REL}/paper_exports"

if [[ "${SYNC_PLOTS}" == "1" ]]; then
  stage_if_exists "${RUN_REL}/plots"
else
  stage_if_exists "${RUN_REL}/plots/plot_manifest.json"
fi

if [[ "${SYNC_JSONL}" == "1" ]]; then
  stage_glob_if_exists "${RUN_REL}/logs/*.jsonl"
  stage_glob_if_exists "${RUN_REL}/metrics/*.jsonl"
fi

if git diff --cached --quiet -- "scripts/final_plan" "diagnostics/final_plan" "${RUN_REL}"; then
  echo "[GitSync] nothing staged for ${RUN_REL}"
  exit 0
fi

git commit -m "${COMMIT_MESSAGE}"
git push "${REMOTE_NAME}" "${BRANCH_NAME}"
echo "[GitSync] pushed ${RUN_REL}"
