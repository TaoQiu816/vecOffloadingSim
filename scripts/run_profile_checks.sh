#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME="train_v2v_competitive_v1"
OUT_ROOT="results_dbg/profile_checks/${PROFILE_NAME}"

mkdir -p "${OUT_ROOT}"
export CFG_PROFILE="${PROFILE_NAME}"

echo "[INFO] profile=${PROFILE_NAME}"
echo "[INFO] out=${OUT_ROOT}"

python scripts/param_sanity_report.py --out_dir "${OUT_ROOT}"

python scripts/decision_dominance_audit.py \
  --episodes 50 --steps 50 --seed 0 \
  --out_dir "${OUT_ROOT}/decision_dominance" \
  --policy_rollout type_balanced_random

python scripts/final_readiness_check.py \
  --out_dir "${OUT_ROOT}/final_ready" \
  --episodes 2 --steps 50 --seed 7 --audit 1 --reward_mode delta_cft

python - <<'PY'
import json
import os
from pathlib import Path

from pathlib import Path
from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from baselines import LocalOnlyPolicy, GreedyPolicy

out_root = Path("results_dbg/profile_checks/train_v2v_competitive_v1")
baseline_dir = out_root / "baselines"
baseline_dir.mkdir(parents=True, exist_ok=True)

# 关键：GreedyPolicy 需要 env
env = VecOffloadingEnv()

policies = [
    ("local_only", LocalOnlyPolicy()),
    ("greedy", GreedyPolicy(env)),
]

episodes = 10
steps = min(200, Cfg.MAX_STEPS)

summary_rows = []

for name, policy in policies:
    jsonl_path = baseline_dir / f"{name}.jsonl"
    os.environ["REWARD_JSONL_PATH"] = str(jsonl_path)
    os.environ["MAX_EPISODES"] = str(episodes)

    env = VecOffloadingEnv()
    for ep in range(episodes):
        obs_list, _ = env.reset(seed=ep)
        policy.reset()
        for _ in range(steps):
            actions = policy.select_action(obs_list)
            obs_list, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

    last_line = jsonl_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    data = json.loads(last_line)
    summary_rows.append({
        "policy": name,
        "success_rate_end": data.get("success_rate_end"),
        "deadline_miss_rate": data.get("deadline_miss_rate"),
        "mean_cft": data.get("mean_cft"),
        "decision_frac_local": data.get("decision_frac_local"),
        "decision_frac_rsu": data.get("decision_frac_rsu"),
        "decision_frac_v2v": data.get("decision_frac_v2v"),
        "terminated": data.get("terminated"),
        "truncated": data.get("truncated"),
    })

summary_md = out_root / "baseline_summary.md"
lines = [
    "# Baseline Summary",
    "",
    "| policy | success_rate_end | deadline_miss_rate | mean_cft | frac_local | frac_rsu | frac_v2v | terminated | truncated |",
    "|---|---|---|---|---|---|---|---|---|",
]
for row in summary_rows:
    lines.append(
        f"| {row['policy']} | {row['success_rate_end']:.4f} | {row['deadline_miss_rate']:.4f} | "
        f"{row['mean_cft']:.4f} | {row['decision_frac_local']:.4f} | {row['decision_frac_rsu']:.4f} | "
        f"{row['decision_frac_v2v']:.4f} | {row['terminated']} | {row['truncated']} |"
    )

lines.append("")
lines.append("Note: rsu_only policy not available in baselines/.")

summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"[OK] wrote {summary_md}")
PY
