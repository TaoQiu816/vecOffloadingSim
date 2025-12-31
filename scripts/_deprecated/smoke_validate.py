import json
import os
import subprocess
import sys
from datetime import datetime

from scripts import summarize_ablation as sa


def run_one(log_dir, reward_mode, bonus_mode, seed, episodes=3, steps=50, run_id=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{reward_mode}__{bonus_mode}__seed{seed}__run{run_id}__{timestamp}.jsonl"
    log_path = os.path.join(log_dir, log_name)

    env = os.environ.copy()
    env["REWARD_MODE"] = reward_mode
    env["BONUS_MODE"] = bonus_mode
    env["SEED"] = str(seed)
    env["MAX_EPISODES"] = str(episodes)
    env["MAX_STEPS"] = str(steps)
    env["EVAL_INTERVAL"] = str(episodes + 1)
    env["SAVE_INTERVAL"] = str(episodes + 1)
    env["DISABLE_BASELINE_EVAL"] = "1"
    env["REWARD_JSONL_PATH"] = log_path
    env["RUN_ID"] = run_id or ""

    subprocess.run([sys.executable, "train.py"], check=True, env=env)
    return log_path


def main():
    log_dir = os.path.join("logs", "smoke")
    out_dir = os.path.join("results", "smoke")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = []
    files.append(run_one(log_dir, "incremental_cost", "none", seed=0, run_id=run_id))
    files.append(run_one(log_dir, "delta_cft", "none", seed=0, run_id=run_id))

    runs, excluded, missing, legacy = sa.load_runs(log_dir, run_id=run_id)
    runs = sa._dedup_latest(runs)
    summary = sa.summarize_groups(runs)
    sa.write_csv(summary, os.path.join(out_dir, "ablation_summary.csv"))
    sa.write_md(summary, os.path.join(out_dir, "ablation_summary.md"), excluded, missing, legacy)

    print("[Smoke] JSONL files:")
    for path in files:
        line_count = sum(1 for _ in open(path, "r", encoding="utf-8"))
        print(f"  - {path} ({line_count} lines)")

    if excluded:
        print("[Smoke] Excluded files:")
        for item in excluded:
            print(f"  - {item['path']}: {item['reason']}")

    # Range checks
    ok = True
    for run in runs:
        for line in run["lines"]:
            vehicle_success = line.get("vehicle_success_rate", line.get("success_rate", 0.0))
            deadline_miss = line.get("deadline_miss_rate", 0.0)
            if not (0.0 <= vehicle_success <= 1.0):
                ok = False
            if not (0.0 <= deadline_miss <= 1.0):
                ok = False

    print("[Smoke] Result:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
