import argparse
import os
import subprocess
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--log_dir", type=str, default="logs/ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    combos = [
        ("incremental_cost", "none"),
        ("incremental_cost", "subtask"),
        ("incremental_cost", "success"),
        ("incremental_cost", "both"),
        ("delta_cft", "none"),
        ("delta_cft", "subtask"),
        ("delta_cft", "success"),
        ("delta_cft", "both"),
    ]

    os.makedirs(args.log_dir, exist_ok=True)

    for reward_mode, bonus_mode in combos:
        for seed in seeds:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"{reward_mode}__{bonus_mode}__seed{seed}__run{run_id}__{timestamp}.jsonl"
            log_path = os.path.join(args.log_dir, log_name)

            env = os.environ.copy()
            env["REWARD_MODE"] = reward_mode
            env["BONUS_MODE"] = bonus_mode
            env["SEED"] = str(seed)
            env["MAX_EPISODES"] = str(args.episodes)
            env["MAX_STEPS"] = str(args.steps)
            env["EVAL_INTERVAL"] = str(args.episodes + 1)
            env["SAVE_INTERVAL"] = str(args.episodes + 1)
            env["DISABLE_BASELINE_EVAL"] = "1"
            env["REWARD_JSONL_PATH"] = log_path
            env["RUN_ID"] = run_id

            print(f"[Run] {reward_mode} | {bonus_mode} | seed={seed} -> {log_path}")
            subprocess.run([sys.executable, "train.py"], check=True, env=env)


if __name__ == "__main__":
    main()
