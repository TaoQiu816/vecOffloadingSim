"""
[基线评估脚本] run_baselines.py
Baseline Evaluation Runner

作用 (Purpose):
    在独立进程中评估所有baseline策略，并将结果写入指定run目录的logs/baseline_stats.csv，
    便于与train.py生成的训练曲线进行统一绘图。

使用方法 (Usage):
    python scripts/run_baselines.py --run-dir runs/run_YYYYMMDD_HHMMSS --num-episodes 20
    python scripts/run_baselines.py --run-id run_YYYYMMDD_HHMMSS --num-episodes 20
"""

import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from train import evaluate_single_baseline_episode, apply_env_overrides


def _find_latest_run(base_dir="runs"):
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        if name.startswith("run_"):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_run_dir(args):
    if args.run_dir:
        return os.path.abspath(args.run_dir)
    if args.run_id:
        return os.path.abspath(os.path.join("runs", args.run_id))
    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        return os.path.abspath(env_run_dir)
    latest = _find_latest_run()
    if latest:
        return os.path.abspath(latest)
    raise FileNotFoundError("未找到可用的run目录，请使用--run-dir或--run-id显式指定。")


def _apply_config_snapshot(snapshot_path):
    if not os.path.exists(snapshot_path):
        return False
    with open(snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    system_cfg = data.get("system_config", {})
    train_cfg = data.get("train_config", {})
    for key, val in system_cfg.items():
        if hasattr(Cfg, key):
            setattr(Cfg, key, val)
    for key, val in train_cfg.items():
        if hasattr(TC, key):
            setattr(TC, key, val)
    return True


def _ensure_reward_jsonl(logs_dir):
    reward_jsonl_path = os.environ.get("REWARD_JSONL_PATH")
    if not reward_jsonl_path:
        reward_jsonl_path = os.path.join(logs_dir, "env_reward.jsonl")
        os.environ["REWARD_JSONL_PATH"] = reward_jsonl_path
    os.makedirs(os.path.dirname(reward_jsonl_path), exist_ok=True)
    if not os.path.exists(reward_jsonl_path):
        with open(reward_jsonl_path, "w", encoding="utf-8") as f:
            f.write("{}\n")
    return reward_jsonl_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Run baseline evaluation and write baseline_stats.csv.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--episode-start", type=int, default=1)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = _resolve_run_dir(args)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 应用环境变量覆盖，再加载快照保证与训练一致，最后应用CLI覆盖
    apply_env_overrides()
    snapshot_path = os.path.join(logs_dir, "config_snapshot.json")
    _apply_config_snapshot(snapshot_path)

    if args.seed is not None:
        Cfg.SEED = int(args.seed)
    if args.max_steps is not None:
        TC.MAX_STEPS = int(args.max_steps)

    _ensure_reward_jsonl(logs_dir)

    env = VecOffloadingEnv()

    baseline_policies = ["Random", "Local-Only", "Greedy", "EFT", "Static"]
    baseline_stats_fields = [
        "episode", "policy", "reward_mean", "reward_total",
        "vehicle_sr", "task_sr", "subtask_sr", "v2v_subtask_sr",
        "ratio_local", "ratio_rsu", "ratio_v2v",
        "avg_power", "avg_queue_len", "avg_rsu_queue",
    ]
    baseline_stats_csv = os.path.join(logs_dir, "baseline_stats.csv")

    if args.append:
        header_written = os.path.exists(baseline_stats_csv) and os.path.getsize(baseline_stats_csv) > 0
        file_mode = "a"
    else:
        header_written = False
        file_mode = "w"

    summary = {policy: [] for policy in baseline_policies}
    start_ep = int(args.episode_start)
    total_episodes = int(args.num_episodes)

    with open(baseline_stats_csv, file_mode, newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=baseline_stats_fields, extrasaction="ignore")
        if not header_written:
            writer.writeheader()
        for ep_offset in range(total_episodes):
            episode = start_ep + ep_offset
            for policy_name in baseline_policies:
                metrics = evaluate_single_baseline_episode(env, policy_name)
                summary[policy_name].append(metrics)
                row = {
                    "episode": episode,
                    "policy": policy_name,
                    "reward_mean": metrics["avg_step_reward"],
                    "reward_total": metrics["total_reward"],
                    "vehicle_sr": metrics["veh_success_rate"],
                    "task_sr": metrics.get("task_success_rate", metrics["veh_success_rate"]),
                    "subtask_sr": metrics["subtask_success_rate"],
                    "v2v_subtask_sr": metrics["v2v_subtask_success_rate"],
                    "ratio_local": metrics["decision_frac_local"],
                    "ratio_rsu": metrics["decision_frac_rsu"],
                    "ratio_v2v": metrics["decision_frac_v2v"],
                    "avg_power": metrics["avg_power"],
                    "avg_queue_len": metrics["avg_queue_len"],
                    "avg_rsu_queue": metrics.get("avg_rsu_queue", 0.0),
                }
                writer.writerow(row)

    print(f"✓ Baseline stats saved: {baseline_stats_csv}")
    print("结果摘要 (均值):")
    for policy_name in baseline_policies:
        records = summary[policy_name]
        if not records:
            continue
        reward = sum(r["avg_step_reward"] for r in records) / len(records)
        task_sr = sum(r.get("task_success_rate", r["veh_success_rate"]) for r in records) / len(records)
        subtask_sr = sum(r["subtask_success_rate"] for r in records) / len(records)
        print(f"  {policy_name:<10} reward={reward:.4f} task_sr={task_sr:.4f} subtask_sr={subtask_sr:.4f}")


if __name__ == "__main__":
    main()
