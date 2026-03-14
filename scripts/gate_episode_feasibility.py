#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from train import evaluate_single_baseline_episode


AXES = ("workload_level", "helper_availability", "rsu_accessibility_or_load")
PILOT_POLICIES = ("Greedy", "EFT", "CP-EFT")
AXIS_BUCKETS = {
    "workload_level": ("light", "medium", "heavy"),
    "helper_availability": ("low", "medium", "high"),
    "rsu_accessibility_or_load": ("low", "medium", "high"),
}


def _safe_mean(rows, key):
    vals = [float(r.get(key, float("nan"))) for r in rows if r.get(key) is not None and np.isfinite(float(r.get(key, float("nan"))))]
    return float(np.mean(vals)) if vals else float("nan")


def _pilot_select_policy(episodes: int, seed: int):
    scored = []
    for idx, policy in enumerate(PILOT_POLICIES):
        env = VecOffloadingEnv(Cfg)
        rows = []
        for ep in range(episodes):
            ep_seed = seed + idx * 1000 + ep
            metrics = evaluate_single_baseline_episode(env, policy, episode_seed=ep_seed)
            rows.append(metrics)
        scored.append({
            "policy": policy,
            "task_success_rate": _safe_mean(rows, "task_success_rate"),
            "deadline_miss_rate": _safe_mean(rows, "deadline_miss_rate"),
            "time_limit_rate": _safe_mean(rows, "time_limit_rate"),
        })
    scored.sort(key=lambda r: (-r["task_success_rate"], r["deadline_miss_rate"], r["time_limit_rate"], r["policy"]))
    return scored[0], scored


def _bucket_summary(records, axis):
    groups = defaultdict(list)
    for row in records:
        groups[str(row.get(axis, "unknown"))].append(row)
    summary = {}
    for bucket, rows in groups.items():
        summary[bucket] = {
            "count": len(rows),
            "task_success_rate": _safe_mean(rows, "task_success_rate"),
            "deadline_miss_rate": _safe_mean(rows, "deadline_miss_rate"),
            "time_limit_rate": _safe_mean(rows, "time_limit_rate"),
        }
    return summary


def _axis_pass(summary):
    raise RuntimeError("use _axis_pass_with_name")


def _axis_pass_with_name(axis, summary):
    ordered = [summary.get(level) for level in AXIS_BUCKETS[axis]]
    if any(item is None or int(item.get("count", 0)) <= 0 for item in ordered):
        return False
    sr = [float(item["task_success_rate"]) for item in ordered]
    miss = [float(item["deadline_miss_rate"]) for item in ordered]
    tl = [float(item["time_limit_rate"]) for item in ordered]
    if min(sr) < 0.35:
        return False
    if max(sr) - min(sr) > 0.35:
        return False
    if max(miss) > 0.60 or (max(miss) - min(miss) > 0.35):
        return False
    if max(tl) > 0.35 or (max(tl) - min(tl) > 0.25):
        return False
    return True


def _write_records(records, out_csv: Path):
    if not records:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    ap = argparse.ArgumentParser(description="Episode feasibility gate grouped by exogenous factors.")
    ap.add_argument("--episodes", type=int, default=45)
    ap.add_argument("--pilot-episodes", type=int, default=6)
    ap.add_argument("--policy", type=str, default="auto", choices=("auto", "Greedy", "EFT", "CP-EFT"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/exogenous_gate_20260314/feasibility_records.csv")
    args = ap.parse_args()

    selected = {"policy": args.policy}
    pilot_rows = []
    if args.policy == "auto":
        selected, pilot_rows = _pilot_select_policy(args.pilot_episodes, args.seed)
        policy_name = selected["policy"]
    else:
        policy_name = args.policy

    env = VecOffloadingEnv(Cfg)
    records = []
    for ep in range(args.episodes):
        ep_seed = args.seed + 10000 + ep
        metrics = evaluate_single_baseline_episode(env, policy_name, episode_seed=ep_seed)
        epm = dict(getattr(env, "_last_episode_metrics", {}) or {})
        record = {
            "episode": ep + 1,
            "policy": policy_name,
            "task_success_rate": float(metrics.get("task_success_rate", 0.0)),
            "deadline_miss_rate": float(metrics.get("deadline_miss_rate", 0.0) or 0.0),
            "time_limit_rate": float(metrics.get("time_limit_rate", 0.0) or 0.0),
        }
        for axis in AXES:
            record[axis] = str(epm.get(axis, "unknown"))
        records.append(record)
        if (ep + 1) % 3 == 0 or (ep + 1) == args.episodes:
            print(
                f"progress episode={ep + 1}/{args.episodes} "
                f"task_success_rate={record['task_success_rate']:.3f} "
                f"deadline_miss_rate={record['deadline_miss_rate']:.3f} "
                f"time_limit_rate={record['time_limit_rate']:.3f}",
                flush=True,
            )

    out_path = Path(args.out)
    _write_records(records, out_path)

    summaries = {axis: _bucket_summary(records, axis) for axis in AXES}
    axis_pass = {axis: _axis_pass_with_name(axis, summary) for axis, summary in summaries.items()}
    gate_pass = all(axis_pass.values())

    print(f"selected_policy {policy_name}")
    if pilot_rows:
        for row in pilot_rows:
            print(
                "pilot "
                f"policy={row['policy']} "
                f"task_success_rate={row['task_success_rate']:.3f} "
                f"deadline_miss_rate={row['deadline_miss_rate']:.3f} "
                f"time_limit_rate={row['time_limit_rate']:.3f}"
            )
    for axis in AXES:
        print(f"[{axis}]")
        for bucket in AXIS_BUCKETS[axis]:
            item = summaries[axis].get(bucket, {})
            count = int(item.get("count", 0))
            sr = float(item.get("task_success_rate", float("nan")))
            miss = float(item.get("deadline_miss_rate", float("nan")))
            tl = float(item.get("time_limit_rate", float("nan")))
            print(
                f"bucket={bucket} count={count} "
                f"task_success_rate={sr:.3f} deadline_miss_rate={miss:.3f} time_limit_rate={tl:.3f}"
            )
        print(f"axis_pass {int(axis_pass[axis])}")
    print(f"gate2_pass {int(gate_pass)}")
    print(f"csv={out_path}")


if __name__ == "__main__":
    main()
