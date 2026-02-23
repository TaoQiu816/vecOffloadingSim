#!/usr/bin/env python3
import argparse
import json
import os
import statistics as stats


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _window(vals, n):
    if not vals:
        return []
    n = max(1, min(n, len(vals)))
    return vals[-n:]


def _mean(vals):
    return float(stats.fmean(vals)) if vals else float("nan")


def _min(vals):
    return float(min(vals)) if vals else float("nan")


def _max(vals):
    return float(max(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_jsonl")
    ap.add_argument("--head-window", type=int, default=20)
    ap.add_argument("--tail-window", type=int, default=20)
    args = ap.parse_args()

    rows = _read_jsonl(args.metrics_jsonl)
    if not rows:
        raise SystemExit(f"No rows found in {args.metrics_jsonl}")

    # Prefer exact aliases if present
    reward_key = "episode_reward" if "episode_reward" in rows[0] else "reward_total"
    illegal_key = "illegal_action_ratio" if "illegal_action_ratio" in rows[0] else "illegal_action_rate"
    value_key = "value_loss"
    ent_key = "policy_entropy"

    def collect(key):
        out = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    rewards = collect(reward_key)
    illegal = collect(illegal_key)
    value = collect(value_key)
    entropy = collect(ent_key)

    head_n = max(1, min(args.head_window, len(rows)))
    tail_n = max(1, min(args.tail_window, len(rows)))
    rows_head = rows[:head_n]
    rows_tail = rows[-tail_n:]

    def collect_rows(rs, key):
        vals = []
        for r in rs:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    summary = {
        "episodes": len(rows),
        "reward_key": reward_key,
        "illegal_key": illegal_key,
        "value_loss": {
            "head_mean": _mean(collect_rows(rows_head, value_key)),
            "head_min": _min(collect_rows(rows_head, value_key)),
            "tail_mean": _mean(collect_rows(rows_tail, value_key)),
            "tail_min": _min(collect_rows(rows_tail, value_key)),
            "tail_max": _max(collect_rows(rows_tail, value_key)),
        },
        "episode_reward": {
            "head_mean": _mean(collect_rows(rows_head, reward_key)),
            "tail_mean": _mean(collect_rows(rows_tail, reward_key)),
            "tail_min": _min(collect_rows(rows_tail, reward_key)),
            "tail_max": _max(collect_rows(rows_tail, reward_key)),
            "global_max": _max(rewards),
        },
        "illegal_action_ratio": {
            "head_mean": _mean(collect_rows(rows_head, illegal_key)),
            "tail_mean": _mean(collect_rows(rows_tail, illegal_key)),
            "tail_min": _min(collect_rows(rows_tail, illegal_key)),
            "tail_max": _max(collect_rows(rows_tail, illegal_key)),
        },
        "policy_entropy": {
            "head_mean": _mean(collect_rows(rows_head, ent_key)),
            "tail_mean": _mean(collect_rows(rows_tail, ent_key)),
            "tail_min": _min(collect_rows(rows_tail, ent_key)),
            "tail_max": _max(collect_rows(rows_tail, ent_key)),
        },
        "tail_episodes": [int(r.get("episode", 0)) for r in rows_tail],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
