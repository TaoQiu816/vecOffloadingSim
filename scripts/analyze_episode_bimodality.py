#!/usr/bin/env python3
"""
Summarize episode bimodality (short vs normal).
Inputs: --run_dir (expects logs/metrics.csv or logs/metrics.jsonl).
Outputs: one-page stats with short_episode_ratio and key p50/p90 by group.
"""
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from statistics import median


def _load_metrics(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def _to_float(val):
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None
        return float(val)
    except Exception:
        return None


def _pstats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    vals = sorted(vals)
    n = len(vals)
    p50 = median(vals)
    p90 = vals[int(0.9 * (n - 1))]
    return p50, p90


def _summarize(rows):
    steps = [_to_float(r.get("steps")) for r in rows]
    steps = [s for s in steps if s is not None]
    if not steps:
        raise RuntimeError("no steps found in metrics")
    max_steps = max(steps)
    rows_short = []
    rows_norm = []
    for r in rows:
        st = _to_float(r.get("steps"))
        if st is None:
            continue
        (rows_short if st < 0.5 * max_steps else rows_norm).append(r)
    def metric(group, key):
        return _pstats([_to_float(r.get(key)) for r in group])
    total_eps = len(rows_short) + len(rows_norm)
    term_flags = [bool(r.get("terminated") == "True") if isinstance(r.get("terminated"), str) else bool(r.get("terminated")) for r in rows]
    trunc_flags = [bool(r.get("truncated") == "True") if isinstance(r.get("truncated"), str) else bool(r.get("truncated")) for r in rows]
    term_ratio = (sum(term_flags) / len(term_flags)) if term_flags else 0.0
    trunc_ratio = (sum(trunc_flags) / len(trunc_flags)) if trunc_flags else 0.0
    tl_rates = [_to_float(r.get("time_limit_rate")) for r in rows]
    tl_rates = [v for v in tl_rates if v is not None]
    time_limit_rate = float(np.mean(tl_rates)) if tl_rates else 0.0
    result = {
        "total_episodes": total_eps,
        "max_steps": max_steps,
        "short_episode_ratio": (len(rows_short) / total_eps) if total_eps > 0 else 0.0,
        "terminated_ratio": term_ratio,
        "truncated_ratio": trunc_ratio,
        "time_limit_rate": time_limit_rate,
        "short": {
            "reward_mean": metric(rows_short, "reward_mean"),
            "task_success_rate": metric(rows_short, "task_success_rate"),
            "deadline_miss_rate": metric(rows_short, "deadline_miss_rate"),
            "decision_frac_rsu": metric(rows_short, "decision_frac_rsu") or metric(rows_short, "decision_rsu_frac"),
            "delta_cft_rem_mean": metric(rows_short, "delta_cft_rem_mean"),
        },
        "normal": {
            "reward_mean": metric(rows_norm, "reward_mean"),
            "task_success_rate": metric(rows_norm, "task_success_rate"),
            "deadline_miss_rate": metric(rows_norm, "deadline_miss_rate"),
            "decision_frac_rsu": metric(rows_norm, "decision_frac_rsu") or metric(rows_norm, "decision_rsu_frac"),
            "delta_cft_rem_mean": metric(rows_norm, "delta_cft_rem_mean"),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze episode bimodality (short vs normal).")
    parser.add_argument("--run_dir", required=True, help="Run directory containing logs/metrics.{csv,jsonl}")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = None
    for cand in [run_dir / "logs" / "metrics.csv", run_dir / "logs" / "metrics.jsonl"]:
        if cand.exists():
            metrics_path = cand
            break
    if metrics_path is None:
        raise FileNotFoundError(f"metrics not found under {run_dir}/logs/")

    rows = list(_load_metrics(metrics_path))
    summary = _summarize(rows)

    print(f"[Bimodality] max_steps={summary['max_steps']:.0f} episodes={summary['total_episodes']}")
    print(f"[Bimodality] short_episode_ratio={summary['short_episode_ratio']:.3f}")
    print(f"[Bimodality] terminated_ratio={summary['terminated_ratio']:.3f} truncated_ratio={summary['truncated_ratio']:.3f} time_limit_rate={summary['time_limit_rate']:.3f}")
    for label in ("short", "normal"):
        stats = summary[label]
        def fmt(name, val):
            if val is None or val[0] is None:
                return f"{name}: n/a"
            p50, p90 = val
            return f"{name}: p50={p50:.3f}, p90={p90:.3f}"
        parts = [
            fmt("reward_mean", stats["reward_mean"]),
            fmt("task_success_rate", stats["task_success_rate"]),
            fmt("deadline_miss_rate", stats["deadline_miss_rate"]),
            fmt("decision_frac_rsu", stats["decision_frac_rsu"]),
            fmt("delta_cft_rem_mean", stats["delta_cft_rem_mean"]),
        ]
        print(f"[{label.upper()}] " + " | ".join(parts))


if __name__ == "__main__":
    main()
