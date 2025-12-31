#!/usr/bin/env python3
"""Plot training metrics for one or more run directories.

Usage:
  python scripts/plot_training_metrics.py --run_dir runs/my_run
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(val):
    if val is None:
        return None
    try:
        if isinstance(val, str) and val.strip() == "":
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


def _load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _series(rows, key):
    xs = []
    ys = []
    for idx, row in enumerate(rows):
        ep = row.get("episode")
        ep_val = _to_float(ep)
        if ep_val is None:
            ep_val = idx + 1
        val = _to_float(row.get(key))
        if val is None:
            continue
        xs.append(ep_val)
        ys.append(val)
    return xs, ys


def _plot_series(out_dir, title, x, y, filename, ylabel=None):
    if not x or not y:
        return False
    plt.figure()
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel or title)
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_multi(out_dir, title, series_dict, filename, ylabel=None):
    has_any = False
    plt.figure()
    for label, (x, y) in series_dict.items():
        if not x or not y:
            continue
        plt.plot(x, y, linewidth=1.2, label=label)
        has_any = True
    if not has_any:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel or title)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _find_reward_key(rows):
    candidates = ["reward_abs.mean", "avg_step_reward", "total_reward"]
    for key in candidates:
        xs, ys = _series(rows, key)
        if xs and ys:
            return key
    return None


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def _plot_for_run(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics" / "train_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics csv not found: {metrics_path}")

    rows = _load_csv(metrics_path)
    if not rows:
        raise RuntimeError(f"metrics csv empty: {metrics_path}")

    plots_dir = run_dir / "plots"
    _ensure_dir(plots_dir)

    reward_key = _find_reward_key(rows)
    if reward_key:
        x, y = _series(rows, reward_key)
        _plot_series(plots_dir, f"{reward_key}", x, y, "reward_mean.png", ylabel=reward_key)

    for key, fname in [
        ("success_rate_end", "success_rate_end.png"),
        ("subtask_success_rate", "subtask_success_rate.png"),
        ("deadline_miss_rate", "deadline_miss_rate.png"),
        ("mean_cft", "mean_cft.png"),
        ("clip_hit_ratio", "clip_hit_ratio.png"),
    ]:
        x, y = _series(rows, key)
        _plot_series(plots_dir, key, x, y, fname, ylabel=key)

    frac_series = {
        "local": _series(rows, "decision_frac_local"),
        "rsu": _series(rows, "decision_frac_rsu"),
        "v2v": _series(rows, "decision_frac_v2v"),
    }
    _plot_multi(plots_dir, "decision_frac", frac_series, "decision_fracs.png", ylabel="fraction")

    ppo_series = {
        "entropy": _series(rows, "entropy"),
        "approx_kl": _series(rows, "approx_kl"),
        "actor_loss": _series(rows, "actor_loss"),
        "critic_loss": _series(rows, "critic_loss"),
        "update_loss": _series(rows, "update_loss"),
    }
    _plot_multi(plots_dir, "ppo_metrics", ppo_series, "ppo_metrics.png", ylabel="value")

    return plots_dir


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics to PNG.")
    parser.add_argument("--run_dir", action="append", required=True, help="Run directory (repeatable)")
    args = parser.parse_args()

    failures = 0
    for run_dir in args.run_dir:
        try:
            plots_dir = _plot_for_run(run_dir)
            print(f"[OK] plots saved: {plots_dir}")
        except Exception as exc:
            failures += 1
            print(f"[ERR] {run_dir}: {exc}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
