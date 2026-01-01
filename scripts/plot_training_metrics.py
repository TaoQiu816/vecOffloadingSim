#!/usr/bin/env python3
"""Purpose: plot training metrics from a run directory.
Inputs: --run_dir (repeatable), expects run_dir/logs/metrics.csv.
Outputs: run_dir/plots/*.png with stable filenames for comparison.
Example: python scripts/plot_training_metrics.py --run_dir runs/my_run
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        ep = row.get("episode") or row.get("ep")
        ep_val = _to_float(ep)
        if ep_val is None:
            ep_val = idx + 1
        val = _to_float(row.get(key))
        if val is None:
            continue
        xs.append(ep_val)
        ys.append(val)
    return xs, ys


def _rolling_mean(values, window=20):
    if not values:
        return []
    if len(values) < window:
        return values
    arr = np.array(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(arr, kernel, mode="valid")
    pad = [arr[0]] * (window - 1)
    return pad + smoothed.tolist()


def _plot_lines(out_dir, title, series, filename, ylabel=None, window=20):
    plt.figure()
    has_any = False
    for label, (x, y) in series.items():
        if not x or not y:
            continue
        has_any = True
        smooth = _rolling_mean(y, window=window)
        plt.plot(x, y, linewidth=1.0, alpha=0.25, label=f"{label}_raw")
        plt.plot(x, smooth, linewidth=1.8, label=label)
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


def _metrics_path(run_dir: Path) -> Path:
    primary = run_dir / "logs" / "metrics.csv"
    if primary.exists():
        return primary
    fallback = run_dir / "metrics" / "metrics.csv"
    if fallback.exists():
        return fallback
    legacy = run_dir / "metrics" / "train_metrics.csv"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"metrics csv not found in {run_dir}/logs or {run_dir}/metrics")


def _plot_for_run(run_dir):
    run_dir = Path(run_dir)
    metrics_path = _metrics_path(run_dir)
    rows = _load_csv(metrics_path)
    if not rows:
        raise RuntimeError(f"metrics csv empty: {metrics_path}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_lines(
        plots_dir,
        "reward_mean",
        {
            "reward_mean": _series(rows, "reward_mean"),
            "reward_p95": _series(rows, "reward_p95"),
        },
        "reward_mean.png",
        ylabel="reward",
    )

    _plot_lines(
        plots_dir,
        "delta_cft_rem_mean",
        {"delta_cft_rem_mean": _series(rows, "delta_cft_rem_mean")},
        "delta_cft_rem_mean.png",
        ylabel="delta_cft_rem_mean",
    )

    _plot_lines(
        plots_dir,
        "mean_cft",
        {"mean_cft": _series(rows, "mean_cft")},
        "mean_cft.png",
        ylabel="mean_cft",
    )

    _plot_lines(
        plots_dir,
        "success_rates",
        {
            "success_rate_end": _series(rows, "success_rate_end"),
            "task_success_rate": _series(rows, "task_success_rate"),
            "subtask_success_rate": _series(rows, "subtask_success_rate"),
        },
        "success_rates.png",
        ylabel="rate",
    )

    _plot_lines(
        plots_dir,
        "deadline_miss_rate",
        {"deadline_miss_rate": _series(rows, "deadline_miss_rate")},
        "deadline_miss_rate.png",
        ylabel="rate",
    )

    _plot_lines(
        plots_dir,
        "safety_rates",
        {
            "illegal_action_rate": _series(rows, "illegal_action_rate"),
            "hard_trigger_rate": _series(rows, "hard_trigger_rate"),
        },
        "safety_rates.png",
        ylabel="rate",
    )

    _plot_lines(
        plots_dir,
        "decision_fractions",
        {
            "local": _series(rows, "decision_frac_local"),
            "rsu": _series(rows, "decision_frac_rsu"),
            "v2v": _series(rows, "decision_frac_v2v"),
        },
        "decision_fracs.png",
        ylabel="fraction",
    )

    _plot_lines(
        plots_dir,
        "ppo_diagnostics",
        {
            "entropy": _series(rows, "entropy"),
            "approx_kl": _series(rows, "approx_kl"),
            "clip_frac": _series(rows, "clip_frac"),
        },
        "ppo_diagnostics.png",
        ylabel="value",
    )

    _plot_lines(
        plots_dir,
        "losses",
        {
            "policy_loss": _series(rows, "policy_loss"),
            "value_loss": _series(rows, "value_loss"),
            "total_loss": _series(rows, "total_loss"),
        },
        "losses.png",
        ylabel="loss",
    )

    _plot_lines(
        plots_dir,
        "power_ratio_mean",
        {
            "power_ratio_mean": _series(rows, "power_ratio_mean"),
            "power_ratio_p95": _series(rows, "power_ratio_p95"),
        },
        "power_ratio_mean.png",
        ylabel="power_ratio",
    )

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
