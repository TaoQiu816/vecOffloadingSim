"""
Merge baseline evaluation CSV parts into a single baseline_stats.csv.

Typical usage (parallel by policy):
  python scripts/merge_baseline_parts.py --run-dir runs/<run_id> --parts "runs/<run_id>/logs/baseline_parts/*.csv"

This script writes to:
  - <run_dir>/logs/baseline_stats.csv
  - <run_dir>/metrics/baseline_stats.csv (if metrics/ exists)
"""

import argparse
import glob
import os
import shutil

import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description="Merge baseline CSV parts into baseline_stats.csv")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--parts", type=str, required=True, help="Glob pattern for CSV parts, e.g. 'logs/baseline_parts/*.csv'")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV path. Default: <run_dir>/logs/baseline_stats.csv")
    return p.parse_args()


def main():
    args = _parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_csv = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "logs", "baseline_stats.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Expand glob relative to run_dir if the user passes a relative pattern.
    pattern = args.parts
    if not os.path.isabs(pattern):
        pattern = os.path.join(run_dir, pattern)

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV parts found for pattern: {pattern}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("All CSV parts are empty.")

    df_all = pd.concat(frames, ignore_index=True)

    # Normalize a few column names if needed (keep existing ones if present)
    if "task_sr" not in df_all.columns and "task_success_rate" in df_all.columns:
        df_all["task_sr"] = df_all["task_success_rate"]
    if "subtask_sr" not in df_all.columns and "subtask_success_rate" in df_all.columns:
        df_all["subtask_sr"] = df_all["subtask_success_rate"]

    # Drop exact duplicates (re-runs / retries), keep the first occurrence.
    df_all = df_all.drop_duplicates()

    # Make ordering stable for plotting.
    if "episode" in df_all.columns and "policy" in df_all.columns:
        df_all = df_all.sort_values(["episode", "policy"]).reset_index(drop=True)

    df_all.to_csv(out_csv, index=False)
    print(f"✓ Merged baseline stats: {out_csv} (rows={len(df_all)})")

    metrics_dir = os.path.join(run_dir, "metrics")
    if os.path.isdir(metrics_dir):
        try:
            dst = os.path.join(metrics_dir, "baseline_stats.csv")
            shutil.copyfile(out_csv, dst)
            print(f"✓ Mirrored: {dst}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

