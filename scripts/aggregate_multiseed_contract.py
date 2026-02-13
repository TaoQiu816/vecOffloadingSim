#!/usr/bin/env python3
"""
Aggregate multi-seed train/baseline CSVs under a run root.

Expected layout:
  <runs_root>/<experiment>/seed_<seed>/logs/training_stats.csv
  <runs_root>/<experiment>/seed_<seed>/logs/baseline_stats.csv
"""

import argparse
import glob
import os
import sys
from typing import List

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import REQUIRED_COMPARE_COLUMNS


def _tail_mean(df: pd.DataFrame, cols: List[str], window: int) -> dict:
    if df.empty:
        return {c: 0.0 for c in cols}
    tail = df.tail(min(window, len(df)))
    out = {}
    for c in cols:
        out[c] = float(tail[c].mean()) if c in tail.columns else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description="Aggregate multi-seed CSVs (same schema contract).")
    ap.add_argument("--runs-root", type=str, required=True, help="e.g. runs/exp_contract")
    ap.add_argument("--window", type=int, default=100, help="tail window for mean aggregation")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(runs_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    seed_dirs = sorted(glob.glob(os.path.join(runs_root, "seed_*")))
    if not seed_dirs:
        raise FileNotFoundError(f"no seed dirs found under {runs_root}")

    train_rows = []
    baseline_rows = []
    for sdir in seed_dirs:
        seed_name = os.path.basename(sdir)
        seed = seed_name.replace("seed_", "")
        train_csv = os.path.join(sdir, "logs", "training_stats.csv")
        baseline_csv = os.path.join(sdir, "logs", "baseline_stats.csv")
        if os.path.exists(train_csv):
            tdf = pd.read_csv(train_csv)
            row = {"seed": seed}
            row.update(_tail_mean(tdf, REQUIRED_COMPARE_COLUMNS, args.window))
            train_rows.append(row)
        if os.path.exists(baseline_csv):
            bdf = pd.read_csv(baseline_csv)
            for policy, g in bdf.groupby("policy"):
                row = {"seed": seed, "policy": policy}
                row.update(_tail_mean(g, REQUIRED_COMPARE_COLUMNS, args.window))
                baseline_rows.append(row)

    train_out = os.path.join(out_dir, "train_tail_summary.csv")
    baseline_out = os.path.join(out_dir, "baseline_tail_summary.csv")
    pd.DataFrame(train_rows).to_csv(train_out, index=False)
    pd.DataFrame(baseline_rows).to_csv(baseline_out, index=False)
    print(f"[OK] wrote: {train_out}")
    print(f"[OK] wrote: {baseline_out}")


if __name__ == "__main__":
    main()
