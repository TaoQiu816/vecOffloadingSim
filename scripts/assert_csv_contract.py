#!/usr/bin/env python3
"""
CSV contract assertion for training/baseline outputs.

Usage:
  python scripts/assert_csv_contract.py --run-dir runs/your_run
"""

import argparse
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import REQUIRED_COMPARE_COLUMNS


def _check_columns(csv_path: str, required):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing csv: {csv_path}")
    df = pd.read_csv(csv_path, nrows=1)
    cols = set(df.columns)
    miss = [c for c in required if c not in cols]
    if miss:
        raise AssertionError(f"{csv_path} missing columns: {miss}")
    print(f"[OK] {csv_path} contains required columns ({len(required)}).")


def main():
    ap = argparse.ArgumentParser(description="Assert training/baseline csv schema contract.")
    ap.add_argument("--run-dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    train_csv = os.path.join(run_dir, "logs", "training_stats.csv")
    baseline_csv = os.path.join(run_dir, "logs", "baseline_stats.csv")

    _check_columns(train_csv, REQUIRED_COMPARE_COLUMNS)
    _check_columns(baseline_csv, REQUIRED_COMPARE_COLUMNS)
    print("[PASS] CSV contract satisfied.")


if __name__ == "__main__":
    main()
