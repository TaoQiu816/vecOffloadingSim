#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"


def _load_last_metrics(run_dir: Path, tail_n: int = 20):
    metrics = {}
    train_csv = run_dir / "logs" / "training_stats.csv"
    metrics_csv = run_dir / "logs" / "metrics.csv"
    reward_jsonl = run_dir / "logs" / "env_reward.jsonl"

    if not train_csv.exists():
        raise FileNotFoundError(f"missing training stats: {train_csv}")
    df = pd.read_csv(train_csv)
    if df.empty:
        raise RuntimeError(f"empty training stats: {train_csv}")
    required = ("task_sr", "task_duration_mean", "task_duration_p95")
    for col in required:
        if col not in df.columns:
            raise KeyError(f"missing column '{col}' in {train_csv}")
    tail = df.tail(min(tail_n, len(df)))
    metrics["success_rate"] = float(tail["task_sr"].mean())
    metrics["makespan_mean"] = float(tail["task_duration_mean"].mean())
    metrics["makespan_p95"] = float(tail["task_duration_p95"].mean())

    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        mtail = mdf.tail(min(tail_n, len(mdf)))
        if "illegal_action_rate" in mtail.columns:
            metrics["illegal_rate"] = float(mtail["illegal_action_rate"].mean())
        else:
            metrics["illegal_rate"] = 0.0
    else:
        metrics["illegal_rate"] = 0.0

    metrics["fallback_rate"] = 0.0
    metrics["dropped_cnt_mean"] = 0.0
    if reward_jsonl.exists():
        records = []
        with reward_jsonl.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and "episode" in obj:
                    records.append(obj)
        if records:
            rdf = pd.DataFrame(records)
            rtail = rdf.tail(min(tail_n, len(rdf)))
            if "fallback_rate" in rtail.columns:
                metrics["fallback_rate"] = float(rtail["fallback_rate"].mean())
            if "candidate_dropped_cnt_mean" in rtail.columns:
                metrics["dropped_cnt_mean"] = float(rtail["candidate_dropped_cnt_mean"].mean())
    return metrics


def _run_one(train_py: Path, run_dir: Path, episodes: int, seed: int, k: int, device: str):
    # Ensure each (K, seed) is isolated and reproducible.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_py),
        "--max-episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--device",
        device,
        "--run-dir",
        str(run_dir),
        "--exact-run-dir",
        "--disable-baseline-eval",
    ]
    # inherit parent env while forcing ALL_FEASIBLE candidate strategy
    parent_env = dict(os.environ)
    parent_env["DISABLE_AUTO_PLOT"] = "1"
    parent_env["CANDIDATE_MODE"] = "ALL"
    parent_env["V2V_TOP_K"] = str(k)
    parent_env["TOPK_K"] = str(k)
    parent_env["RANDOMK_K"] = str(k)
    subprocess.run(cmd, cwd=str(ROOT), env=parent_env, check=True)


def main():
    parser = argparse.ArgumentParser(description="K-sweep runner (ALL_FEASIBLE mode, isolated subprocess runs).")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="logs/k_sweep")
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    rows = []
    base_dir = Path(args.logdir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for k in args.ks:
        run_dir = base_dir / f"k{k}" / f"seed{args.seed}"
        _run_one(TRAIN_PY, run_dir, episodes=args.episodes, seed=args.seed, k=int(k), device=args.device)
        metrics = _load_last_metrics(run_dir, tail_n=args.tail)
        rows.append(
            {
                "K": int(k),
                "mode": "ALL",
                "run_dir": str(run_dir),
                "success_rate": metrics["success_rate"],
                "makespan_mean": metrics["makespan_mean"],
                "makespan_p95": metrics["makespan_p95"],
                "illegal_rate": metrics["illegal_rate"],
                "fallback_rate": metrics["fallback_rate"],
                "dropped_cnt_mean": metrics["dropped_cnt_mean"],
            }
        )

    out_path = base_dir / "k_sweep_summary.csv"
    df = pd.DataFrame(rows)
    if df[["success_rate", "makespan_mean", "makespan_p95"]].isna().any().any():
        raise RuntimeError("k_sweep_summary contains NaN in key metrics.")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
