#!/usr/bin/env python3
"""
Short-run RL pilot wrapper for the current vecOffloadingSim mainline.

Runs train.py with the current default config and summarizes tail metrics from
logs/metrics.csv so pilot and formal training share the same code path.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"


def _tail_mean(rows, key: str) -> float:
    vals = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, "", "None"):
            continue
        try:
            vals.append(float(raw))
        except Exception:
            continue
    return float(statistics.fmean(vals)) if vals else 0.0


def _load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_run_dir(run_id_prefix: str) -> Path:
    direct = ROOT / "runs" / run_id_prefix
    if direct.exists():
        return direct
    matches = sorted((ROOT / "runs").glob(f"{run_id_prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No run directory found for prefix: {run_id_prefix}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a short mainline RL pilot and summarize tail metrics.")
    ap.add_argument("--episodes", type=int, default=10, help="Number of training episodes.")
    ap.add_argument("--tail", type=int, default=5, help="Tail episodes used for summary.")
    ap.add_argument("--seed", type=int, default=42, help="Training seed.")
    ap.add_argument("--device", type=str, default="cpu", help="Training device passed to train.py.")
    ap.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra raw arg forwarded to train.py.")
    args = ap.parse_args()

    run_id = args.run_id or f"pilot_mainline_e{args.episodes}_s{args.seed}"
    cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--max-episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--run-id", run_id,
        "--disable-baseline-eval",
    ]
    cmd.extend(args.extra_arg)

    print("[Pilot] command:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    run_dir = _resolve_run_dir(run_id)
    metrics_csv = run_dir / "logs" / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics csv: {metrics_csv}")

    rows = _load_rows(metrics_csv)
    if not rows:
        raise RuntimeError(f"Empty metrics csv: {metrics_csv}")
    tail_n = max(1, min(int(args.tail), len(rows)))
    tail_rows = rows[-tail_n:]

    summary = {
        "episodes_tail": tail_n,
        "decision_frac_local": _tail_mean(tail_rows, "decision_frac_local"),
        "decision_frac_rsu": _tail_mean(tail_rows, "decision_frac_rsu"),
        "decision_frac_v2v": _tail_mean(tail_rows, "decision_frac_v2v"),
        "task_success_rate": _tail_mean(tail_rows, "task_success_rate"),
        "mean_cft_completed": _tail_mean(tail_rows, "mean_cft_completed"),
        "avg_power": _tail_mean(tail_rows, "avg_power"),
        "power_ratio_mean": _tail_mean(tail_rows, "power_ratio_mean"),
        "I_total_mean": _tail_mean(tail_rows, "I_total_mean"),
        "I_caused_mean": _tail_mean(tail_rows, "I_caused_mean"),
        "v2v_link_break_rate": _tail_mean(tail_rows, "v2v_link_break_rate"),
        "illegal_action_rate": _tail_mean(tail_rows, "illegal_action_rate"),
        "reward_mean": _tail_mean(tail_rows, "reward_mean"),
        "r_time": _tail_mean(tail_rows, "r_time"),
        "r_energy": _tail_mean(tail_rows, "r_energy"),
        "r_interf": _tail_mean(tail_rows, "r_interf"),
        "r_illegal": _tail_mean(tail_rows, "r_illegal"),
        "r_term": _tail_mean(tail_rows, "r_term"),
        "r_step": _tail_mean(tail_rows, "r_step"),
        "r_total": _tail_mean(tail_rows, "r_total"),
    }

    print("\n[Pilot] tail summary", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"metrics_csv: {metrics_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
