#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np


def _read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _write_t_est_vs_real(records, out_path):
    groups = {}
    for rec in records:
        action = rec.get("action_type", "Unknown")
        groups.setdefault(action, []).append(rec)

    rows = []
    for action, items in sorted(groups.items()):
        errors = np.array([r.get("est_error", 0.0) for r in items], dtype=np.float32)
        t_est = np.array([r.get("t_actual_est", 0.0) for r in items], dtype=np.float32)
        t_real = np.array([r.get("t_actual_real", 0.0) for r in items], dtype=np.float32)
        if errors.size == 0:
            continue
        rows.append({
            "action_type": action,
            "count": int(errors.size),
            "est_error_p50": float(np.percentile(errors, 50)),
            "est_error_p90": float(np.percentile(errors, 90)),
            "est_error_p95": float(np.percentile(errors, 95)),
            "t_est_mean": float(np.mean(t_est)),
            "t_real_mean": float(np.mean(t_real)),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "action_type",
                "count",
                "est_error_p50",
                "est_error_p90",
                "est_error_p95",
                "t_est_mean",
                "t_real_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _check_scheme_activation(audit_csv):
    if not os.path.exists(audit_csv):
        raise RuntimeError(f"missing scheme activation csv: {audit_csv}")
    with open(audit_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("scheme activation csv empty")
    last = rows[-1]
    if last.get("passed") != "yes":
        raise RuntimeError(f"scheme activation failed: {last.get('reason')}")
    return last


def main():
    parser = argparse.ArgumentParser(description="Smoke test reward schemes with fail-fast audit.")
    parser.add_argument("--scheme", required=True, choices=["PBRS_KP", "PBRS_KP_V2", "LEGACY_CFT"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join("runs", f"smoke_{args.scheme.lower()}_{ts}"))
    env = os.environ.copy()
    env["REWARD_SCHEME"] = args.scheme
    env["MAX_EPISODES"] = str(args.episodes)
    env["SEED"] = str(args.seed)
    env["DEVICE_NAME"] = "cpu"
    env["DISABLE_BASELINE_EVAL"] = "1"

    cmd = [
        sys.executable,
        "train.py",
        "--max-episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
    ]
    print(f"[smoke] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    audit_dir = os.path.join(run_dir, "audit_results")
    audit_csv = os.path.join(audit_dir, "scheme_activation_check.csv")
    last_row = _check_scheme_activation(audit_csv)
    print(f"[smoke] scheme activation ok: {last_row.get('scheme')}")

    t_est_path = os.path.join(audit_dir, "t_est_real_records.jsonl")
    records = _read_jsonl(t_est_path)
    if not records:
        raise RuntimeError(f"no t_est records found at {t_est_path}")
    t_est_csv = os.path.join(audit_dir, "t_est_vs_real.csv")
    _write_t_est_vs_real(records, t_est_csv)
    print(f"[smoke] wrote {t_est_csv}")


if __name__ == "__main__":
    main()
