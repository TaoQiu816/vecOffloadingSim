"""
Post-process a finished run directory to make downstream analysis/comparison easier.

This script is intentionally "display/audit layer only":
- It does NOT modify environment dynamics or reward definitions.
- It only reads existing run artifacts (metrics CSV + env_reward.jsonl) and writes
  augmented metrics files for plotting and paper tables.

Inputs (if present):
- <run_dir>/metrics/train_metrics.csv
- <run_dir>/logs/env_reward.jsonl   (episode-level reward_stats dumps)

Outputs:
- <run_dir>/metrics/train_metrics_full.csv
"""

import argparse
import json
import os
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description="Post-process run artifacts into richer metrics for plotting/comparison.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--overwrite", action="store_true", default=False, help="Overwrite train_metrics_full.csv if exists.")
    return p.parse_args()


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _load_env_reward_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            ep = obj.get("episode")
            if ep is None:
                continue

            out: Dict[str, Any] = {"episode": int(ep)}

            metrics = obj.get("metrics") or {}
            # reward_stats style: metrics[metric_name] = {mean, abs_mean, p95, ...}
            for k, v in metrics.items():
                if not isinstance(v, dict):
                    continue
                mean_v = v.get("mean")
                abs_mean_v = v.get("abs_mean")
                p95_v = v.get("p95")
                if mean_v is not None:
                    out[f"{k}.mean"] = _safe_float(mean_v)
                if abs_mean_v is not None:
                    out[f"{k}.abs_mean"] = _safe_float(abs_mean_v)
                if p95_v is not None:
                    out[f"{k}.p95"] = _safe_float(p95_v)

            # Also capture top-level episode stats if present (newer runs).
            for key in (
                "episode_time_seconds",
                "mean_cft_est",
                "mean_cft_completed",
                "task_duration_mean",
                "task_duration_p95",
                "deadline_miss_rate",
                "time_limit_rate",
                "I_total_mean",
                "I_total_p95",
                "rho_selected_mean",
                "rho_selected_p10",
                "rho_selected_p50",
                "rho_selected_p95",
                "rho_selected_lt_0p6_rate",
                "rho_selected_lt_0p7_rate",
                "risk_penalty_mean",
                "chain_tx_total",
                "chain_p95_mean",
                "chain_pfail_mean",
                "chain_risk_cost_total",
                "trust_attempts",
                "trust_failures",
                "trust_failure_rate",
                "trust_retry_count",
                "unified_nonfinite_count",
                "unified_consistency_mismatch_count",
                "unified_illegal_trigger_count",
            ):
                if key in obj:
                    out[key] = obj.get(key)

            rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["episode"])
    df = pd.DataFrame(rows)
    df = df.sort_values("episode").drop_duplicates(subset=["episode"], keep="last").reset_index(drop=True)
    return df


def _compute_abs_ratios(df: pd.DataFrame) -> pd.DataFrame:
    # We compute dominance shares from abs_mean columns if present.
    parts = [
        "r_time",
        "r_energy",
        "r_interf",
        "r_risk",
        "r_illegal",
        "r_pbrs",
        "r_term",
    ]
    abs_cols = {p: f"{p}.abs_mean" for p in parts if f"{p}.abs_mean" in df.columns}
    if not abs_cols:
        return df

    abs_sum = None
    for p, c in abs_cols.items():
        s = df[c].fillna(0.0).astype(float)
        abs_sum = s if abs_sum is None else (abs_sum + s)
    abs_sum = abs_sum.replace(0.0, np.nan)

    for p, c in abs_cols.items():
        df[f"abs_ratio_{p}"] = (df[c].astype(float) / abs_sum).fillna(0.0)

    return df


def main():
    args = _parse_args()
    run_dir = os.path.abspath(args.run_dir)
    metrics_dir = os.path.join(run_dir, "metrics")
    logs_dir = os.path.join(run_dir, "logs")
    in_csv = os.path.join(metrics_dir, "train_metrics.csv")
    env_jsonl = os.path.join(logs_dir, "env_reward.jsonl")
    out_csv = os.path.join(metrics_dir, "train_metrics_full.csv")

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing: {in_csv}")
    if os.path.exists(out_csv) and not args.overwrite:
        print(f"[Postprocess] exists (skip): {out_csv} (use --overwrite to regenerate)")
        return

    df = pd.read_csv(in_csv)
    if "episode" not in df.columns:
        raise ValueError("train_metrics.csv missing 'episode' column")

    if os.path.exists(env_jsonl):
        df_env = _load_env_reward_jsonl(env_jsonl)
        df_env = _compute_abs_ratios(df_env)

        # Map env_reward metric keys to flatter column names.
        rename = {
            "r_time.mean": "r_time_mean",
            "r_energy.mean": "r_energy_mean",
            "r_interf.mean": "r_interf_mean",
            "r_risk.mean": "r_risk_mean",
            "r_illegal.mean": "r_illegal_mean",
            "r_pbrs.mean": "r_pbrs_mean",
            "r_term.mean": "r_term_mean",
            "r_step.mean": "r_step_mean",
            "reward.mean": "r_total_mean",
            "r_time.abs_mean": "r_time_abs_mean",
            "r_energy.abs_mean": "r_energy_abs_mean",
            "r_interf.abs_mean": "r_interf_abs_mean",
            "r_risk.abs_mean": "r_risk_abs_mean",
            "r_illegal.abs_mean": "r_illegal_abs_mean",
            "r_pbrs.abs_mean": "r_pbrs_abs_mean",
            "r_term.abs_mean": "r_term_abs_mean",
            "abs_ratio_r_time": "abs_ratio_r_time",
            "abs_ratio_r_energy": "abs_ratio_r_energy",
            "abs_ratio_r_interf": "abs_ratio_r_interf",
            "abs_ratio_r_risk": "abs_ratio_r_risk",
            "abs_ratio_r_illegal": "abs_ratio_r_illegal",
            "abs_ratio_r_pbrs": "abs_ratio_r_pbrs",
            "abs_ratio_r_term": "abs_ratio_r_term",
        }
        for src, dst in list(rename.items()):
            if src not in df_env.columns:
                rename.pop(src, None)
        df_env = df_env.rename(columns=rename)

        # Merge and prefer the postprocessed values when the base is missing/NaN.
        merged = df.merge(df_env, on="episode", how="left", suffixes=("", "_pp"))

        # If train_metrics already has these columns (newer runs), only fill NaNs.
        for col in df_env.columns:
            if col == "episode":
                continue
            if col in df.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_pp"])
                merged = merged.drop(columns=[f"{col}_pp"])
            else:
                merged = merged.rename(columns={f"{col}_pp": col}) if f"{col}_pp" in merged.columns else merged

        df = merged
        print(f"[Postprocess] loaded env_reward.jsonl: {env_jsonl}")
    else:
        print(f"[Postprocess] env_reward.jsonl missing (skip): {env_jsonl}")

    os.makedirs(metrics_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✓ Wrote: {out_csv} (rows={len(df)} cols={len(df.columns)})")


if __name__ == "__main__":
    main()
