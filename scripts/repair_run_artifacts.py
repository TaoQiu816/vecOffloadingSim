#!/usr/bin/env python3
"""
Repair and audit a run's CSV artifacts without mutating the original run logs.

Outputs a self-contained cleaned copy under:
  <run_dir>/diagnostics/repair/
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


BASELINE_POLICIES = ["Local-Only", "Greedy", "EFT", "CP-EFT"]
NON_NUMERIC_COLUMNS = {"policy", "termination_reason", "termination_reason_raw", "termination_reason_bucket", "abs_ratio_basis"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = int(df[col].notna().sum())
        if non_null == 0:
            continue
        if int(converted.notna().sum()) >= max(1, int(non_null * 0.95)):
            df[col] = converted
    return df


def _header_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "episode" in df.columns:
        mask |= df["episode"].astype(str).str.strip().eq("episode")
    if "policy" in df.columns:
        mask |= df["policy"].astype(str).str.strip().eq("policy")
    return mask


def _clean_by_kind(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(df)
    header_mask = _header_mask(df)
    header_rows_removed = int(header_mask.sum())
    df = df.loc[~header_mask].copy()
    df = _coerce_numeric_columns(df)

    if "episode" not in df.columns:
        raise ValueError("CSV 缺少 episode 列")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    invalid_episode_rows = int(df["episode"].isna().sum())
    df = df[df["episode"].notna()].copy()
    df["episode"] = df["episode"].astype(int)

    duplicate_rows_removed = 0
    if kind == "episode_log":
        if "policy" in df.columns:
            df["policy"] = df["policy"].fillna("").astype(str)
            baseline_mask = df["policy"].isin(BASELINE_POLICIES)
            train_df = df[~baseline_mask].copy()
            base_df = df[baseline_mask].copy()
            duplicate_rows_removed += int(len(train_df) - len(train_df.drop_duplicates(["episode"], keep="last")))
            duplicate_rows_removed += int(len(base_df) - len(base_df.drop_duplicates(["policy", "episode"], keep="last")))
            train_df = train_df.sort_values("episode").drop_duplicates(["episode"], keep="last")
            base_df = base_df.sort_values(["policy", "episode"]).drop_duplicates(["policy", "episode"], keep="last")
            df = pd.concat([train_df, base_df], ignore_index=True, sort=False)
        elif "duration" in df.columns:
            duration_str = df["duration"].astype(str)
            baseline_mask = duration_str.isin(BASELINE_POLICIES)
            train_df = df[~baseline_mask].copy()
            base_df = df[baseline_mask].copy()
            duplicate_rows_removed += int(len(train_df) - len(train_df.drop_duplicates(["episode"], keep="last")))
            duplicate_rows_removed += int(len(base_df) - len(base_df.drop_duplicates(["duration", "episode"], keep="last")))
            train_df = train_df.sort_values("episode").drop_duplicates(["episode"], keep="last")
            base_df = base_df.sort_values(["duration", "episode"]).drop_duplicates(["duration", "episode"], keep="last")
            df = pd.concat([train_df, base_df], ignore_index=True, sort=False)
        else:
            duplicate_rows_removed += int(len(df) - len(df.drop_duplicates(["episode"], keep="last")))
            df = df.sort_values("episode").drop_duplicates(["episode"], keep="last")
    elif kind == "baseline":
        if "policy" not in df.columns:
            raise ValueError("baseline_stats.csv 缺少 policy 列")
        df["policy"] = df["policy"].fillna("").astype(str)
        duplicate_rows_removed += int(len(df) - len(df.drop_duplicates(["policy", "episode"], keep="last")))
        df = df[df["policy"].isin(BASELINE_POLICIES)].sort_values(["policy", "episode"]).drop_duplicates(["policy", "episode"], keep="last")
    else:
        duplicate_rows_removed += int(len(df) - len(df.drop_duplicates(["episode"], keep="last")))
        df = df.sort_values("episode").drop_duplicates(["episode"], keep="last")

    summary = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "header_rows_removed": header_rows_removed,
        "invalid_episode_rows_removed": invalid_episode_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "object_columns_after": [c for c in df.columns if df[c].dtype == "object"],
    }
    return df.reset_index(drop=True), summary


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        _ensure_dir(dst.parent)
        shutil.copyfile(src, dst)


def _repair_one(src: Path, dst: Path, kind: str, manifest: dict) -> None:
    if not src.exists():
        manifest[str(src)] = {"status": "missing"}
        return
    raw = pd.read_csv(src, dtype=str)
    cleaned, summary = _clean_by_kind(raw, kind)
    _ensure_dir(dst.parent)
    cleaned.to_csv(dst, index=False)
    summary["status"] = "ok"
    summary["source"] = str(src)
    summary["output"] = str(dst)
    manifest[str(src)] = summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair run CSV artifacts into diagnostics/repair")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repair_dir = run_dir / "diagnostics" / "repair"
    _ensure_dir(repair_dir / "logs")
    _ensure_dir(repair_dir / "metrics")
    _ensure_dir(repair_dir / "plots")

    manifest = {
        "run_dir": str(run_dir),
        "repair_dir": str(repair_dir),
        "baseline_policies": list(BASELINE_POLICIES),
        "files": {},
    }

    targets = [
        (run_dir / "episode_log.csv", repair_dir / "episode_log.csv", "episode_log"),
        (run_dir / "logs" / "training_stats.csv", repair_dir / "logs" / "training_stats.csv", "episode"),
        (run_dir / "logs" / "metrics.csv", repair_dir / "logs" / "metrics.csv", "episode"),
        (run_dir / "metrics" / "train_metrics.csv", repair_dir / "metrics" / "train_metrics.csv", "episode"),
        (run_dir / "metrics" / "metrics.csv", repair_dir / "metrics" / "metrics.csv", "episode"),
    ]
    for src, dst, kind in targets:
        _repair_one(src, dst, kind, manifest["files"])

    baseline_src = run_dir / "logs" / "baseline_stats.csv"
    if baseline_src.exists():
        _repair_one(baseline_src, repair_dir / "logs" / "baseline_stats.csv", "baseline", manifest["files"])

    for rel in [
        "config.json",
        "config_dump.json",
        "run_meta.json",
        os.path.join("logs", "config_snapshot.json"),
        "baseline_run_meta.json",
    ]:
        src = run_dir / rel
        dst = repair_dir / rel
        _copy_if_exists(src, dst)

    manifest_path = repair_dir / "repair_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✓ Repaired artifacts saved to: {repair_dir}")
    print(f"✓ Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
