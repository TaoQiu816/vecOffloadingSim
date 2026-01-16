#!/usr/bin/env python3
"""
Two-stage sweep for VALUE_CLIP_RANGE and MIN_ACTIVE_SAMPLES.
Stage1: quick filter, Stage2: refined runs on Top-K.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


def _parse_args():
    parser = argparse.ArgumentParser(description="Sweep VALUE_CLIP_RANGE and MIN_ACTIVE_SAMPLES.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--clip-ranges", type=str, default="0.2,0.3,0.4")
    parser.add_argument("--min-active", type=str, default="64,96,128")
    parser.add_argument("--smoke-only", action="store_true", default=False)
    parser.add_argument("--stage1-results", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--disable-baseline", action="store_true", default=False)
    parser.add_argument("--disable-auto-plot", action="store_true", default=False)
    return parser.parse_args()


def _parse_list(text, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _run_train(run_dir: Path, seed: int, episodes: int, clip_range: float, min_active: int, args):
    env = os.environ.copy()
    env["VALUE_CLIP_RANGE"] = str(clip_range)
    env["MIN_ACTIVE_SAMPLES"] = str(min_active)
    env["USE_VALUE_CLIP"] = "1"
    env["USE_VALUE_TARGET_NORM"] = "1"
    if args.disable_baseline:
        env["DISABLE_BASELINE_EVAL"] = "1"
    if args.disable_auto_plot:
        env["DISABLE_AUTO_PLOT"] = "1"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "train.py"),
        "--max-episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--log-interval",
        "10",
        "--eval-interval",
        str(episodes),
    ]

    start = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT_DIR))
    return time.time() - start


def _nan_inf_count(df, cols):
    count = 0
    for col in cols:
        if col not in df:
            continue
        series = df[col]
        count += int(series.isna().sum())
        count += int(np.isinf(series).sum())
    return count


def _write_summary(metrics_path: Path, summary_path: Path, run_info: dict):
    df = pd.read_csv(metrics_path)
    cols = [
        "reward_mean",
        "value_loss",
        "policy_loss",
        "value_clip_fraction",
        "skipped_update_count",
        "active_ratio",
        "active_samples",
    ]
    summary = dict(run_info)
    summary["episodes"] = int(len(df))
    for col in cols:
        if col in df:
            val_mean = float(df[col].mean())
            val_std = float(df[col].std())
        else:
            val_mean = 0.0
            val_std = 0.0
        if col == "active_samples":
            summary["active_samples_mean"] = val_mean
            summary["active_samples_std"] = val_std
        else:
            summary[col] = val_mean
            summary[f"{col}_std"] = val_std
    skipped = df["skipped_update_count"].sum() if "skipped_update_count" in df else 0.0
    summary["skipped_update_ratio"] = float(skipped) / float(len(df)) if len(df) > 0 else 0.0
    if "active_samples" in df:
        summary["active_samples_p10"] = float(df["active_samples"].quantile(0.10))
        summary["active_samples_p50"] = float(df["active_samples"].quantile(0.50))
    else:
        summary["active_samples_p10"] = 0.0
        summary["active_samples_p50"] = 0.0
    summary["nan_inf_count"] = _nan_inf_count(df, ["value_loss", "policy_loss", "grad_norm"])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return summary


def _aggregate_results(summaries, out_path: Path):
    rows = []
    for key, items in summaries.items():
        stage, episodes, clip_range, min_active = key
        row = {
            "stage": stage,
            "episodes": episodes,
            "value_clip_range": clip_range,
            "min_active_samples": min_active,
        }
        for metric in [
            "reward_mean",
            "value_loss",
            "policy_loss",
            "value_clip_fraction",
            "skipped_update_ratio",
            "active_ratio",
            "active_samples_mean",
            "active_samples_p10",
            "active_samples_p50",
        ]:
            vals = [it.get(metric, 0.0) for it in items]
            row[f"{metric}_mean"] = float(mean(vals)) if vals else 0.0
            row[f"{metric}_std"] = float(stdev(vals)) if len(vals) > 1 else 0.0
        row["nan_inf_runs"] = int(sum(1 for it in items if it.get("nan_inf_count", 0) > 0))
        row["elapsed_sec_mean"] = float(mean([it.get("elapsed_sec", 0.0) for it in items])) if items else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def _write_recommendation(df, out_path: Path):
    filtered = df.copy()
    filtered = filtered[filtered["nan_inf_runs"] == 0]
    filtered = filtered[filtered["skipped_update_ratio_mean"] <= 0.30]
    if filtered.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("No valid configuration after filtering.\n")
        return

    filtered["clip_score"] = (filtered["value_clip_fraction_mean"] - 0.5).abs()
    ranked = filtered.sort_values(
        by=["reward_mean_mean", "value_loss_std", "clip_score"],
        ascending=[False, True, True],
    )
    top = ranked.head(3)

    lines = []
    best = top.iloc[0]
    lines.append("Top-1 Recommendation\n")
    lines.append(
        f"clip_range={best['value_clip_range']} min_active_samples={best['min_active_samples']} "
        f"reward_mean={best['reward_mean_mean']:.4f}±{best['reward_mean_std']:.4f} "
        f"value_loss={best['value_loss_mean']:.4f}±{best['value_loss_std']:.4f} "
        f"clip_frac={best['value_clip_fraction_mean']:.4f}±{best['value_clip_fraction_std']:.4f}\n"
    )
    lines.append("\nTop-3 Candidates\n")
    for _, row in top.iterrows():
        lines.append(
            f"- clip_range={row['value_clip_range']} min_active_samples={row['min_active_samples']} "
            f"reward_mean={row['reward_mean_mean']:.4f}±{row['reward_mean_std']:.4f} "
            f"value_loss={row['value_loss_mean']:.4f}±{row['value_loss_std']:.4f} "
            f"clip_frac={row['value_clip_fraction_mean']:.4f}±{row['value_clip_fraction_std']:.4f}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_stage1_topk(path: Path, topk: int):
    import json
    if not path.exists():
        return []
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return _recommend_topk(df, topk)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [(float(item["clip_range"]), int(item["min_active_samples"])) for item in payload]
    return []


def _write_stage1_topk(path: Path, combos):
    # Format example:
    # [{"clip_range": 0.2, "min_active_samples": 64}, ...]
    import json
    payload = [{"clip_range": c[0], "min_active_samples": c[1]} for c in combos]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _recommend_topk(df, topk: int):
    filtered = df.copy()
    filtered = filtered[filtered["nan_inf_count"] == 0]
    filtered = filtered[filtered["skipped_update_ratio"] <= 0.30]
    filtered["clip_score"] = (filtered["value_clip_fraction"] - 0.5).abs()
    ranked = filtered.sort_values(
        by=["reward_mean", "value_loss_std", "clip_score"],
        ascending=[False, True, True],
    )
    rows = ranked.head(topk)
    return [(float(r["value_clip_range"]), int(r["min_active_samples"])) for _, r in rows.iterrows()]


def _write_stage1_recommendation(path: Path, df, topk_list):
    lines = []
    lines.append("Stage1 Filtering Summary\n")
    lines.append(f"total_candidates={len(df)}")
    lines.append(f"filtered_nan_inf={int((df['nan_inf_count'] > 0).sum())}")
    lines.append(f"filtered_skipped_ratio={int((df['skipped_update_ratio'] > 0.30).sum())}")
    lines.append("\nTop-K candidates:")
    for clip_range, min_active in topk_list:
        row = df[(df["value_clip_range"] == clip_range) & (df["min_active_samples"] == min_active)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"- clip={clip_range} min_active={min_active} "
            f"reward_mean={r['reward_mean']:.4f} "
            f"value_loss_std={r['value_loss_std']:.4f} "
            f"clip_frac={r['value_clip_fraction']:.4f}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = _parse_args()
    stage_flag_present = "--stage" in sys.argv
    seeds = _parse_list(args.seeds, int)
    clip_ranges = _parse_list(args.clip_ranges, float)
    min_actives = _parse_list(args.min_active, int)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"sweep_results/valueclip_active_samples_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_only:
        args.episodes = 1
        seeds = seeds[:1]

    if not stage_flag_present:
        stage = None
    else:
        stage = args.stage

    def _run_stage(stage_id: int, episodes: int, combos, seeds_list, stage_dir: Path):
        summaries = {}
        stage_dir.mkdir(parents=True, exist_ok=True)
        for clip_range, min_active in combos:
            for seed in seeds_list:
                clip_str = str(clip_range).replace(".", "p")
                run_id = f"stage{stage_id}_ep{episodes}_s{seed}_clip{clip_str}_min{min_active}_{ts}"
                run_dir = (
                    Path("runs")
                    / f"stage{stage_id}_ep{episodes}"
                    / f"clip{clip_str}_min{min_active}"
                    / f"seed{seed}"
                    / run_id
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                elapsed = _run_train(run_dir, seed, episodes, clip_range, min_active, args)

                metrics_src = run_dir / "logs" / "metrics.csv"
                metrics_dst = stage_dir / f"metrics_{run_id}.csv"
                summary_dst = stage_dir / f"summary_{run_id}.csv"
                metrics_dst.write_bytes(metrics_src.read_bytes())

                config_dump = run_dir / "config_dump.json"
                config_dump.write_text(
                    f'{{"run_id":"{run_id}","stage":{stage_id},"episodes":{episodes},"seed":{seed},'
                    f'"value_clip_range":{clip_range},"min_active_samples":{min_active}}}',
                    encoding="utf-8",
                )

                summary = _write_summary(
                    metrics_src,
                    summary_dst,
                    {
                        "run_id": run_id,
                        "stage": stage_id,
                        "episodes": episodes,
                        "seed": seed,
                        "value_clip_range": clip_range,
                        "min_active_samples": min_active,
                        "elapsed_sec": elapsed,
                        "run_dir": str(run_dir),
                    },
                )

                key = (stage_id, episodes, clip_range, min_active)
                summaries.setdefault(key, []).append(summary)

        results_table_path = stage_dir / "results_table.csv"
        results_table = pd.DataFrame([row for items in summaries.values() for row in items])
        results_table.to_csv(results_table_path, index=False)

        results_table_agg_path = stage_dir / "results_table_agg.csv"
        results_table_agg = _aggregate_results(summaries, results_table_agg_path)

        return results_table, results_table_agg

    full_combos = [(c, m) for c in clip_ranges for m in min_actives]

    if stage is None:
        stage_dir = out_dir / "single_stage"
        results_table, results_table_agg = _run_stage(0, args.episodes, full_combos, seeds, stage_dir)
        recommendation_path = stage_dir / "recommendation.txt"
        _write_recommendation(results_table_agg, recommendation_path)
        print(f"[Sweep] results_table: {stage_dir / 'results_table.csv'}")
        print(f"[Sweep] results_table_agg: {stage_dir / 'results_table_agg.csv'}")
        print(f"[Sweep] recommendation: {recommendation_path}")
        return

    if stage == 1:
        stage1_dir = out_dir / "stage1"
        results_table, results_table_agg = _run_stage(1, args.episodes, full_combos, seeds[:1], stage1_dir)
        stage1_results_path = stage1_dir / "stage1_results_table.csv"
        results_table.to_csv(stage1_results_path, index=False)
        stage1_agg_path = stage1_dir / "stage1_results_table_agg.csv"
        results_table_agg.to_csv(stage1_agg_path, index=False)
        topk_list = _recommend_topk(results_table, args.topk)
        topk_path = stage1_dir / "stage1_topk.json"
        _write_stage1_topk(topk_path, topk_list)
        rec_path = stage1_dir / "stage1_recommendation.txt"
        _write_stage1_recommendation(rec_path, results_table, topk_list)
        print(f"[Stage1] results_table: {stage1_results_path}")
        print(f"[Stage1] topk: {topk_path}")
        print(f"[Stage1] recommendation: {rec_path}")
        return

    if stage == 2:
        stage2_dir = out_dir / "stage2"
        if args.topk:
            stage1_path = Path(args.stage1_results) if args.stage1_results else (out_dir / "stage1" / "stage1_topk.json")
            topk_list = _load_stage1_topk(stage1_path, args.topk)
            combos = list(topk_list)
            if not combos:
                raise RuntimeError(f"[Stage2] No Top-K combos found from {stage1_path}")
        else:
            combos = full_combos
        results_table, results_table_agg = _run_stage(2, args.episodes, combos, seeds, stage2_dir)
        final_table_path = stage2_dir / "final_results_table.csv"
        final_agg_path = stage2_dir / "final_results_table_agg.csv"
        results_table.to_csv(final_table_path, index=False)
        results_table_agg.to_csv(final_agg_path, index=False)
        recommendation_path = stage2_dir / "final_recommendation.txt"
        _write_recommendation(results_table_agg, recommendation_path)
        print(f"[Stage2] results_table: {final_table_path}")
        print(f"[Stage2] results_table_agg: {final_agg_path}")
        print(f"[Stage2] recommendation: {recommendation_path}")


if __name__ == "__main__":
    main()
