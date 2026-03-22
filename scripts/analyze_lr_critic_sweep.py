#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURVE_COLORS = ["#1f4e79", "#d97904", "#2c7a3f", "#9b2c2c"]
BASELINE_COLORS = {
    "Local-Only": "#444444",
    "Greedy": "#b22222",
    "CP-EFT": "#5b8c5a",
}
TAIL_METRICS = [
    ("reward_mean", "higher", "Average Reward"),
    ("task_sr", "higher", "Task Success Rate"),
    ("subtask_sr", "higher", "Subtask Success Rate"),
    ("deadline_miss_rate", "lower", "Deadline Miss Rate"),
    ("avg_rsu_queue", "lower", "Average RSU Queue"),
    ("avg_power", "lower", "Average Power"),
    ("approx_kl", "lower", "Approx KL"),
    ("entropy", "neutral", "Entropy"),
    ("clip_frac", "lower", "Clip Fraction"),
    ("ratio_local", "neutral", "Local Ratio"),
    ("ratio_rsu", "neutral", "RSU Ratio"),
    ("ratio_v2v", "neutral", "V2V Ratio"),
]


def _parse_args():
    ap = argparse.ArgumentParser(description="Analyze LR critic sweep runs.")
    ap.add_argument("--run", action="append", required=True, help="Run directory path.")
    ap.add_argument("--label", action="append", default=None, help="Optional label for each run.")
    ap.add_argument("--out-dir", type=str, required=True)
    return ap.parse_args()


def _load_snapshot(run_dir: Path) -> Dict[str, object]:
    with (run_dir / "logs" / "config_snapshot.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_training(run_dir: Path) -> pd.DataFrame:
    repair_csv = run_dir / "diagnostics" / "repair" / "logs" / "training_stats.csv"
    raw_csv = run_dir / "logs" / "training_stats.csv"
    csv_path = repair_csv if repair_csv.exists() else raw_csv
    df = pd.read_csv(csv_path)
    if "episode" in df.columns:
        df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
        df = df.dropna(subset=["episode"]).copy()
        df["episode"] = df["episode"].astype(int)
        df = df.sort_values("episode").drop_duplicates(subset=["episode"], keep="last")
    return df


def _load_baseline_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "logs" / "baseline_eval_core_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_baseline_stats(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "logs" / "baseline_stats.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _smooth_series(series: pd.Series, window: int = 30) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(window=window, min_periods=1).mean()


def _best50_task_sr(df: pd.DataFrame) -> Dict[str, float]:
    if "task_sr" not in df.columns or len(df) < 50:
        return {"best50_task_sr": float("nan"), "best50_episode": float("nan")}
    rolling = df["task_sr"].rolling(50).mean()
    idx = int(rolling.idxmax())
    return {
        "best50_task_sr": float(rolling.loc[idx]),
        "best50_episode": int(df.loc[idx, "episode"]),
    }


def _tail_mean(df: pd.DataFrame, n: int) -> Dict[str, float]:
    tail = df.tail(min(n, len(df)))
    out = {}
    for metric_id, _, _ in TAIL_METRICS:
        if metric_id in tail.columns:
            out[f"{metric_id}_last{n}"] = float(pd.to_numeric(tail[metric_id], errors="coerce").mean())
    return out


def _run_summary(run_dir: Path, label: str) -> Dict[str, object]:
    snapshot = _load_snapshot(run_dir)
    tc = snapshot["train_config"]
    df = _load_training(run_dir)
    baseline_df = _load_baseline_summary(run_dir)
    summary = {
        "label": label,
        "run_dir": str(run_dir),
        "lr_actor": float(tc["LR_ACTOR"]),
        "lr_critic": float(tc["LR_CRITIC"]),
        "entropy_coef_start": float(tc["ENTROPY_COEF_START"]),
        "entropy_coef_end": float(tc["ENTROPY_COEF_END"]),
        "episodes": int(len(df)),
    }
    summary.update(_tail_mean(df, 100))
    summary.update(_tail_mean(df, 300))
    summary.update(_best50_task_sr(df))
    if not baseline_df.empty and "policy" in baseline_df.columns:
        for _, row in baseline_df.iterrows():
            key = f"baseline_task_sr_{row['policy']}"
            summary[key] = float(row.get("task_success_rate_mean", float("nan")))
    return summary


def _build_run_frames(args: argparse.Namespace) -> List[Dict[str, object]]:
    labels = args.label or []
    if labels and len(labels) != len(args.run):
        raise ValueError("--label count must match --run count.")

    run_frames: List[Dict[str, object]] = []
    for idx, run in enumerate(args.run):
        run_dir = Path(run).resolve()
        label = labels[idx] if idx < len(labels) else run_dir.name
        snapshot = _load_snapshot(run_dir)
        tc = snapshot["train_config"]
        df = _load_training(run_dir)
        baseline_stats = _load_baseline_stats(run_dir)
        run_frames.append(
            {
                "label": label,
                "run_dir": run_dir,
                "df": df,
                "baseline_stats": baseline_stats,
                "lr_actor": float(tc["LR_ACTOR"]),
                "lr_critic": float(tc["LR_CRITIC"]),
            }
        )
    run_frames.sort(key=lambda item: item["lr_critic"])
    return run_frames


def _write_report(out_dir: Path, summary_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# LR Critic Sweep Report")
    lines.append("")
    lines.append("## Config Summary")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['label']}: lr_actor={row['lr_actor']:.4g}, lr_critic={row['lr_critic']:.4g}, "
            f"episodes={int(row['episodes'])}, best50_task_sr={row.get('best50_task_sr', float('nan')):.4f}"
        )
    lines.append("")
    lines.append("## Tail-100 Highlights")

    def _winner(metric: str, direction: str) -> str:
        vals = summary_df[["label", metric]].dropna()
        if vals.empty:
            return "n/a"
        if direction == "higher":
            return str(vals.loc[vals[metric].idxmax(), "label"])
        if direction == "lower":
            return str(vals.loc[vals[metric].idxmin(), "label"])
        return "n/a"

    for metric_id, direction, metric_name in TAIL_METRICS[:8]:
        col = f"{metric_id}_last100"
        if col not in summary_df.columns:
            continue
        lines.append(f"- {metric_name}: winner={_winner(col, direction)}")
        for _, row in summary_df.iterrows():
            lines.append(f"  - {row['label']}: {row[col]:.6f}")
    lines.append("")
    lines.append("## Baseline Task-SR")
    baseline_cols = [c for c in summary_df.columns if c.startswith("baseline_task_sr_")]
    if baseline_cols:
        for _, row in summary_df.iterrows():
            parts = [f"{c.replace('baseline_task_sr_', '')}={row[c]:.4f}" for c in baseline_cols if pd.notna(row[c])]
            lines.append(f"- {row['label']}: " + ", ".join(parts))
    else:
        lines.append("- No baseline summary available.")
    lines.append("")
    lines.append("## Key Findings")

    def _find_label_by(metric: str, direction: str) -> str:
        return _winner(metric, direction)

    lines.append(
        f"- Highest tail-100 task success: {_find_label_by('task_sr_last100', 'higher')}; "
        f"lowest deadline miss: {_find_label_by('deadline_miss_rate_last100', 'lower')}."
    )
    lines.append(
        f"- Lowest tail-100 RSU queue: {_find_label_by('avg_rsu_queue_last100', 'lower')}; "
        f"lowest power: {_find_label_by('avg_power_last100', 'lower')}."
    )
    lines.append(
        f"- Best tail-100 reward: {_find_label_by('reward_mean_last100', 'higher')}."
    )

    with (out_dir / "LR_CRITIC_SWEEP_REPORT.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_key_metrics(out_dir: Path, summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("task_sr_last100", "Task SR"),
        ("deadline_miss_rate_last100", "Deadline Miss"),
        ("avg_rsu_queue_last100", "RSU Queue"),
        ("avg_power_last100", "Power"),
    ]
    for ax, (col, title) in zip(axes.flat, plot_specs):
        vals = summary_df[col]
        ax.bar(summary_df["label"], vals)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "lr_key_metrics.png", dpi=180)
    plt.close(fig)


def _plot_offloading(out_dir: Path, summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summary_df))
    ax.bar([i - 0.25 for i in x], summary_df["ratio_local_last100"], width=0.25, label="Local")
    ax.bar(x, summary_df["ratio_rsu_last100"], width=0.25, label="RSU")
    ax.bar([i + 0.25 for i in x], summary_df["ratio_v2v_last100"], width=0.25, label="V2V")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["label"], rotation=15)
    ax.set_ylabel("Tail-100 ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lr_offloading_ratios.png", dpi=180)
    plt.close(fig)


def _plot_reward_curves(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, item in enumerate(run_frames):
        df = item["df"]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        ax.plot(
            df["episode"],
            _smooth_series(df["reward_mean"], 30),
            linewidth=2.2,
            color=color,
            label=item["label"],
        )
    ax.set_title("Reward Curve Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed reward_mean (window=30)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lr_reward_curves.png", dpi=180)
    plt.close(fig)


def _plot_raw_run_curves(run_frames: List[Dict[str, object]]) -> None:
    specs = [
        ("reward_mean", "Raw Reward Mean"),
        ("task_sr", "Raw Task Success Rate"),
        ("deadline_miss_rate", "Raw Deadline Miss Rate"),
        ("avg_rsu_queue", "Raw Average RSU Queue"),
    ]
    for idx, item in enumerate(run_frames):
        df = item["df"]
        plots_dir = Path(item["run_dir"]) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for ax, (metric, title) in zip(axes.flat, specs):
            if metric not in df.columns:
                ax.text(0.5, 0.5, f"{metric} unavailable", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(
                    df["episode"],
                    pd.to_numeric(df[metric], errors="coerce"),
                    linewidth=1.0,
                    alpha=0.95,
                    color=color,
                )
            ax.set_title(title)
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Value")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 1].set_xlabel("Episode")
        fig.suptitle(f"Raw Training Curves: {item['label']}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plots_dir / "fig_raw_convergence_curves.png", dpi=180)
        plt.close(fig)


def _plot_raw_combined_curves(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    specs = [
        ("reward_mean", "Raw Reward Mean"),
        ("task_sr", "Raw Task Success Rate"),
        ("deadline_miss_rate", "Raw Deadline Miss Rate"),
        ("avg_rsu_queue", "Raw Average RSU Queue"),
    ]
    for ax, (metric, title) in zip(axes.flat, specs):
        for idx, item in enumerate(run_frames):
            df = item["df"]
            if metric not in df.columns:
                continue
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            ax.plot(
                df["episode"],
                pd.to_numeric(df[metric], errors="coerce"),
                linewidth=0.95,
                alpha=0.9,
                color=color,
                label=item["label"],
            )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_frames)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "lr_raw_convergence_curves.png", dpi=180)
    plt.close(fig)


def _plot_hybrid_run_curves(run_frames: List[Dict[str, object]]) -> None:
    specs = [
        ("reward_mean", "Reward Mean", 50),
        ("task_sr", "Task Success Rate", 50),
        ("deadline_miss_rate", "Deadline Miss Rate", 50),
        ("avg_rsu_queue", "Average RSU Queue", 30),
    ]
    for idx, item in enumerate(run_frames):
        df = item["df"]
        plots_dir = Path(item["run_dir"]) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for ax, (metric, title, window) in zip(axes.flat, specs):
            if metric not in df.columns:
                ax.text(0.5, 0.5, f"{metric} unavailable", ha="center", va="center", transform=ax.transAxes)
            else:
                raw = pd.to_numeric(df[metric], errors="coerce")
                smooth = raw.rolling(window, min_periods=1).mean()
                ax.plot(df["episode"], raw, linewidth=0.7, alpha=0.22, color=color)
                ax.plot(df["episode"], smooth, linewidth=2.2, alpha=0.98, color=color)
            ax.set_title(f"{title} (raw + MA)")
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Value")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 1].set_xlabel("Episode")
        fig.suptitle(f"Hybrid Convergence Curves: {item['label']}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plots_dir / "fig_hybrid_convergence_curves.png", dpi=180)
        plt.close(fig)


def _plot_hybrid_combined_curves(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    specs = [
        ("reward_mean", "Reward Mean", 50),
        ("task_sr", "Task Success Rate", 50),
        ("deadline_miss_rate", "Deadline Miss Rate", 50),
        ("avg_rsu_queue", "Average RSU Queue", 30),
    ]
    for ax, (metric, title, window) in zip(axes.flat, specs):
        for idx, item in enumerate(run_frames):
            df = item["df"]
            if metric not in df.columns:
                continue
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            raw = pd.to_numeric(df[metric], errors="coerce")
            smooth = raw.rolling(window, min_periods=1).mean()
            ax.plot(df["episode"], raw, linewidth=0.45, alpha=0.10, color=color)
            ax.plot(df["episode"], smooth, linewidth=2.0, alpha=0.98, color=color, label=item["label"])
        ax.set_title(f"{title} (raw + MA)")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_frames)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "lr_hybrid_convergence_curves.png", dpi=180)
    plt.close(fig)


def _plot_reward_convergence_style(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LR Critic Reward Convergence Analysis", fontsize=16, fontweight="bold")
    baseline_ref = run_frames[0]["baseline_stats"] if run_frames else pd.DataFrame()

    ax = axes[0, 0]
    for idx, item in enumerate(run_frames):
        df = item["df"]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        reward_mean = pd.to_numeric(df["reward_mean"], errors="coerce").rolling(50, min_periods=1).mean()
        reward_std = pd.to_numeric(df["reward_mean"], errors="coerce").rolling(50, min_periods=1).std().fillna(0.0)
        ax.plot(df["episode"], reward_mean, color=color, linewidth=2.2, label=item["label"])
        ax.fill_between(df["episode"], reward_mean - reward_std, reward_mean + reward_std, color=color, alpha=0.12)
    for policy in ("Local-Only", "Greedy"):
        if not baseline_ref.empty and policy in set(baseline_ref["policy"]):
            sub = baseline_ref[baseline_ref["policy"] == policy]
            mean_v = float(pd.to_numeric(sub["reward_mean"], errors="coerce").mean())
            std_v = float(pd.to_numeric(sub["reward_mean"], errors="coerce").std())
            bcolor = BASELINE_COLORS[policy]
            ax.axhline(mean_v, color=bcolor, linestyle="--", linewidth=1.6, label=f"{policy} mean")
            ax.axhspan(mean_v - std_v, mean_v + std_v, color=bcolor, alpha=0.08)
    ax.set_ylabel("Reward (50-ep window)")
    ax.set_title("Reward Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for idx, item in enumerate(run_frames):
        df = item["df"]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        reward_std = pd.to_numeric(df["reward_mean"], errors="coerce").rolling(50, min_periods=1).std().fillna(0.0)
        ax.plot(df["episode"], reward_std, color=color, linewidth=2.0, label=item["label"])
    ax.set_ylabel("Reward Std (50-ep window)")
    ax.set_title("Reward Variance Over Training")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for idx, item in enumerate(run_frames):
        df = item["df"]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        sr_smooth = _smooth_series(df["task_sr"], 50) * 100.0
        sr_change = np.abs(np.diff(sr_smooth.to_numpy(), prepend=float(sr_smooth.iloc[0])))
        ax.plot(df["episode"], sr_change, color=color, linewidth=1.8, label=item["label"])
    ax.set_ylabel("|ΔTask SR| (%)")
    ax.set_title("Task Success Volatility")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for idx, item in enumerate(run_frames):
        df = item["df"]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        p_local = pd.to_numeric(df["ratio_local"], errors="coerce").fillna(0.0) + 1e-10
        p_rsu = pd.to_numeric(df["ratio_rsu"], errors="coerce").fillna(0.0) + 1e-10
        p_v2v = pd.to_numeric(df["ratio_v2v"], errors="coerce").fillna(0.0) + 1e-10
        entropy = -(p_local * np.log(p_local) + p_rsu * np.log(p_rsu) + p_v2v * np.log(p_v2v))
        entropy = entropy / np.log(3.0)
        ax.plot(df["episode"], entropy, alpha=0.18, color=color, linewidth=0.4)
        ax.plot(df["episode"], _smooth_series(entropy, 50), color=color, linewidth=2.0, label=item["label"])
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("Policy Diversity")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "lr_reward_convergence_style.png", dpi=180)
    plt.close(fig)


def _plot_mixed_convergence(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    specs = [
        ("task_sr", "Task Success Rate", "Smoothed task_sr"),
        ("reward_mean", "Reward Convergence", "Smoothed reward_mean"),
        ("deadline_miss_rate", "Deadline Miss Convergence", "Smoothed deadline_miss_rate"),
        ("avg_rsu_queue", "RSU Queue Convergence", "Smoothed avg_rsu_queue"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for idx, item in enumerate(run_frames):
            df = item["df"]
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            ax.plot(
                df["episode"],
                _smooth_series(df[metric], 30),
                linewidth=2.0,
                color=color,
                label=item["label"],
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(run_frames), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "lr_mixed_convergence_curves.png", dpi=180)
    plt.close(fig)


def _plot_offloading_curves(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    specs = [
        ("ratio_local", "Local Ratio"),
        ("ratio_rsu", "RSU Ratio"),
        ("ratio_v2v", "V2V Ratio"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        for idx, item in enumerate(run_frames):
            df = item["df"]
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            ax.plot(
                df["episode"],
                _smooth_series(df[metric], 30),
                linewidth=2.0,
                color=color,
                label=item["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Smoothed ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(run_frames), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "lr_offloading_curves.png", dpi=180)
    plt.close(fig)


def _plot_training_diagnostics_curves(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    specs = [
        ("approx_kl", "Approx KL"),
        ("entropy", "Entropy"),
        ("clip_frac", "Clip Fraction"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        for idx, item in enumerate(run_frames):
            df = item["df"]
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            ax.plot(
                df["episode"],
                _smooth_series(df[metric], 30),
                linewidth=2.0,
                color=color,
                label=item["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Smoothed metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(run_frames), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "lr_training_diagnostics_curves.png", dpi=180)
    plt.close(fig)


def _plot_comprehensive_metrics(out_dir: Path, run_frames: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
    specs = [
        ("task_sr", "Task Success Rate", ("Local-Only", "Greedy", "CP-EFT")),
        ("deadline_miss_rate", "Deadline Miss Rate", ("Local-Only", "Greedy", "CP-EFT")),
        ("reward_mean", "Reward Mean", ("Local-Only", "Greedy", "CP-EFT")),
        ("energy_mean", "Energy Mean", ("Local-Only", "Greedy", "CP-EFT")),
        ("avg_rsu_queue", "Average RSU Queue", ("Local-Only", "Greedy", "CP-EFT")),
        ("avg_power", "Average Power", ("Local-Only", "Greedy", "CP-EFT")),
        ("task_duration_mean", "Task Duration Mean", ("Local-Only", "Greedy", "CP-EFT")),
        ("mean_cft_completed", "Mean CFT Completed", ("Local-Only", "Greedy", "CP-EFT")),
    ]
    for ax, (metric, title, baseline_policies) in zip(axes.flat, specs):
        for idx, item in enumerate(run_frames):
            df = item["df"]
            color = CURVE_COLORS[idx % len(CURVE_COLORS)]
            ax.plot(
                df["episode"],
                _smooth_series(df[metric], 50),
                linewidth=2.0,
                color=color,
                label=item["label"],
            )
        baseline_df = run_frames[0]["baseline_stats"] if run_frames else pd.DataFrame()
        if baseline_policies and not baseline_df.empty and "policy" in baseline_df.columns:
            for policy in baseline_policies:
                sub = baseline_df[baseline_df["policy"] == policy]
                if sub.empty or metric not in sub.columns:
                    continue
                series = pd.to_numeric(sub[metric], errors="coerce").dropna()
                if series.empty:
                    continue
                mean_v = float(series.mean())
                std_v = float(series.std())
                bcolor = BASELINE_COLORS.get(policy, "#555555")
                ax.axhline(mean_v, linestyle="--", linewidth=1.5, color=bcolor, label=f"{policy} mean")
                ax.axhspan(mean_v - std_v, mean_v + std_v, color=bcolor, alpha=0.08)
        elif not baseline_policies:
            ax.text(
                0.98,
                0.06,
                "baseline unavailable",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#666666",
            )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[2, 0].set_xlabel("Episode")
    axes[2, 1].set_xlabel("Episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "lr_comprehensive_metrics_with_baselines.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_frames = _build_run_frames(args)
    summaries = []
    for item in run_frames:
        summaries.append(_run_summary(item["run_dir"], item["label"]))

    summary_df = pd.DataFrame(summaries).sort_values("lr_critic")
    summary_df.to_csv(out_dir / "lr_tail_summary.csv", index=False)
    _plot_key_metrics(out_dir, summary_df)
    _plot_offloading(out_dir, summary_df)
    _plot_reward_curves(out_dir, run_frames)
    _plot_raw_run_curves(run_frames)
    _plot_raw_combined_curves(out_dir, run_frames)
    _plot_hybrid_run_curves(run_frames)
    _plot_hybrid_combined_curves(out_dir, run_frames)
    _plot_reward_convergence_style(out_dir, run_frames)
    _plot_mixed_convergence(out_dir, run_frames)
    _plot_offloading_curves(out_dir, run_frames)
    _plot_training_diagnostics_curves(out_dir, run_frames)
    _plot_comprehensive_metrics(out_dir, run_frames)
    _write_report(out_dir, summary_df)
    print(f"Saved LR critic sweep analysis to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
