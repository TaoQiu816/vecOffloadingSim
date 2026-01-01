#!/usr/bin/env python
"""
绘制关键训练指标图（v4）。读取指定 run_dir 下的 metrics.csv（优先 logs/metrics.csv，其次 metrics/metrics.csv），
输出到 run_dir/plots/*.png。示例：
  python scripts/plot_key_metrics_v4.py --run-dir runs/v4_fixed200_s7_... --window 15
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "episode",
    "steps",
    "reward_mean",
    "success_rate_end",
    "deadline_miss_rate",
    "time_limit_rate",
]


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def find_metrics(run_dir: Path) -> Path:
    candidates = [run_dir / "logs" / "metrics.csv", run_dir / "metrics" / "metrics.csv"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"metrics.csv not found under {run_dir}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clip_series(series: pd.Series, pclip: float) -> tuple[pd.Series, float]:
    if series.dropna().empty:
        return series, np.nan
    clip_val = float(series.quantile(pclip))
    return series.clip(upper=clip_val), clip_val


def plot_lines(x, ys, labels, title, ylabel, out_path, ylim=None):
    plt.figure(figsize=(12, 4))
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label, alpha=1.0)
    plt.grid(alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_reward_sum(df, window, out_dir):
    reward_sum = df["reward_mean"] * df["steps"]
    roll = rolling(reward_sum, window)
    plt.figure(figsize=(12, 4))
    plt.plot(df["episode"], reward_sum, label="reward_sum raw", alpha=0.35)
    plt.plot(df["episode"], roll, label=f"roll{window}", alpha=1.0)
    plt.grid(alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("reward_sum (=reward_mean*steps)")
    plt.title(f"reward_sum with rolling window {window}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_reward_sum.png", dpi=160)
    plt.close()
    return reward_sum, roll


def plot_success_miss(df, window, out_dir):
    x = df["episode"]
    plot_lines(
        x,
        [rolling(df["success_rate_end"], window), rolling(df["deadline_miss_rate"], window), rolling(df["time_limit_rate"], window)],
        [f"success roll{window}", f"miss roll{window}", f"time_limit roll{window}"],
        f"roll{window} success/miss/time_limit",
        "rate",
        out_dir / "02_success_miss_timelimit.png",
        ylim=(-0.02, 1.02),
    )


def plot_actions(df, window, out_dir):
    x = df["episode"]
    plot_lines(
        x,
        [
            rolling(df.get("decision_frac_local", pd.Series(np.nan, index=x.index)), window),
            rolling(df.get("decision_frac_rsu", pd.Series(np.nan, index=x.index)), window),
            rolling(df.get("decision_frac_v2v", pd.Series(np.nan, index=x.index)), window),
        ],
        [f"local roll{window}", f"rsu roll{window}", f"v2v roll{window}"],
        f"action fractions roll{window}",
        "fraction",
        out_dir / "03_action_frac_LRV.png",
        ylim=(-0.02, 1.02),
    )


def plot_v2v_costs(df, window, out_dir):
    x = df["episode"]
    v2v_win = rolling(df.get("v2v_beats_rsu_rate", pd.Series(np.nan, index=x.index)), window)
    plot_lines(
        x,
        [v2v_win],
        [f"v2v_beats_rsu roll{window}"],
        f"v2v beats rsu rate roll{window}",
        "rate (0-1)",
        out_dir / "04a_v2v_beats_rsu_rate.png",
        ylim=(-0.02, 1.02),
    )
    gap = rolling(df.get("mean_cost_gap_v2v_minus_rsu", pd.Series(np.nan, index=x.index)), window)
    cost_rsu = rolling(df.get("mean_cost_rsu", pd.Series(np.nan, index=x.index)), window)
    cost_v2v = rolling(df.get("mean_cost_v2v", pd.Series(np.nan, index=x.index)), window)
    plt.figure(figsize=(12, 4))
    plt.plot(x, gap, label=f"gap roll{window} (v2v - rsu)")
    plt.plot(x, cost_rsu, label=f"cost_rsu roll{window}")
    plt.plot(x, cost_v2v, label=f"cost_v2v roll{window}")
    plt.grid(alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("estimated cost/time")
    plt.title("cost gap>0 => V2V slower than RSU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04b_cost_gap_and_costs.png", dpi=160)
    plt.close()


def plot_ppo(df, window, out_dir):
    x = df["episode"]
    entropy_col = "policy_entropy" if "policy_entropy" in df.columns else "entropy"
    entropy_label = entropy_col
    plot_lines(
        x,
        [
            rolling(df.get("approx_kl", pd.Series(np.nan, index=x.index)), window),
            rolling(df.get("clip_frac", pd.Series(np.nan, index=x.index)), window),
            rolling(df.get(entropy_col, pd.Series(np.nan, index=x.index)), window),
        ],
        [f"kl roll{window}", f"clip roll{window}", f"{entropy_label} roll{window}"],
        f"PPO diag roll{window} (entropy from {entropy_label})",
        "value",
        out_dir / "05_ppo_kl_clip_entropy.png",
    )


def plot_value_loss(df, window, pclip, out_dir):
    x = df["episode"]
    series = df.get("value_loss", pd.Series(np.nan, index=x.index))
    clipped, clip_val = clip_series(series, pclip)
    roll = rolling(clipped, window)
    plt.figure(figsize=(12, 4))
    plt.plot(x, clipped, label=f"value_loss raw clipped@p{pclip}", alpha=0.35)
    plt.plot(x, roll, label=f"roll{window}", alpha=1.0)
    plt.grid(alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("value_loss (clipped display)")
    plt.title(f"value_loss display clipped at p{pclip}={clip_val:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "06_value_loss.png", dpi=160)
    plt.close()


def plot_steps_hist(df, short_th, out_dir):
    steps = df["steps"]
    plt.figure(figsize=(12, 4))
    plt.hist(steps, bins=30, alpha=0.7, edgecolor="black")
    plt.axvline(short_th, color="red", linestyle="--", label=f"short_th={short_th}")
    plt.grid(alpha=0.3)
    plt.xlabel("steps")
    plt.title(f"steps histogram, short_th={short_th}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "07_steps_distribution.png", dpi=160)
    plt.close()


def plot_reward_sum_vs_steps(df, out_dir):
    reward_sum = df["reward_mean"] * df["steps"]
    plt.figure(figsize=(12, 4))
    plt.scatter(df["steps"], reward_sum, alpha=0.4, s=10, label="episodes")
    if "termination_reason" in df.columns:
        # simple visual hint: color by termination_reason
        tl_mask = df["termination_reason"] == "time_limit"
        plt.scatter(df.loc[tl_mask, "steps"], reward_sum[tl_mask], alpha=0.6, s=12, label="time_limit")
    plt.grid(alpha=0.3)
    plt.xlabel("steps")
    plt.ylabel("reward_sum (=reward_mean*steps)")
    plt.title("reward_sum vs steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "08_reward_sum_vs_steps.png", dpi=160)
    plt.close()


def print_summary(df, reward_sum, roll_reward, window):
    def stats(name, series):
        s = series.dropna()
        mean = s.mean()
        med = s.median()
        roll_last = rolling(series, window).dropna().iloc[-1] if not rolling(series, window).dropna().empty else np.nan
        print(f"{name}: mean={mean:.4f} med={med:.4f} roll_last={roll_last:.4f}")

    print(f"run_dir={args.run_dir}")
    print(f"window={args.window}, short_th={args.short_th}, pclip={args.pclip}")
    stats("reward_sum", reward_sum)
    stats("success_rate_end", df["success_rate_end"])
    stats("deadline_miss_rate", df["deadline_miss_rate"])
    stats("time_limit_rate", df["time_limit_rate"])
    if "decision_frac_v2v" in df:
        stats("decision_frac_v2v", df["decision_frac_v2v"])
    if "v2v_beats_rsu_rate" in df:
        stats("v2v_beats_rsu_rate", df["v2v_beats_rsu_rate"])
    if "approx_kl" in df:
        stats("approx_kl", df["approx_kl"])
    if "clip_frac" in df:
        stats("clip_frac", df["clip_frac"])


def main():
    parser = argparse.ArgumentParser(description="Plot key metrics for v4 runs")
    parser.add_argument("--run-dir", required=True, help="Path to run directory (runs/<id>)")
    parser.add_argument("--window", type=int, default=15, help="Rolling window")
    parser.add_argument("--short-th", type=int, default=150, help="Threshold for short episode in histogram")
    parser.add_argument("--pclip", type=float, default=0.99, help="Quantile for clipping heavy-tailed losses")
    global args
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = find_metrics(run_dir)
    df = pd.read_csv(metrics_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"metrics.csv missing required columns: {missing}")
    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    reward_sum, roll_reward = plot_reward_sum(df, args.window, out_dir)
    plot_success_miss(df, args.window, out_dir)
    plot_actions(df, args.window, out_dir)
    plot_v2v_costs(df, args.window, out_dir)
    plot_ppo(df, args.window, out_dir)
    plot_value_loss(df, args.window, args.pclip, out_dir)
    plot_steps_hist(df, args.short_th, out_dir)
    plot_reward_sum_vs_steps(df, out_dir)

    print_summary(df, reward_sum, roll_reward, args.window)


if __name__ == "__main__":
    main()
