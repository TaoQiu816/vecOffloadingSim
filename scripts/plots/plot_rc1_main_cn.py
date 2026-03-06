"""
[RC1论文绘图] plot_rc1_main_cn.py

基于主 run 的 logs/metrics.csv（训练/评估曲线）以及“统一评估CSV”（mappo+baselines），
输出论文可用的中文图表：
  - 收敛曲线：成功率 / 完工时间 / 风险代价
  - 与基线对比：柱状图（均值±标准差）
  - 行为特征：Local/RSU/V2V 占比
  - 功率统计：远端功率比例分布（箱线/直方图二选一）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def _set_cn_style():
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.size"] = 11


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=max(1, window // 3)).mean().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--eval-episode-csv", type=str, default=None)
    ap.add_argument("--eval-summary-csv", type=str, default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--smooth-window", type=int, default=10)
    args = ap.parse_args()

    _set_cn_style()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    metrics_csv = run_dir / "logs" / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        ep = df["episode"].to_numpy()

        # 1) 成功率/完工时间/风险代价收敛曲线
        fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.4), sharex=True)

        y1 = _rolling_mean(df["task_success_rate"].to_numpy(dtype=float), args.smooth_window)
        axes[0].plot(ep, y1, linewidth=1.8)
        axes[0].set_ylabel("按时完成率 / 成功率")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(True, alpha=0.25)

        y2 = _rolling_mean(df["mean_cft_est"].to_numpy(dtype=float), args.smooth_window)
        axes[1].plot(ep, y2, linewidth=1.8, color="#d55e00")
        axes[1].set_ylabel("平均完工时间（估计）/s")
        axes[1].grid(True, alpha=0.25)

        y3 = _rolling_mean(df["risk_penalty_mean"].to_numpy(dtype=float), args.smooth_window)
        axes[2].plot(ep, y3, linewidth=1.8, color="#0072b2")
        axes[2].set_ylabel("平均风险代价")
        axes[2].set_xlabel("训练轮次 / Episode")
        axes[2].grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_dir / "fig_main_convergence_cn.png", dpi=220)
        plt.close(fig)

    # Baseline comparison plots (from unified eval CSV)
    if args.eval_summary_csv:
        s = pd.read_csv(Path(args.eval_summary_csv).resolve())
        # Expect sweep="main"
        if "sweep" in s.columns:
            s = s[s["sweep"] == "main"].copy()
        if not s.empty:
            order = list(s.sort_values("task_success_rate_mean", ascending=False)["policy"].to_list())

            def bar_metric(metric: str, ylabel: str, fname: str):
                fig, ax = plt.subplots(figsize=(6.6, 3.8))
                vals = s.set_index("policy").loc[order]
                y = vals[f"{metric}_mean"].to_numpy(dtype=float)
                yerr = vals[f"{metric}_std"].to_numpy(dtype=float)
                ax.bar(range(len(order)), y, yerr=yerr, capsize=3, alpha=0.85)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=20, ha="right")
                ax.set_ylabel(ylabel)
                ax.grid(True, axis="y", alpha=0.25)
                fig.tight_layout()
                fig.savefig(out_dir / fname, dpi=220)
                plt.close(fig)

            bar_metric("task_success_rate", "成功率", "fig_baseline_success_cn.png")
            bar_metric("mean_cft_est", "平均完工时间（估计）/s", "fig_baseline_cft_cn.png")
            bar_metric("risk_penalty_mean", "平均风险代价", "fig_baseline_risk_cn.png")

            # Behavior (stacked Local/RSU/V2V)
            fig, ax = plt.subplots(figsize=(6.6, 3.8))
            vals = s.set_index("policy").loc[order]
            local = vals["decision_frac_local_mean"].to_numpy(dtype=float)
            rsu = vals["decision_frac_rsu_mean"].to_numpy(dtype=float)
            v2v = vals["decision_frac_v2v_mean"].to_numpy(dtype=float)
            x = np.arange(len(order))
            ax.bar(x, local, label="Local", alpha=0.9)
            ax.bar(x, rsu, bottom=local, label="RSU", alpha=0.9)
            ax.bar(x, v2v, bottom=local + rsu, label="V2V", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=20, ha="right")
            ax.set_ylabel("决策占比")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(ncol=3, frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_behavior_ratio_cn.png", dpi=220)
            plt.close(fig)

    if args.eval_episode_csv:
        e = pd.read_csv(Path(args.eval_episode_csv).resolve())
        if "sweep" in e.columns:
            e = e[e["sweep"] == "main"].copy()
        # Power distribution (remote decisions) – use per-episode mean as a proxy
        if not e.empty and "power_ratio_mean" in e.columns:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            # compare only a few policies to keep plot clean
            keep = [p for p in ["mappo", "Greedy", "EFT"] if p in set(e["policy"].unique())]
            data = [e.loc[e["policy"] == p, "power_ratio_mean"].dropna().to_numpy(dtype=float) for p in keep]
            ax.boxplot(data, labels=keep, showfliers=False)
            ax.set_ylabel("远端功率比例（每回合均值）")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_power_box_cn.png", dpi=220)
            plt.close(fig)

    print(f"✓ Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

