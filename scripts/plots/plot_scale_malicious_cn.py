"""
[RC1论文绘图] plot_scale_malicious_cn.py

读取 sweep_eval_scale_malicious.py 输出的 summary CSV，
绘制“不同规模 U”与“不同恶意比例”下的核心指标对比图（中文标注）。
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


def _plot_with_errorbar(ax, x, y, yerr, label, marker="o"):
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        label=label,
        marker=marker,
        capsize=3,
        linewidth=1.6,
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--policies", type=str, default="mappo,Greedy,EFT,LB-Greedy,Local-Only")
    args = ap.parse_args()

    _set_cn_style()

    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    df = pd.read_csv(summary_csv)
    wanted = [x.strip() for x in (args.policies or "").split(",") if x.strip()]
    if wanted:
        df = df[df["policy"].isin(wanted)]

    def plot_sweep(sweep: str, xlabel: str, x_cast=float, xfmt=None, tag=""):
        d = df[df["sweep"] == sweep].copy()
        if d.empty:
            return
        d["x"] = d["value"].map(x_cast)
        d = d.sort_values(["policy", "x"])

        # 1) 成功率
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for p, g in d.groupby("policy"):
            _plot_with_errorbar(
                ax,
                g["x"].to_numpy(),
                g["task_success_rate_mean"].to_numpy(),
                g["task_success_rate_std"].to_numpy(),
                label=p,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("按时完成率 / 成功率")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{tag}{sweep}_success_cn.png", dpi=200)
        plt.close(fig)

        # 2) 完工时间（估计）
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for p, g in d.groupby("policy"):
            _plot_with_errorbar(
                ax,
                g["x"].to_numpy(),
                g["mean_cft_est_mean"].to_numpy(),
                g["mean_cft_est_std"].to_numpy(),
                label=p,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("平均完工时间（估计）/s")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{tag}{sweep}_cft_cn.png", dpi=200)
        plt.close(fig)

        # 3) 风险代价
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for p, g in d.groupby("policy"):
            _plot_with_errorbar(
                ax,
                g["x"].to_numpy(),
                g["risk_penalty_mean_mean"].to_numpy(),
                g["risk_penalty_mean_std"].to_numpy(),
                label=p,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("平均风险代价")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{tag}{sweep}_risk_cn.png", dpi=200)
        plt.close(fig)

        # 4) 行为：远端比例（1-local）
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for p, g in d.groupby("policy"):
            remote = 1.0 - g["decision_frac_local_mean"].to_numpy()
            remote_std = g["decision_frac_local_std"].to_numpy()
            _plot_with_errorbar(ax, g["x"].to_numpy(), remote, remote_std, label=p)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("远端卸载比例")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{tag}{sweep}_remote_cn.png", dpi=200)
        plt.close(fig)

    plot_sweep("scale", "车辆数量 $U$", x_cast=int, tag="")
    plot_sweep("malicious", "恶意邻车比例", x_cast=float, tag="")

    print(f"✓ Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

