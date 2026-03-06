"""
[RC1论文绘图] plot_rc1_ablation_cn.py

基于若干 evaluation-only（不重训）的开关结果，绘制消融对比图（中文标注）。
注意：该图仅用于说明“模块开关对推理阶段行为/指标的影响”，并不等价于训练期消融。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_one(summary_csv: Path) -> pd.Series:
    df = pd.read_csv(summary_csv)
    # expect single row for policy=mappo
    if len(df) != 1:
        df = df[df["policy"] == "mappo"]
    if df.empty:
        raise ValueError(f"No mappo row in {summary_csv}")
    return df.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-summary", type=str, required=True)
    ap.add_argument("--no-pbca-summary", type=str, required=True)
    ap.add_argument("--fixed-power-summary", type=str, required=True)
    ap.add_argument("--no-trust-summary", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    _set_cn_style()
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    items: List[Tuple[str, pd.Series]] = [
        ("完整模型", _load_one(Path(args.full_summary))),
        ("去PBCA(物理偏置)", _load_one(Path(args.no_pbca_summary))),
        ("固定功率", _load_one(Path(args.fixed_power_summary))),
        ("去信誉/风险输入", _load_one(Path(args.no_trust_summary))),
    ]

    labels = [x[0] for x in items]
    sr = np.array([float(x[1]["task_success_rate_mean"]) for x in items])
    cft = np.array([float(x[1]["mean_cft_est_mean"]) for x in items])
    risk = np.array([float(x[1]["risk_penalty_mean_mean"]) for x in items])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    x = np.arange(len(labels))

    axes[0].bar(x, sr, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("成功率")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, cft, alpha=0.85, color="#d55e00")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("平均完工时间（估计）/s")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, risk, alpha=0.85, color="#0072b2")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_ylabel("平均风险代价")
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.suptitle("消融对比（评估期模块开关，不重训）")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_cn.png", dpi=220)
    plt.close(fig)

    print(f"✓ Saved: {out_dir / 'fig_ablation_cn.png'}")


if __name__ == "__main__":
    main()

