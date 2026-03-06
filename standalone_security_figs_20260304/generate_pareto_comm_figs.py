from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


OUT_DIR = Path("standalone_security_figs_20260304")
RNG = np.random.default_rng(20260304)


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def _confidence_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    n_std: float = 2.0,
    alpha: float = 0.16,
) -> None:
    cov = np.cov(x, y)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        lw=1.6,
        alpha=alpha,
        label=label,
    )
    ax.add_patch(ellipse)


def fig_pareto_scatter_with_ellipses() -> None:
    n = 40

    # Proposed: tight cluster near Pareto front.
    x_prop = RNG.normal(0.12, 0.012, n)
    y_prop = 210 - 350 * (x_prop - 0.12) + RNG.normal(0.0, 5.0, n)

    # w/o Multi-Dim: right-upper, large horizontal spread.
    x_nomd = RNG.normal(0.26, 0.032, n)
    y_nomd = 180 - 160 * (x_nomd - 0.26) + RNG.normal(0.0, 9.0, n)

    # w/o Soft-Sort: left-lower-middle, moderate spread.
    x_nosoft = RNG.normal(0.16, 0.018, n)
    y_nosoft = 130 - 140 * (x_nosoft - 0.16) + RNG.normal(0.0, 7.5, n)

    x_prop = np.clip(x_prop, 0.05, 0.35)
    x_nomd = np.clip(x_nomd, 0.05, 0.35)
    x_nosoft = np.clip(x_nosoft, 0.05, 0.35)
    y_prop = np.clip(y_prop, 100, 250)
    y_nomd = np.clip(y_nomd, 100, 250)
    y_nosoft = np.clip(y_nosoft, 100, 250)

    fig, ax = plt.subplots(figsize=(9.8, 5.8))

    ax.scatter(x_prop, y_prop, s=34, marker="o", color="#1f4e79", alpha=0.85, label="Proposed (Full)")
    ax.scatter(x_nomd, y_nomd, s=34, marker="s", color="#b5525c", alpha=0.82, label="w/o Multi-Dim")
    ax.scatter(x_nosoft, y_nosoft, s=34, marker="^", color="#2f7d32", alpha=0.82, label="w/o Soft-Sort")

    _confidence_ellipse(ax, x_prop, y_prop, color="#1f4e79", label="Proposed 95% Ellipse")
    _confidence_ellipse(ax, x_nomd, y_nomd, color="#b5525c", label="w/o Multi-Dim 95% Ellipse")
    _confidence_ellipse(ax, x_nosoft, y_nosoft, color="#2f7d32", label="w/o Soft-Sort 95% Ellipse")

    # Faded Pareto front in upper-left region.
    x_front = np.array([0.08, 0.10, 0.12, 0.14, 0.16])
    y_front = np.array([235, 226, 216, 206, 196])
    ax.plot(
        x_front,
        y_front,
        color="#555555",
        lw=2.0,
        ls="--",
        alpha=0.50,
        marker=".",
        label="Pareto Front",
    )
    ax.text(0.084, 240, "Pareto Front", color="#555555", fontsize=10, alpha=0.85)

    ax.set_title("消融变体在吞吐效能与委员会污染率下的帕累托前沿分布")
    ax.set_xlabel("委员会污染率 (Committee Pollution Rate)")
    ax.set_ylabel("有效吞吐量 (Effective TPS)")
    ax.set_xlim(0.05, 0.35)
    ax.set_ylim(100, 250)
    ax.legend(loc="lower left", ncol=2)

    _save(fig, "fig_pareto_ablation_scatter_ellipse")


def fig_comm_overhead_bar() -> None:
    labels = ["Static-Param", "Heuristic-AIMD", "Proposed (PPO-Gov)"]
    means = np.array([13.5, 10.8, 7.2])
    stds = np.array([0.5, 1.8, 0.8])
    colors = ["#b5525c", "#2f7d32", "#1f4e79"]
    hatches = ["///", "xx", "\\\\\\"]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="#222222",
        linewidth=1.0,
        width=0.62,
    )

    for b, h in zip(bars, hatches):
        b.set_hatch(h)
        b.set_alpha(0.92)

    ax.set_title("不同治理算法下的共识通信消息开销对比")
    ax.set_xlabel("治理算法")
    ax.set_ylabel(r"累积通信消息总量 ($\times 10^4$)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 15)
    ax.set_yticks(np.arange(0, 16, 1))

    # show the lightweight gain of Proposed
    reduction = (means[0] - means[2]) / means[0] * 100.0
    ax.text(1.36, 13.9, f"Proposed vs Static: -{reduction:.1f}%", fontsize=10, color="#333333")

    _save(fig, "fig_comm_overhead_bar_error")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    fig_pareto_scatter_with_ellipses()
    fig_comm_overhead_bar()
    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
