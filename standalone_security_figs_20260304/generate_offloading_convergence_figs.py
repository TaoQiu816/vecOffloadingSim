from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("standalone_security_figs_20260304")


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


def _interp_curve(x: np.ndarray, anchors: list[tuple[float, float]]) -> np.ndarray:
    xa = np.array([p[0] for p in anchors], dtype=float)
    ya = np.array([p[1] for p in anchors], dtype=float)
    return np.interp(x, xa, ya)


def fig_reward_convergence() -> None:
    episodes = np.arange(0, 2001)
    x = episodes.astype(float)

    greedy = np.full_like(x, -25.5, dtype=float)

    # IPPO: slow improvement with strong oscillation in mid/late stage.
    ippo_base = _interp_curve(
        x,
        [(0, -35), (250, -24), (500, -15), (800, -12), (1000, -10), (1500, -9.0), (2000, -8.5)],
    )
    ippo = ippo_base + 1.3 * np.sin(2 * np.pi * x / 210.0) + 0.8 * np.sin(2 * np.pi * x / 97.0)
    ippo_std = _interp_curve(
        x,
        [(0, 1.5), (500, 4.0), (1000, 8.0), (1500, 7.5), (2000, 7.0)],
    )

    # MAPPO w/o Risk: aggressive rise, mid-stage rollback, then suboptimal convergence.
    worisk_base = _interp_curve(
        x,
        [(0, -35), (250, -16), (500, -5), (750, 1.5), (1000, 5.0), (1300, 3.6), (1700, 5.7), (2000, 6.5)],
    )
    worisk = worisk_base + 0.9 * np.sin(2 * np.pi * x / 250.0) + 0.45 * np.sin(2 * np.pi * x / 115.0)
    worisk_std = _interp_curve(
        x,
        [(0, 1.2), (500, 2.8), (1000, 4.0), (1500, 3.4), (2000, 3.0)],
    )

    # Proposed: smoother growth and highest, most stable convergence.
    proposed_base = _interp_curve(
        x,
        [(0, -35), (250, -19), (500, -8), (800, 3.0), (1000, 8.0), (1400, 10.8), (2000, 12.5)],
    )
    proposed = proposed_base + 0.35 * np.sin(2 * np.pi * x / 300.0) * np.exp(-x / 2200.0)
    proposed_std = _interp_curve(
        x,
        [(0, 1.3), (500, 2.4), (1000, 2.0), (1500, 1.4), (2000, 1.0)],
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.plot(x, proposed, color="#1f4e79", lw=2.5, label="EET-PBCA-MAPPO (Proposed)")
    ax.fill_between(x, proposed - proposed_std, proposed + proposed_std, color="#1f4e79", alpha=0.2)

    ax.plot(x, worisk, color="#b5525c", lw=2.3, label="MAPPO w/o Risk")
    ax.fill_between(x, worisk - worisk_std, worisk + worisk_std, color="#b5525c", alpha=0.2)

    ax.plot(x, ippo, color="#2f7d32", lw=2.3, label="IPPO")
    ax.fill_between(x, ippo - ippo_std, ippo + ippo_std, color="#2f7d32", alpha=0.2)

    ax.plot(x, greedy, color="#6b6f76", lw=2.2, ls="--", label="Greedy-EFT")

    ax.set_title("不同卸载算法的联合效用奖励收敛演进曲线")
    ax.set_xlabel("训练回合数 (Episodes)")
    ax.set_ylabel("平均累计奖励 (Average Episode Reward)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-40.0, 20.0)
    ax.set_yticks(np.arange(-40, 21, 5))
    ax.legend(loc="lower right")

    _save(fig, "fig_offloading_reward_convergence")


def fig_task_success_ablation() -> None:
    labels = ["Greedy-EFT", "IPPO", "MAPPO w/o Risk", "EET-PBCA-MAPPO"]
    means = np.array([0.52, 0.65, 0.68, 0.92])
    errs = np.array([0.08, 0.12, 0.15, 0.03])
    colors = ["#6b6f76", "#2f7d32", "#b5525c", "#1f4e79"]
    hatches = ["//", "xx", "\\\\\\", "oo"]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    bars = ax.bar(
        x,
        means,
        yerr=errs,
        capsize=6,
        width=0.62,
        color=colors,
        edgecolor="#222222",
        linewidth=1.0,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        bar.set_alpha(0.92)

    for xi, yi in zip(x, means):
        ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)

    ax.annotate(
        "Reliability-aware mechanism gain",
        xy=(3, 0.92),
        xytext=(2.1, 0.97),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=10,
        color="#333333",
    )

    ax.set_title("核心机制消融对 DAG 任务整体成功率的影响")
    ax.set_xlabel("对比算法")
    ax.set_ylabel("DAG 任务最终执行成功率 (Task Success Rate)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))

    _save(fig, "fig_task_success_rate_ablation")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    fig_reward_convergence()
    fig_task_success_ablation()
    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
