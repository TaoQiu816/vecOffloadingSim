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


def _interp_curve(epochs: np.ndarray, anchors: list[tuple[int, float]]) -> np.ndarray:
    xs = np.array([p[0] for p in anchors], dtype=float)
    ys = np.array([p[1] for p in anchors], dtype=float)
    return np.interp(epochs, xs, ys)


def fig_mempool_backlog_dynamics() -> None:
    epochs = np.arange(0, 151)

    proposed = _interp_curve(
        epochs,
        [(0, 0), (30, 10), (40, 12), (50, 150), (60, 60), (65, 5), (70, 20), (100, 15), (150, 10)],
    )
    heuristic = _interp_curve(
        epochs,
        [(0, 0), (30, 10), (40, 15), (50, 300), (70, 450), (100, 200), (120, 50), (150, 50)],
    )
    static = _interp_curve(
        epochs,
        [(0, 0), (30, 10), (40, 12), (50, 200), (70, 400), (100, 700), (150, 950)],
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.axvspan(40, 150, color="#d9d9d9", alpha=0.2)
    ax.text(86, 930, "High Load Region", color="#555555", fontsize=10, ha="center")
    ax.axvline(40, color="#404040", lw=1.6, ls="--")
    ax.text(42, 875, "Burst Load Triggered", color="#404040", fontsize=10)

    ax.plot(epochs, proposed, color="#1f4e79", lw=2.4, marker="o", markevery=10, ms=4, label="Proposed (Maskable PPO-Gov)")
    ax.plot(epochs, heuristic, color="#2f7d32", lw=2.3, marker="s", markevery=10, ms=4, label="Heuristic-AIMD")
    ax.plot(epochs, static, color="#b5525c", lw=2.5, marker="^", markevery=10, ms=4, label="Static-Param")

    ax.set_title("突发阶跃负载下的内存池队列积压 (Mempool Backlog) 动态演进")
    ax.set_xlabel("治理周期 (Epoch)")
    ax.set_ylabel(r"队列积压量 $Q_e$ (Transactions)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1000)
    ax.set_yticks(np.arange(0, 1001, 100))
    ax.legend(loc="upper left")

    _save(fig, "fig_mempool_backlog_burst_load")


def fig_proposed_param_step_dynamics() -> None:
    epochs = np.arange(0, 151)

    block_capacity = np.full_like(epochs, 50, dtype=float)
    block_capacity[epochs >= 42] = 150
    block_capacity[epochs >= 45] = 250
    block_capacity[epochs >= 95] = 200

    timeout = np.full_like(epochs, 1.5, dtype=float)
    timeout[epochs >= 41] = 1.0
    timeout[epochs >= 45] = 0.5
    timeout[epochs >= 95] = 1.0

    fig, ax1 = plt.subplots(figsize=(10.0, 5.8))
    ax2 = ax1.twinx()

    l1 = ax1.step(epochs, block_capacity, where="post", color="#1f4e79", lw=2.6, label=r"Block Capacity $b_e$ (Tx/Block)")
    l2 = ax2.step(epochs, timeout, where="post", color="#b5525c", lw=2.4, ls="--", label=r"Batch Timeout $\tau_e$ (Seconds)")

    ax1.axvline(40, color="#404040", lw=1.4, ls=":")
    ax1.text(42, 240, "Burst Load Triggered", color="#404040", fontsize=10)

    ax1.set_title("Proposed 算法在突发负载下的区块参数自适应演进轨迹")
    ax1.set_xlabel("治理周期 (Epoch)")
    ax1.set_ylabel(r"区块容量 $b_e$ (Tx/Block)")
    ax2.set_ylabel(r"批处理超时 $\tau_e$ (Seconds)")

    ax1.set_xlim(0, 150)
    ax1.set_ylim(50, 250)
    ax2.set_ylim(0.5, 2.0)

    ax1.set_yticks([50, 100, 150, 200, 250])
    ax2.set_yticks([0.5, 1.0, 1.5, 2.0])

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    _save(fig, "fig_proposed_adaptive_params_step")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    fig_mempool_backlog_dynamics()
    fig_proposed_param_step_dynamics()
    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
