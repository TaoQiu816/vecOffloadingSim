from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None


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


def fig_scalability_delay_energy_dual_axis() -> None:
    n = np.array([5, 10, 15, 20, 25], dtype=float)

    makespan_greedy = np.array([0.75, 1.20, 2.10, 3.20, 3.95])
    makespan_ippo = np.array([0.80, 1.10, 1.65, 2.15, 2.60])
    makespan_prop = np.array([0.70, 0.95, 1.20, 1.45, 1.65])

    energy_greedy = np.array([12.5, 18.0, 26.5, 35.0, 42.0])
    energy_ippo = np.array([13.0, 16.5, 22.0, 26.5, 31.0])
    energy_prop = np.array([11.5, 14.0, 16.5, 18.0, 19.5])

    colors = {
        "prop": "#1f4e79",   # deep blue
        "ippo": "#2f7d32",   # forest green
        "greedy": "#b5525c",  # brick red
    }

    fig, ax1 = plt.subplots(figsize=(10.2, 5.9))
    ax2 = ax1.twinx()

    # Left axis: makespan (solid)
    l1 = ax1.plot(n, makespan_prop, color=colors["prop"], lw=2.6, marker="o", ms=6, label="Proposed Makespan (Left Y)")
    l2 = ax1.plot(n, makespan_ippo, color=colors["ippo"], lw=2.4, marker="s", ms=6, label="IPPO Makespan (Left Y)")
    l3 = ax1.plot(n, makespan_greedy, color=colors["greedy"], lw=2.5, marker="^", ms=6, label="Greedy-EFT Makespan (Left Y)")

    # Right axis: energy (dashed + marker)
    l4 = ax2.plot(
        n,
        energy_prop,
        color=colors["prop"],
        lw=2.2,
        ls="--",
        marker="D",
        ms=5,
        label="Proposed Energy (Right Y)",
    )
    l5 = ax2.plot(
        n,
        energy_ippo,
        color=colors["ippo"],
        lw=2.1,
        ls="--",
        marker="P",
        ms=5,
        label="IPPO Energy (Right Y)",
    )
    l6 = ax2.plot(
        n,
        energy_greedy,
        color=colors["greedy"],
        lw=2.2,
        ls="--",
        marker="X",
        ms=5,
        label="Greedy-EFT Energy (Right Y)",
    )

    ax1.set_title("不同车辆并发规模下的任务完成时延与能耗折中")
    ax1.set_xlabel(r"车辆并发规模 $N$ (Number of Vehicles)")
    ax1.set_ylabel("平均完成时延 (Average Makespan) [s]")
    ax2.set_ylabel("系统平均能耗 (Average Energy Consumption) [J]")

    ax1.set_xlim(5, 25)
    ax1.set_xticks(n)
    ax1.set_ylim(0.5, 4.5)
    ax1.set_yticks(np.arange(0.5, 4.51, 0.5))
    ax2.set_ylim(10.0, 45.0)
    ax2.set_yticks(np.arange(10, 46, 5))

    lines = l1 + l2 + l3 + l4 + l5 + l6
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", ncol=2, frameon=True)

    _save(fig, "fig_vehicle_scale_delay_energy_dual_axis")


def fig_action_distribution_stacked_area() -> None:
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    rsu = np.array([68, 55, 42, 30, 23], dtype=float)
    v2v = np.array([22, 28, 32, 36, 39], dtype=float)
    local = np.array([10, 17, 26, 34, 38], dtype=float)

    x_dense = np.linspace(x.min(), x.max(), 300)
    if PchipInterpolator is not None:
        rsu_dense = PchipInterpolator(x, rsu)(x_dense)
        v2v_dense = PchipInterpolator(x, v2v)(x_dense)
        local_dense = PchipInterpolator(x, local)(x_dense)
    else:
        rsu_dense = np.interp(x_dense, x, rsu)
        v2v_dense = np.interp(x_dense, x, v2v)
        local_dense = np.interp(x_dense, x, local)

    total = rsu_dense + v2v_dense + local_dense
    rsu_dense = 100.0 * rsu_dense / total
    v2v_dense = 100.0 * v2v_dense / total
    local_dense = 100.0 * local_dense / total

    colors = ["#1f4e79", "#8fb9dd", "#d5d9de"]  # deep blue, light blue, light gray

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.stackplot(
        x_dense,
        rsu_dense,
        v2v_dense,
        local_dense,
        labels=["边缘卸载 (RSU Offloading)", "侧行协作 (V2V Collaboration)", "本地执行 (Local Execution)"],
        colors=colors,
        alpha=0.95,
    )

    # Overlay anchor points to emphasize provided percentages.
    ax.scatter(x, rsu, color="#11395b", s=24, zorder=5)
    ax.scatter(x, rsu + v2v, color="#5d92c0", s=24, zorder=5)

    ax.set_title("不同任务数据量下的多智能体卸载动作分布演化")
    ax.set_xlabel("子任务平均数据量 (Task Data Size) [MB]")
    ax.set_ylabel("动作决策占比 (Action Distribution Proportion) [%]")
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(loc="upper right")

    _save(fig, "fig_action_distribution_stacked_area")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    fig_scalability_delay_energy_dual_axis()
    fig_action_distribution_stacked_area()
    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
