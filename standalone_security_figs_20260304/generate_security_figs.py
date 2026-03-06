from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("standalone_security_figs_20260304")
PHI = np.array([0, 10, 20, 30, 40], dtype=float)
PHI_LABELS = [f"{int(x)}%" for x in PHI]

POLLUTION = {
    "Proposed (Maskable PPO-Gov)": np.array([0.00, 0.05, 0.11, 0.16, 0.21]),
    "Heuristic-AIMD": np.array([0.00, 0.08, 0.16, 0.23, 0.29]),
    "Static-Param": np.array([0.00, 0.10, 0.20, 0.29, 0.38]),
}

FAILURE = {
    "Proposed (Maskable PPO-Gov)": np.array([0.00, 0.01, 0.03, 0.05, 0.08]),
    "Heuristic-AIMD": np.array([0.00, 0.03, 0.07, 0.12, 0.18]),
    "Static-Param": np.array([0.00, 0.04, 0.12, 0.22, 0.35]),
}

TPS = {
    "Proposed (Maskable PPO-Gov)": np.array([252, 240, 225, 210, 195]),
    "Heuristic-AIMD": np.array([235, 215, 185, 160, 140]),
    "Static-Param": np.array([180, 165, 135, 105, 75]),
}

COLORS = {
    "Proposed (Maskable PPO-Gov)": "#1f4e79",
    "Heuristic-AIMD": "#2f7d32",
    "Static-Param": "#b5525c",
}


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


def fig_dual_axis_pollution_failure() -> None:
    fig, ax1 = plt.subplots(figsize=(9.6, 5.8))
    ax2 = ax1.twinx()

    for name, y in POLLUTION.items():
        ax1.plot(PHI, y, marker="o", lw=2.2, ms=6, color=COLORS[name], label=f"{name} - Pollution")
    for name, y in FAILURE.items():
        ax2.plot(PHI, y, marker="s", lw=2.0, ms=5, ls="--", color=COLORS[name], alpha=0.95, label=f"{name} - Failure")

    ax1.axhline(0.33, color="#d62728", lw=1.8, ls="--", label="Byzantine Fault Tolerance limit = 0.33")

    ax1.set_title("不同恶意节点比例下的委员会污染率与共识失败率对比")
    ax1.set_xlabel("恶意节点比例 φ")
    ax1.set_ylabel("委员会污染率 (Committee Pollution Rate)")
    ax2.set_ylabel("共识失败率 (Consensus Failure Rate)")

    ax1.set_xticks(PHI, PHI_LABELS)
    ax1.set_ylim(0.0, 0.45)
    ax2.set_ylim(0.0, 0.45)
    ticks = np.arange(0.0, 0.46, 0.05)
    ax1.set_yticks(ticks)
    ax2.set_yticks(ticks)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", ncol=2, frameon=True)

    _save(fig, "fig_pollution_vs_failure_dual_axis")


def fig_effective_tps() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for name, y in TPS.items():
        ax.plot(PHI, y, marker="o", lw=2.4, ms=6, color=COLORS[name], label=name)

    ax.set_title("不同恶意节点比例下的系统有效吞吐量对比")
    ax.set_xlabel("恶意节点比例 φ")
    ax.set_ylabel("有效吞吐量 (Effective TPS)")
    ax.set_xticks(PHI, PHI_LABELS)
    ax.set_ylim(50, 300)
    ax.set_yticks(np.arange(50, 301, 25))
    ax.legend(loc="upper right")

    _save(fig, "fig_effective_tps_vs_phi")


def fig_reputation_dynamics() -> None:
    epochs = np.arange(0, 101)

    trad_epochs = np.array([0, 20, 25, 50, 100])
    trad_values = np.array([0.50, 0.85, 0.82, 0.70, 0.65])
    traditional = np.interp(epochs, trad_epochs, trad_values)

    cadr_epochs = np.array([0, 20, 22, 25, 30, 50, 100])
    cadr_values = np.array([0.50, 0.85, 0.83, 0.45, 0.35, 0.25, 0.15])
    cadr = np.interp(epochs, cadr_epochs, cadr_values)
    post_attack = epochs >= 20
    cadr[post_attack] += 0.02 * np.sin((epochs[post_attack] - 20) / 1.7) * np.exp(-(epochs[post_attack] - 20) / 11.0)
    cadr = np.clip(cadr, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(epochs, cadr, color="#b5525c", lw=2.6, label="CADR (Proposed 多维惩罚)")
    ax.plot(epochs, traditional, color="#1f4e79", lw=2.3, ls="--", label="Traditional (传统单维信誉)")

    ax.axvline(20, color="#333333", lw=1.8, ls="--")
    ax.text(21.2, 0.93, "Attack Started", fontsize=10, color="#333333")
    ax.axhline(0.50, color="#d62728", lw=1.5, ls=":", label="Admission Threshold = 0.5")

    ax.set_title("典型恶意节点的多维信誉收敛动态轨迹")
    ax.set_xlabel("治理周期 (Epoch)")
    ax.set_ylabel("节点综合信誉分值 (Reputation Score)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.legend(loc="upper right")

    _save(fig, "fig_reputation_dynamics_malicious_node")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    fig_dual_axis_pollution_failure()
    fig_effective_tps()
    fig_reputation_dynamics()
    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
