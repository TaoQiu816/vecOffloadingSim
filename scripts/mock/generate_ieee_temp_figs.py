"""
Temporary IEEE-style figure generator for progress checking.

This script synthesizes physically plausible expected data and draws 6 figures:
  - fig1_lr.pdf
  - fig2_ablation.pdf
  - fig3_scalability.pdf
  - fig4_delay_energy.pdf
  - fig5_action_distribution.pdf
  - fig6_pareto.pdf

Dependencies: matplotlib, numpy, scipy
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse
import glob


OUTPUT_DIR = Path("mock_runs/ieee_temp_figs")
OUTPUT_FORMAT = "png"
OUTPUT_DPI = 300
REF_METRICS_PATH: str | None = None
REFERENCE_CANDIDATES = [
    "runs/rc1_margin_w03_1000ep_20260225_184215/logs/metrics.csv",
    "runs/run_20260222_135030/logs/metrics.csv",
]

COLOR = {
    "proposed": "#1f4e79",   # 深海蓝
    "greedy": "#b5525c",     # 砖红
    "rsu": "#2f7d32",        # 森林绿
    "local": "#6b6f76",      # 深灰
    "violet": "#7a4fa3",     # 紫罗兰
    "gold": "#b08d2f",       # 暗金
}


def _save_current(fig_stem: str) -> None:
    out_path = OUTPUT_DIR / f"{fig_stem}.{OUTPUT_FORMAT}"
    plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight")


def set_ieee_style() -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.facecolor": "#F2F2F2",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
        }
    )


def _smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    return uniform_filter1d(x, size=w, mode="nearest")


def _asym_pulse(x: np.ndarray, center: float, width_left: float, width_right: float) -> np.ndarray:
    """Asymmetric pulse used to model drop-and-recovery events."""
    y = np.zeros_like(x, dtype=float)
    left = x <= center
    y[left] = np.exp(-0.5 * ((x[left] - center) / max(width_left, 1e-6)) ** 2)
    y[~left] = np.exp(-0.5 * ((x[~left] - center) / max(width_right, 1e-6)) ** 2)
    return y


def _gen_curve_runs(mean_curve: np.ndarray, std_curve: np.ndarray, n_runs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        noise = rng.normal(0.0, std_curve, size=mean_curve.shape[0])
        # Add low-frequency perturbation to emulate realistic training wobble.
        lowf = _smooth(rng.normal(0.0, 1.0, size=mean_curve.shape[0]), w=45) * 0.15
        y = mean_curve + noise + lowf
        runs.append(_smooth(y, w=11))
    return np.asarray(runs)


def _find_recent_reference_metrics() -> str | None:
    candidates = glob.glob("runs/**/logs/metrics.csv", recursive=True)
    scored = []
    for p in candidates:
        try:
            data = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if data.size < 500:
                continue
            cols = set(data.dtype.names or [])
            if not {"episode", "task_success_rate", "reward_mean"}.issubset(cols):
                continue
            reward = np.asarray(data["reward_mean"], dtype=float)
            success = np.asarray(data["task_success_rate"], dtype=float)
            length = int(reward.shape[0])
            if length < 500:
                continue
            # Favor 1000ep and non-degenerate curves.
            score = 0.0
            score += min(length, 2000) / 2000.0
            tail_n = min(100, length)
            score += float(np.mean(success[-tail_n:]))
            score += max(float(np.max(reward)), 0.0)
            score += 0.3 if "1000" in p else 0.0
            score += 0.15 if "main" in p or "margin_w03" in p else 0.0
            scored.append((score, p))
        except Exception:
            continue
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def _load_reference_curve(path: str | None, col: str, out_len: int) -> np.ndarray | None:
    ref_path = path or _find_recent_reference_metrics()
    if not ref_path:
        return None
    try:
        data = np.genfromtxt(ref_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        cols = set(data.dtype.names or [])
        if col not in cols or "episode" not in cols:
            return None
        episode = np.asarray(data["episode"], dtype=float)
        y = np.asarray(data[col], dtype=float)
        if len(y) < 20:
            return None
        # Keep temporal ordering stable in case the file is not strictly sorted.
        order = np.argsort(episode)
        y = y[order]
        y = _smooth(y, w=21)
        if len(y) != out_len:
            src = np.linspace(0.0, 1.0, len(y))
            tgt = np.linspace(0.0, 1.0, out_len)
            y = np.interp(tgt, src, y)
        return _smooth(y, w=13)
    except Exception:
        return None


def _normalize01(y: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(y, 5)), float(np.percentile(y, 95))
    if hi - lo < 1e-9:
        return np.zeros_like(y)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


def _blended_cadence(col: str, out_len: int) -> np.ndarray:
    paths = []
    if REF_METRICS_PATH:
        paths.append(REF_METRICS_PATH)
    for p in REFERENCE_CANDIDATES:
        if p not in paths:
            paths.append(p)

    cadences = []
    for p in paths:
        curve = _load_reference_curve(p, col=col, out_len=out_len)
        if curve is None:
            continue
        trend = _smooth(curve, w=121)
        dev = _smooth(curve - trend, w=17)
        dev_std = float(np.std(dev))
        if dev_std < 1e-9:
            continue
        cadences.append(dev / dev_std)

    if not cadences:
        x = np.arange(out_len, dtype=float)
        return np.sin(2 * np.pi * x / 240.0) * np.exp(-x / 1700.0)

    return np.mean(np.stack(cadences, axis=0), axis=0)


def _plot_with_raw_and_smooth(x: np.ndarray, smooth: np.ndarray, std: np.ndarray, color: str, label: str, seed: int = 0, ls: str = "-") -> None:
    rng = np.random.default_rng(seed)
    raw = smooth + rng.normal(0.0, std, size=smooth.shape[0])
    raw = _smooth(raw, w=3)
    plt.plot(x, raw, color=color, lw=1.0, alpha=0.18, ls=ls)
    plt.plot(x, smooth, color=color, lw=2.7, ls=ls, label=label)


def fig1_lr() -> None:
    episodes = np.arange(0, 1001)
    x = episodes.astype(float)

    # Real-curve-driven cadence from referenced runs, but with new trajectories.
    cadence = _blended_cadence(col="reward_mean", out_len=episodes.size)
    stable_gate = 1.0 - 0.75 / (1.0 + np.exp(-(x - 700.0) / 70.0))
    cadence_amp = (0.22 * np.exp(-x / 1200.0) + 0.035) * stable_gate

    # LR=5e-4: best tradeoff, fast rise and stable late stage.
    best = 1.12 - (1.12 + 0.78) * np.exp(-x / 95.0)
    best += 0.95 * cadence_amp * cadence
    best += 0.028 * np.sin(2 * np.pi * x / 290.0) * stable_gate
    best -= 0.42 * _asym_pulse(x, 210.0, 30.0, 45.0)
    best -= 0.38 * _asym_pulse(x, 520.0, 24.0, 34.0)
    best -= 0.74 * _asym_pulse(x, 675.0, 16.0, 22.0)
    best = _smooth(best, w=9)
    best_std = (0.24 * np.exp(-x / 350.0) + 0.08) * stable_gate + 0.035

    # LR=1e-3: faster but less stable, larger oscillation and sporadic drops.
    high = 1.05 - (1.05 + 0.70) * np.exp(-x / 72.0)
    high -= 0.14 / (1.0 + np.exp(-(x - 930.0) / 120.0))
    high += 1.20 * cadence_amp * cadence
    high += 0.065 * np.sin(2 * np.pi * x / 175.0) * stable_gate
    high -= 0.58 * _asym_pulse(x, 190.0, 22.0, 30.0)
    high -= 0.95 * _asym_pulse(x, 740.0, 18.0, 25.0)
    high -= 0.42 * _asym_pulse(x, 1380.0, 16.0, 32.0)
    high = _smooth(high, w=9)
    high_std = (0.30 * np.exp(-x / 280.0) + 0.11) * stable_gate + 0.045

    # LR=1e-4: slow but stable, underfitting at 2000 episodes.
    low = 0.82 - (0.82 + 0.62) * np.exp(-x / 370.0)
    low += 0.55 * cadence_amp * cadence
    low += 0.020 * np.sin(2 * np.pi * x / 260.0) * stable_gate
    low -= 0.20 * _asym_pulse(x, 650.0, 28.0, 44.0)
    low -= 0.30 * _asym_pulse(x, 1260.0, 24.0, 40.0)
    low = _smooth(low, w=9)
    low_std = (0.18 * np.exp(-x / 850.0) + 0.06) * stable_gate + 0.028

    runs_best = _gen_curve_runs(best, best_std, n_runs=8, seed=11)
    runs_high = _gen_curve_runs(high, high_std, n_runs=8, seed=22)
    runs_low = _gen_curve_runs(low, low_std, n_runs=8, seed=33)

    mean_best, std_best = runs_best.mean(axis=0), runs_best.std(axis=0)
    mean_high, std_high = runs_high.mean(axis=0), runs_high.std(axis=0)
    mean_low, std_low = runs_low.mean(axis=0), runs_low.std(axis=0)

    colors = {"best": COLOR["proposed"], "high": COLOR["greedy"], "low": COLOR["rsu"]}

    plt.figure(figsize=(7.2, 4.8))
    plt.title("Reward Convergence")
    plt.plot(episodes, mean_best, color=colors["best"], lw=2.8, ls="-", label="LR=5e-4 (Best)")
    plt.fill_between(episodes, mean_best - 0.9 * std_best, mean_best + 0.9 * std_best, color=colors["best"], alpha=0.22)
    plt.plot(episodes, mean_high, color=colors["high"], lw=2.6, ls="--", label="LR=1e-3 (Too Large)")
    plt.fill_between(episodes, mean_high - 0.8 * std_high, mean_high + 0.8 * std_high, color=colors["high"], alpha=0.12)
    plt.plot(episodes, mean_low, color=colors["low"], lw=2.6, ls="-.", label="LR=1e-4 (Too Small)")
    plt.fill_between(episodes, mean_low - 0.8 * std_low, mean_low + 0.8 * std_low, color=colors["low"], alpha=0.12)

    plt.xlabel("Episode")
    plt.ylabel("Reward (50-ep window)")
    plt.xlim(0, 1000)
    plt.ylim(-1.8, 1.7)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    _save_current("fig1_lr")
    plt.close()


def fig2_ablation() -> None:
    episodes = np.arange(0, 1001)
    x = episodes.astype(float)

    # Real-curve-driven cadence: preserve realistic training jitter style while keeping stable convergence.
    cadence = _blended_cadence(col="task_success_rate", out_len=episodes.size)
    stable_gate = 1.0 - 0.62 / (1.0 + np.exp(-(x - 700.0) / 75.0))
    cadence_amp = (0.085 * np.exp(-x / 1600.0) + 0.015) * stable_gate

    proposed = 0.86 - (0.86 - 0.06) * np.exp(-x / 145.0)
    proposed += 0.65 * cadence_amp * cadence
    proposed += 0.016 * np.sin(2 * np.pi * x / 310.0) * stable_gate
    proposed -= 0.16 * _asym_pulse(x, 355.0, 22.0, 34.0)
    proposed -= 0.22 * _asym_pulse(x, 515.0, 18.0, 24.0)
    proposed -= 0.28 * _asym_pulse(x, 680.0, 15.0, 22.0)
    proposed = np.clip(_smooth(proposed, w=7), 0.0, 1.0)
    proposed_std = 0.058 * np.exp(-x / 620.0) + 0.010

    # w/o Graph-Encoder: lower plateau, slower recovery, and different fluctuation cadence.
    no_graph = 0.52 - (0.52 - 0.05) * np.exp(-x / 170.0)
    no_graph += 0.022 * np.sin(2 * np.pi * x / 180.0) * np.exp(-x / 2000.0)
    no_graph += 0.010 * np.sin(2 * np.pi * x / 510.0) * stable_gate
    no_graph -= 0.30 * _asym_pulse(x, 355.0, 18.0, 28.0)
    no_graph -= 0.22 * _asym_pulse(x, 680.0, 18.0, 26.0)
    no_graph -= 0.10 * _asym_pulse(x, 560.0, 16.0, 24.0)
    no_graph -= 0.022 * (1.0 - np.exp(-x / 900.0))
    no_graph = np.clip(_smooth(no_graph, w=9), 0.0, 1.0)
    no_graph_std = 0.068 * np.exp(-x / 700.0) + 0.014

    # w/o Trust-Penalty: similar rise, but mid-late instability from risky selections.
    no_trust = 0.74 - (0.74 - 0.05) * np.exp(-x / 155.0)
    no_trust += 0.032 * np.sin(2 * np.pi * x / 210.0) * stable_gate
    no_trust += 0.010 * np.sin(2 * np.pi * x / 470.0) * stable_gate
    no_trust -= 0.26 * _asym_pulse(x, 350.0, 18.0, 30.0)
    no_trust -= 0.34 * _asym_pulse(x, 350.0, 18.0, 28.0)
    no_trust -= 0.58 * _asym_pulse(x, 680.0, 15.0, 24.0)
    # Irregular late-stage reliability shocks (non-periodic).
    no_trust -= 0.08 * _asym_pulse(x, 760.0, 18.0, 36.0)
    no_trust -= 0.07 * _asym_pulse(x, 860.0, 18.0, 34.0)
    no_trust = np.clip(_smooth(no_trust, w=9), 0.0, 1.0)
    no_trust_std = 0.074 * np.exp(-x / 650.0) + 0.016

    runs_p = _gen_curve_runs(proposed, proposed_std, n_runs=8, seed=44)
    runs_g = _gen_curve_runs(no_graph, no_graph_std, n_runs=8, seed=55)
    runs_t = _gen_curve_runs(no_trust, no_trust_std, n_runs=8, seed=66)

    mp, sp = np.clip(runs_p.mean(axis=0), 0, 1), runs_p.std(axis=0)
    mg, sg = np.clip(runs_g.mean(axis=0), 0, 1), runs_g.std(axis=0)
    mt, st = np.clip(runs_t.mean(axis=0), 0, 1), runs_t.std(axis=0)

    colors = {"p": COLOR["proposed"], "g": COLOR["violet"], "t": COLOR["greedy"]}

    plt.figure(figsize=(7.2, 4.8))
    plt.title("Task Success Convergence")
    _plot_with_raw_and_smooth(episodes, mp, 0.62 * sp, colors["p"], "Proposed", seed=404, ls="-")
    _plot_with_raw_and_smooth(episodes, mg, 0.62 * sg, colors["g"], "w/o Graph-Encoder", seed=505, ls="--")
    _plot_with_raw_and_smooth(episodes, mt, 0.62 * st, colors["t"], "w/o Trust-Penalty", seed=606, ls="-.")

    plt.xlabel("Episode")
    plt.ylabel("Task Success Rate (50-ep window)")
    plt.xlim(0, 1000)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    _save_current("fig2_ablation")
    plt.close()


def fig3_scalability() -> None:
    n = np.array([4, 8, 12, 16])

    proposed = np.array([0.92, 0.86, 0.78, 0.68])
    greedy = np.array([0.60, 0.48, 0.35, 0.25])
    rsu_pref = np.array([0.50, 0.40, 0.24, 0.08])
    local_only = np.array([0.06, 0.05, 0.05, 0.04])

    width = 0.18
    x = np.arange(len(n))

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - 1.5 * width, proposed, width=width, color=COLOR["proposed"], edgecolor="black", linewidth=0.4, label="Proposed (MAPPO-DAG)")
    plt.bar(x - 0.5 * width, greedy, width=width, color=COLOR["greedy"], edgecolor="black", linewidth=0.4, label="Greedy-EFT")
    plt.bar(x + 0.5 * width, rsu_pref, width=width, color=COLOR["rsu"], edgecolor="black", linewidth=0.4, label="RSU-Prefer")
    plt.bar(x + 1.5 * width, local_only, width=width, color=COLOR["local"], edgecolor="black", linewidth=0.4, label="Local-Only")

    plt.xticks(x, n)
    plt.xlabel("Number of Vehicles N")
    plt.ylabel("Task Success Rate")
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    _save_current("fig3_scalability")
    plt.close()


def fig4_delay_energy() -> None:
    algos = ["Proposed", "Greedy", "RSU-Prefer", "Local-Only"]
    makespan = np.array([3.2, 4.5, 5.8, 8.5])
    energy = np.array([5.8, 7.5, 4.0, 15.0])  # aligned with algo order

    x = np.arange(len(algos))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    bar_colors = [COLOR["proposed"], COLOR["greedy"], COLOR["rsu"], COLOR["local"]]
    ax1.bar(x, makespan, color=bar_colors, alpha=0.88, edgecolor="black", linewidth=0.4, label="Average Makespan")
    ax1.set_ylabel("Average Makespan (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(x, energy, color=COLOR["violet"], lw=2.2, marker="o", ms=6, linestyle="-", label="Average Energy")
    ax2.set_ylabel("Average Energy (J)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    _save_current("fig4_delay_energy")
    plt.close()


def fig5_action_distribution() -> None:
    data_mb = np.array([0.5, 1.0, 1.5, 2.0])
    local = np.array([10, 12, 16, 20], dtype=float)
    rsu = np.array([70, 55, 40, 25], dtype=float)
    v2v = np.array([20, 33, 44, 55], dtype=float)

    colors = {"local": COLOR["local"], "rsu": COLOR["proposed"], "v2v": COLOR["rsu"]}

    plt.figure(figsize=(7.2, 4.8))
    plt.stackplot(data_mb, local, rsu, v2v, colors=[colors["local"], colors["rsu"], colors["v2v"]],
                  alpha=0.9, labels=["Local", "RSU", "V2V"])
    plt.xlabel("Average Task Data Size (MB)")
    plt.ylabel("Action Ratio (%)")
    plt.ylim(0, 100)
    plt.xlim(0.5, 2.0)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    _save_current("fig5_action_distribution")
    plt.close()


def _pareto_front(points: np.ndarray) -> np.ndarray:
    """Return non-dominated points for minimization (delay, energy)."""
    # Sort by delay asc, then keep strictly decreasing energy envelope.
    pts = points[np.argsort(points[:, 0])]
    frontier = []
    best_energy = np.inf
    for p in pts:
        if p[1] < best_energy:
            frontier.append(p)
            best_energy = p[1]
    return np.asarray(frontier)


def fig6_pareto() -> None:
    rng = np.random.default_rng(2026)

    # Means and covariances for 4 clusters (80 points each).
    # Proposed: near lower-left Pareto region.
    proposed = rng.multivariate_normal(mean=[3.2, 5.8], cov=[[0.20, 0.10], [0.10, 0.60]], size=80)
    greedy = rng.multivariate_normal(mean=[5.2, 7.8], cov=[[0.45, 0.20], [0.20, 1.10]], size=80)
    rsu_pref = rng.multivariate_normal(mean=[6.4, 4.8], cov=[[0.40, 0.05], [0.05, 0.80]], size=80)
    local = rng.multivariate_normal(mean=[8.6, 15.0], cov=[[0.45, 0.20], [0.20, 1.50]], size=80)

    all_points = np.vstack([proposed, greedy, rsu_pref, local])
    all_points[:, 0] = np.clip(all_points[:, 0], 2.0, 10.0)
    all_points[:, 1] = np.clip(all_points[:, 1], 2.0, 18.0)

    frontier = _pareto_front(all_points)

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(proposed[:, 0], proposed[:, 1], s=28, alpha=0.6, c=COLOR["proposed"], marker="o", label="Proposed (MAPPO-DAG)")
    plt.scatter(greedy[:, 0], greedy[:, 1], s=28, alpha=0.6, c=COLOR["greedy"], marker="s", label="Greedy-EFT")
    plt.scatter(rsu_pref[:, 0], rsu_pref[:, 1], s=28, alpha=0.6, c=COLOR["rsu"], marker="^", label="RSU-Prefer")
    plt.scatter(local[:, 0], local[:, 1], s=28, alpha=0.6, c=COLOR["local"], marker="d", label="Local-Only")

    if frontier.shape[0] >= 2:
        plt.plot(frontier[:, 0], frontier[:, 1], "k--", lw=2.0, label="Pareto Front")

    plt.xlabel("Time Cost (Delay, s)")
    plt.ylabel("Energy Cost (J)")
    plt.xlim(2.0, 10.0)
    plt.ylim(2.0, 18.0)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    _save_current("fig6_pareto")
    plt.close()


def main() -> None:
    global OUTPUT_DIR, OUTPUT_FORMAT, OUTPUT_DPI, REF_METRICS_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR), help="Directory for output figures.")
    parser.add_argument("--format", type=str, default=OUTPUT_FORMAT, choices=["png", "jpg", "jpeg", "svg", "pdf"], help="Image format.")
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI, help="Output DPI.")
    parser.add_argument("--reference-metrics", type=str, default=None, help="Recent training metrics.csv used as convergence-shape template.")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.out_dir).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FORMAT = args.format.lower()
    OUTPUT_DPI = int(args.dpi)
    REF_METRICS_PATH = args.reference_metrics
    selected_ref = REF_METRICS_PATH or _find_recent_reference_metrics()
    if selected_ref is not None:
        REF_METRICS_PATH = selected_ref

    set_ieee_style()
    fig1_lr()
    fig2_ablation()
    fig3_scalability()
    fig4_delay_energy()
    fig5_action_distribution()
    fig6_pareto()
    print(
        f"Saved to {OUTPUT_DIR}: "
        f"fig1_lr.{OUTPUT_FORMAT}, fig2_ablation.{OUTPUT_FORMAT}, fig3_scalability.{OUTPUT_FORMAT}, "
        f"fig4_delay_energy.{OUTPUT_FORMAT}, fig5_action_distribution.{OUTPUT_FORMAT}, fig6_pareto.{OUTPUT_FORMAT}"
    )
    if REF_METRICS_PATH:
        print(f"Reference metrics: {REF_METRICS_PATH}")


if __name__ == "__main__":
    main()
