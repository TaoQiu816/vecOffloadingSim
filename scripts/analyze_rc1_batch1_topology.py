from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = ROOT / "runs" / "rc1_batch1_part1_topology_20260323_182712"
OUT_DIR = SUITE_ROOT / "batch_analysis"

TOPOLOGY_RUNS = [
    ("parallel", "Parallel"),
    ("balanced", "Balanced"),
    ("deep", "Deep"),
]

POLICY_ORDER = [
    ("full", "Full-MAPPO", "#1f4e79"),
    ("wo_dag", "w/o DAG-Feature", "#c0392b"),
    ("local_only", "Local-Only", "#2e7d32"),
    ("greedy", "Greedy", "#f39c12"),
]


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _tail_mean(path: Path, n: int = 100) -> dict:
    df = pd.read_csv(path).copy()
    win = df.tail(n)
    return {
        "episodes": int(len(df)),
        "window": n,
        "reward_mean": float(win["reward_mean"].mean()),
        "task_sr": float(win["task_sr"].mean()),
        "subtask_sr": float(win["subtask_sr"].mean()),
        "deadline_miss_rate": float(win["deadline_miss_rate"].mean()),
        "mean_cft_completed": float(win["mean_cft_completed"].mean()),
        "avg_rsu_queue": float(win["avg_rsu_queue"].mean()),
        "avg_power": float(win["avg_power"].mean()),
        "decision_frac_local": float(win["decision_frac_local"].mean()),
        "decision_frac_rsu": float(win["decision_frac_rsu"].mean()),
        "decision_frac_v2v": float(win["decision_frac_v2v"].mean()),
        "approx_kl": float(win["approx_kl"].mean()),
        "entropy": float(win["entropy"].mean()),
        "clip_frac": float(win["clip_frac"].mean()),
    }


def _baseline_mean(path: Path) -> dict:
    df = pd.read_csv(path).copy()
    row = df.iloc[0]
    return {
        "episodes": int(row["episodes"]),
        "window": int(row["episodes"]),
        "reward_mean": math.nan,
        "task_sr": float(row["task_success_rate_mean"]),
        "subtask_sr": math.nan,
        "deadline_miss_rate": float(row["deadline_miss_rate_mean"]),
        "mean_cft_completed": float(row["mean_cft_completed_mean"]),
        "avg_rsu_queue": float(row["avg_rsu_queue_mean"]),
        "avg_power": float(row["avg_power_mean"]),
        "decision_frac_local": float(row["decision_frac_local_mean"]),
        "decision_frac_rsu": float(row["decision_frac_rsu_mean"]),
        "decision_frac_v2v": float(row["decision_frac_v2v_mean"]),
        "approx_kl": math.nan,
        "entropy": math.nan,
        "clip_frac": math.nan,
    }


def build_topology_summary() -> pd.DataFrame:
    rows = []
    for topo_key, topo_label in TOPOLOGY_RUNS:
        for policy_key, policy_label, _ in POLICY_ORDER:
            base_dir = SUITE_ROOT / f"topology_{topo_key}" / policy_key
            if policy_key in {"full", "wo_dag"}:
                metrics = _tail_mean(base_dir / "logs" / "training_stats.csv", n=100)
                metrics["source"] = "train_tail100"
            else:
                metrics = _baseline_mean(base_dir / "logs" / "baseline_eval_core_summary.csv")
                metrics["source"] = "baseline_eval_mean"
            metrics["topology_key"] = topo_key
            metrics["topology"] = topo_label
            metrics["policy_key"] = policy_key
            metrics["policy"] = policy_label
            rows.append(metrics)
    return pd.DataFrame(rows)


def build_ippo_main_summary() -> pd.DataFrame:
    rows = []
    ippo = _tail_mean(SUITE_ROOT / "ippo_main" / "logs" / "training_stats.csv", n=100)
    ippo.update(
        {
            "run_name": "ippo_main",
            "label": "IPPO Main (base scene)",
        }
    )
    rows.append(ippo)
    return pd.DataFrame(rows)


def build_space_audit() -> pd.DataFrame:
    rows = []
    for path in SUITE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.relative_to(SUITE_ROOT)),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix,
                "top_level": path.relative_to(SUITE_ROOT).parts[0],
            }
        )
    df = pd.DataFrame(rows).sort_values("size_bytes", ascending=False)
    return df


def _load_train_curve(topo_key: str, policy_key: str) -> pd.DataFrame:
    return pd.read_csv(SUITE_ROOT / f"topology_{topo_key}" / policy_key / "logs" / "training_stats.csv")


def _load_main_curve(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "logs" / "training_stats.csv")


def build_strict_train_inventory() -> pd.DataFrame:
    rows = []
    ippo_df = _load_main_curve(SUITE_ROOT / "ippo_main")
    rows.append(
        {
            "run_key": "ippo_main",
            "label": "IPPO Main (base scene)",
            "topology": "Base",
            "policy": "IPPO",
            "episodes": int(len(ippo_df)),
            "path": str((SUITE_ROOT / "ippo_main").relative_to(ROOT)),
        }
    )
    for topo_key, topo_label in TOPOLOGY_RUNS:
        for policy_key, policy_label, _ in POLICY_ORDER[:2]:
            run_dir = SUITE_ROOT / f"topology_{topo_key}" / policy_key
            df = _load_main_curve(run_dir)
            rows.append(
                {
                    "run_key": f"{topo_key}_{policy_key}",
                    "label": f"{topo_label}-{policy_label}",
                    "topology": topo_label,
                    "policy": policy_label,
                    "episodes": int(len(df)),
                    "path": str(run_dir.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows)


def plot_batch_overview(topology_df: pd.DataFrame, ippo_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    metrics = [
        ("task_sr", "Task Success Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("mean_cft_completed", "Mean CFT Completed"),
    ]
    x = range(len(TOPOLOGY_RUNS))
    width = 0.2

    for idx, (metric, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        for offset_idx, (_, policy_label, color) in enumerate(POLICY_ORDER):
            vals = []
            for topo_key, _ in TOPOLOGY_RUNS:
                row = topology_df[
                    (topology_df["topology_key"] == topo_key)
                    & (topology_df["policy"] == policy_label)
                ].iloc[0]
                vals.append(row[metric])
            xs = [i + (offset_idx - 1.5) * width for i in x]
            ax.bar(xs, vals, width=width, label=policy_label, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels([label for _, label in TOPOLOGY_RUNS])
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.legend(frameon=False, fontsize=9)

    train_curve_specs = [
        ("parallel", "Parallel"),
        ("balanced", "Balanced"),
        ("deep", "Deep"),
    ]
    for row_idx, (topo_key, topo_label) in enumerate(train_curve_specs, start=1):
        ax1 = fig.add_subplot(gs[row_idx, 0:2])
        for policy_key, policy_label, color in POLICY_ORDER[:2]:
            path = SUITE_ROOT / f"topology_{topo_key}" / policy_key / "logs" / "training_stats.csv"
            df = pd.read_csv(path)
            smooth = df["task_sr"].rolling(50, min_periods=1).mean()
            ax1.plot(df["episode"], smooth, color=color, linewidth=2.2, label=policy_label)
        base_rows = topology_df[topology_df["topology_key"] == topo_key]
        for _, policy_label, color in POLICY_ORDER[2:]:
            val = float(base_rows[base_rows["policy"] == policy_label]["task_sr"].iloc[0])
            ax1.axhline(val, color=color, linestyle="--", linewidth=1.8, label=policy_label)
        ax1.set_title(f"{topo_label}: Task Success Rate")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Task SR")
        ax1.grid(alpha=0.25)
        if row_idx == 1:
            ax1.legend(frameon=False, ncol=4, fontsize=9)

        ax2 = fig.add_subplot(gs[row_idx, 2])
        for policy_key, policy_label, color in POLICY_ORDER[:2]:
            path = SUITE_ROOT / f"topology_{topo_key}" / policy_key / "logs" / "training_stats.csv"
            df = pd.read_csv(path)
            smooth = df["avg_rsu_queue"].rolling(50, min_periods=1).mean()
            ax2.plot(df["episode"], smooth, color=color, linewidth=2.2, label=policy_label)
        for _, policy_label, color in POLICY_ORDER[2:]:
            val = float(base_rows[base_rows["policy"] == policy_label]["avg_rsu_queue"].iloc[0])
            ax2.axhline(val, color=color, linestyle="--", linewidth=1.8)
        ax2.set_title(f"{topo_label}: Avg RSU Queue")
        ax2.set_xlabel("Episode")
        ax2.grid(alpha=0.25)

    fig.suptitle("RC1 Batch1 Topology Study Overview", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_topology_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ippo_metrics = [
        ("task_sr", "Task Success Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("mean_cft_completed", "Mean CFT Completed"),
        ("avg_rsu_queue", "Avg RSU Queue"),
        ("avg_power", "Avg Power"),
        ("decision_frac_rsu", "RSU Ratio"),
    ]
    colors = ["#8e44ad"]
    labels = ippo_df["label"].tolist()
    for ax, (metric, title) in zip(axes.flat, ippo_metrics):
        ax.bar(labels, ippo_df[metric], color=colors, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Base Scene: IPPO Main (tail100 summary)", fontsize=15)
    fig.savefig(OUT_DIR / "batch1_ippo_main_summary_bars.png", dpi=180)
    plt.close(fig)


def plot_extended_metric_bars(topology_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    metric_specs = [
        ("task_sr", "Task Success Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("mean_cft_completed", "Mean CFT Completed"),
        ("avg_rsu_queue", "Avg RSU Queue"),
        ("avg_power", "Avg Power"),
        ("decision_frac_local", "Local Ratio"),
        ("decision_frac_rsu", "RSU Ratio"),
        ("decision_frac_v2v", "V2V Ratio"),
        ("subtask_sr", "Subtask Success Rate"),
    ]
    x = np.arange(len(TOPOLOGY_RUNS))
    width = 0.2
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for offset_idx, (_, policy_label, color) in enumerate(POLICY_ORDER):
            vals = []
            for topo_key, _ in TOPOLOGY_RUNS:
                row = topology_df[
                    (topology_df["topology_key"] == topo_key)
                    & (topology_df["policy"] == policy_label)
                ].iloc[0]
                vals.append(row[metric] if pd.notna(row[metric]) else 0.0)
            xs = x + (offset_idx - 1.5) * width
            ax.bar(xs, vals, width=width, color=color, alpha=0.9, label=policy_label)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in TOPOLOGY_RUNS])
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Batch1 Topology: Extended Metric Comparison", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_topology_extended_bars.png", dpi=180)
    plt.close(fig)


def plot_policy_mix_bars(topology_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    mix_metrics = [
        ("decision_frac_local", "Local Ratio"),
        ("decision_frac_rsu", "RSU Ratio"),
        ("decision_frac_v2v", "V2V Ratio"),
    ]
    labels = [f"{topo}\n{policy}" for topo in topology_df["topology"] for policy in []]
    plot_df = topology_df.copy()
    plot_df["x_label"] = plot_df["topology"] + "\n" + plot_df["policy"]
    colors = [dict((label, color) for _, label, color in POLICY_ORDER)[p] for p in plot_df["policy"]]
    for ax, (metric, title) in zip(axes.flat, mix_metrics):
        ax.bar(plot_df["x_label"], plot_df[metric], color=colors, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Batch1 Topology: Policy Mix Comparison", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_topology_policy_mix_bars.png", dpi=180)
    plt.close(fig)


def plot_topology_training_diagnostics() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    metric_specs = [
        ("reward_mean", "Reward Mean"),
        ("task_sr", "Task Success Rate"),
        ("avg_rsu_queue", "Avg RSU Queue"),
        ("mean_cft_completed", "Mean CFT Completed"),
        ("decision_frac_rsu", "RSU Ratio"),
        ("decision_frac_local", "Local Ratio"),
        ("approx_kl", "Approx KL"),
        ("entropy", "Entropy"),
        ("clip_frac", "Clip Fraction"),
    ]
    line_styles = {
        "parallel": ("Parallel", "-"),
        "balanced": ("Balanced", "--"),
        "deep": ("Deep", ":"),
    }
    policy_colors = {
        "full": "#1f4e79",
        "wo_dag": "#c0392b",
    }
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for topo_key, (topo_label, ls) in line_styles.items():
            for policy_key, policy_label, _ in POLICY_ORDER[:2]:
                df = _load_train_curve(topo_key, policy_key)
                smooth = df[metric].rolling(50, min_periods=1).mean()
                ax.plot(
                    df["episode"],
                    smooth,
                    color=policy_colors[policy_key],
                    linestyle=ls,
                    linewidth=2.0,
                    label=f"{topo_label}-{policy_label}",
                )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0, 0].legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Batch1 Topology: Training Dynamics (Full vs w/o DAG)", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_topology_training_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_ippo_main_curves() -> None:
    ippo_df = _load_main_curve(SUITE_ROOT / "ippo_main")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    metric_specs = [
        ("reward_mean", "Reward Mean"),
        ("task_sr", "Task Success Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("mean_cft_completed", "Mean CFT Completed"),
        ("avg_rsu_queue", "Avg RSU Queue"),
        ("decision_frac_rsu", "RSU Ratio"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        smooth = ippo_df[metric].rolling(50, min_periods=1).mean()
        ax.plot(ippo_df["episode"], smooth, color="#8e44ad", linewidth=2.2, label="IPPO Main")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Base Scene: IPPO Main Training Curves (1500ep)", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_ippo_main_training_curves_1500ep.png", dpi=180)
    plt.close(fig)


def plot_strict_train_convergence_1500ep(train_inventory: pd.DataFrame) -> None:
    if not (train_inventory["episodes"] == 1500).all():
        bad = train_inventory[train_inventory["episodes"] != 1500]
        raise ValueError(f"Found non-1500ep runs in strict convergence set:\n{bad}")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    metric_specs = [
        ("reward_mean", "Reward Mean"),
        ("task_sr", "Task Success Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("avg_rsu_queue", "Avg RSU Queue"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        ippo_df = _load_main_curve(SUITE_ROOT / "ippo_main")
        ax.plot(
            ippo_df["episode"],
            ippo_df[metric].rolling(50, min_periods=1).mean(),
            color="#8e44ad",
            linewidth=2.5,
            label="Base-IPPO",
        )
        for topo_key, topo_label in TOPOLOGY_RUNS:
            for policy_key, policy_label, color in POLICY_ORDER[:2]:
                df = _load_train_curve(topo_key, policy_key)
                ls = "-" if policy_key == "full" else "--"
                ax.plot(
                    df["episode"],
                    df[metric].rolling(50, min_periods=1).mean(),
                    color=color,
                    linewidth=1.9,
                    linestyle=ls,
                    label=f"{topo_label}-{policy_label}",
                )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0, 0].legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Batch1 Strict 1500ep Convergence Curves (all trainable runs)", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_strict_train_convergence_1500ep.png", dpi=180)
    plt.close(fig)


def plot_topology_score_scatter(topology_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    color_map = {label: color for _, label, color in POLICY_ORDER}
    for ax, (topo_key, topo_label) in zip(axes.flat, TOPOLOGY_RUNS):
        sub = topology_df[topology_df["topology_key"] == topo_key].copy()
        for _, row in sub.iterrows():
            ax.scatter(
                row["mean_cft_completed"],
                row["task_sr"],
                s=140,
                color=color_map[row["policy"]],
                alpha=0.9,
                label=row["policy"],
            )
            ax.annotate(row["policy"], (row["mean_cft_completed"], row["task_sr"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(topo_label)
        ax.set_xlabel("Mean CFT Completed")
        ax.set_ylabel("Task Success Rate")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8)
    fig.suptitle("Batch1 Topology: Score Scatter (CFT vs Task SR)", fontsize=16)
    fig.savefig(OUT_DIR / "batch1_topology_score_scatter.png", dpi=180)
    plt.close(fig)


def write_report(topology_df: pd.DataFrame, ippo_df: pd.DataFrame, space_df: pd.DataFrame, train_inventory: pd.DataFrame) -> None:
    ckpt_df = space_df[space_df["path"].str.contains("/models/checkpoints/|/models/best_model.pth|/models/last_model.pth", regex=True)].copy()
    plots_df = space_df[space_df["path"].str.contains("/plots/", regex=True)].copy()
    lines = []
    lines.append("# RC1 Batch1 Analysis Report\n")
    lines.append("## Space Audit\n")
    lines.append(f"- Batch size: `{space_df['size_bytes'].sum() / (1024 ** 2):.1f} MB`\n")
    lines.append(f"- Model files total: `{ckpt_df['size_bytes'].sum() / (1024 ** 2):.1f} MB`\n")
    lines.append(f"- Plot files total: `{plots_df['size_bytes'].sum() / (1024 ** 2):.1f} MB`\n")
    lines.append(f"- Sparse checkpoints total: `{ckpt_df[ckpt_df['path'].str.contains('/models/checkpoints/')]['size_bytes'].sum() / (1024 ** 2):.1f} MB`\n")
    lines.append("\n## Topology Summary\n")
    cols = [
        "topology",
        "policy",
        "source",
        "task_sr",
        "deadline_miss_rate",
        "mean_cft_completed",
        "avg_rsu_queue",
        "avg_power",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
    ]
    lines.append(topology_df[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Base Scene Reference\n")
    lines.append(ippo_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Strict 1500ep Train Inventory\n")
    lines.append(train_inventory.to_markdown(index=False))
    lines.append("\n\n## Generated Figures\n")
    for name in [
        "batch1_topology_overview.png",
        "batch1_topology_extended_bars.png",
        "batch1_topology_policy_mix_bars.png",
        "batch1_topology_training_diagnostics.png",
        "batch1_topology_score_scatter.png",
        "batch1_ippo_main_summary_bars.png",
        "batch1_ippo_main_training_curves_1500ep.png",
        "batch1_strict_train_convergence_1500ep.png",
    ]:
        lines.append(f"- `{name}`")

    pivot_sr = topology_df.pivot(index="topology", columns="policy", values="task_sr")
    pivot_cft = topology_df.pivot(index="topology", columns="policy", values="mean_cft_completed")
    pivot_queue = topology_df.pivot(index="topology", columns="policy", values="avg_rsu_queue")
    lines.append("\n\n## Evidence-Based Findings\n")
    for topo_label in [label for _, label in TOPOLOGY_RUNS]:
        full_sr = float(pivot_sr.loc[topo_label, "Full-MAPPO"])
        wodag_sr = float(pivot_sr.loc[topo_label, "w/o DAG-Feature"])
        full_cft = float(pivot_cft.loc[topo_label, "Full-MAPPO"])
        wodag_cft = float(pivot_cft.loc[topo_label, "w/o DAG-Feature"])
        full_q = float(pivot_queue.loc[topo_label, "Full-MAPPO"])
        wodag_q = float(pivot_queue.loc[topo_label, "w/o DAG-Feature"])
        lines.append(
            f"- `{topo_label}`: `w/o DAG` vs `Full` -> "
            f"`task_sr {wodag_sr:.4f} vs {full_sr:.4f}`, "
            f"`mean_cft {wodag_cft:.4f} vs {full_cft:.4f}`, "
            f"`avg_rsu_queue {wodag_q:.4f} vs {full_q:.4f}`."
        )

    lines.append(
        "- Strict convergence figures in this batch now use only internal `1500ep` runs. "
        "External `run_1000ep_A_20260320` is no longer used in convergence plotting."
    )

    (OUT_DIR / "BATCH1_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_out_dir()
    topology_df = build_topology_summary().sort_values(["topology_key", "policy_key"])
    ippo_df = build_ippo_main_summary()
    space_df = build_space_audit()
    train_inventory = build_strict_train_inventory()

    topology_df.to_csv(OUT_DIR / "batch1_topology_summary.csv", index=False)
    ippo_df.to_csv(OUT_DIR / "batch1_ippo_main_summary.csv", index=False)
    space_df.to_csv(OUT_DIR / "batch1_space_audit.csv", index=False)
    train_inventory.to_csv(OUT_DIR / "batch1_strict_train_inventory.csv", index=False)

    plot_batch_overview(topology_df, ippo_df)
    plot_extended_metric_bars(topology_df)
    plot_policy_mix_bars(topology_df)
    plot_topology_training_diagnostics()
    plot_ippo_main_curves()
    plot_strict_train_convergence_1500ep(train_inventory)
    plot_topology_score_scatter(topology_df)
    write_report(topology_df, ippo_df, space_df, train_inventory)
    print(f"Wrote analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
