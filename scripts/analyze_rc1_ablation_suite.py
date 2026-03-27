from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = ROOT / "runs" / "rc1_ablation_1500ep_20260322_180707"
OUT_DIR = SUITE_ROOT / "ablation_compare"

RUNS = [
    ("full", "TERA-MAPPO", "#1f4e79"),
    ("wo_dag", "w/o TDE", "#c0392b"),
    ("wo_resource", "w/o CARE", "#16a085"),
    ("wo_dag_resource", "w/o DAG & Resource", "#8e44ad"),
]

BASELINE_ORDER = ["Local-Only", "Greedy-Local", "Legal-Random"]


def load_training_df(run_name: str) -> pd.DataFrame:
    path = SUITE_ROOT / run_name / "logs" / "training_stats.csv"
    df = pd.read_csv(path).copy()
    df["run_name"] = run_name
    return df


def load_formal_summary() -> list[dict]:
    path = SUITE_ROOT / "authoritative_eval" / "formal_eval_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best50_stats(df: pd.DataFrame) -> dict:
    idx = int(df["task_sr"].rolling(50).mean().idxmax())
    win = df.iloc[idx - 49 : idx + 1]
    return {
        "best50_end_episode": int(df.iloc[idx]["episode"]),
        "best50_task_sr": float(win["task_sr"].mean()),
        "best50_deadline_miss_rate": float(win["deadline_miss_rate"].mean()),
        "best50_mean_cft_completed": float(win["mean_cft_completed"].mean()),
        "best50_avg_rsu_queue": float(win["avg_rsu_queue"].mean()),
        "best50_avg_power": float(win["avg_power"].mean()),
        "best50_decision_frac_local": float(win["decision_frac_local"].mean()),
        "best50_decision_frac_rsu": float(win["decision_frac_rsu"].mean()),
        "best50_decision_frac_v2v": float(win["decision_frac_v2v"].mean()),
    }


def tail_stats(df: pd.DataFrame, n: int) -> dict:
    win = df.tail(n)
    prefix = f"last{n}"
    return {
        f"{prefix}_reward_mean": float(win["reward_mean"].mean()),
        f"{prefix}_task_sr": float(win["task_sr"].mean()),
        f"{prefix}_deadline_miss_rate": float(win["deadline_miss_rate"].mean()),
        f"{prefix}_mean_cft_completed": float(win["mean_cft_completed"].mean()),
        f"{prefix}_avg_rsu_queue": float(win["avg_rsu_queue"].mean()),
        f"{prefix}_avg_power": float(win["avg_power"].mean()),
        f"{prefix}_decision_frac_local": float(win["decision_frac_local"].mean()),
        f"{prefix}_decision_frac_rsu": float(win["decision_frac_rsu"].mean()),
        f"{prefix}_decision_frac_v2v": float(win["decision_frac_v2v"].mean()),
        f"{prefix}_approx_kl": float(win["approx_kl"].mean()),
        f"{prefix}_entropy": float(win["entropy"].mean()),
        f"{prefix}_clip_frac": float(win["clip_frac"].mean()),
    }


def build_training_summary() -> pd.DataFrame:
    rows = []
    for run_name, label, _ in RUNS:
        df = load_training_df(run_name)
        row = {
            "run_name": run_name,
            "label": label,
            "num_episodes": int(len(df)),
        }
        row.update(best50_stats(df))
        row.update(tail_stats(df, 100))
        row.update(tail_stats(df, 300))
        rows.append(row)
    return pd.DataFrame(rows)


def build_formal_summary_table(formal_summary: list[dict]) -> pd.DataFrame:
    rows = []
    for item in formal_summary:
        rows.append(
            {
                "policy": item["policy"],
                "task_success_rate_B": item["task_success_rate_B_mean"],
                "deadline_miss_rate": item["deadline_miss_rate_mean"],
                "mean_cft": item["mean_cft_mean"],
                "subtask_success_rate": item["subtask_success_rate_mean"],
                "finished_vehicle_count": item["finished_vehicle_count_mean"],
                "failed_vehicle_count": item["failed_vehicle_count_mean"],
                "decision_frac_local": item["decision_frac_local_mean"],
                "decision_frac_rsu": item["decision_frac_rsu_mean"],
                "decision_frac_v2v": item["decision_frac_v2v_mean"],
                "reward_sum": item["reward_sum_mean"],
                "score_tuple": tuple(item["score_tuple"]),
                "run_name": item.get("run_name", ""),
                "checkpoint_kind": item.get("checkpoint_kind", ""),
            }
        )
    return pd.DataFrame(rows)


def plot_training_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    metric_specs = [
        ("reward_mean", "Reward Mean"),
        ("task_sr", "Task Success Rate"),
        ("mean_cft_completed", "Mean CFT Completed"),
        ("avg_rsu_queue", "Average RSU Queue"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for run_name, label, color in RUNS:
            df = load_training_df(run_name)
            smooth = df[metric].rolling(50, min_periods=1).mean()
            ax.plot(df["episode"], smooth, label=label, color=color, linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.savefig(OUT_DIR / "ablation_training_curves.png", dpi=180)
    plt.close(fig)


def plot_policy_mix_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    metric_specs = [
        ("decision_frac_local", "Local Ratio"),
        ("decision_frac_rsu", "RSU Ratio"),
        ("decision_frac_v2v", "V2V Ratio"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for run_name, label, color in RUNS:
            df = load_training_df(run_name)
            smooth = df[metric].rolling(50, min_periods=1).mean()
            ax.plot(df["episode"], smooth, label=label, color=color, linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(OUT_DIR / "ablation_policy_mix_curves.png", dpi=180)
    plt.close(fig)


def plot_diagnostic_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    metric_specs = [
        ("approx_kl", "Approx KL"),
        ("entropy", "Entropy"),
        ("clip_frac", "Clip Fraction"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for run_name, label, color in RUNS:
            df = load_training_df(run_name)
            smooth = df[metric].rolling(50, min_periods=1).mean()
            ax.plot(df["episode"], smooth, label=label, color=color, linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(OUT_DIR / "ablation_diagnostic_curves.png", dpi=180)
    plt.close(fig)


def plot_formal_bars(formal_df: pd.DataFrame) -> None:
    winners = []
    for run_name, label, _ in RUNS:
        sub = formal_df[formal_df["run_name"] == run_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            by=["task_success_rate_B", "deadline_miss_rate", "mean_cft"],
            ascending=[False, True, True],
        )
        winners.append(sub.iloc[0])
    winners_df = pd.DataFrame(winners)
    base_df = formal_df[formal_df["policy"].isin(BASELINE_ORDER)].copy()
    plot_df = pd.concat([winners_df, base_df], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panels = [
        ("task_success_rate_B", "Formal Task Success Rate"),
        ("mean_cft", "Formal Mean CFT"),
        ("decision_frac_local", "Formal Local Ratio"),
        ("decision_frac_v2v", "Formal V2V Ratio"),
    ]
    labels = plot_df["policy"].tolist()
    colors = []
    color_map = {label: color for _, label, color in RUNS}
    for policy in labels:
        base = policy.split("::")[0]
        colors.append(color_map.get(base, "#666666"))
    for ax, (metric, title) in zip(axes.flat, panels):
        ax.bar(range(len(plot_df)), plot_df[metric], color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "ablation_formal_eval_bars.png", dpi=180)
    plt.close(fig)


def write_report(training_df: pd.DataFrame, formal_df: pd.DataFrame) -> None:
    winners = []
    for run_name, label, _ in RUNS:
        sub = formal_df[formal_df["run_name"] == run_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            by=["task_success_rate_B", "deadline_miss_rate", "mean_cft"],
            ascending=[False, True, True],
        )
        winners.append(sub.iloc[0].to_dict())
    winners_df = pd.DataFrame(winners)

    lines = []
    lines.append("# RC1 Ablation Compare Report\n")
    lines.append("## Training Summary\n")
    lines.append(
        training_df[
            [
                "label",
                "best50_end_episode",
                "best50_task_sr",
                "last100_task_sr",
                "last100_deadline_miss_rate",
                "last100_mean_cft_completed",
                "last100_avg_rsu_queue",
                "last100_decision_frac_local",
                "last100_decision_frac_rsu",
                "last100_decision_frac_v2v",
            ]
        ].to_markdown(index=False)
    )
    lines.append("\n## Formal Winner Per Run\n")
    lines.append(
        winners_df[
            [
                "policy",
                "task_success_rate_B",
                "deadline_miss_rate",
                "mean_cft",
                "subtask_success_rate",
                "decision_frac_local",
                "decision_frac_rsu",
                "decision_frac_v2v",
                "score_tuple",
            ]
        ].to_markdown(index=False)
    )
    lines.append("\n## Notes\n")
    lines.append("- 所有图和表均基于现有真实训练结果与 authoritative eval，未对任何模型结果做人工减值或改写。")
    lines.append("- 正式协议基线仅包含 `Local-Only / Greedy-Local / Legal-Random`。")
    (OUT_DIR / "ABLATION_COMPARE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training_df = build_training_summary()
    formal_df = build_formal_summary_table(load_formal_summary())

    training_df.to_csv(OUT_DIR / "ablation_training_summary.csv", index=False)
    formal_df.to_csv(OUT_DIR / "ablation_formal_summary.csv", index=False)

    plot_training_curves()
    plot_policy_mix_curves()
    plot_diagnostic_curves()
    plot_formal_bars(formal_df)
    write_report(training_df, formal_df)


if __name__ == "__main__":
    main()
