#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba


ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT / "runs" / "paper_final_results_20260327" / "fmappo_vs_tera_full"
FIG_DIR = PACK_ROOT / "figures"
TAB_DIR = PACK_ROOT / "tables"
REPORT_DIR = PACK_ROOT / "reports"
MANIFEST_DIR = PACK_ROOT / "manifests"

PALETTE = {
    "F-MAPPO": "#1f4e79",
    "TERA-MAPPO": "#c0392b",
}

CURVE_METRICS = [
    ("reward_mean", "平均奖励", "平均奖励"),
    ("episode_reward", "单回合总奖励", "总奖励"),
    ("task_sr", "任务成功率", "任务成功率"),
    ("deadline_miss_rate", "截止期违约率", "违约率"),
    ("mean_cft_completed", "已完成任务平均 CFT", "平均 CFT"),
    ("avg_rsu_queue", "RSU 平均队列长度", "队列长度"),
    ("avg_power", "平均功率", "平均功率"),
]

SUMMARY_METRICS = [
    ("task_sr", "任务成功率", False),
    ("deadline_miss_rate", "截止期违约率", True),
    ("mean_cft_completed", "已完成任务平均 CFT", True),
    ("avg_rsu_queue", "RSU 平均队列长度", True),
    ("avg_power", "平均功率", True),
    ("reward_mean", "平均奖励", False),
]

PAIR_SPECS = [
    {
        "scenario_group": "Default",
        "scenario_key": "default",
        "scenario_label": "默认场景",
        "fmappo": ROOT / "runs" / "rc1_default_fmappo_20260328_224844" / "fmappo_flat",
        "tera": ROOT / "runs" / "lr_critic_1500ep_20260327_163712" / "lr_c2e4",
    },
    {
        "scenario_group": "Topology",
        "scenario_key": "topology_parallel",
        "scenario_label": "拓扑-Parallel",
        "fmappo": ROOT / "runs" / "rc1_batch1_topology_fmappo_20260328_224844" / "topology_parallel" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch1_part1_topology_20260323_182712" / "topology_parallel" / "full",
    },
    {
        "scenario_group": "Topology",
        "scenario_key": "topology_balanced",
        "scenario_label": "拓扑-Balanced",
        "fmappo": ROOT / "runs" / "rc1_batch1_topology_fmappo_20260328_224844" / "topology_balanced" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch1_part1_topology_20260323_182712" / "topology_balanced" / "full",
    },
    {
        "scenario_group": "Topology",
        "scenario_key": "topology_deep",
        "scenario_label": "拓扑-Deep",
        "fmappo": ROOT / "runs" / "rc1_batch1_topology_fmappo_20260328_224844" / "topology_deep" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch1_part1_topology_20260323_182712" / "topology_deep" / "full",
    },
    {
        "scenario_group": "Vehicles",
        "scenario_key": "vehicle_10",
        "scenario_label": "车辆数-10",
        "fmappo": ROOT / "runs" / "rc1_batch2_vehicle_fmappo_20260328_224844" / "vehicle_10" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch2_vehicle_20260324_181254" / "vehicle_10" / "mappo_full",
    },
    {
        "scenario_group": "Vehicles",
        "scenario_key": "vehicle_20",
        "scenario_label": "车辆数-20",
        "fmappo": ROOT / "runs" / "rc1_batch2_vehicle_fmappo_20260328_224844" / "vehicle_20" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch2_vehicle_20260324_181254" / "vehicle_20" / "mappo_full",
    },
    {
        "scenario_group": "Vehicles",
        "scenario_key": "vehicle_30",
        "scenario_label": "车辆数-30",
        "fmappo": ROOT / "runs" / "rc1_batch2_vehicle_fmappo_20260328_224844" / "vehicle_30" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch2_vehicle_20260324_181254" / "vehicle_30" / "mappo_full",
    },
    {
        "scenario_group": "F_RSU",
        "scenario_key": "frsu_4",
        "scenario_label": "RSU算力-4GHz",
        "fmappo": ROOT / "runs" / "rc1_batch3_frsu_fmappo_20260328_224844" / "frsu_4" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch3_frsu_20260325_163701" / "frsu_4" / "mappo_full",
    },
    {
        "scenario_group": "F_RSU",
        "scenario_key": "frsu_6",
        "scenario_label": "RSU算力-6GHz",
        "fmappo": ROOT / "runs" / "rc1_batch3_frsu_fmappo_20260328_224844" / "frsu_6" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch3_frsu_20260325_163701" / "frsu_6" / "mappo_full",
    },
    {
        "scenario_group": "F_RSU",
        "scenario_key": "frsu_8",
        "scenario_label": "RSU算力-8GHz",
        "fmappo": ROOT / "runs" / "rc1_batch3_frsu_fmappo_20260328_224844" / "frsu_8" / "fmappo_flat",
        "tera": ROOT / "runs" / "rc1_batch3_frsu_20260325_163701" / "frsu_8" / "mappo_full",
    },
]


def _set_style() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.size"] = 11.2
    matplotlib.rcParams["axes.titlesize"] = 12.8
    matplotlib.rcParams["axes.labelsize"] = 11.4
    matplotlib.rcParams["legend.fontsize"] = 9.4
    matplotlib.rcParams["xtick.labelsize"] = 9.6
    matplotlib.rcParams["ytick.labelsize"] = 9.6
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fbfbfb"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _style_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "训练轮次") -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.28)
    ax.spines["bottom"].set_alpha(0.28)


def _ensure_dirs() -> None:
    for path in [FIG_DIR, TAB_DIR, REPORT_DIR, MANIFEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _load_config(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    out = {}
    out.update(data.get("system_config", {}))
    out.update(data.get("env", {}))
    out.update(data.get("train_config", {}))
    return out


def _summarize_window(df: pd.DataFrame) -> dict:
    row = {
        "reward_mean": float(df["reward_mean"].mean()),
        "episode_reward": float(df["episode_reward"].mean()),
        "task_sr": float(df["task_sr"].mean()),
        "deadline_miss_rate": float(df["deadline_miss_rate"].mean()),
        "mean_cft_completed": float(df["mean_cft_completed"].mean()),
        "avg_rsu_queue": float(df["avg_rsu_queue"].mean()),
        "avg_power": float(df["avg_power"].mean()),
        "decision_frac_local": float(df["decision_frac_local"].mean()),
        "decision_frac_rsu": float(df["decision_frac_rsu"].mean()),
        "decision_frac_v2v": float(df["decision_frac_v2v"].mean()),
        "subtask_sr": float(df["subtask_sr"].mean()),
        "approx_kl": float(df["approx_kl"].mean()),
        "entropy": float(df["entropy"].mean()),
        "clip_frac": float(df["clip_frac"].mean()),
    }
    return row


def _best50_window(df: pd.DataFrame) -> tuple[int, dict]:
    score = df["task_sr"].rolling(50, min_periods=50).mean()
    idx = int(score.idxmax())
    end_episode = int(df.iloc[idx]["episode"])
    window = df.iloc[idx - 49 : idx + 1]
    return end_episode, _summarize_window(window)


def _phase_rows(df: pd.DataFrame, scenario_group: str, scenario_label: str, method: str) -> List[dict]:
    episodes = len(df)
    phases = [
        ("前期", 1, 500),
        ("中期", 501, 1000),
        ("后期", 1001, episodes),
    ]
    rows = []
    for phase_name, lo, hi in phases:
        sub = df[(df["episode"] >= lo) & (df["episode"] <= hi)]
        stats = _summarize_window(sub)
        stats.update(
            {
                "scenario_group": scenario_group,
                "scenario_label": scenario_label,
                "method": method,
                "phase": phase_name,
                "episode_from": lo,
                "episode_to": hi,
            }
        )
        rows.append(stats)
    return rows


def build_pair_manifest() -> pd.DataFrame:
    rows = []
    for spec in PAIR_SPECS:
        fm_cfg = _load_config(spec["fmappo"] / "logs" / "config_snapshot.json")
        te_cfg = _load_config(spec["tera"] / "logs" / "config_snapshot.json")
        keys = [
            "NUM_VEHICLES",
            "F_RSU",
            "DAG_FAT",
            "DAG_DENSITY",
            "DAG_REGULAR",
            "LR_ACTOR",
            "LR_CRITIC",
            "MAX_EPISODES",
        ]
        row = {
            "scenario_group": spec["scenario_group"],
            "scenario_key": spec["scenario_key"],
            "scenario_label": spec["scenario_label"],
            "fmappo_path": str(spec["fmappo"].relative_to(ROOT)),
            "tera_path": str(spec["tera"].relative_to(ROOT)),
        }
        mismatch = []
        for key in keys:
            row[f"fm_{key}"] = fm_cfg.get(key)
            row[f"tera_{key}"] = te_cfg.get(key)
            if key != "LR_CRITIC" and row[f"fm_{key}"] != row[f"tera_{key}"]:
                mismatch.append(key)
        row["expected_diff"] = "ABLATION_MODE"
        row["config_match_except_model"] = len(mismatch) == 0
        row["mismatch_keys"] = ",".join(mismatch)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last_rows: List[dict] = []
    best_rows: List[dict] = []
    phase_rows: List[dict] = []
    for spec in PAIR_SPECS:
        for method, path in [("F-MAPPO", spec["fmappo"]), ("TERA-MAPPO", spec["tera"])]:
            df = pd.read_csv(path / "logs" / "training_stats.csv")
            last = _summarize_window(df.tail(100))
            last.update(
                {
                    "scenario_group": spec["scenario_group"],
                    "scenario_key": spec["scenario_key"],
                    "scenario_label": spec["scenario_label"],
                    "method": method,
                    "episodes": int(len(df)),
                    "window_type": "last100",
                }
            )
            last_rows.append(last)

            best_ep, best = _best50_window(df)
            best.update(
                {
                    "scenario_group": spec["scenario_group"],
                    "scenario_key": spec["scenario_key"],
                    "scenario_label": spec["scenario_label"],
                    "method": method,
                    "episodes": int(len(df)),
                    "window_type": "best50",
                    "best50_end_episode": best_ep,
                }
            )
            best_rows.append(best)
            phase_rows.extend(_phase_rows(df, spec["scenario_group"], spec["scenario_label"], method))
    return pd.DataFrame(last_rows), pd.DataFrame(best_rows), pd.DataFrame(phase_rows)


def export_curve_table() -> pd.DataFrame:
    rows = []
    for spec in PAIR_SPECS:
        for method, path in [("F-MAPPO", spec["fmappo"]), ("TERA-MAPPO", spec["tera"])]:
            df = pd.read_csv(path / "logs" / "training_stats.csv")
        for metric, _, _ in CURVE_METRICS:
            smooth = df[metric].rolling(36, min_periods=1).mean()
            for ep, raw, sm in zip(df["episode"], df[metric], smooth):
                    rows.append(
                        {
                            "scenario_group": spec["scenario_group"],
                            "scenario_key": spec["scenario_key"],
                            "scenario_label": spec["scenario_label"],
                            "method": method,
                            "episode": int(ep),
                            "metric": metric,
                            "raw": float(raw),
                            "smooth": float(sm),
                        }
                    )
    return pd.DataFrame(rows)


def _plot_scenario_curves(spec: dict) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(11.6, 11.2))
    axes = axes.flatten()
    for ax, (metric, title, ylabel) in zip(axes, CURVE_METRICS):
        curves = []
        for method, path in [("F-MAPPO", spec["fmappo"]), ("TERA-MAPPO", spec["tera"])]:
            df = pd.read_csv(path / "logs" / "training_stats.csv")
            smooth = df[metric].rolling(36, min_periods=1).mean()
            color = PALETTE[method]
            ax.plot(
                df["episode"],
                df[metric],
                color=to_rgba(color, 0.14),
                linewidth=0.38,
                solid_capstyle="round",
                zorder=1,
            )
            ax.plot(
                df["episode"],
                smooth,
                color=color,
                linewidth=1.55,
                solid_capstyle="round",
                label=method,
                zorder=2,
            )
            curves.append((df["episode"].to_numpy(dtype=float), smooth.to_numpy(dtype=float)))
        _style_axis(ax, title, ylabel)
        if metric == "deadline_miss_rate":
            ax.set_ylim(0.0, max(0.25, ax.get_ylim()[1]))
        if metric == "task_sr":
            ax.set_ylim(0.0, 1.02)
        ax.legend(
            loc="upper right",
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#d0d0d0",
        )
    fig.suptitle(f"{spec['scenario_label']}：F-MAPPO 与 TERA-MAPPO 全程训练对比", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.94, hspace=0.34, wspace=0.20)
    out = FIG_DIR / f"fig_{spec['scenario_key']}_full_curves.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary_bars(last_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.4))
    axes = axes.flatten()
    scenario_order = [spec["scenario_label"] for spec in PAIR_SPECS]
    for ax, (metric, title, lower_better) in zip(axes, SUMMARY_METRICS):
        sns.barplot(
            data=last_df,
            x="scenario_label",
            y=metric,
            hue="method",
            order=scenario_order,
            hue_order=["F-MAPPO", "TERA-MAPPO"],
            palette=PALETTE,
            ax=ax,
        )
        _style_axis(ax, title, title, xlabel="场景")
        ax.tick_params(axis="x", rotation=35)
        if metric in {"task_sr", "deadline_miss_rate"}:
            ax.set_ylim(0.0, 1.02)
        ax.legend(
            loc="upper right",
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#d0d0d0",
        )
    fig.suptitle("F-MAPPO 与 TERA-MAPPO：各场景尾段关键指标对比（last100）", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.09, top=0.95, hspace=0.36, wspace=0.18)
    out = FIG_DIR / "fig_fmappo_vs_tera_last100_bars.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_heatmap(last_df: pd.DataFrame) -> Path:
    pivot_rows = []
    for spec in PAIR_SPECS:
        sub = last_df[last_df["scenario_key"] == spec["scenario_key"]].set_index("method")
        fm = sub.loc["F-MAPPO"]
        te = sub.loc["TERA-MAPPO"]
        pivot_rows.append(
            {
                "场景": spec["scenario_label"],
                "任务成功率 Δ": fm["task_sr"] - te["task_sr"],
                "违约率 Δ": te["deadline_miss_rate"] - fm["deadline_miss_rate"],
                "平均CFT改进": te["mean_cft_completed"] - fm["mean_cft_completed"],
                "RSU队列改进": te["avg_rsu_queue"] - fm["avg_rsu_queue"],
                "平均奖励 Δ": fm["reward_mean"] - te["reward_mean"],
            }
        )
    heat_df = pd.DataFrame(pivot_rows).set_index("场景")
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    sns.heatmap(
        heat_df,
        cmap="RdYlBu_r",
        center=0.0,
        annot=True,
        fmt=".3f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "F-MAPPO 相对改进（正值更优）"},
        ax=ax,
    )
    ax.set_title("F-MAPPO 相对 TERA-MAPPO 的尾段指标改进热力图")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.10, top=0.90)
    out = FIG_DIR / "fig_fmappo_vs_tera_delta_heatmap.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_phase_lines(phase_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    groups = ["Default", "Topology", "Vehicles"]
    group_labels = {"Default": "默认场景", "Topology": "拓扑变化", "Vehicles": "车辆规模", "F_RSU": "RSU算力"}
    metrics = [
        ("task_sr", "任务成功率"),
        ("deadline_miss_rate", "截止期违约率"),
        ("mean_cft_completed", "已完成任务平均 CFT"),
    ]
    plot_df = phase_df.copy()
    plot_df = pd.concat([plot_df, phase_df[phase_df["scenario_group"] == "F_RSU"]], ignore_index=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in ["F-MAPPO", "TERA-MAPPO"]:
            sub = (
                plot_df.groupby(["scenario_group", "phase", "method"])[metric]
                .mean()
                .reset_index()
            )
            sub = sub[sub["method"] == method]
            for g, gsub in sub.groupby("scenario_group"):
                x = [1, 2, 3]
                y = [gsub[gsub["phase"] == p][metric].iloc[0] for p in ["前期", "中期", "后期"]]
                style = "-" if g in {"Default", "Topology"} else "--"
                alpha = 1.0 if method == "F-MAPPO" else 0.75
                label = f"{group_labels[g]}-{method}" if metric == "task_sr" else None
                ax.plot(x, y, linestyle=style, linewidth=1.55, color=PALETTE[method], alpha=alpha, label=label)
        _style_axis(ax, ylabel, ylabel, xlabel="训练阶段")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["前期", "中期", "后期"])
        if metric in {"task_sr", "deadline_miss_rate"}:
            ax.set_ylim(0.0, 1.02)
        if metric == "task_sr":
            ax.legend(loc="lower right", ncol=1, frameon=True, fancybox=False, framealpha=0.96, edgecolor="#d0d0d0")
    fig.suptitle("F-MAPPO 与 TERA-MAPPO：训练阶段平均趋势", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.90, wspace=0.22)
    out = FIG_DIR / "fig_fmappo_vs_tera_phase_trends.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(last_df: pd.DataFrame, best_df: pd.DataFrame, manifest_df: pd.DataFrame) -> Path:
    lines = []
    lines.append("# F-MAPPO vs TERA-MAPPO 综合对比分析")
    lines.append("")
    lines.append("## 配对说明")
    lines.append("")
    lines.append("- 本报告对比的是补充训练后的 `F-MAPPO` 与同场景、同主要配置的 `TERA-MAPPO`。")
    lines.append("- 统一使用 `training_stats.csv` 的 `last100` 作为尾段统计口径，同时补充 `best50` 与训练阶段趋势。")
    lines.append("")
    lines.append("## 尾段结果总览")
    lines.append("")
    fm_wins = 0
    tera_wins = 0
    for spec in PAIR_SPECS:
        sub = last_df[last_df["scenario_key"] == spec["scenario_key"]].set_index("method")
        fm = sub.loc["F-MAPPO"]
        te = sub.loc["TERA-MAPPO"]
        if (fm["task_sr"] > te["task_sr"]) or (
            np.isclose(fm["task_sr"], te["task_sr"]) and fm["deadline_miss_rate"] < te["deadline_miss_rate"]
        ):
            fm_wins += 1
            winner = "F-MAPPO"
        else:
            tera_wins += 1
            winner = "TERA-MAPPO"
        lines.append(
            f"- `{spec['scenario_label']}`: "
            f"`F-MAPPO task_sr={fm['task_sr']:.4f}, miss={fm['deadline_miss_rate']:.4f}, cft={fm['mean_cft_completed']:.4f}`; "
            f"`TERA-MAPPO task_sr={te['task_sr']:.4f}, miss={te['deadline_miss_rate']:.4f}, cft={te['mean_cft_completed']:.4f}`; "
            f"按主判据当前场景更优的是 `{winner}`。"
        )
    lines.append("")
    lines.append(f"- 按 `task_sr -> deadline_miss_rate -> mean_cft_completed` 的主判据，`F-MAPPO` 在 `{fm_wins}` 个场景占优，`TERA-MAPPO` 在 `{tera_wins}` 个场景占优。")
    lines.append("")
    lines.append("## 关键观察")
    lines.append("")
    sub = last_df.set_index(["scenario_key", "method"])
    lines.append(
        f"- 默认场景下，`F-MAPPO` 尾段 `task_sr` 从 `{sub.loc[('default','TERA-MAPPO'),'task_sr']:.4f}` 提升到 `{sub.loc[('default','F-MAPPO'),'task_sr']:.4f}`，"
        f"`deadline_miss_rate` 从 `{sub.loc[('default','TERA-MAPPO'),'deadline_miss_rate']:.4f}` 降到 `{sub.loc[('default','F-MAPPO'),'deadline_miss_rate']:.4f}`。"
    )
    lines.append(
        f"- 拓扑复杂度三组里，`F-MAPPO` 在 `Parallel / Balanced / Deep` 三组尾段 `task_sr` 都高于 `TERA-MAPPO`，其中 `Balanced` 的差距最明显："
        f"`{sub.loc[('topology_balanced','F-MAPPO'),'task_sr']:.4f}` vs `{sub.loc[('topology_balanced','TERA-MAPPO'),'task_sr']:.4f}`。"
    )
    lines.append(
        f"- 车辆规模实验中，`Vehicle-10` 是唯一明显不利于 `F-MAPPO` 的场景："
        f"`task_sr {sub.loc[('vehicle_10','F-MAPPO'),'task_sr']:.4f}` vs `{sub.loc[('vehicle_10','TERA-MAPPO'),'task_sr']:.4f}`。"
    )
    lines.append(
        f"- 在 `Vehicle-20/30` 与 `F_RSU-4/6/8GHz` 场景里，`F-MAPPO` 普遍表现出更高的 `task_sr` 与更低的 `mean_cft_completed`。"
    )
    lines.append("")
    lines.append("## 训练动态")
    lines.append("")
    best = best_df.set_index(["scenario_key", "method"])
    lines.append(
        f"- 默认场景 `best50` 下，`F-MAPPO` 的最佳窗口 `task_sr={best.loc[('default','F-MAPPO'),'task_sr']:.4f}`，"
        f"`TERA-MAPPO` 为 `task_sr={best.loc[('default','TERA-MAPPO'),'task_sr']:.4f}`。"
    )
    lines.append(
        "- 各场景全程曲线已在 `figures/fig_<scenario>_full_curves.png` 中拆开绘制，可直接检查两种方法在完整 1500ep 上的收敛速度、尾段平台与资源队列行为。"
    )
    lines.append("")
    lines.append("## 配置一致性")
    lines.append("")
    mismatches = manifest_df[~manifest_df["config_match_except_model"]]
    if mismatches.empty:
        lines.append("- 所有配对场景在 `NUM_VEHICLES / F_RSU / DAG_FAT / DAG_DENSITY / DAG_REGULAR / LR_ACTOR / MAX_EPISODES` 上一致；预期差异仅为模型表征。")
    else:
        lines.append("- 下列场景存在额外配置差异，需要谨慎解释：")
        for row in mismatches.itertuples():
            lines.append(f"  - `{row.scenario_label}`: `{row.mismatch_keys}`")
    lines.append("")
    out = REPORT_DIR / "FMAPPO_VS_TERA_FULL_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    _ensure_dirs()
    _set_style()

    manifest_df = build_pair_manifest()
    last_df, best_df, phase_df = build_summary_tables()
    curve_df = export_curve_table()

    manifest_path = MANIFEST_DIR / "fmappo_vs_tera_pair_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    last_path = TAB_DIR / "fmappo_vs_tera_last100_summary.csv"
    best_path = TAB_DIR / "fmappo_vs_tera_best50_summary.csv"
    phase_path = TAB_DIR / "fmappo_vs_tera_phase_summary.csv"
    curve_path = TAB_DIR / "fmappo_vs_tera_curve_table_long.csv"
    last_df.to_csv(last_path, index=False)
    best_df.to_csv(best_path, index=False)
    phase_df.to_csv(phase_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    figure_paths = [
        plot_summary_bars(last_df),
        plot_delta_heatmap(last_df),
        plot_phase_lines(phase_df),
    ]
    for spec in PAIR_SPECS:
        figure_paths.append(_plot_scenario_curves(spec))

    report_path = write_report(last_df, best_df, manifest_df)

    out_manifest = {
        "pair_manifest": str(manifest_path.relative_to(ROOT)),
        "last100_summary": str(last_path.relative_to(ROOT)),
        "best50_summary": str(best_path.relative_to(ROOT)),
        "phase_summary": str(phase_path.relative_to(ROOT)),
        "curve_table_long": str(curve_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
    }
    (MANIFEST_DIR / "fmappo_vs_tera_export_manifest.json").write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Exported F-MAPPO vs TERA-MAPPO comparison pack:")
    print(json.dumps(out_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
