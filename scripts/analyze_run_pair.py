#!/usr/bin/env python3
"""
Generate per-run diagnosis reports and a combined A/B comparison report.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


KEY_METRICS = [
    ("reward_mean", "平均步奖励", "higher"),
    ("task_sr", "任务成功率", "higher"),
    ("subtask_sr", "子任务成功率", "higher"),
    ("deadline_miss_rate", "超时失败率", "lower"),
    ("avg_rsu_queue", "平均RSU队列", "lower"),
    ("avg_power", "平均功率", "lower"),
    ("approx_kl", "Approx KL", "lower"),
    ("entropy", "策略熵", "neutral"),
]

BASELINE_METRICS = [
    ("reward_mean", "reward_mean", "平均步奖励", "higher"),
    ("task_sr", "task_sr", "任务成功率", "higher"),
    ("subtask_sr", "subtask_sr", "子任务成功率", "higher"),
    ("deadline_miss_rate", "deadline_miss_rate", "超时失败率", "lower"),
    ("avg_power", "avg_power", "平均功率", "lower"),
    ("avg_rsu_queue", "avg_rsu_queue", "平均RSU队列", "lower"),
]


def _rolling(series: pd.Series, window: int = 50) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.rolling(window=min(window, max(1, len(series))), min_periods=1).mean()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_snapshot(run_dir: Path) -> dict:
    snapshot = run_dir / "logs" / "config_snapshot.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    config = run_dir / "config.json"
    if config.exists():
        return json.loads(config.read_text())
    return {}


def _read_manifest(repair_dir: Path) -> dict:
    manifest = repair_dir / "repair_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    return {}


def _pick_col(df: pd.DataFrame, *cols: str) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def _metric_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _summarize_run(run_dir: Path, repair_dir: Path) -> dict:
    metrics = _load_csv(repair_dir / "metrics" / "train_metrics.csv")
    training = _load_csv(repair_dir / "logs" / "training_stats.csv")
    episode = _load_csv(repair_dir / "episode_log.csv")
    baseline = _load_csv(repair_dir / "logs" / "baseline_stats.csv")
    snapshot = _load_snapshot(run_dir)
    manifest = _read_manifest(repair_dir)

    if "task_sr" not in metrics.columns and "task_success_rate" in metrics.columns:
        metrics["task_sr"] = metrics["task_success_rate"]
    if "subtask_sr" not in metrics.columns and "subtask_success_rate" in metrics.columns:
        metrics["subtask_sr"] = metrics["subtask_success_rate"]

    tail100 = metrics.tail(100).copy()
    tail300 = metrics.tail(min(300, len(metrics))).copy()
    roll50 = _rolling(pd.to_numeric(metrics["task_sr"], errors="coerce"), 50) if "task_sr" in metrics.columns else None
    best50_task_sr = float(roll50.max()) if roll50 is not None else float("nan")
    best50_episode = int(metrics.loc[roll50.idxmax(), "episode"]) if roll50 is not None and roll50.notna().any() else -1

    summary = {
        "run_dir": str(run_dir),
        "repair_dir": str(repair_dir),
        "metrics": metrics,
        "training": training,
        "episode": episode,
        "baseline": baseline,
        "snapshot": snapshot,
        "manifest": manifest,
        "lr_actor": snapshot.get("train_config", {}).get("LR_ACTOR"),
        "lr_critic": snapshot.get("train_config", {}).get("LR_CRITIC"),
        "max_steps": snapshot.get("system_config", {}).get("MAX_STEPS"),
        "tail100": {},
        "tail300": {},
        "best50_task_sr": best50_task_sr,
        "best50_episode": best50_episode,
    }

    for metric_id, _, _ in KEY_METRICS:
        if metric_id in metrics.columns:
            summary["tail100"][metric_id] = float(pd.to_numeric(tail100[metric_id], errors="coerce").mean())
            summary["tail300"][metric_id] = float(pd.to_numeric(tail300[metric_id], errors="coerce").mean())

    baseline_rows = []
    for policy in sorted(baseline["policy"].unique().tolist()):
        base_df = baseline[baseline["policy"] == policy].copy()
        for rl_col, bl_col, metric_name, direction in BASELINE_METRICS:
            if rl_col not in metrics.columns or bl_col not in base_df.columns:
                continue
            rl = pd.to_numeric(metrics[rl_col], errors="coerce").dropna().to_numpy()
            bl = pd.to_numeric(base_df[bl_col], errors="coerce").dropna().to_numpy()
            k = int(min(100, len(rl), len(bl)))
            if k <= 0:
                continue
            rl_tail = rl[-k:]
            bl_tail = bl[-k:]
            rl_mean = float(np.mean(rl_tail))
            bl_mean = float(np.mean(bl_tail))
            oriented_delta = rl_mean - bl_mean if direction == "higher" else bl_mean - rl_mean
            baseline_rows.append(
                {
                    "policy": policy,
                    "metric_id": rl_col,
                    "metric_name": metric_name,
                    "direction": direction,
                    "matched_tail_k": k,
                    "rl_mean": rl_mean,
                    "baseline_mean": bl_mean,
                    "oriented_delta": oriented_delta,
                }
            )
    summary["baseline_compare"] = pd.DataFrame(baseline_rows)
    return summary


def _write_single_run_report(run_name: str, run_summary: dict, out_path: Path) -> None:
    metrics = run_summary["metrics"]
    manifest = run_summary["manifest"].get("files", {})
    baseline_compare = run_summary["baseline_compare"]
    tail100 = run_summary["tail100"]
    tail300 = run_summary["tail300"]

    lines = [
        f"# {run_name} 训练诊断报告",
        "",
        "## 真实生效配置",
        f"- lr_actor={run_summary['lr_actor']}",
        f"- lr_critic={run_summary['lr_critic']}",
        f"- max_steps={run_summary['max_steps']}",
        f"- 训练回合数={len(metrics)}",
        "",
        "## 修复审计",
    ]
    for src, meta in manifest.items():
        if meta.get("status") != "ok":
            continue
        lines.append(
            f"- {Path(src).name}: raw={meta['raw_rows']} clean={meta['clean_rows']} "
            f"header_removed={meta['header_rows_removed']} duplicate_removed={meta['duplicate_rows_removed']} "
            f"invalid_episode_removed={meta['invalid_episode_rows_removed']}"
        )
    lines.extend(
        [
            "",
            "## 尾段训练表现",
            f"- last100 reward_mean={tail100.get('reward_mean', float('nan')):.6f}",
            f"- last100 task_sr={tail100.get('task_sr', float('nan')):.6f}",
            f"- last100 subtask_sr={tail100.get('subtask_sr', float('nan')):.6f}",
            f"- last100 deadline_miss_rate={tail100.get('deadline_miss_rate', float('nan')):.6f}",
            f"- last100 avg_rsu_queue={tail100.get('avg_rsu_queue', float('nan')):.6f}",
            f"- last100 avg_power={tail100.get('avg_power', float('nan')):.6f}",
            f"- best50 task_sr={run_summary['best50_task_sr']:.6f} @ episode={run_summary['best50_episode']}",
            f"- last300 vs last100 reward drift={tail100.get('reward_mean', float('nan')) - tail300.get('reward_mean', float('nan')):.6f}",
            "",
            "## Baseline matched-tail 对比",
        ]
    )
    if baseline_compare.empty:
        lines.append("- 无 baseline 结果。")
    else:
        focus = baseline_compare[baseline_compare["metric_id"].isin(["reward_mean", "task_sr", "deadline_miss_rate"])]
        for _, row in focus.sort_values(["metric_id", "policy"]).iterrows():
            lines.append(
                f"- {row['metric_name']} vs {row['policy']}: rl={row['rl_mean']:.6f}, "
                f"baseline={row['baseline_mean']:.6f}, oriented_delta={row['oriented_delta']:.6f}"
            )
    lines.extend(
        [
            "",
            "## 诊断结论",
            "- 若 last100 与 last300 差异仍明显，说明训练仍在漂移而非完全稳定。",
            "- 若 deadline_miss_rate 与 avg_rsu_queue 同步升高，说明策略在更激进卸载下出现队列拥塞。",
            "- 若 reward 提升但 task_sr / deadline_miss 未同步改善，说明奖励项与主任务质量可能存在权衡。",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_comparison(summary_a: dict, summary_b: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()
    colors = {"A": "#2563eb", "B": "#dc2626"}
    labels = {
        "A": f"Run A (lr_critic={summary_a['lr_critic']})",
        "B": f"Run B (lr_critic={summary_b['lr_critic']})",
    }
    for ax, (metric_id, metric_name, _) in zip(axes, KEY_METRICS):
        for tag, summary in [("A", summary_a), ("B", summary_b)]:
            df = summary["metrics"]
            if metric_id not in df.columns:
                continue
            y = pd.to_numeric(df[metric_id], errors="coerce")
            ax.plot(df["episode"], _rolling(y, 50), label=labels[tag], color=colors[tag], linewidth=2)
        ax.set_title(metric_name)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        if metric_id in {"task_sr", "subtask_sr"}:
            ax.set_ylim(0, 1.05)
    axes[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "ab_key_metrics.png", dpi=220, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (tag, summary) in zip(axes, [("A", summary_a), ("B", summary_b)]):
        df = summary["episode"].tail(100)
        values = [
            pd.to_numeric(df[col], errors="coerce").mean() * 100
            for col in ["decision_frac_local", "decision_frac_rsu", "decision_frac_v2v"]
        ]
        ax.bar(["Local", "RSU", "V2V"], values, color=["#16a34a", "#2563eb", "#f59e0b"])
        ax.set_ylim(0, 100)
        ax.set_title(f"Run {tag} 最后100回合卸载分布")
    plt.tight_layout()
    plt.savefig(out_dir / "ab_offloading_tail100.png", dpi=220, bbox_inches="tight")
    plt.close()


def _config_diff(summary_a: dict, summary_b: dict) -> pd.DataFrame:
    snap_a = summary_a["snapshot"]
    snap_b = summary_b["snapshot"]
    rows = []
    for section in ["train_config", "system_config"]:
        sa = snap_a.get(section, {})
        sb = snap_b.get(section, {})
        for key in sorted(set(sa.keys()) | set(sb.keys())):
            if sa.get(key) != sb.get(key):
                rows.append({"section": section, "key": key, "run_a": sa.get(key), "run_b": sb.get(key)})
    return pd.DataFrame(rows)


def _winner(a: float, b: float, direction: str) -> str:
    if pd.isna(a) or pd.isna(b):
        return "N/A"
    if direction == "higher":
        return "A" if a > b else "B"
    if direction == "lower":
        return "A" if a < b else "B"
    return "A" if a > b else "B"


def _write_combined_report(summary_a: dict, summary_b: dict, config_diff: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for metric_id, metric_name, direction in KEY_METRICS:
        a = summary_a["tail100"].get(metric_id, float("nan"))
        b = summary_b["tail100"].get(metric_id, float("nan"))
        rows.append(
            {
                "metric_id": metric_id,
                "metric_name": metric_name,
                "direction": direction,
                "run_a_last100": a,
                "run_b_last100": b,
                "winner": _winner(a, b, direction),
            }
        )
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(out_dir / "ab_tail100_summary.csv", index=False)
    config_diff.to_csv(out_dir / "ab_config_diff.csv", index=False)
    summary_a["baseline_compare"].to_csv(out_dir / "run_a_baseline_compare.csv", index=False)
    summary_b["baseline_compare"].to_csv(out_dir / "run_b_baseline_compare.csv", index=False)

    lines = [
        "# A/B 双 Run 深度对比报告",
        "",
        "## 概览",
        f"- Run A: lr_critic={summary_a['lr_critic']}, repaired_metrics_rows={len(summary_a['metrics'])}",
        f"- Run B: lr_critic={summary_b['lr_critic']}, repaired_metrics_rows={len(summary_b['metrics'])}",
        "",
        "## 真实配置差异",
    ]
    if config_diff.empty:
        lines.append("- 未检测到有效配置差异。")
    else:
        for _, row in config_diff.iterrows():
            lines.append(f"- {row['section']}.{row['key']}: A={row['run_a']} | B={row['run_b']}")

    lines.extend(["", "## last100 指标对比"])
    for _, row in compare_df.iterrows():
        lines.append(
            f"- {row['metric_name']}: A={row['run_a_last100']:.6f}, "
            f"B={row['run_b_last100']:.6f}, winner={row['winner']}"
        )

    lines.extend(
        [
            "",
            "## 结论与解释",
            f"- 在任务成功率上，赢家={compare_df[compare_df['metric_id'] == 'task_sr']['winner'].iloc[0] if not compare_df.empty else 'N/A'}。",
            f"- 在平均奖励上，赢家={compare_df[compare_df['metric_id'] == 'reward_mean']['winner'].iloc[0] if not compare_df.empty else 'N/A'}。",
            f"- 在超时失败率上，赢家={compare_df[compare_df['metric_id'] == 'deadline_miss_rate']['winner'].iloc[0] if not compare_df.empty else 'N/A'}。",
            "- 若 B 的 reward 更高但 task_sr / deadline_miss_rate 更差，说明更高 critic 学习率提升了优化速度，但牺牲了稳定可交付性。",
            "- 若 A 的 avg_rsu_queue 明显更高而 deadline_miss 更低，说明 A 通过更积极利用 RSU 获得成功率，但系统拥塞代价更大。",
            "- 需同时看 reward、success、deadline、queue、power，而不能只看 reward 单指标。",
            "",
            "## 已确认代码问题",
            "- DataRecorder 原先仅靠进程内标志写 header，复用 exact run-dir 时会把表头重复写入 episode_log.csv。",
            "- train.py 原先允许 exact run-dir 在已有非空训练 CSV 时继续写入，容易产生脏日志和重复 episode。",
            "- generate_all_plots.py 对 object 列直接 rolling，遇到脏行会抛 No numeric types to aggregate。",
            "- 多个绘图模块使用过期 baseline 列表，和当前训练实际 baseline 集合不一致。",
            "",
            "## 建议",
            "- 立即修复项: 保持本次已实现的日志保护和 header 防护，不再复用脏 run-dir 直接追加训练。",
            "- 下一轮实验项: 在 lr_critic=2e-4、3e-4、5e-4 三档做固定 seed 对照，重点看 task_sr、deadline_miss_rate、avg_rsu_queue 的联动。",
            "- 下一轮实验项: 保持当前 reward 权重不变，单独扫描 queue / timeout 相关项，确认 reward 提升是否来自真实任务质量而非指标偏置。",
            "- 口径治理项: 所有 baseline 名称、CSV schema、后处理脚本字段别名保持单一来源，不再各脚本手写一套列表。",
        ]
    )
    (out_dir / "AB_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze two repaired runs and write reports")
    parser.add_argument("--run-a", type=str, required=True)
    parser.add_argument("--run-b", type=str, required=True)
    parser.add_argument("--repair-a", type=str, required=True)
    parser.add_argument("--repair-b", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    repair_a = Path(args.repair_a).resolve()
    repair_b = Path(args.repair_b).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_a = _summarize_run(run_a, repair_a)
    summary_b = _summarize_run(run_b, repair_b)

    _write_single_run_report("Run A", summary_a, run_a / "diagnostics" / "run_diagnosis.md")
    _write_single_run_report("Run B", summary_b, run_b / "diagnostics" / "run_diagnosis.md")
    cfg_diff = _config_diff(summary_a, summary_b)
    _plot_comparison(summary_a, summary_b, out_dir)
    _write_combined_report(summary_a, summary_b, cfg_diff, out_dir)

    print(f"✓ Combined report directory: {out_dir}")
    print(f"✓ Run A report: {run_a / 'diagnostics' / 'run_diagnosis.md'}")
    print(f"✓ Run B report: {run_b / 'diagnostics' / 'run_diagnosis.md'}")


if __name__ == "__main__":
    main()
