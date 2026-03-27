#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGO_FULL_NAME = "TERA-MAPPO (Topology-Enhanced and Resource-Aware MAPPO)"
PALETTE = ["#1f77b4", "#d62728", "#2ca02c"]
SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _set_cn_style() -> None:
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
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["axes.labelsize"] = 11
    matplotlib.rcParams["legend.fontsize"] = 9.5
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fcfcfc"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="中文论文版学习率 sweep 绘图脚本")
    ap.add_argument("--run", action="append", required=True, help="训练 run 目录")
    ap.add_argument("--label", action="append", default=None, help="可选图例标签")
    ap.add_argument("--out-dir", type=str, required=True, help="输出目录")
    ap.add_argument("--band-window", type=int, default=50, help="波动带窗口")
    ap.add_argument("--smooth-window", type=int, default=50, help="主曲线平滑窗口")
    ap.add_argument("--tail-window", type=int, default=100, help="尾部统计窗口")
    return ap.parse_args()


def _load_snapshot(run_dir: Path) -> Dict[str, object]:
    with (run_dir / "logs" / "config_snapshot.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_training(run_dir: Path) -> pd.DataFrame:
    repair_csv = run_dir / "diagnostics" / "repair" / "logs" / "training_stats.csv"
    raw_csv = run_dir / "logs" / "training_stats.csv"
    csv_path = repair_csv if repair_csv.exists() else raw_csv
    df = pd.read_csv(csv_path)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"]).copy()
    df["episode"] = df["episode"].astype(int)
    df = df.sort_values("episode").drop_duplicates(subset=["episode"], keep="last")
    return df


def _format_sci(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coeff = value / (10 ** exponent)
    coeff_round = round(coeff, 2)
    if abs(coeff_round - 1.0) < 1e-8:
        return f"10{str(exponent).translate(SUPERSCRIPTS)}"
    coeff_text = f"{coeff_round:g}"
    return f"{coeff_text}×10{str(exponent).translate(SUPERSCRIPTS)}"


def _auto_label(lr_critic: float) -> str:
    return f"lr_c={_format_sci(lr_critic)}"


def _rolling_mean_std(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.rolling(window=window, min_periods=1).mean()
    std = numeric.rolling(window=window, min_periods=1).std().fillna(0.0)
    return mean, std


def _smooth(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(window=window, min_periods=1).mean()


def _clip_bounds(metric: str, lower: pd.Series, upper: pd.Series) -> tuple[pd.Series, pd.Series]:
    if metric in {"task_sr", "subtask_sr", "vehicle_sr", "deadline_miss_rate", "ratio_local", "ratio_rsu", "ratio_v2v", "clip_frac"}:
        return lower.clip(0.0, 1.0), upper.clip(0.0, 1.0)
    if metric in {"entropy", "active_ratio", "actor_update_active_frac", "value_clip_fraction"}:
        return lower.clip(lower=0.0), upper.clip(lower=0.0)
    return lower, upper


def _first_convergence_episode(
    series: pd.Series,
    *,
    direction: str,
    tail_window: int,
    smooth_window: int = 35,
    stable_len: int = 30,
    ratio: float = 0.95,
) -> float:
    values = _smooth(series, smooth_window)
    tail = pd.to_numeric(series, errors="coerce").tail(min(tail_window, len(series))).mean()
    if pd.isna(tail):
        return float("nan")

    if direction == "higher":
        threshold = tail * ratio
        flags = values >= threshold
    else:
        threshold = tail / ratio if ratio > 0 else tail
        flags = values <= threshold

    run_len = 0
    for idx, flag in enumerate(flags.fillna(False).to_list()):
        run_len = run_len + 1 if flag else 0
        if run_len >= stable_len:
            return float(idx + 1 - stable_len + 1)
    return float("nan")


def _tail_mean(df: pd.DataFrame, metric: str, tail_window: int) -> float:
    if metric not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[metric], errors="coerce").tail(min(tail_window, len(df))).mean())


def _build_run_frames(args: argparse.Namespace) -> List[Dict[str, object]]:
    labels = args.label or []
    if labels and len(labels) != len(args.run):
        raise ValueError("--label 的数量必须与 --run 保持一致。")

    frames: List[Dict[str, object]] = []
    for idx, run in enumerate(args.run):
        run_dir = Path(run).resolve()
        snapshot = _load_snapshot(run_dir)
        tc = snapshot["train_config"]
        lr_actor = float(tc["LR_ACTOR"])
        lr_critic = float(tc["LR_CRITIC"])
        frames.append(
            {
                "run_dir": run_dir,
                "df": _load_training(run_dir),
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "label": labels[idx] if idx < len(labels) else _auto_label(lr_critic),
            }
        )

    frames.sort(key=lambda item: item["lr_critic"])
    for idx, item in enumerate(frames):
        item["color"] = PALETTE[idx % len(PALETTE)]
    return frames


def _style_axis(ax: plt.Axes, *, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None) -> None:
    if title:
        ax.set_title(title, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _add_axis_legend(ax: plt.Axes, loc: str = "best", ncol: int = 1) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    dedup: Dict[str, object] = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc=loc,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="#cccccc",
    )


def _plot_metric_with_band(
    ax: plt.Axes,
    run_frames: List[Dict[str, object]],
    metric: str,
    title: str,
    ylabel: str,
    smooth_window: int,
    band_window: int,
) -> None:
    for item in run_frames:
        df = item["df"]
        mean, std = _rolling_mean_std(df[metric], band_window)
        if smooth_window != band_window:
            mean = _smooth(mean, smooth_window)
        lower, upper = _clip_bounds(metric, mean - std, mean + std)
        ax.fill_between(df["episode"], lower, upper, color=item["color"], alpha=0.12)
        ax.plot(df["episode"], mean, color=item["color"], linewidth=2.2, label=item["label"])
    _style_axis(ax, ylabel=ylabel, title=title)


def _plot_main_training(
    out_dir: Path,
    run_frames: List[Dict[str, object]],
    smooth_window: int,
    band_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), sharex=True)
    _plot_metric_with_band(axes[0, 0], run_frames, "reward_mean", "回报收敛曲线", "平均奖励", smooth_window, band_window)
    _plot_metric_with_band(axes[0, 1], run_frames, "task_sr", "任务成功率收敛", "任务成功率", smooth_window, band_window)
    _plot_metric_with_band(axes[1, 0], run_frames, "deadline_miss_rate", "超期率收敛", "超期率", smooth_window, band_window)
    _plot_metric_with_band(axes[1, 1], run_frames, "avg_rsu_queue", "RSU 平均排队长度", "平均 RSU 队列长度", smooth_window, band_window)

    axes[1, 0].set_xlabel("训练回合 / Episode")
    axes[1, 1].set_xlabel("训练回合 / Episode")
    _add_axis_legend(axes[0, 0], loc="lower right")
    _add_axis_legend(axes[0, 1], loc="lower right")
    _add_axis_legend(axes[1, 0], loc="upper right")
    _add_axis_legend(axes[1, 1], loc="upper right")
    fig.suptitle(
        f"{ALGO_FULL_NAME}\n不同评论器学习率下的训练收敛对比（$lr_a$ 固定为 {_format_sci(run_frames[0]['lr_actor'])}）",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(0.5, 0.02, "注：实线表示滚动均值，浅色阴影表示相同窗口下的波动范围（均值±1σ）。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, out_dir, "fig_lr_main_training_cn")
    plt.close(fig)


def _plot_stability_diagnostics(
    out_dir: Path,
    run_frames: List[Dict[str, object]],
    smooth_window: int,
    band_window: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.2), sharex=True)

    for item in run_frames:
        df = item["df"]
        reward_mean, reward_std = _rolling_mean_std(df["reward_mean"], band_window)
        task_sr = _smooth(df["task_sr"], smooth_window)
        task_vol = pd.Series(np.abs(np.diff(task_sr.to_numpy(), prepend=float(task_sr.iloc[0]))))
        entropy = _smooth(df["entropy"], smooth_window)
        approx_kl = _smooth(df["approx_kl"], smooth_window)
        clip_frac = _smooth(df["clip_frac"], smooth_window)
        grad_norm = _smooth(df["grad_norm"], smooth_window)

        axes[0, 0].plot(df["episode"], reward_std, color=item["color"], linewidth=2.0, label=item["label"])
        axes[0, 1].plot(df["episode"], task_vol, color=item["color"], linewidth=2.0, label=item["label"])
        axes[0, 2].plot(df["episode"], entropy, color=item["color"], linewidth=2.0, label=item["label"])
        axes[1, 0].plot(df["episode"], approx_kl, color=item["color"], linewidth=2.0, label=item["label"])
        axes[1, 1].plot(df["episode"], clip_frac, color=item["color"], linewidth=2.0, label=item["label"])
        axes[1, 2].plot(df["episode"], grad_norm, color=item["color"], linewidth=2.0, label=item["label"])

    _style_axis(axes[0, 0], ylabel="奖励滚动标准差", title="训练阶段奖励波动")
    _style_axis(axes[0, 1], ylabel="相邻回合变化幅度", title="任务成功率波动强度")
    _style_axis(axes[0, 2], ylabel="策略熵", title="策略探索程度")
    _style_axis(axes[1, 0], xlabel="训练回合 / Episode", ylabel="Approx KL", title="策略更新幅度")
    _style_axis(axes[1, 1], xlabel="训练回合 / Episode", ylabel="Clip Fraction", title="裁剪比例变化")
    _style_axis(axes[1, 2], xlabel="训练回合 / Episode", ylabel="梯度范数", title="梯度稳定性")

    for ax, loc in zip(
        axes.flat,
        ["upper right", "upper right", "upper right", "upper right", "upper right", "upper right"],
    ):
        _add_axis_legend(ax, loc=loc)
    fig.suptitle("TERA-MAPPO 在不同学习率下的训练稳定性与优化诊断", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.02, "注：该图用于展示优化过程是否平稳，所有曲线均采用滑动平均以减少随机抖动干扰。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, out_dir, "fig_lr_stability_diagnostics_cn")
    plt.close(fig)


def _plot_decision_mix(
    out_dir: Path,
    run_frames: List[Dict[str, object]],
    smooth_window: int,
    band_window: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharex=True, sharey=True)
    specs = [
        ("ratio_local", "本地执行比例"),
        ("ratio_rsu", "RSU 卸载比例"),
        ("ratio_v2v", "V2V 卸载比例"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        _plot_metric_with_band(ax, run_frames, metric, title, "比例", smooth_window, band_window)
        ax.set_xlabel("训练回合 / Episode")
        _add_axis_legend(ax, loc="best")

    fig.suptitle("TERA-MAPPO 在不同学习率下的决策结构演化", fontsize=15, fontweight="bold", y=1.08)
    fig.text(0.5, 0.01, "注：比例曲线反映不同训练阶段的本地执行、RSU 卸载与 V2V 卸载偏好。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_dir, "fig_lr_decision_mix_cn")
    plt.close(fig)


def _build_summary_df(run_frames: List[Dict[str, object]], tail_window: int) -> pd.DataFrame:
    rows = []
    for item in run_frames:
        df = item["df"]
        rows.append(
            {
                "label": item["label"],
                "run_dir": str(item["run_dir"]),
                "lr_actor": item["lr_actor"],
                "lr_critic": item["lr_critic"],
                "episodes": int(len(df)),
                "reward_mean_last_tail": _tail_mean(df, "reward_mean", tail_window),
                "task_sr_last_tail": _tail_mean(df, "task_sr", tail_window),
                "deadline_miss_rate_last_tail": _tail_mean(df, "deadline_miss_rate", tail_window),
                "avg_rsu_queue_last_tail": _tail_mean(df, "avg_rsu_queue", tail_window),
                "entropy_last_tail": _tail_mean(df, "entropy", tail_window),
                "approx_kl_last_tail": _tail_mean(df, "approx_kl", tail_window),
                "task_sr_convergence_ep": _first_convergence_episode(
                    df["task_sr"], direction="higher", tail_window=tail_window
                ),
                "reward_convergence_ep": _first_convergence_episode(
                    df["reward_mean"], direction="higher", tail_window=tail_window
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("lr_critic")


def _plot_tail_tradeoff(out_dir: Path, summary_df: pd.DataFrame) -> None:
    x_labels = [_format_sci(v) for v in summary_df["lr_critic"].to_numpy(dtype=float)]
    x = np.arange(len(summary_df))

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
    specs = [
        ("reward_mean_last_tail", "尾部平均奖励", "平均奖励"),
        ("task_sr_last_tail", "尾部任务成功率", "任务成功率"),
        ("deadline_miss_rate_last_tail", "尾部超期率", "超期率"),
        ("task_sr_convergence_ep", "达到稳定性能的回合数", "回合数"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        y = summary_df[metric].to_numpy(dtype=float)
        ax.plot(x, y, color="#7a7a7a", linewidth=1.4, alpha=0.7, zorder=1)
        for idx, row in summary_df.reset_index(drop=True).iterrows():
            ax.scatter(idx, row[metric], s=90, color=PALETTE[idx % len(PALETTE)], zorder=3)
            label_text = f"{row[metric]:.3f}" if metric != "task_sr_convergence_ep" else f"{int(row[metric])}"
            ax.text(idx, row[metric], f" {label_text}", va="bottom", ha="left", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        _style_axis(ax, ylabel=ylabel, title=title)
        ax.set_xlabel("评论器学习率 $lr_c$")
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[i % len(PALETTE)], markersize=8, label=summary_df.iloc[i]["label"])
            for i in range(len(summary_df))
        ]
        ax.legend(handles=legend_handles, loc="best", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    fig.suptitle("TERA-MAPPO 不同学习率下的尾部性能与收敛速度对比", fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.02, "注：尾部性能按最后 100 回合统计；收敛回合数定义为首次稳定达到尾部性能 95% 的训练位置。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, out_dir, "fig_lr_tail_tradeoff_cn")
    plt.close(fig)


def _write_report(out_dir: Path, summary_df: pd.DataFrame, tail_window: int) -> None:
    best_reward = summary_df.loc[summary_df["reward_mean_last_tail"].idxmax()]
    best_task_sr = summary_df.loc[summary_df["task_sr_last_tail"].idxmax()]
    best_deadline = summary_df.loc[summary_df["deadline_miss_rate_last_tail"].idxmin()]
    fastest = summary_df.loc[summary_df["task_sr_convergence_ep"].idxmin()]

    lines = [
        "# 不同学习率训练对比报告",
        "",
        f"算法首次标注：{ALGO_FULL_NAME}",
        "",
        "## 对比设置",
        f"- 比较对象：TERA-MAPPO 在不同评论器学习率下的训练过程。",
        f"- 固定设置：actor 学习率固定为 {_format_sci(float(summary_df['lr_actor'].iloc[0]))}。",
        f"- 统计窗口：尾部 {tail_window} 回合。",
        "",
        "## 核心结论",
        f"- 尾部平均奖励最高：{best_reward['label']}，对应奖励 {best_reward['reward_mean_last_tail']:.4f}。",
        f"- 尾部任务成功率最高：{best_task_sr['label']}，对应成功率 {best_task_sr['task_sr_last_tail']:.4f}。",
        f"- 尾部超期率最低：{best_deadline['label']}，对应超期率 {best_deadline['deadline_miss_rate_last_tail']:.4f}。",
        f"- 收敛速度最快：{fastest['label']}，稳定达到尾部性能 95% 的回合约为 {int(fastest['task_sr_convergence_ep'])}。",
        "",
        "## 各学习率尾部统计",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "- "
            f"{row['label']}: reward={row['reward_mean_last_tail']:.4f}, "
            f"task_sr={row['task_sr_last_tail']:.4f}, "
            f"deadline_miss={row['deadline_miss_rate_last_tail']:.4f}, "
            f"avg_rsu_queue={row['avg_rsu_queue_last_tail']:.4f}, "
            f"task_sr_convergence_ep={int(row['task_sr_convergence_ep']) if pd.notna(row['task_sr_convergence_ep']) else 'nan'}"
        )

    with (out_dir / "LR_SWEEP_CN_REPORT.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = _parse_args()
    _set_cn_style()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_frames = _build_run_frames(args)
    summary_df = _build_summary_df(run_frames, tail_window=args.tail_window)
    summary_df.to_csv(out_dir / "lr_sweep_cn_summary.csv", index=False)

    _plot_main_training(out_dir, run_frames, smooth_window=args.smooth_window, band_window=args.band_window)
    _plot_stability_diagnostics(out_dir, run_frames, smooth_window=args.smooth_window, band_window=args.band_window)
    _plot_decision_mix(out_dir, run_frames, smooth_window=args.smooth_window, band_window=args.band_window)
    _plot_tail_tradeoff(out_dir, summary_df)
    _write_report(out_dir, summary_df, tail_window=args.tail_window)

    print(f"已生成中文论文版学习率对比图：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
