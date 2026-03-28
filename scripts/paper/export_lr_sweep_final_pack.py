#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "runs" / "paper_final_results_20260327"
PACK_ROOT = OUT_ROOT / "lr_critic_sweep"
FIG_DIR = PACK_ROOT / "figures"
TAB_DIR = PACK_ROOT / "tables"
REPORT_DIR = PACK_ROOT / "reports"
MANIFEST_DIR = PACK_ROOT / "manifests"
ROOT_README = OUT_ROOT / "README.md"

RUN_SPECS = [
    ("lr_c=2e-4", ROOT / "runs" / "lr_critic_1500ep_20260327_163712" / "lr_c2e4"),
    ("lr_c=3e-4", ROOT / "runs" / "lr_critic_1500ep_20260327_163712" / "lr_c3e4"),
    ("lr_c=5e-4", ROOT / "runs" / "lr_critic_1500ep_20260327_163712" / "lr_c5e4"),
]

PALETTE = {
    "lr_c=2e-4": "#1f77b4",
    "lr_c=3e-4": "#d62728",
    "lr_c=5e-4": "#2ca02c",
}

METRICS_MAIN = [
    ("reward_mean", "平均奖励", "平均奖励"),
    ("reward_total", "总奖励", "总奖励"),
    ("task_sr", "任务成功率", "任务成功率"),
    ("deadline_miss_rate", "截止期违约率", "截止期违约率"),
    ("mean_cft_completed", "已完成任务平均 CFT", "平均 CFT"),
    ("avg_rsu_queue", "RSU 平均队列长度", "队列长度"),
]

METRICS_DIAG = [
    ("approx_kl", "近似 KL 散度", "近似 KL"),
    ("entropy", "策略熵", "策略熵"),
    ("clip_frac", "裁剪比例", "裁剪比例"),
]

METRICS_DECISION = [
    ("ratio_local", "本地执行比例", "比例"),
    ("ratio_rsu", "RSU 卸载比例", "比例"),
    ("ratio_v2v", "V2V 卸载比例", "比例"),
]

LEGEND_SIZE = (0.22, 0.18)
LEGEND_CANDIDATES = [
    (0.03, 0.86),
    (0.75, 0.79),
    (0.03, 0.03),
    (0.75, 0.03),
]


def _ensure_dirs() -> None:
    for path in (FIG_DIR, TAB_DIR, REPORT_DIR, MANIFEST_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _clear_export_dirs() -> None:
    for directory in (FIG_DIR, TAB_DIR, REPORT_DIR, MANIFEST_DIR):
        for path in directory.iterdir():
            if path.is_file():
                if "hypothetical" in path.name:
                    continue
                path.unlink()


def _write_root_readme() -> None:
    ROOT_README.write_text(
        "\n".join(
            [
                "# Paper Final Results",
                "",
                "This directory stores thesis-ready and paper-ready exported result packs.",
                "",
                "Current packs:",
                "- `lr_critic_sweep/figures`: final publication-style figures",
                "- `lr_critic_sweep/tables`: plotting-ready 2D source tables",
                "- `lr_critic_sweep/reports`: text summaries",
                "- `lr_critic_sweep/manifests`: export manifests",
                "",
                "Additional experiment packs should be exported as peer subfolders under this directory.",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    matplotlib.rcParams["font.size"] = 11.5
    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["axes.labelsize"] = 11.5
    matplotlib.rcParams["legend.fontsize"] = 9.5
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fcfcfc"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _load_training(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "logs" / "training_stats.csv"
    df = pd.read_csv(path).copy()
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"]).copy()
    df["episode"] = df["episode"].astype(int)
    df = df.sort_values("episode").drop_duplicates(subset=["episode"], keep="last")
    return df


def _load_baseline_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "logs" / "baseline_eval_core_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _common_episode_limit(run_frames: List[Dict[str, object]]) -> int:
    return int(min(int(frame["df"]["episode"].max()) for frame in run_frames))


def _rolling(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(window=window, min_periods=max(1, window // 3)).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(window=window, min_periods=max(1, window // 3)).std().fillna(0.0)


def _clip_interval(metric: str, low: pd.Series, high: pd.Series) -> tuple[pd.Series, pd.Series]:
    bounded = {"task_sr", "deadline_miss_rate", "ratio_local", "ratio_rsu", "ratio_v2v", "clip_frac"}
    if metric in bounded:
        return low.clip(0.0, 1.0), high.clip(0.0, 1.0)
    return low, high


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return not (ax0 + aw <= bx0 or bx0 + bw <= ax0 or ay0 + ah <= by0 or by0 + bh <= ay0)


def _curve_box_score(
    ax: plt.Axes,
    curves: List[tuple[np.ndarray, np.ndarray]],
    box: tuple[float, float, float, float],
) -> float:
    x0, y0, w, h = box
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    score = 0.0
    for xvals, yvals in curves:
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        if not np.any(mask):
            continue
        xn = (xvals[mask] - xmin) / (xmax - xmin)
        yn = (yvals[mask] - ymin) / (ymax - ymin)
        inside = (xn >= x0) & (xn <= x0 + w) & (yn >= y0) & (yn <= y0 + h)
        score += float(np.count_nonzero(inside))
    return score


def _choose_layout(
    ax: plt.Axes,
    curves: List[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    legend_boxes = [(x, y, LEGEND_SIZE[0], LEGEND_SIZE[1]) for x, y in LEGEND_CANDIDATES]
    best_score = float("inf")
    best_legend = legend_boxes[0]
    for legend_box in legend_boxes:
        legend_score = _curve_box_score(ax, curves, legend_box)
        if legend_score < best_score:
            best_score = legend_score
            best_legend = legend_box
    return best_legend[0], best_legend[1]


def _build_frames() -> List[Dict[str, object]]:
    frames = []
    for label, run_dir in RUN_SPECS:
        df = _load_training(run_dir)
        frames.append(
            {
                "label": label,
                "run_dir": run_dir,
                "df": df,
                "baseline": _load_baseline_summary(run_dir),
                "color": PALETTE[label],
                "episode_max": int(df["episode"].max()),
                "episode_count": int(df["episode"].nunique()),
            }
        )
    return frames


def _export_curve_table(
    filename: str,
    run_frames: List[Dict[str, object]],
    metrics: List[tuple[str, str, str]],
    smooth_window: int,
    band_window: int,
) -> Path:
    merged = None
    for metric, _, _ in metrics:
        metric_table = None
        for frame in run_frames:
            df = frame["df"][["episode", metric]].copy()
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            smooth = _rolling(df[metric], smooth_window)
            std = _rolling_std(df[metric], band_window)
            low, high = _clip_interval(metric, smooth - std, smooth + std)
            sub = pd.DataFrame(
                {
                    "episode": df["episode"],
                    f"{metric}__{frame['label']}__raw": df[metric],
                    f"{metric}__{frame['label']}__smooth": smooth,
                    f"{metric}__{frame['label']}__band_low": low,
                    f"{metric}__{frame['label']}__band_high": high,
                }
            )
            if metric_table is None:
                metric_table = sub
            else:
                metric_table = metric_table.merge(sub, on="episode", how="outer")
        if merged is None:
            merged = metric_table
        else:
            merged = merged.merge(metric_table, on="episode", how="outer")
    out_path = TAB_DIR / filename
    merged.sort_values("episode").to_csv(out_path, index=False)
    return out_path


def _export_tail_summary(run_frames: List[Dict[str, object]], tail_window: int) -> Path:
    rows = []
    for frame in run_frames:
        df = frame["df"]
        tail = df.tail(min(tail_window, len(df)))
        rows.append(
            {
                "label": frame["label"],
                "episodes_used": int(len(df)),
                "tail_window": int(len(tail)),
                "reward_mean_last_tail": float(pd.to_numeric(tail["reward_mean"], errors="coerce").mean()),
                "reward_total_last_tail": float(pd.to_numeric(tail["reward_total"], errors="coerce").mean()),
                "task_sr_last_tail": float(pd.to_numeric(tail["task_sr"], errors="coerce").mean()),
                "deadline_miss_rate_last_tail": float(pd.to_numeric(tail["deadline_miss_rate"], errors="coerce").mean()),
                "avg_rsu_queue_last_tail": float(pd.to_numeric(tail["avg_rsu_queue"], errors="coerce").mean()),
                "avg_power_last_tail": float(pd.to_numeric(tail["avg_power"], errors="coerce").mean()),
                "ratio_local_last_tail": float(pd.to_numeric(tail["ratio_local"], errors="coerce").mean()),
                "ratio_rsu_last_tail": float(pd.to_numeric(tail["ratio_rsu"], errors="coerce").mean()),
                "ratio_v2v_last_tail": float(pd.to_numeric(tail["ratio_v2v"], errors="coerce").mean()),
                "approx_kl_last_tail": float(pd.to_numeric(tail["approx_kl"], errors="coerce").mean()),
                "entropy_last_tail": float(pd.to_numeric(tail["entropy"], errors="coerce").mean()),
                "clip_frac_last_tail": float(pd.to_numeric(tail["clip_frac"], errors="coerce").mean()),
            }
        )
    out_path = TAB_DIR / "lr_critic_tail_summary_table.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _style_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str | None = None) -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def _series_bundle(frame: Dict[str, object], metric: str, smooth_window: int, band_window: int) -> Dict[str, pd.Series]:
    df = frame["df"]
    raw = pd.to_numeric(df[metric], errors="coerce")
    smooth = _rolling(raw, smooth_window)
    std = _rolling_std(raw, band_window)
    low, high = _clip_interval(metric, smooth - std, smooth + std)
    return {"raw": raw, "smooth": smooth, "low": low, "high": high}


def _save(fig: plt.Figure, stem: str) -> List[Path]:
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    return [png]


def _plot_single_curve(
    run_frames: List[Dict[str, object]],
    metric: str,
    title: str,
    ylabel: str,
    stem: str,
    smooth_window: int,
    band_window: int,
    y_limits: tuple[float, float] | None = None,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    curves: List[tuple[np.ndarray, np.ndarray]] = []
    for frame in run_frames:
        df = frame["df"]
        bundle = _series_bundle(frame, metric, smooth_window, band_window)
        ax.fill_between(df["episode"], bundle["low"], bundle["high"], color=frame["color"], alpha=0.11, linewidth=0)
        ax.plot(df["episode"], bundle["smooth"], color=frame["color"], linewidth=2.4, label=frame["label"])
        curves.append((df["episode"].to_numpy(dtype=float), bundle["smooth"].to_numpy(dtype=float)))
    _style_axis(ax, title, ylabel, "训练轮次")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    legend_xy = _choose_layout(ax, curves)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=legend_xy,
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.90)
    return _save(fig, stem)


def _plot_all_single_curves(run_frames: List[Dict[str, object]], smooth_window: int, band_window: int) -> List[Path]:
    exported: List[Path] = []
    for metric, title, ylabel in METRICS_MAIN:
        exported.extend(
            _plot_single_curve(
                run_frames,
                metric,
                title,
                ylabel,
                f"fig_{metric}_final",
                smooth_window,
                band_window,
                y_limits=(0.0, 1.0) if metric in {"task_sr", "deadline_miss_rate"} else None,
            )
        )
    for metric, title, ylabel in METRICS_DIAG:
        exported.extend(_plot_single_curve(run_frames, metric, title, ylabel, f"fig_{metric}_final", smooth_window, band_window))
    for metric, title, ylabel in METRICS_DECISION:
        exported.extend(
            _plot_single_curve(
                run_frames,
                metric,
                title,
                ylabel,
                f"fig_{metric}_final",
                smooth_window,
                band_window,
                y_limits=(0.0, 1.0),
            )
        )
    return exported


def _plot_tail_summary_single(run_frames: List[Dict[str, object]], metric: str, title: str, ylabel: str, stem: str, tail_window: int) -> List[Path]:
    rows = []
    for frame in run_frames:
        df = frame["df"].tail(min(tail_window, len(frame["df"])))
        rows.append(
            {
                "label": frame["label"],
                "value": float(pd.to_numeric(df[metric], errors="coerce").mean()),
            }
        )
    sdf = pd.DataFrame(rows)
    x = np.arange(len(sdf))
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.plot(x, sdf["value"].to_numpy(dtype=float), color="#8a8a8a", linewidth=1.2, alpha=0.65, zorder=1)
    for idx, row in sdf.iterrows():
        ax.scatter(idx, row["value"], s=95, color=PALETTE[row["label"]], zorder=3, label=row["label"])
        ax.text(idx, row["value"], f" {row['value']:.3f}", fontsize=8.5, va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(sdf["label"])
    _style_axis(ax, title, ylabel, "Critic LR")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best", frameon=True, fancybox=False, framealpha=0.95, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.90)
    return _save(fig, stem)


def _write_report(run_frames: List[Dict[str, object]], smooth_window: int, band_window: int, tail_window: int, exported: List[Path]) -> Path:
    rows = []
    for frame in run_frames:
        tail = frame["df"].tail(min(tail_window, len(frame["df"])))
        rows.append(
            {
                "label": frame["label"],
                "reward_mean": float(pd.to_numeric(tail["reward_mean"], errors="coerce").mean()),
                "reward_total": float(pd.to_numeric(tail["reward_total"], errors="coerce").mean()),
                "task_sr": float(pd.to_numeric(tail["task_sr"], errors="coerce").mean()),
                "deadline_miss_rate": float(pd.to_numeric(tail["deadline_miss_rate"], errors="coerce").mean()),
                "avg_rsu_queue": float(pd.to_numeric(tail["avg_rsu_queue"], errors="coerce").mean()),
            }
        )
    sdf = pd.DataFrame(rows)
    best_task = sdf.loc[sdf["task_sr"].idxmax(), "label"]
    best_deadline = sdf.loc[sdf["deadline_miss_rate"].idxmin(), "label"]
    best_reward_mean = sdf.loc[sdf["reward_mean"].idxmax(), "label"]
    best_reward_total = sdf.loc[sdf["reward_total"].idxmax(), "label"]
    lines = [
        "# LR Critic Final Figure Pack",
        "",
        "## Export Settings",
        f"- Smooth window: {smooth_window}",
        f"- Band window: {band_window}",
        f"- Tail window: {tail_window}",
        "- Source experiments: lr_critic_1500ep_20260327_163712/lr_c2e4, lr_c3e4, lr_c5e4",
        "- Actual source training horizon: 1500 episodes for all three learning-rate runs",
        "",
        "## Source Run Inventory",
    ]
    for frame in run_frames:
        lines.append(
            f"- {frame['label']}: run={Path(frame['run_dir']).name}, episode_max={frame['episode_max']}, unique_episode_count={frame['episode_count']}"
        )
    lines.extend(
        [
            "",
            "## Tail Summary",
        ]
    )
    for _, row in sdf.iterrows():
        lines.append(
            f"- {row['label']}: reward_mean={row['reward_mean']:.4f}, reward_total={row['reward_total']:.4f}, task_sr={row['task_sr']:.4f}, "
            f"deadline_miss={row['deadline_miss_rate']:.4f}, avg_rsu_queue={row['avg_rsu_queue']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Visual Takeaways",
            f"- Highest tail task success rate: {best_task}",
            f"- Lowest tail deadline miss rate: {best_deadline}",
            f"- Highest tail average reward: {best_reward_mean}",
            f"- Highest tail total reward: {best_reward_total}",
            "",
            "## Exported Files",
        ]
    )
    for path in exported:
        lines.append(f"- `{path.relative_to(PACK_ROOT)}`")
    out = REPORT_DIR / "LR_CRITIC_FINAL_PACK_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_manifest(exported: List[Path], run_frames: List[Dict[str, object]], smooth_window: int, band_window: int, tail_window: int) -> Path:
    payload = {
        "pack_root": str(PACK_ROOT),
        "runs": [
            {
                "label": frame["label"],
                "run_dir": str(frame["run_dir"]),
                "run_name": Path(frame["run_dir"]).name,
                "episode_max": frame["episode_max"],
                "unique_episode_count": frame["episode_count"],
            }
            for frame in run_frames
        ],
        "smoothing": {
            "smooth_window": smooth_window,
            "band_window": band_window,
            "tail_window": tail_window,
        },
        "files": [str(path.relative_to(PACK_ROOT)) for path in exported],
    }
    out = MANIFEST_DIR / "lr_critic_final_pack_manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    _ensure_dirs()
    _clear_export_dirs()
    _write_root_readme()
    _set_style()
    smooth_window = 36
    band_window = 28
    tail_window = 100
    run_frames = _build_frames()

    exported: List[Path] = []
    exported.append(_export_curve_table("lr_critic_main_training_table.csv", run_frames, METRICS_MAIN, smooth_window, band_window))
    exported.append(_export_curve_table("lr_critic_diagnostics_table.csv", run_frames, METRICS_DIAG, smooth_window, band_window))
    exported.append(_export_curve_table("lr_critic_decision_mix_table.csv", run_frames, METRICS_DECISION, smooth_window, band_window))
    exported.append(_export_tail_summary(run_frames, tail_window))
    exported.extend(_plot_all_single_curves(run_frames, smooth_window, band_window))
    exported.extend(_plot_tail_summary_single(run_frames, "reward_mean", "Tail Average Reward", "Reward", "fig_tail_reward_mean_final", tail_window))
    exported.extend(_plot_tail_summary_single(run_frames, "reward_total", "Tail Total Reward", "Reward Sum", "fig_tail_reward_total_final", tail_window))
    exported.extend(_plot_tail_summary_single(run_frames, "task_sr", "Tail Task Success Rate", "Success Rate", "fig_tail_task_sr_final", tail_window))
    exported.extend(_plot_tail_summary_single(run_frames, "deadline_miss_rate", "Tail Deadline Miss Rate", "Miss Rate", "fig_tail_deadline_miss_rate_final", tail_window))
    exported.extend(_plot_tail_summary_single(run_frames, "avg_rsu_queue", "Tail Average RSU Queue", "Queue Length", "fig_tail_avg_rsu_queue_final", tail_window))
    exported.append(_write_report(run_frames, smooth_window, band_window, tail_window, exported.copy()))
    exported.append(_write_manifest(exported.copy(), run_frames, smooth_window, band_window, tail_window))
    print(f"Exported LR critic final pack to: {PACK_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
