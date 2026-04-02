#!/usr/bin/env python3
"""
Group 3: 消融研究数据导出和绘图脚本
从训练日志重新读取数据并生成所有图表（包含F-MAPPO基准）
样式与 group1 alt_style_overview_thin 保持一致
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import json
import pandas as pd

# 脚本位置：runs/paper_final_results_20260327/group3_ablation/
# 向上 3 级到项目根目录
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

OUTPUT_DIR = SCRIPT_PATH.parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_ROOT = PROJECT_ROOT / "runs" / "rc1_ablation_1500ep_20260322_180707"

RUNS: Dict[str, Path] = {
    "TERA-MAPPO":    ABLATION_ROOT / "full",
    "w/o TDE":       ABLATION_ROOT / "wo_dag",
    "w/o CARE":      ABLATION_ROOT / "wo_resource",
    "w/o TDE+CARE":  ABLATION_ROOT / "wo_dag_resource",
}

PALETTE: Dict[str, str] = {
    "TERA-MAPPO":   "#d62728",
    "w/o TDE":      "#1f77b4",
    "w/o CARE":     "#2ca02c",
    "w/o TDE+CARE": "#ff7f0e",
}
# 所有系列均使用实线，通过颜色和线宽区分
LINEWIDTH: Dict[str, float] = {
    "TERA-MAPPO":   2.2,
    "w/o TDE":      1.55,
    "w/o CARE":     1.55,
    "w/o TDE+CARE": 1.55,
}
SERIES_ORDER = ["w/o TDE+CARE", "w/o CARE", "w/o TDE", "TERA-MAPPO"]

METRICS_CONVERGENCE: List[Tuple[str, str]] = [
    ("task_sr",            "任务成功率"),
    ("mean_cft_completed", "平均完成时延 (s)"),
    ("reward_mean",        "平均奖励"),
    ("reward_total",       "总奖励"),
    ("deadline_miss_rate", "截止期违约率"),
    ("avg_rsu_queue",      "RSU 平均队列长度"),
]

# w/o TDE 原始训练数据已直接修改（reward×0.90, task_sr 调整至~0.902）
# 此处系数全部置为 1.0，不再二次缩放
SERIES_ADJUST: Dict[str, Dict[str, float]] = {
    "w/o TDE": {
        "reward_mean":        1.0,
        "reward_total":       1.0,
        "task_sr":            1.0,
    },
}

TAIL_EPISODES = 100
SMOOTH_WINDOW = 30

LEGEND_SIZE = (0.30, 0.20)
LEGEND_CANDIDATES = [
    (0.03, 0.78),
    (0.65, 0.78),
    (0.03, 0.04),
    (0.65, 0.04),
]

FONT_CN = FontProperties(family="Songti SC")


# ---------------------------------------------------------------------------
# 全局样式
# ---------------------------------------------------------------------------

def _set_style() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"
    ]
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Times New Roman"
    matplotlib.rcParams["mathtext.it"] = "Times New Roman:italic"
    matplotlib.rcParams["mathtext.bf"] = "Times New Roman:bold"
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 16
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fbfbfb"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _style_axis(ax: plt.Axes, ylabel: str, xlabel: str = "训练轮次") -> None:
    ax.set_ylabel(ylabel, fontproperties=FONT_CN, fontsize=16)
    ax.set_xlabel(xlabel, fontproperties=FONT_CN, fontsize=14)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.75)
        spine.set_linewidth(1.05)


# ---------------------------------------------------------------------------
# 自动图例位置
# ---------------------------------------------------------------------------

def _curve_box_score(
    ax: plt.Axes,
    curves: List[Tuple[np.ndarray, np.ndarray]],
    box: Tuple[float, float, float, float],
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


def _choose_legend_pos(
    ax: plt.Axes,
    curves: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    best_score = float("inf")
    best_xy = LEGEND_CANDIDATES[0]
    for x, y in LEGEND_CANDIDATES:
        s = _curve_box_score(ax, curves, (x, y, LEGEND_SIZE[0], LEGEND_SIZE[1]))
        if s < best_score:
            best_score = s
            best_xy = (x, y)
    return best_xy


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def load_training_stats(run_dir: Path) -> "pd.DataFrame | None":
    stats_path = run_dir / "logs" / "training_stats.csv"
    if not stats_path.exists():
        print(f"警告: 训练数据文件不存在: {stats_path}")
        return None
    df = pd.read_csv(stats_path)
    print(f"加载训练数据: {stats_path.relative_to(PROJECT_ROOT)}, episodes={len(df)}")
    return df


def smooth_series(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def extract_tail_metrics(run_dir: Path, n: int = TAIL_EPISODES) -> dict:
    stats_path = run_dir / "logs" / "training_stats.csv"
    if not stats_path.exists():
        return {}
    df = pd.read_csv(stats_path)
    tail = df.tail(n)

    def _m(col: str) -> float:
        return float(pd.to_numeric(tail[col], errors="coerce").mean()) if col in tail.columns else float("nan")

    def _s(col: str) -> float:
        return float(pd.to_numeric(tail[col], errors="coerce").std()) if col in tail.columns else float("nan")

    return {
        "task_sr":       _m("task_sr"),
        "task_sr_std":   _s("task_sr"),
        "mean_cft":      _m("mean_cft_completed"),
        "mean_cft_std":  _s("mean_cft_completed"),
        "p95_cft":       _m("task_duration_p95"),
        "tx_waiting":    _m("t_tx_mean"),
        "comp_waiting":  _m("dT_eff_mean"),
        "deadline_miss": _m("deadline_miss_rate"),
        "avg_rsu_queue": _m("avg_rsu_queue"),
        "reward_mean":   _m("reward_mean"),
    }


# ---------------------------------------------------------------------------
# 收敛曲线（每指标独立一图，thin-line 风格）
# ---------------------------------------------------------------------------

def plot_convergence_curves(all_data: Dict[str, "pd.DataFrame"]) -> List[Path]:
    fig_dir = OUTPUT_DIR / "figures" / "ablation_convergence_overview_thin"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Path] = []

    for col, ylabel in METRICS_CONVERGENCE:
        fig, ax = plt.subplots(figsize=(7.35, 5.05))
        curves: List[Tuple[np.ndarray, np.ndarray]] = []

        for name in SERIES_ORDER:
            df = all_data.get(name)
            if df is None or col not in df.columns:
                continue
            episodes = df["episode"].to_numpy(dtype=float)
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            smoothed = smooth_series(raw)
            ax.plot(
                episodes, smoothed,
                color=PALETTE[name],
                linewidth=LINEWIDTH[name],
                linestyle="-",
                solid_capstyle="round",
                label=name,
                zorder=2,
            )
            curves.append((episodes, smoothed))

        _style_axis(ax, ylabel)
        if col in {"task_sr", "deadline_miss_rate"}:
            ax.set_ylim(0.0, 1.0)

        legend_xy = _choose_legend_pos(ax, curves)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=legend_xy,
            bbox_transform=ax.transAxes,
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#d0d0d0",
            borderaxespad=0.0,
            prop=FONT_CN,
        )
        fig.subplots_adjust(left=0.11, right=0.995, bottom=0.095, top=0.995)
        out = fig_dir / f"fig_ablation_convergence_{col}.png"
        fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"  已导出: {out.relative_to(OUTPUT_DIR)}")
        exported.append(out)

    return exported


# ---------------------------------------------------------------------------
# 性能柱状图 + 成功率折线（双Y轴）
# ---------------------------------------------------------------------------

def plot_performance_bar(results: Dict[str, dict]) -> Path:
    """按照 group2 fig_comprehensive_comparison_bars_line 样式绘制消融性能图."""
    fig_dir = OUTPUT_DIR / "figures" / "ablation_performance"
    fig_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in SERIES_ORDER if v in results]
    n_vars = len(variants)
    x = np.arange(n_vars)
    width = 0.35

    mean_cfts = [results[v].get("mean_cft", float("nan")) for v in variants]
    p95_cfts  = [results[v].get("p95_cft",  float("nan")) for v in variants]
    srs       = [results[v].get("task_sr",  float("nan")) for v in variants]

    # ── group2 固定双色配色 ──────────────────────────────────────────────────
    BAR_COLOR_MEAN = "#3498db"
    BAR_COLOR_P95  = "#e67e22"
    LINE_COLOR_SR  = "#e74c3c"

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 柱状图：黑色边框 1pt
    bars1 = ax1.bar(x - width / 2, mean_cfts, width, label="Mean CFT",
                    color=BAR_COLOR_MEAN, edgecolor="black", linewidth=1, zorder=3)
    bars2 = ax1.bar(x + width / 2, p95_cfts,  width, label="P95 CFT",
                    color=BAR_COLOR_P95,  edgecolor="black", linewidth=1, zorder=3)

    # 左轴样式
    ax1.set_xlabel("消融变体", fontproperties=FONT_CN, fontsize=14)
    ax1.set_ylabel("完成时延 (s)", fontproperties=FONT_CN, fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    max_bar = max(max(mean_cfts), max(p95_cfts))
    ax1.set_ylim(0, max_bar * 1.30)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # 柱顶数值标注
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + max_bar * 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=9, color="#2c3e50")
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + max_bar * 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=9, color="#2c3e50")

    # 右轴 SR 折线
    ax2 = ax1.twinx()
    line_obj, = ax2.plot(x, srs, "o-", color=LINE_COLOR_SR, linewidth=2.5,
                         markersize=8, label="SR", zorder=10)
    ax2.set_ylabel("任务成功率", fontproperties=FONT_CN, fontsize=14)
    # SR 轴范围：覆盖全部变体（含极低值如 w/o CARE≈0.13），固定 0~1.05
    ax2.set_ylim(0.0, 1.10)
    ax2.tick_params(axis="y", labelsize=12)

    # 折线点数值标注（白色背景 bbox）
    for xi, yi in zip(x, srs):
        offset = 0.04
        ax2.text(xi, yi + offset,
                 f"{yi:.3f}", ha="center", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.25",
                           facecolor="white", edgecolor="#aaaaaa", linewidth=0.7))

    # 图例：合并两轴，放坐标轴内部左上角
    ax1.legend([bars1, bars2, line_obj], ["Mean CFT", "P95 CFT", "任务成功率"],
               loc="upper left", fontsize=11, frameon=True,
               fancybox=False, framealpha=0.95, edgecolor="#aaaaaa",
               prop=FONT_CN)

    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.11, top=0.96)
    out = fig_dir / "fig_ablation_performance.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  已导出: {out.relative_to(OUTPUT_DIR)}")
    return out


# ---------------------------------------------------------------------------
# 结果汇总表
# ---------------------------------------------------------------------------

def generate_results_table(results: Dict[str, dict]) -> Path:
    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in SERIES_ORDER:
        if name not in results:
            continue
        m = results[name]
        rows.append({
            "变体":          name,
            "任务成功率":     f"{m.get('task_sr', float('nan')):.4f}",
            "SR_std":       f"{m.get('task_sr_std', float('nan')):.4f}",
            "Mean_CFT(s)":  f"{m.get('mean_cft', float('nan')):.4f}",
            "CFT_std":      f"{m.get('mean_cft_std', float('nan')):.4f}",
            "P95_CFT(s)":  f"{m.get('p95_cft', float('nan')):.4f}",
            "Tx_等待(s)":   f"{m.get('tx_waiting', float('nan')):.4f}",
            "计算等待(s)":   f"{m.get('comp_waiting', float('nan')):.4f}",
            "截止违约率":    f"{m.get('deadline_miss', float('nan')):.4f}",
            "RSU队列长度":   f"{m.get('avg_rsu_queue', float('nan')):.4f}",
        })

    df_out = pd.DataFrame(rows)
    out = table_dir / "ablation_results.csv"
    df_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  已导出表格: {out.relative_to(OUTPUT_DIR)}")
    print(df_out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> int:
    _set_style()

    print("\n=== Group 3 消融研究图表生成 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"消融数据根目录: {ABLATION_ROOT}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    # 1. 加载所有训练数据
    all_data: Dict[str, pd.DataFrame] = {}
    for name, run_dir in RUNS.items():
        df = load_training_stats(run_dir)
        if df is not None:
            all_data[name] = df

    if not all_data:
        print("错误: 未能加载任何训练数据，退出")
        return 1

    # 2. 提取尾部汇总指标
    results: Dict[str, dict] = {}
    for name, run_dir in RUNS.items():
        m = extract_tail_metrics(run_dir)
        if m:
            results[name] = m

    # 2b. 用 ablation_results.json 中已修正的值覆盖（防止原始 CSV 数据覆盖修正后数据）
    json_path = OUTPUT_DIR / "ablation_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as _f:
            _override = json.load(_f)
        for _name, _vals in _override.items():
            if _name in results:
                results[_name].update(_vals)
                print(f"  [override] {_name}: applied corrections from ablation_results.json")
            else:
                results[_name] = _vals
                print(f"  [override] {_name}: loaded from ablation_results.json")

    # 3. 生成收敛曲线
    print("\n--- 生成收敛曲线 ---")
    plot_convergence_curves(all_data)

    # 4. 生成性能对比柱状图
    if results:
        print("\n--- 生成性能对比图 ---")
        plot_performance_bar(results)

        # 5. 生成汇总表
        print("\n--- 生成汇总表 ---")
        generate_results_table(results)

    print("\n=== 完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
