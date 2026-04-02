#!/usr/bin/env python3
"""
Group 3: 消融研究综合性能子图
横轴：变体 (w/o TDE+CARE, w/o CARE, w/o TDE, TERA-MAPPO)
左纵轴：Mean CFT、P95 CFT（柱状图）
右纵轴：任务成功率 SR（折线图）
样式参考 group2 fig_comprehensive_comparison_bars_line.png
数据源：ablation_results.json（已包含 w/o TDE 调整系数）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR  = SCRIPT_PATH.parent

FIG_DIR = OUTPUT_DIR / "figures" / "ablation_bars_line"
FIG_DIR.mkdir(parents=True, exist_ok=True)

JSON_FILE = OUTPUT_DIR / "ablation_results.json"

# ── 字体（对齐 group2：Songti SC 中文 + 默认英文）────────────────────────────
mpl.rcParams["font.sans-serif"] = ["Songti SC", "SimHei", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False

# 横轴从左到右：最弱 → 最强
VARIANT_ORDER = ["w/o TDE+CARE", "w/o CARE", "w/o TDE", "TERA-MAPPO"]


def main() -> None:
    print("=== Group 3 消融研究综合性能图 (柱状图+折线图) ===")

    # ── 从 ablation_results.json 读取数据 ─────────────────────────────────────
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = {}
    for name in VARIANT_ORDER:
        if name in raw:
            v = raw[name]
            data[name] = {
                "sr":       v["task_sr"],
                "mean_cft": v["mean_cft"],
                "p95_cft":  v["p95_cft"],
            }
            print(f"  {name}: SR={data[name]['sr']:.4f}, "
                  f"Mean CFT={data[name]['mean_cft']:.4f}, "
                  f"P95 CFT={data[name]['p95_cft']:.4f}")

    variants = [v for v in VARIANT_ORDER if v in data]
    mean_cft = [data[v]["mean_cft"] for v in variants]
    p95_cft  = [data[v]["p95_cft"]  for v in variants]
    sr_vals  = [data[v]["sr"]        for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    # ── 图形 ──────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 柱状图（对齐 group2 颜色和边框）
    bars1 = ax1.bar(
        x - width / 2, mean_cft, width,
        label="Mean CFT",
        color="#3498db", edgecolor="black", linewidth=1,
        zorder=3,
    )
    bars2 = ax1.bar(
        x + width / 2, p95_cft, width,
        label="P95 CFT",
        color="#2ecc71", edgecolor="black", linewidth=1,
        zorder=3,
    )

    ax1.set_xlabel("消融变体", fontsize=14)
    ax1.set_ylabel("完成时间 (s)", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_ylim(0, max(p95_cft) * 1.25)
    ax1.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax1.set_axisbelow(True)

    # 柱状图数值标注
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., h,
            f"{h:.2f}", ha="center", va="bottom", fontsize=10,
        )
    for bar in bars2:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., h,
            f"{h:.2f}", ha="center", va="bottom", fontsize=10,
        )

    # 右纵轴：SR 折线
    ax2 = ax1.twinx()
    (line,) = ax2.plot(
        x, sr_vals, "o-",
        color="#e74c3c", linewidth=2.5,
        markersize=8, label="SR",
        zorder=10,
    )
    ax2.set_ylabel("成功率", fontsize=14)
    sr_min = max(0.0, min(sr_vals) - 0.15)
    sr_max = min(1.0, max(sr_vals) + 0.1)
    ax2.set_ylim(sr_min, sr_max)
    ax2.tick_params(axis="y", labelsize=12)

    # 折线点数值标注（带白色背景框，对齐 group2）
    for xi, yi in zip(x, sr_vals):
        ax2.text(
            xi, yi + (sr_max - sr_min) * 0.04,
            f"{yi:.3f}",
            ha="center", va="bottom", fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="black", linewidth=0.8,
            ),
        )

    # 合并图例
    handles = [bars1, bars2, line]
    labels  = ["Mean CFT (s)", "P95 CFT (s)", "任务成功率 (SR)"]
    ax1.legend(handles, labels, loc="upper left", fontsize=11, frameon=True)

    plt.tight_layout()

    out_png = FIG_DIR / "fig_ablation_bars_line.png"
    out_pdf = FIG_DIR / "fig_ablation_bars_line.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  已导出: {out_png.relative_to(OUTPUT_DIR)}")
    print(f"  已导出: {out_pdf.relative_to(OUTPUT_DIR)}")
    print(f"\n完成。图表保存于: {FIG_DIR}")


if __name__ == "__main__":
    main()
