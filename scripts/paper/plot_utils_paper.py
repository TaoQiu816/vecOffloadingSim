#!/usr/bin/env python3
"""
统一绘图工具模块 - 用于论文图表生成
符合高水平期刊/会议标准
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties


# 配色方案 - 参考现有高质量图表
COLORS = {
    "LO": "#e74c3c",           # 红色
    "NRO": "#3498db",          # 蓝色
    "EFT-H": "#2ecc71",        # 绿色
    "IPPO-H": "#9b59b6",       # 紫色
    "F-MAPPO": "#e67e22",      # 橙色
    "TERA-MAPPO": "#1abc9c",   # 青色
    "w/o TDE": "#f39c12",      # 金色
    "w/o CARE": "#95a5a6",     # 灰色
}

# 中文字体
FONT_CN = FontProperties(family="Songti SC")


def set_paper_style():
    """设置论文级别的绘图风格"""
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ]
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Times New Roman"
    matplotlib.rcParams["mathtext.it"] = "Times New Roman:italic"
    matplotlib.rcParams["mathtext.bf"] = "Times New Roman:bold"
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 16
    matplotlib.rcParams["legend.fontsize"] = 12
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fbfbfb"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def style_axis(ax: plt.Axes, ylabel: str, xlabel: str = ""):
    """统一的坐标轴样式"""
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=FONT_CN, fontsize=18)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=FONT_CN, fontsize=18)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.28)
    ax.spines["bottom"].set_alpha(0.28)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 320):
    """保存图表"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def plot_bar_chart(
    data: Dict[str, float],
    ylabel: str,
    xlabel: str = "",
    title: str = "",
    figsize: Tuple[float, float] = (7.35, 5.05),
    color_map: Optional[Dict[str, str]] = None,
    order: Optional[List[str]] = None,
) -> plt.Figure:
    """
    绘制高质量柱状图
    
    Args:
        data: {method: value} 字典
        ylabel: Y轴标签
        xlabel: X轴标签
        title: 图标题
        figsize: 图尺寸
        color_map: 颜色映射字典
        order: 方法顺序
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = order if order else list(data.keys())
    values = [data[m] for m in methods]
    colors = [color_map[m] if color_map and m in color_map else COLORS.get(m, "#34495e") for m in methods]
    
    bars = ax.bar(range(len(methods)), values, color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=0, ha="center")
    
    # 在柱子上添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontproperties=FONT_CN,
        )
    
    if title:
        ax.set_title(title, fontproperties=FONT_CN, fontsize=16)
    style_axis(ax, ylabel, xlabel)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.12, top=0.95)
    
    return fig


def plot_line_chart(
    data: Dict[str, List[Tuple[float, float]]],
    ylabel: str,
    xlabel: str = "",
    title: str = "",
    figsize: Tuple[float, float] = (7.35, 5.05),
    color_map: Optional[Dict[str, str]] = None,
    legend_loc: str = "best",
) -> plt.Figure:
    """
    绘制高质量折线图
    
    Args:
        data: {method: [(x, y), ...]} 字典
        ylabel: Y轴标签
        xlabel: X轴标签
        title: 图标题
        figsize: 图尺寸
        color_map: 颜色映射字典
        legend_loc: 图例位置
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for method, points in data.items():
        if not points:
            continue
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        color = color_map[method] if color_map and method in color_map else COLORS.get(method, "#34495e")
        
        ax.plot(x_vals, y_vals, color=color, linewidth=2.2, marker="o", markersize=6, label=method)
    
    if title:
        ax.set_title(title, fontproperties=FONT_CN, fontsize=16)
    style_axis(ax, ylabel, xlabel)
    
    if data:
        ax.legend(loc=legend_loc, frameon=True, fancybox=False, framealpha=0.96, edgecolor="#d0d0d0")
    
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.095, top=0.95)
    
    return fig


def plot_stacked_bar(
    data: Dict[str, Dict[str, float]],
    ylabel: str,
    xlabel: str = "",
    title: str = "",
    figsize: Tuple[float, float] = (7.35, 5.05),
    stack_labels: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    绘制堆叠柱状图
    
    Args:
        data: {method: {component: value}} 字典
        ylabel: Y轴标签
        xlabel: X轴标签
        title: 图标题
        figsize: 图尺寸
        stack_labels: 堆叠组件标签顺序
        color_map: 颜色映射字典
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(data.keys())
    
    # 确定堆叠组件
    if stack_labels is None:
        all_components = set()
        for comp_dict in data.values():
            all_components.update(comp_dict.keys())
        stack_labels = sorted(all_components)
    
    # 默认颜色
    default_stack_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    
    # 准备数据
    bottoms = np.zeros(len(methods))
    for i, component in enumerate(stack_labels):
        values = [data[m].get(component, 0) for m in methods]
        color = color_map[component] if color_map and component in color_map else default_stack_colors[i % len(default_stack_colors)]
        
        ax.bar(range(len(methods)), values, bottom=bottoms, color=color, label=component, alpha=0.9, edgecolor="white", linewidth=0.5)
        bottoms += values
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=0, ha="center")
    
    if title:
        ax.set_title(title, fontproperties=FONT_CN, fontsize=16)
    style_axis(ax, ylabel, xlabel)
    ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=0.96, edgecolor="#d0d0d0")
    
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.12, top=0.95)
    
    return fig


def plot_boxplot(
    data: Dict[str, List[float]],
    ylabel: str,
    xlabel: str = "",
    title: str = "",
    figsize: Tuple[float, float] = (7.35, 5.05),
    color_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    绘制箱线图
    
    Args:
        data: {group: [values]} 字典
        ylabel: Y轴标签
        xlabel: X轴标签
        title: 图标题
        figsize: 图尺寸
        color_map: 颜色映射字典
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = list(data.keys())
    values = [data[g] for g in groups]
    
    # 创建箱线图
    bp = ax.boxplot(values, labels=groups, patch_artist=True)
    
    # 设置颜色
    for patch, group in zip(bp["boxes"], groups):
        color = color_map[group] if color_map and group in color_map else COLORS.get(group, "#34995e")
        patch.set_facecolor(to_rgba(color, 0.7))
        patch.set_alpha(0.7)
    
    # 设置其他元素样式
    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color="#2c3e50", linewidth=1.5)
    
    if title:
        ax.set_title(title, fontproperties=FONT_CN, fontsize=16)
    style_axis(ax, ylabel, xlabel)
    
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.12, top=0.95)
    
    return fig
