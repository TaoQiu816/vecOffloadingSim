#!/usr/bin/env python3
"""论文最终结果的绘图工具"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 论文级别的绘图参数
rcParams.update({
    'figure.dpi': 320,
    'savefig.dpi': 320,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.autolayout': True,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# 定义颜色方案（按照技术设计文档）
COLOR_SCHEMES = {
    # Group 2: 综合对比
    "group2": {
        "LO": "#95a5a6",          # 灰色
        "NRO": "#3498db",         # 蓝色
        "EFT-H": "#2ecc71",       # 绿色
        "IPPO-H": "#9b59b6",      # 紫色
        "F-MAPPO": "#e67e22",     # 橙色
        "TERA-MAPPO": "#e74c3c",  # 红色（主方法）
    },
    # Group 3: 消融实验
    "group3": {
        "w/o-TDE": "#95a5a6",     # 灰色
        "w/o-CARE": "#3498db",    # 蓝色
        "TERA-MAPPO": "#e74c3c",  # 红色（完整方法）
    },
    # Group 4 & 5: 敏感性分析（线图）
    "sensitivity": {
        "DAG Size": "#3498db",    # 蓝色
        "Deadline Factor": "#2ecc71",  # 绿色
        "Vehicle Count": "#e67e22",    # 橙色
        "RSU CPU": "#9b59b6",     # 紫色
    },
    # Group 6: 机制分析
    "group6": {
        "Local": "#95a5a6",
        "V2V": "#3498db",
        "V2I": "#2ecc71",
        "Transmission": "#e67e22",
        "Computation": "#9b59b6",
        "Queueing": "#f39c12",
    },
}

# 延迟分解颜色
DELAY_COLORS = {
    "Transmission": "#e67e22",
    "Computation": "#9b59b6",
    "Queueing": "#f39c12",
}


class PaperPlotter:
    """论文级别的高质量绘图工具"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_colors(self, methods: List[str], scheme: str = "group2") -> List[str]:
        """获取指定方案的颜色"""
        palette = COLOR_SCHEMES.get(scheme, COLOR_SCHEMES["group2"])
        return [palette.get(m, "#34495e") for m in methods]
    
    def plot_comparison_bars(
        self,
        data: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str,
        filename: str,
        ylabel_map: Optional[Dict[str, str]] = None,
        scheme: str = "group2",
        figsize: Tuple[float, float] = (10, 3),
    ) -> Path:
        """
        绘制对比柱状图（Group 2 & 3）
        
        Args:
            data: {method: {metric: value}}
            metrics: 要绘制的指标列表
            title: 图标题
            filename: 保存文件名
            ylabel_map: 指标到Y轴标签的映射
            scheme: 颜色方案
            figsize: 图形大小
        """
        methods = list(data.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        colors = self._get_colors(methods, scheme)
        
        for ax, metric in zip(axes, metrics):
            values = [data[m][metric] for m in methods]
            ylabel = ylabel_map.get(metric, metric) if ylabel_map else metric
            
            bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
            
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(metric, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # 旋转x轴标签
            ax.set_xticklabels(methods, rotation=30, ha='right')
        
        plt.suptitle(title, fontsize=12, y=1.02)
        plt.tight_layout()
        
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_sensitivity_lines(
        self,
        data: Dict[str, Dict[str, List[float]]],
        x_values: Dict[str, List[float]],
        title: str,
        filename: str,
        ylabel: str,
        xlabel_map: Optional[Dict[str, str]] = None,
        figsize: Tuple[float, float] = (10, 4),
    ) -> Path:
        """
        绘制敏感性分析线图（Group 4 & 5）
        
        Args:
            data: {factor: {metric: [values]}}
            x_values: {factor: [x_values]}
            title: 图标题
            filename: 保存文件名
            ylabel: Y轴标签
            xlabel_map: 因子到X轴标签的映射
            figsize: 图形大小
        """
        n_factors = len(data)
        metrics = list(next(iter(data.values())).keys())
        
        fig, axes = plt.subplots(1, n_factors, figsize=figsize)
        if n_factors == 1:
            axes = [axes]
        
        for ax, (factor, factor_data) in zip(axes, data.items()):
            xs = x_values[factor]
            xlabel = xlabel_map.get(factor, factor) if xlabel_map else factor
            
            for i, (metric, values) in enumerate(factor_data.items()):
                color = COLOR_SCHEMES["sensitivity"].get(metric, "#34495e")
                ax.plot(xs, values, marker='o', label=metric, color=color, linewidth=2)
            
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(factor, fontsize=11)
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.suptitle(title, fontsize=12, y=1.02)
        plt.tight_layout()
        
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_stacked_bars(
        self,
        data: Dict[str, Dict[str, float]],
        title: str,
        filename: str,
        ylabel: str,
        xlabel: str = "Method",
        figsize: Tuple[float, float] = (6, 4),
        color_map: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        绘制堆叠柱状图（Group 3 延迟分解, Group 6）
        
        Args:
            data: {method: {component: value}}
            title: 图标题
            filename: 保存文件名
            ylabel: Y轴标签
            xlabel: X轴标签
            figsize: 图形大小
            color_map: 组件到颜色的映射
        """
        methods = list(data.keys())
        components = list(next(iter(data.values())).keys())
        
        if color_map is None:
            color_map = DELAY_COLORS
        
        colors = [color_map.get(c, "#34495e") for c in components]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bottoms = np.zeros(len(methods))
        
        for comp in components:
            values = [data[m][comp] for m in methods]
            ax.bar(methods, values, bottom=bottoms, label=comp, 
                   color=color_map.get(comp, "#34495e"), alpha=0.8, edgecolor='white', linewidth=0.5)
            bottoms += values
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_box_plots(
        self,
        data: Dict[str, List[float]],
        title: str,
        filename: str,
        ylabel: str,
        figsize: Tuple[float, float] = (6, 4),
        scheme: str = "group2",
    ) -> Path:
        """
        绘制箱线图（Group 6 功率分布）
        
        Args:
            data: {method: [values]}
            title: 图标题
            filename: 保存文件名
            ylabel: Y轴标签
            figsize: 图形大小
            scheme: 颜色方案
        """
        methods = list(data.keys())
        values = [data[m] for m in methods]
        colors = self._get_colors(methods, scheme)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bp = ax.boxplot(values, labels=methods, patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        
        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        plt.close()
        
        return output_path


def load_csv_data(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """加载CSV数据并转换为绘图格式"""
    import csv
    
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['policy']
            data[method] = {
                'task_success_rate': float(row['task_success_rate_mean']),
                'mean_cft': float(row['mean_cft_mean']),
                'mean_energy': float(row['mean_energy_mean']),
            }
    
    return data
