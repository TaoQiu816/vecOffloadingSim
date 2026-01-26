#!/usr/bin/env python3
"""
消融实验详细对比绘图脚本
Ablation Study Comprehensive Comparison Plotting

生成图表：
1. 成功率对比曲线 (所有方案)
2. 奖励曲线对比
3. 收敛速度对比 (达到90%成功率的episode)
4. 动作分布柱状图对比
5. RSU队列负载对比
6. 策略熵对比
7. 消融效果汇总表格
8. 综合仪表板
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Optional
import json

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 实验标签和颜色配置
EXPERIMENT_CONFIG = {
    "full": {"label": "Full Model (Ours)", "color": "#2E86AB", "linestyle": "-"},
    "no_transformer": {"label": "w/o Transformer", "color": "#E94F37", "linestyle": "--"},
    "no_edge": {"label": "w/o Edge Bias", "color": "#44AF69", "linestyle": "-."},
    "no_physics": {"label": "w/o Physics Bias", "color": "#F18F01", "linestyle": ":"},
    "exp_b": {"label": "High Pressure (B)", "color": "#C73E1D", "linestyle": "-"},
    "hard": {"label": "Hard Mode", "color": "#6B2D5C", "linestyle": "-"},
}


def load_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    """加载训练指标CSV"""
    if run_dir is None or not run_dir.exists():
        return None
    
    csv_path = run_dir / "metrics" / "train_metrics.csv"
    if not csv_path.exists():
        csv_path = run_dir / "logs" / "training_stats.csv"
    if not csv_path.exists():
        print(f"Warning: No metrics found in {run_dir}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def smooth(data: pd.Series, window: int = 20) -> pd.Series:
    """平滑处理"""
    return data.rolling(window=window, min_periods=1).mean()


def find_convergence_episode(df: pd.DataFrame, threshold: float = 0.9) -> int:
    """找到收敛到threshold成功率的episode"""
    if 'task_success_rate' not in df.columns:
        return -1
    
    smoothed = smooth(df['task_success_rate'], window=30)
    above_threshold = smoothed >= threshold
    
    if above_threshold.any():
        # 找到连续20个episode都超过threshold的起始点
        for i in range(len(above_threshold) - 20):
            if above_threshold.iloc[i:i+20].all():
                return int(df['episode'].iloc[i])
    return -1


def plot_success_rate_comparison(data: Dict, output_dir: str):
    """绘制成功率对比曲线"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for key, df in data.items():
        if df is None or 'task_success_rate' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['task_success_rate']) * 100
        ax.plot(df['episode'], smoothed, 
                label=cfg["label"], color=cfg["color"], 
                linestyle=cfg["linestyle"], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Task Success Rate (%)', fontsize=12)
    ax.set_title('Task Success Rate Comparison - Ablation Study', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_success_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_success_rate.png")


def plot_reward_comparison(data: Dict, output_dir: str):
    """绘制奖励曲线对比"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for key, df in data.items():
        if df is None or 'reward_mean' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['reward_mean'])
        ax.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Mean Reward Comparison - Ablation Study', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_reward.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_reward.png")


def plot_convergence_comparison(data: Dict, output_dir: str):
    """绘制收敛速度对比（柱状图）"""
    convergence_data = []
    
    for key, df in data.items():
        if df is None:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray"})
        conv_ep = find_convergence_episode(df, threshold=0.85)
        if conv_ep > 0:
            convergence_data.append({
                "name": cfg["label"],
                "episode": conv_ep,
                "color": cfg["color"]
            })
    
    if not convergence_data:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [d["name"] for d in convergence_data]
    episodes = [d["episode"] for d in convergence_data]
    colors = [d["color"] for d in convergence_data]
    
    bars = ax.barh(names, episodes, color=colors, alpha=0.8)
    
    for bar, ep in zip(bars, episodes):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{ep} ep', va='center', fontsize=10)
    
    ax.set_xlabel('Episodes to Convergence (85% Success Rate)', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_convergence.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_convergence.png")


def plot_action_distribution(data: Dict, output_dir: str):
    """绘制动作分布对比"""
    action_cols = ['decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v']
    action_labels = ['Local', 'RSU', 'V2V']
    
    n_exps = len([k for k, v in data.items() if v is not None])
    if n_exps == 0:
        return
    
    fig, axes = plt.subplots(1, n_exps, figsize=(4 * n_exps, 5))
    if n_exps == 1:
        axes = [axes]
    
    idx = 0
    for key, df in data.items():
        if df is None:
            continue
        if not all(c in df.columns for c in action_cols):
            continue
        
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray"})
        last_100 = df.tail(100)
        values = [last_100[c].mean() * 100 for c in action_cols]
        
        bars = axes[idx].bar(action_labels, values, color=cfg["color"], alpha=0.8)
        axes[idx].set_title(f'{cfg["label"]}', fontsize=11)
        axes[idx].set_ylabel('Fraction (%)')
        axes[idx].set_ylim(0, 100)
        
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{val:.1f}%', ha='center', fontsize=9)
        idx += 1
    
    plt.suptitle('Action Distribution Comparison (Last 100 Episodes)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_action_dist.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_action_dist.png")


def plot_rsu_queue_comparison(data: Dict, output_dir: str):
    """绘制RSU队列负载对比"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for key, df in data.items():
        if df is None or 'avg_rsu_queue' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['avg_rsu_queue'])
        ax.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average RSU Queue Load', fontsize=12)
    ax.set_title('RSU Queue Load Comparison - Ablation Study', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_rsu_queue.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_rsu_queue.png")


def plot_entropy_comparison(data: Dict, output_dir: str):
    """绘制策略熵对比"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for key, df in data.items():
        if df is None or 'policy_entropy' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['policy_entropy'])
        ax.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Policy Entropy', fontsize=12)
    ax.set_title('Policy Entropy Comparison - Ablation Study', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_entropy.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_entropy.png")


def generate_summary_table(data: Dict, output_dir: str):
    """生成消融效果汇总表格"""
    summary = []
    
    for key, df in data.items():
        if df is None:
            continue
        
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key})
        last_100 = df.tail(100)
        
        row = {
            "Method": cfg["label"],
            "Success Rate (%)": f"{last_100['task_success_rate'].mean()*100:.1f}" if 'task_success_rate' in df.columns else "N/A",
            "Mean Reward": f"{last_100['reward_mean'].mean():.3f}" if 'reward_mean' in df.columns else "N/A",
            "Convergence Ep": find_convergence_episode(df, 0.85),
            "Local (%)": f"{last_100['decision_frac_local'].mean()*100:.1f}" if 'decision_frac_local' in df.columns else "N/A",
            "RSU (%)": f"{last_100['decision_frac_rsu'].mean()*100:.1f}" if 'decision_frac_rsu' in df.columns else "N/A",
            "V2V (%)": f"{last_100['decision_frac_v2v'].mean()*100:.1f}" if 'decision_frac_v2v' in df.columns else "N/A",
            "RSU Queue": f"{last_100['avg_rsu_queue'].mean():.1f}" if 'avg_rsu_queue' in df.columns else "N/A",
        }
        summary.append(row)
    
    # 保存为CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/ablation_summary.csv", index=False)
    print(f"Saved: {output_dir}/ablation_summary.csv")
    
    # 绘制表格图
    fig, ax = plt.subplots(figsize=(16, len(summary) * 0.8 + 2))
    ax.axis('off')
    
    table_data = [[row[col] for col in summary_df.columns] for row in summary]
    
    table = ax.table(
        cellText=table_data,
        colLabels=summary_df.columns.tolist(),
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 着色表头
    for i, col in enumerate(summary_df.columns):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Ablation Study Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/ablation_summary_table.png")
    
    return summary_df


def plot_comprehensive_dashboard(data: Dict, output_dir: str):
    """绘制综合仪表板"""
    fig = plt.figure(figsize=(20, 16))
    
    # 布局：2x3
    ax1 = fig.add_subplot(2, 3, 1)  # 成功率
    ax2 = fig.add_subplot(2, 3, 2)  # 奖励
    ax3 = fig.add_subplot(2, 3, 3)  # RSU队列
    ax4 = fig.add_subplot(2, 3, 4)  # 策略熵
    ax5 = fig.add_subplot(2, 3, 5)  # 收敛速度
    ax6 = fig.add_subplot(2, 3, 6)  # 汇总表
    
    # 1. 成功率
    for key, df in data.items():
        if df is None or 'task_success_rate' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['task_success_rate']) * 100
        ax1.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Task Success Rate')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. 奖励
    for key, df in data.items():
        if df is None or 'reward_mean' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['reward_mean'])
        ax2.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Mean Reward')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. RSU队列
    for key, df in data.items():
        if df is None or 'avg_rsu_queue' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['avg_rsu_queue'])
        ax3.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('RSU Queue')
    ax3.set_title('RSU Queue Load')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 策略熵
    for key, df in data.items():
        if df is None or 'policy_entropy' not in df.columns:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray", "linestyle": "-"})
        smoothed = smooth(df['policy_entropy'])
        ax4.plot(df['episode'], smoothed,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 收敛速度柱状图
    convergence_data = []
    for key, df in data.items():
        if df is None:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key, "color": "gray"})
        conv_ep = find_convergence_episode(df, threshold=0.85)
        if conv_ep > 0:
            convergence_data.append({
                "name": cfg["label"].split("(")[0].strip()[:15],
                "episode": conv_ep,
                "color": cfg["color"]
            })
    
    if convergence_data:
        names = [d["name"] for d in convergence_data]
        episodes = [d["episode"] for d in convergence_data]
        colors = [d["color"] for d in convergence_data]
        ax5.barh(names, episodes, color=colors, alpha=0.8)
        ax5.set_xlabel('Episodes to 85% SR')
        ax5.set_title('Convergence Speed')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. 汇总表
    ax6.axis('off')
    summary_data = []
    for key, df in data.items():
        if df is None:
            continue
        cfg = EXPERIMENT_CONFIG.get(key, {"label": key})
        last_100 = df.tail(100)
        row = [
            cfg["label"].split("(")[0].strip()[:12],
            f"{last_100['task_success_rate'].mean()*100:.1f}%" if 'task_success_rate' in df.columns else "N/A",
            f"{last_100['reward_mean'].mean():.2f}" if 'reward_mean' in df.columns else "N/A",
            f"{find_convergence_episode(df, 0.85)}" if find_convergence_episode(df, 0.85) > 0 else "N/C",
        ]
        summary_data.append(row)
    
    if summary_data:
        table = ax6.table(
            cellText=summary_data,
            colLabels=['Method', 'Success', 'Reward', 'Conv.Ep'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.3, 2.0)
        ax6.set_title('Summary (Last 100 Ep)', fontsize=11, pad=10)
    
    plt.suptitle('VEC DAG Offloading MAPPO - Ablation Study Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_dashboard.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ablation_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Comparison Plotting")
    parser.add_argument("--full", type=str, default=None, help="Full model run directory")
    parser.add_argument("--no-transformer", type=str, default=None, help="w/o Transformer run directory")
    parser.add_argument("--no-edge", type=str, default=None, help="w/o Edge Bias run directory")
    parser.add_argument("--no-physics", type=str, default=None, help="w/o Physics Bias run directory")
    parser.add_argument("--exp-b", type=str, default=None, help="Exp B (High Pressure) run directory")
    parser.add_argument("--hard", type=str, default=None, help="Hard Mode run directory")
    parser.add_argument("--output", type=str, default="runs/ablation_comparison", help="Output directory")
    args = parser.parse_args()
    
    # 自动查找或使用指定目录
    data = {}
    
    dir_mapping = {
        "full": args.full,
        "no_transformer": args.no_transformer,
        "no_edge": args.no_edge,
        "no_physics": args.no_physics,
        "exp_b": args.exp_b,
        "hard": args.hard,
    }
    
    # 如果没有指定目录，尝试自动查找最新的
    if all(v is None for v in dir_mapping.values()):
        print("No directories specified, searching for latest runs...")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                name = d.name.lower()
                if "ablation_full" in name and "full" not in data:
                    data["full"] = load_metrics(d)
                elif "no_transformer" in name and "no_transformer" not in data:
                    data["no_transformer"] = load_metrics(d)
                elif "no_edge" in name and "no_edge" not in data:
                    data["no_edge"] = load_metrics(d)
                elif "no_physics" in name and "no_physics" not in data:
                    data["no_physics"] = load_metrics(d)
                elif "exp_b" in name and "exp_b" not in data:
                    data["exp_b"] = load_metrics(d)
                elif "hard_mode" in name and "hard" not in data:
                    data["hard"] = load_metrics(d)
    else:
        for key, path in dir_mapping.items():
            if path:
                data[key] = load_metrics(Path(path))
    
    print(f"\nLoaded experiments:")
    for key, df in data.items():
        if df is not None:
            print(f"  {key}: {len(df)} episodes")
    
    if not any(df is not None for df in data.values()):
        print("No data found!")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 生成所有图表
    print("\nGenerating plots...")
    plot_success_rate_comparison(data, args.output)
    plot_reward_comparison(data, args.output)
    plot_convergence_comparison(data, args.output)
    plot_action_distribution(data, args.output)
    plot_rsu_queue_comparison(data, args.output)
    plot_entropy_comparison(data, args.output)
    generate_summary_table(data, args.output)
    plot_comprehensive_dashboard(data, args.output)
    
    print(f"\n=== All plots saved to {args.output}/ ===")


if __name__ == "__main__":
    main()
