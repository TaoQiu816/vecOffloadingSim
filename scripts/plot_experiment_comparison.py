#!/usr/bin/env python3
"""
三组实验对比绘图脚本
Plot Comparison for A/B/C Experiments

使用方法:
    python scripts/plot_experiment_comparison.py
    python scripts/plot_experiment_comparison.py --exp-a runs/exp_A_xxx --exp-b runs/exp_B_xxx --exp-c runs/exp_C_xxx
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_latest_runs():
    """自动找到最新的三组实验目录"""
    runs_dir = Path("runs")
    exp_dirs = {"A": None, "B": None, "C": None}
    
    for d in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        name = d.name.lower()
        if "exp_a" in name and exp_dirs["A"] is None:
            exp_dirs["A"] = d
        elif "exp_b" in name and exp_dirs["B"] is None:
            exp_dirs["B"] = d
        elif "exp_c" in name and exp_dirs["C"] is None:
            exp_dirs["C"] = d
        
        if all(v is not None for v in exp_dirs.values()):
            break
    
    return exp_dirs

def load_metrics(run_dir):
    """加载训练指标CSV"""
    csv_path = run_dir / "metrics" / "train_metrics.csv"
    if not csv_path.exists():
        csv_path = run_dir / "logs" / "training_stats.csv"
    if not csv_path.exists():
        print(f"Warning: No metrics found in {run_dir}")
        return None
    return pd.read_csv(csv_path)

def plot_comparison(exp_dirs, output_dir):
    """绘制对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data = {}
    labels = {
        "A": "基准延长 (Baseline)",
        "B": "高压力场景 (High Pressure)",
        "C": "大规模 (Large Scale)"
    }
    colors = {"A": "#2E86AB", "B": "#E94F37", "C": "#44AF69"}
    
    for key, run_dir in exp_dirs.items():
        if run_dir is not None:
            df = load_metrics(run_dir)
            if df is not None:
                data[key] = df
                print(f"Loaded {key}: {run_dir} ({len(df)} episodes)")
    
    if not data:
        print("No data found!")
        return
    
    # =========================================================================
    # 图1: 任务成功率对比
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, df in data.items():
        if 'task_success_rate' in df.columns:
            # 平滑处理
            smoothed = df['task_success_rate'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed * 100, label=labels[key], color=colors[key], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Task Success Rate (%)', fontsize=12)
    ax.set_title('任务成功率对比 / Task Success Rate Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_success_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison_success_rate.png")
    
    # =========================================================================
    # 图2: 平均奖励对比
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, df in data.items():
        if 'reward_mean' in df.columns:
            smoothed = df['reward_mean'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed, label=labels[key], color=colors[key], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('平均奖励对比 / Mean Reward Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_reward.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison_reward.png")
    
    # =========================================================================
    # 图3: 动作分布对比 (最后100ep平均)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    action_cols = ['decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v']
    action_labels = ['Local', 'RSU', 'V2V']
    
    for idx, (key, df) in enumerate(data.items()):
        if all(c in df.columns for c in action_cols):
            last_100 = df.tail(100)
            values = [last_100[c].mean() * 100 for c in action_cols]
            bars = axes[idx].bar(action_labels, values, color=[colors[key]]*3, alpha=0.8)
            axes[idx].set_title(f'{labels[key]}', fontsize=12)
            axes[idx].set_ylabel('Fraction (%)')
            axes[idx].set_ylim(0, 100)
            for bar, val in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                              f'{val:.1f}%', ha='center', fontsize=10)
    
    plt.suptitle('动作分布对比 (最后100ep) / Action Distribution Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_action_dist.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison_action_dist.png")
    
    # =========================================================================
    # 图4: RSU队列负载对比
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, df in data.items():
        if 'avg_rsu_queue' in df.columns:
            smoothed = df['avg_rsu_queue'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed, label=labels[key], color=colors[key], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average RSU Queue Load', fontsize=12)
    ax.set_title('RSU队列负载对比 / RSU Queue Load Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_rsu_queue.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison_rsu_queue.png")
    
    # =========================================================================
    # 图5: 策略熵对比
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, df in data.items():
        if 'policy_entropy' in df.columns:
            smoothed = df['policy_entropy'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed, label=labels[key], color=colors[key], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Policy Entropy', fontsize=12)
    ax.set_title('策略熵对比 / Policy Entropy Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_entropy.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison_entropy.png")
    
    # =========================================================================
    # 图6: 综合对比仪表板
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 成功率
    ax = axes[0, 0]
    for key, df in data.items():
        if 'task_success_rate' in df.columns:
            smoothed = df['task_success_rate'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed * 100, label=labels[key], color=colors[key], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('任务成功率')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 奖励
    ax = axes[0, 1]
    for key, df in data.items():
        if 'reward_mean' in df.columns:
            smoothed = df['reward_mean'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed, label=labels[key], color=colors[key], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.set_title('平均奖励')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # RSU队列
    ax = axes[1, 0]
    for key, df in data.items():
        if 'avg_rsu_queue' in df.columns:
            smoothed = df['avg_rsu_queue'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['episode'], smoothed, label=labels[key], color=colors[key], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('RSU Queue')
    ax.set_title('RSU队列负载')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 最终指标汇总表
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = []
    for key, df in data.items():
        last_100 = df.tail(100)
        row = [
            labels[key].split('(')[0].strip(),
            f"{last_100['task_success_rate'].mean()*100:.1f}%" if 'task_success_rate' in df.columns else "N/A",
            f"{last_100['reward_mean'].mean():.2f}" if 'reward_mean' in df.columns else "N/A",
            f"{last_100['decision_frac_local'].mean()*100:.1f}%" if 'decision_frac_local' in df.columns else "N/A",
            f"{last_100['decision_frac_rsu'].mean()*100:.1f}%" if 'decision_frac_rsu' in df.columns else "N/A",
            f"{last_100['decision_frac_v2v'].mean()*100:.1f}%" if 'decision_frac_v2v' in df.columns else "N/A",
        ]
        summary_data.append(row)
    
    table = ax.table(
        cellText=summary_data,
        colLabels=['方案', '成功率', '平均奖励', 'Local', 'RSU', 'V2V'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('最终指标汇总 (最后100ep)', fontsize=12, pad=20)
    
    plt.suptitle('VEC DAG卸载 MAPPO 对比实验总览', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_dashboard.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison_dashboard.png")
    
    print(f"\n=== 所有对比图已保存到 {output_dir}/ ===")

def main():
    parser = argparse.ArgumentParser(description="Plot experiment comparison")
    parser.add_argument("--exp-a", type=str, default=None, help="Experiment A directory")
    parser.add_argument("--exp-b", type=str, default=None, help="Experiment B directory")
    parser.add_argument("--exp-c", type=str, default=None, help="Experiment C directory")
    parser.add_argument("--output", type=str, default="runs/comparison_plots", help="Output directory")
    args = parser.parse_args()
    
    # 自动查找或使用指定目录
    if args.exp_a or args.exp_b or args.exp_c:
        exp_dirs = {
            "A": Path(args.exp_a) if args.exp_a else None,
            "B": Path(args.exp_b) if args.exp_b else None,
            "C": Path(args.exp_c) if args.exp_c else None,
        }
    else:
        print("自动查找最新的三组实验目录...")
        exp_dirs = find_latest_runs()
    
    print(f"实验目录:")
    for key, path in exp_dirs.items():
        print(f"  {key}: {path}")
    
    plot_comparison(exp_dirs, args.output)

if __name__ == "__main__":
    main()
