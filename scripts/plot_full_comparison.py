#!/usr/bin/env python3
"""
完整实验对比绘图脚本 - Full Experiment Comparison Plots

生成论文级别的详细对比图：
1. 任务成功率曲线对比
2. 奖励收敛曲线对比
3. 动作分布演化对比
4. RSU队列负载对比
5. 策略熵对比
6. 收敛速度分析
7. 消融实验汇总表
8. 综合仪表板

使用方法:
    python scripts/plot_full_comparison.py --timestamp 20260125_120000
    python scripts/plot_full_comparison.py --output runs/my_comparison
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def find_experiment_dirs(timestamp=None):
    """查找实验目录"""
    runs_dir = Path("runs")
    exp_dirs = {}
    
    # 定义实验名称映射
    exp_patterns = {
        "Full Model": ["exp_full_model", "exp_a_baseline", "ablation_full"],
        "w/o Transformer": ["exp_no_transformer", "ablation_no_transformer"],
        "w/o Edge Bias": ["exp_no_edge_bias", "ablation_no_edge_bias"],
        "w/o Physics Bias": ["exp_no_physics_bias", "ablation_no_physics_bias"],
        "High Pressure": ["exp_high_pressure", "exp_b_highpressure"],
        "Hard Mode": ["exp_hard_mode", "ablation_hard_mode"],
        "Large Scale": ["exp_c_largescale"],
    }
    
    for d in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        name_lower = d.name.lower()
        
        # 如果指定了timestamp，只匹配该timestamp的实验
        if timestamp and timestamp not in d.name:
            continue
        
        for exp_name, patterns in exp_patterns.items():
            if exp_name in exp_dirs:
                continue
            for pattern in patterns:
                if pattern in name_lower:
                    exp_dirs[exp_name] = d
                    break
    
    return exp_dirs


def load_metrics(run_dir):
    """加载训练指标CSV"""
    csv_paths = [
        run_dir / "metrics" / "train_metrics.csv",
        run_dir / "logs" / "training_stats.csv",
        run_dir / "episode_log.csv",
    ]
    
    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    return df
            except Exception as e:
                print(f"Warning: Failed to load {csv_path}: {e}")
    
    return None


def smooth(data, window=20):
    """平滑曲线"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def get_convergence_episode(data, threshold=0.9, target_col='task_success_rate'):
    """计算收敛episode（首次达到阈值的90%）"""
    if target_col not in data.columns:
        return None
    
    max_val = data[target_col].max()
    target = max_val * threshold
    
    smoothed = smooth(data[target_col], window=30)
    above_threshold = smoothed >= target
    
    if above_threshold.any():
        return above_threshold.idxmax()
    return None


def plot_success_rate_comparison(exp_data, output_dir):
    """图1: 任务成功率曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))
    
    for (name, df), color in zip(exp_data.items(), colors):
        if 'task_success_rate' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['task_success_rate'] * 100, window=30)
            ax.plot(x, y, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Task Success Rate (%)', fontsize=12)
    ax.set_title('Task Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_success_rate_comparison.png")
    plt.close()
    print(f"  Saved: 01_success_rate_comparison.png")


def plot_reward_comparison(exp_data, output_dir):
    """图2: 奖励收敛曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))
    
    for (name, df), color in zip(exp_data.items(), colors):
        if 'reward_mean' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['reward_mean'], window=30)
            ax.plot(x, y, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Reward Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_reward_comparison.png")
    plt.close()
    print(f"  Saved: 02_reward_comparison.png")


def plot_action_distribution(exp_data, output_dir):
    """图3: 动作分布对比（最后100ep平均）"""
    action_cols = ['decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v']
    action_labels = ['Local', 'RSU', 'V2V']
    
    n_exps = len(exp_data)
    fig, axes = plt.subplots(1, min(n_exps, 6), figsize=(4*min(n_exps, 6), 5))
    if n_exps == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_exps))
    
    for idx, ((name, df), color) in enumerate(zip(exp_data.items(), colors)):
        if idx >= 6:
            break
        ax = axes[idx]
        
        if all(c in df.columns for c in action_cols):
            last_100 = df.tail(100)
            values = [last_100[c].mean() * 100 for c in action_cols]
            bars = ax.bar(action_labels, values, color=[color]*3, alpha=0.8)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel('Fraction (%)')
            ax.set_ylim(0, 100)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', fontsize=9)
    
    plt.suptitle('Action Distribution (Last 100 Episodes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_action_distribution.png")
    plt.close()
    print(f"  Saved: 03_action_distribution.png")


def plot_rsu_queue_comparison(exp_data, output_dir):
    """图4: RSU队列负载对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))
    
    for (name, df), color in zip(exp_data.items(), colors):
        if 'avg_rsu_queue' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['avg_rsu_queue'], window=30)
            ax.plot(x, y, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average RSU Queue Load', fontsize=12)
    ax.set_title('RSU Queue Load Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_rsu_queue_comparison.png")
    plt.close()
    print(f"  Saved: 04_rsu_queue_comparison.png")


def plot_entropy_comparison(exp_data, output_dir):
    """图5: 策略熵对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))
    
    for (name, df), color in zip(exp_data.items(), colors):
        if 'policy_entropy' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['policy_entropy'], window=30)
            ax.plot(x, y, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Policy Entropy', fontsize=12)
    ax.set_title('Policy Entropy Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_entropy_comparison.png")
    plt.close()
    print(f"  Saved: 05_entropy_comparison.png")


def plot_convergence_analysis(exp_data, output_dir):
    """图6: 收敛速度分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 收敛episode对比
    ax1 = axes[0]
    names = []
    conv_eps = []
    final_sr = []
    
    for name, df in exp_data.items():
        if 'task_success_rate' not in df.columns:
            continue
        names.append(name)
        
        # 收敛episode
        conv = get_convergence_episode(df)
        conv_eps.append(conv if conv else len(df))
        
        # 最终成功率
        final_sr.append(df['task_success_rate'].tail(100).mean() * 100)
    
    x = range(len(names))
    bars = ax1.bar(x, conv_eps, color=plt.cm.tab10(np.linspace(0, 1, len(names))), alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Convergence Episode', fontsize=11)
    ax1.set_title('Convergence Speed Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, conv_eps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val}', ha='center', fontsize=9)
    
    # 右图: 最终成功率对比
    ax2 = axes[1]
    bars = ax2.bar(x, final_sr, color=plt.cm.tab10(np.linspace(0, 1, len(names))), alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Final Success Rate (%)', fontsize=11)
    ax2.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, final_sr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_convergence_analysis.png")
    plt.close()
    print(f"  Saved: 06_convergence_analysis.png")


def plot_ablation_table(exp_data, output_dir):
    """图7: 消融实验汇总表"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['Method', 'Success Rate', 'Mean Reward', 'Conv. Episode', 'Local%', 'RSU%', 'V2V%', 'RSU Queue']
    
    for name, df in exp_data.items():
        last_100 = df.tail(100)
        
        sr = f"{last_100['task_success_rate'].mean()*100:.1f}%" if 'task_success_rate' in df.columns else "N/A"
        reward = f"{last_100['reward_mean'].mean():.3f}" if 'reward_mean' in df.columns else "N/A"
        conv = get_convergence_episode(df)
        conv_str = str(conv) if conv else "N/A"
        local = f"{last_100['decision_frac_local'].mean()*100:.1f}%" if 'decision_frac_local' in df.columns else "N/A"
        rsu = f"{last_100['decision_frac_rsu'].mean()*100:.1f}%" if 'decision_frac_rsu' in df.columns else "N/A"
        v2v = f"{last_100['decision_frac_v2v'].mean()*100:.1f}%" if 'decision_frac_v2v' in df.columns else "N/A"
        queue = f"{last_100['avg_rsu_queue'].mean():.1f}" if 'avg_rsu_queue' in df.columns else "N/A"
        
        table_data.append([name, sr, reward, conv_str, local, rsu, v2v, queue])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 高亮表头
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 高亮Full Model行
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#E2EFDA')
    
    ax.set_title('Ablation Study Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_ablation_summary_table.png")
    plt.close()
    print(f"  Saved: 07_ablation_summary_table.png")
    
    # 同时保存CSV
    df_table = pd.DataFrame(table_data, columns=headers)
    df_table.to_csv(f"{output_dir}/ablation_summary.csv", index=False)
    print(f"  Saved: ablation_summary.csv")


def plot_comprehensive_dashboard(exp_data, output_dir):
    """图8: 综合仪表板"""
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))
    
    # 1. 成功率曲线 (占左上2x1)
    ax1 = fig.add_subplot(gs[0, :2])
    for (name, df), color in zip(exp_data.items(), colors):
        if 'task_success_rate' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['task_success_rate'] * 100, window=30)
            ax1.plot(x, y, label=name, color=color, linewidth=1.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Task Success Rate')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. 最终成功率柱状图 (右上)
    ax2 = fig.add_subplot(gs[0, 2])
    names = list(exp_data.keys())
    final_sr = []
    for df in exp_data.values():
        if 'task_success_rate' in df.columns:
            final_sr.append(df['task_success_rate'].tail(100).mean() * 100)
        else:
            final_sr.append(0)
    bars = ax2.barh(range(len(names)), final_sr, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Success Rate (%)')
    ax2.set_title('Final Performance')
    ax2.set_xlim(0, 105)
    for bar, val in zip(bars, final_sr):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)
    
    # 3. 奖励曲线 (中左)
    ax3 = fig.add_subplot(gs[1, 0])
    for (name, df), color in zip(exp_data.items(), colors):
        if 'reward_mean' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['reward_mean'], window=30)
            ax3.plot(x, y, label=name, color=color, linewidth=1.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Reward Convergence')
    ax3.grid(True, alpha=0.3)
    
    # 4. RSU队列负载 (中中)
    ax4 = fig.add_subplot(gs[1, 1])
    for (name, df), color in zip(exp_data.items(), colors):
        if 'avg_rsu_queue' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['avg_rsu_queue'], window=30)
            ax4.plot(x, y, label=name, color=color, linewidth=1.5)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('RSU Queue')
    ax4.set_title('RSU Queue Load')
    ax4.grid(True, alpha=0.3)
    
    # 5. 策略熵 (中右)
    ax5 = fig.add_subplot(gs[1, 2])
    for (name, df), color in zip(exp_data.items(), colors):
        if 'policy_entropy' in df.columns:
            x = df['episode'] if 'episode' in df.columns else range(len(df))
            y = smooth(df['policy_entropy'], window=30)
            ax5.plot(x, y, label=name, color=color, linewidth=1.5)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Entropy')
    ax5.set_title('Policy Entropy')
    ax5.grid(True, alpha=0.3)
    
    # 6. 汇总表 (下方3x1)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    table_data = []
    for name, df in exp_data.items():
        last_100 = df.tail(100)
        row = [
            name[:20],
            f"{last_100['task_success_rate'].mean()*100:.1f}%" if 'task_success_rate' in df.columns else "N/A",
            f"{last_100['reward_mean'].mean():.3f}" if 'reward_mean' in df.columns else "N/A",
            f"{last_100['decision_frac_local'].mean()*100:.1f}%" if 'decision_frac_local' in df.columns else "N/A",
            f"{last_100['decision_frac_rsu'].mean()*100:.1f}%" if 'decision_frac_rsu' in df.columns else "N/A",
            f"{last_100['decision_frac_v2v'].mean()*100:.1f}%" if 'decision_frac_v2v' in df.columns else "N/A",
        ]
        table_data.append(row)
    
    headers = ['Method', 'Success Rate', 'Mean Reward', 'Local', 'RSU', 'V2V']
    table = ax6.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.suptitle('VEC DAG Offloading - Full Experiment Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{output_dir}/08_comprehensive_dashboard.png")
    plt.close()
    print(f"  Saved: 08_comprehensive_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description="Generate full experiment comparison plots")
    parser.add_argument("--timestamp", type=str, default=None, help="Experiment timestamp filter")
    parser.add_argument("--output", type=str, default="runs/full_comparison", help="Output directory")
    args = parser.parse_args()
    
    print("="*60)
    print("Full Experiment Comparison Plot Generator")
    print("="*60)
    
    # 查找实验目录
    print("\nSearching for experiment directories...")
    exp_dirs = find_experiment_dirs(args.timestamp)
    
    if not exp_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(exp_dirs)} experiments:")
    for name, path in exp_dirs.items():
        print(f"  - {name}: {path}")
    
    # 加载数据
    print("\nLoading experiment data...")
    exp_data = {}
    for name, run_dir in exp_dirs.items():
        df = load_metrics(run_dir)
        if df is not None and len(df) > 0:
            exp_data[name] = df
            print(f"  - {name}: {len(df)} episodes")
        else:
            print(f"  - {name}: No data found")
    
    if not exp_data:
        print("No valid data found!")
        return
    
    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # 生成所有图表
    print("\nGenerating plots...")
    plot_success_rate_comparison(exp_data, output_dir)
    plot_reward_comparison(exp_data, output_dir)
    plot_action_distribution(exp_data, output_dir)
    plot_rsu_queue_comparison(exp_data, output_dir)
    plot_entropy_comparison(exp_data, output_dir)
    plot_convergence_analysis(exp_data, output_dir)
    plot_ablation_table(exp_data, output_dir)
    plot_comprehensive_dashboard(exp_data, output_dir)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
