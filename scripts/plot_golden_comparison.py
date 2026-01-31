#!/usr/bin/env python3
"""
黄金场景实验对比图生成
生成能体现方法优越性的对比分析图
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_data(run_dir):
    """加载单个实验的数据"""
    metrics_file = os.path.join(run_dir, "metrics", "train_metrics.csv")
    if not os.path.exists(metrics_file):
        return None
    
    df = pd.read_csv(metrics_file)
    return df

def get_final_metrics(df, window=50):
    """获取最终指标（最后window个episode的均值）"""
    if df is None or len(df) < window:
        return {}
    
    final = df.tail(window)
    metrics = {
        'success_rate': final['subtask_success_rate'].mean() if 'subtask_success_rate' in final else 0,
        'mean_reward': final['mean_reward'].mean() if 'mean_reward' in final else 0,
        'rsu_queue': final['rsu_queue_mean'].mean() if 'rsu_queue_mean' in final else 0,
        'convergence_ep': find_convergence_episode(df),
    }
    return metrics

def find_convergence_episode(df, threshold=0.95, window=20):
    """找到收敛episode"""
    if 'subtask_success_rate' not in df.columns:
        return -1
    
    rolling = df['subtask_success_rate'].rolling(window).mean()
    max_rate = rolling.max()
    target = max_rate * threshold
    
    converged = rolling >= target
    if converged.any():
        return converged.idxmax()
    return -1

def plot_scene_comparison(experiments, scene_name, output_dir):
    """绘制单个场景的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{scene_name} - 方法对比', fontsize=14, fontweight='bold')
    
    colors = {
        'full': '#2196F3',
        'no_transformer': '#FF9800',
        'no_edge_bias': '#4CAF50',
        'no_physics_bias': '#F44336',
    }
    
    labels = {
        'full': 'Full Model (Ours)',
        'no_transformer': 'w/o Transformer',
        'no_edge_bias': 'w/o Edge/Spatial Bias',
        'no_physics_bias': 'w/o Physics Bias',
    }
    
    for name, df in experiments.items():
        if df is None:
            continue
        
        color = colors.get(name, '#9E9E9E')
        label = labels.get(name, name)
        
        # Success Rate
        if 'subtask_success_rate' in df.columns:
            smoothed = df['subtask_success_rate'].rolling(20).mean()
            axes[0, 0].plot(df['episode'], smoothed, label=label, color=color, linewidth=2)
        
        # Mean Reward
        if 'mean_reward' in df.columns:
            smoothed = df['mean_reward'].rolling(20).mean()
            axes[0, 1].plot(df['episode'], smoothed, label=label, color=color, linewidth=2)
        
        # RSU Queue
        if 'rsu_queue_mean' in df.columns:
            smoothed = df['rsu_queue_mean'].rolling(20).mean()
            axes[1, 0].plot(df['episode'], smoothed, label=label, color=color, linewidth=2)
        
        # Entropy
        if 'policy_entropy' in df.columns:
            smoothed = df['policy_entropy'].rolling(20).mean()
            axes[1, 1].plot(df['episode'], smoothed, label=label, color=color, linewidth=2)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('子任务成功率')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].set_title('平均奖励')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('RSU Queue (GCycles)')
    axes[1, 0].set_title('RSU队列负载')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('策略熵')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{scene_name}_comparison.png'), dpi=150)
    plt.close()

def plot_ablation_contribution(all_data, output_dir):
    """绘制消融贡献分析图"""
    # 收集各场景的数据
    scenes = ['medium', 'largedag', 'resource']
    scene_labels = ['Medium-Challenge', 'Large-DAG', 'Resource-Contention']
    
    methods = ['full', 'no_transformer', 'no_edge_bias', 'no_physics_bias']
    method_labels = ['Full Model', 'w/o Transformer', 'w/o Edge Bias', 'w/o Physics Bias']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('消融实验：各组件在不同场景下的贡献', fontsize=14, fontweight='bold')
    
    x = np.arange(len(scenes))
    width = 0.2
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        success_rates = []
        for scene in scenes:
            key = f"{scene}_{method}"
            if key in all_data and all_data[key] is not None:
                metrics = get_final_metrics(all_data[key])
                success_rates.append(metrics.get('success_rate', 0))
            else:
                success_rates.append(0)
        
        axes[0].bar(x + i * width, success_rates, width, label=label, color=color)
    
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('成功率对比')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(scene_labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 计算相对于Full Model的性能下降
    for i, (method, label, color) in enumerate(zip(methods[1:], method_labels[1:], colors[1:])):
        drops = []
        for scene in scenes:
            full_key = f"{scene}_full"
            ablation_key = f"{scene}_{method}"
            
            if full_key in all_data and ablation_key in all_data:
                full_metrics = get_final_metrics(all_data[full_key])
                ablation_metrics = get_final_metrics(all_data[ablation_key])
                
                full_rate = full_metrics.get('success_rate', 0)
                ablation_rate = ablation_metrics.get('success_rate', 0)
                
                if full_rate > 0:
                    drop = (full_rate - ablation_rate) / full_rate * 100
                else:
                    drop = 0
                drops.append(drop)
            else:
                drops.append(0)
        
        axes[1].bar(x + i * width, drops, width, label=label, color=color)
    
    axes[1].set_ylabel('Performance Drop (%)')
    axes[1].set_title('相对Full Model的性能下降')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(scene_labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 绘制收敛速度对比
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        convergence_eps = []
        for scene in scenes:
            key = f"{scene}_{method}"
            if key in all_data and all_data[key] is not None:
                metrics = get_final_metrics(all_data[key])
                conv_ep = metrics.get('convergence_ep', -1)
                convergence_eps.append(conv_ep if conv_ep > 0 else 1000)
            else:
                convergence_eps.append(1000)
        
        axes[2].bar(x + i * width, convergence_eps, width, label=label, color=color)
    
    axes[2].set_ylabel('Convergence Episode')
    axes[2].set_title('收敛速度对比')
    axes[2].set_xticks(x + width * 1.5)
    axes[2].set_xticklabels(scene_labels)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_contribution_analysis.png'), dpi=150)
    plt.close()

def generate_summary_table(all_data, output_dir):
    """生成汇总表格"""
    rows = []
    
    scenes = ['medium', 'largedag', 'resource']
    methods = ['full', 'no_transformer', 'no_edge_bias', 'no_physics_bias']
    
    for scene in scenes:
        for method in methods:
            key = f"{scene}_{method}"
            if key in all_data and all_data[key] is not None:
                metrics = get_final_metrics(all_data[key])
                rows.append({
                    'Scene': scene,
                    'Method': method,
                    'Success Rate (%)': f"{metrics.get('success_rate', 0):.1f}",
                    'Mean Reward': f"{metrics.get('mean_reward', 0):.3f}",
                    'RSU Queue': f"{metrics.get('rsu_queue', 0):.1f}",
                    'Convergence Ep': metrics.get('convergence_ep', -1),
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'golden_summary.csv'), index=False)
    
    # 生成表格图片
    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#E3F2FD'] * len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('黄金场景实验汇总', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'golden_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--runs-dir', type=str, default='runs')
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = args.timestamp or 'latest'
    output_dir = os.path.join(args.runs_dir, f'golden_study_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有实验数据
    all_data = {}
    
    # 定义实验映射
    experiment_map = {
        'medium_full': 'golden_medium_full',
        'medium_no_transformer': 'golden_medium_no_transformer',
        'medium_no_edge_bias': 'golden_medium_no_edge_bias',
        'medium_no_physics_bias': 'golden_medium_no_physics_bias',
        'largedag_full': 'golden_largedag_full',
        'largedag_no_transformer': 'golden_largedag_no_transformer',
        'resource_full': 'golden_resource_full',
        'resource_no_physics_bias': 'golden_resource_no_physics_bias',
        'resource_no_edge_bias': 'golden_resource_no_edge_bias',
    }
    
    for key, run_id in experiment_map.items():
        # 查找最新的匹配目录
        pattern = os.path.join(args.runs_dir, f'{run_id}_*')
        matches = sorted(glob.glob(pattern))
        if matches:
            run_dir = matches[-1]
            df = load_experiment_data(run_dir)
            all_data[key] = df
            print(f"Loaded: {key} from {run_dir}")
        else:
            print(f"Not found: {key}")
            all_data[key] = None
    
    # 生成各场景对比图
    scenes = {
        'medium': ['full', 'no_transformer', 'no_edge_bias', 'no_physics_bias'],
        'largedag': ['full', 'no_transformer'],
        'resource': ['full', 'no_physics_bias', 'no_edge_bias'],
    }
    
    for scene, methods in scenes.items():
        experiments = {m: all_data.get(f'{scene}_{m}') for m in methods}
        if any(v is not None for v in experiments.values()):
            plot_scene_comparison(experiments, scene, output_dir)
            print(f"Generated: {scene}_comparison.png")
    
    # 生成消融贡献分析图
    plot_ablation_contribution(all_data, output_dir)
    print("Generated: ablation_contribution_analysis.png")
    
    # 生成汇总表格
    summary_df = generate_summary_table(all_data, output_dir)
    print("Generated: golden_summary.csv, golden_summary_table.png")
    
    print(f"\n所有结果保存在: {output_dir}")

if __name__ == '__main__':
    main()
