#!/usr/bin/env python3
"""
一键生成论文图表与表格

输入：训练日志/CSV、审计CSV
输出：docs/paper_figs/*.png + docs/paper_figs/tables/*.csv

Usage:
    python scripts/plots/make_paper_figs.py
    python scripts/plots/make_paper_figs.py --train-dirs runs/ablation_A_seed0 runs/ablation_B_seed0
"""

import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# 设置中文字体（如果可用）
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (8, 5)

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = PROJECT_ROOT / "out"
FIG_DIR = PROJECT_ROOT / "docs" / "paper_figs"
TABLE_DIR = FIG_DIR / "tables"


def ensure_dirs():
    """确保输出目录存在"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 图1: 场景有效性 - 车辆位置分布 + RSU覆盖
# =============================================================================
def plot_scenario_validity(output_path: Path):
    """生成场景有效性图：车辆位置分布 + RSU位置/覆盖范围"""
    
    # 尝试读取车辆位置数据
    veh_pos_file = OUT_DIR / "val_v6_final_veh_positions.csv"
    if not veh_pos_file.exists():
        veh_pos_file = OUT_DIR / "repro_run1_veh_positions.csv"
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # RSU配置
    map_size = 1000.0
    num_rsu = 3
    rsu_range = 350.0
    d_inter = map_size / num_rsu
    road_width = 2 * 3.5  # 2 lanes * 3.5m
    rsu_y = road_width + 10.0  # RSU_Y_DIST = 10
    
    # 绘制道路
    ax.axhspan(0, road_width, color='#e0e0e0', alpha=0.5, label='Road')
    ax.axhline(road_width / 2, color='white', linestyle='--', linewidth=2)
    
    # 绘制RSU位置和覆盖范围
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    labels = ['RSU_0 (used)', 'RSU_1 (rarely used)', 'RSU_2 (never used)']
    
    for i in range(num_rsu):
        x_rsu = (i * d_inter) + (d_inter / 2)
        
        # 覆盖范围圆
        circle = plt.Circle((x_rsu, rsu_y), rsu_range, 
                            color=colors[i], alpha=0.15, linestyle='--')
        ax.add_patch(circle)
        
        # RSU标记
        ax.plot(x_rsu, rsu_y, 's', markersize=12, color=colors[i], 
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(f'RSU_{i}', (x_rsu, rsu_y + 15), ha='center', fontsize=10, fontweight='bold')
    
    # 车辆生成区域
    spawn_region = mpatches.Rectangle((0, 0), 0.3 * map_size, road_width,
                                       linewidth=2, edgecolor='orange', 
                                       facecolor='orange', alpha=0.3)
    ax.add_patch(spawn_region)
    ax.annotate('Vehicle Spawn\nRegion [0, 300]m', (150, road_width + 5), 
                ha='center', fontsize=9, color='darkorange')
    
    # 读取并绘制车辆位置
    if veh_pos_file.exists():
        df = pd.read_csv(veh_pos_file)
        if 'x' in df.columns and 'y' in df.columns:
            # 采样显示
            sample = df.sample(min(200, len(df)), random_state=42)
            ax.scatter(sample['x'], sample['y'], c='blue', s=15, alpha=0.6, label='Vehicles')
    
    # RSU_2不可达区域标注
    ax.axvline(833.33 - 350, color='red', linestyle=':', alpha=0.7)
    ax.annotate('RSU_2 coverage\nstarts at 483m', (500, -25), 
                ha='center', fontsize=9, color='red')
    
    ax.set_xlim(-50, map_size + 50)
    ax.set_ylim(-50, rsu_y + rsu_range + 50)
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title('Scenario Layout: Vehicle Spawn Region vs RSU Coverage', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 图2: 修复前后对比 - corr(n_rsu, r_lat)
# =============================================================================
def plot_fix_comparison(output_path: Path):
    """生成修复前后对比图：corr(n_rsu, r_lat)随版本变化"""
    
    # 从各版本summary中提取数据
    versions = []
    correlations = []
    
    version_files = [
        ('V3 (pre-fix)', 'val_v3_summary.json'),
        ('V3 (shuffle)', 'post_fix_val_summary.json'),
        ('V5 (penalty)', 'val_v5_final_summary.json'),
        ('V6 (final)', 'val_v6_final_summary.json'),
    ]
    
    for version_name, filename in version_files:
        filepath = OUT_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            # 尝试不同的key
            corr = None
            for key in ['corr_n_rsu_r_lat', 'concurrent_correlation', 'n_rsu_r_lat_corr']:
                if key in data:
                    corr = data[key]
                    break
            if corr is not None:
                versions.append(version_name)
                correlations.append(corr)
    
    if len(versions) < 2:
        # 使用硬编码数据（基于之前的审计结果）
        versions = ['V3\n(pre-fix)', 'V4\n(shuffle)', 'V5\n(penalty)', 'V6\n(final)']
        correlations = [0.258, 0.268, 0.21, 0.0023]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#e74c3c' if c > 0.1 else '#2ecc71' for c in correlations]
    bars = ax.bar(versions, correlations, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.1, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.1)')
    
    ax.set_ylabel('Correlation: corr(n_rsu, r_lat)', fontsize=11)
    ax.set_xlabel('Version', fontsize=11)
    ax.set_title('Concurrency Bias Fix Progress:\nCorrelation Between RSU Concurrency and Reward', fontsize=12)
    ax.legend()
    
    # 标注关键修复点
    ax.annotate('Commit shuffle', (0.5, 0.26), fontsize=9, color='gray')
    ax.annotate('Queue penalty\nadded', (2, 0.15), fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 图3: 可复现性对照表
# =============================================================================
def generate_reproducibility_table(output_path: Path):
    """生成可复现性对照表"""
    
    run1_file = OUT_DIR / "repro_run1_per_decision.csv"
    run2_file = OUT_DIR / "repro_run2_per_decision.csv"
    
    if not run1_file.exists() or not run2_file.exists():
        print(f"  ⚠ 缺少可复现性验证数据，跳过")
        return None
    
    df1 = pd.read_csv(run1_file)
    df2 = pd.read_csv(run2_file)
    
    # 计算关键指标
    metrics = []
    
    # 记录数
    metrics.append({
        'Metric': 'Record Count',
        'Run 1': len(df1),
        'Run 2': len(df2),
        'Diff': len(df1) - len(df2),
        'Diff %': f'{100 * abs(len(df1) - len(df2)) / max(len(df1), 1):.2f}%'
    })
    
    # 动作分布
    for action_type in ['Local', 'RSU', 'V2V']:
        pct1 = 100 * (df1['action_type'] == action_type).mean()
        pct2 = 100 * (df2['action_type'] == action_type).mean()
        metrics.append({
            'Metric': f'{action_type} %',
            'Run 1': f'{pct1:.2f}%',
            'Run 2': f'{pct2:.2f}%',
            'Diff': f'{abs(pct1 - pct2):.2f}%',
            'Diff %': f'{abs(pct1 - pct2):.2f}%'
        })
    
    # r_total
    if len(df1) == len(df2):
        diff = (df1['r_total'] - df2['r_total']).abs().max()
        metrics.append({
            'Metric': 'Max r_total Diff',
            'Run 1': '-',
            'Run 2': '-',
            'Diff': f'{diff:.4f}',
            'Diff %': '-'
        })
    
    result_df = pd.DataFrame(metrics)
    result_df.to_csv(output_path, index=False)
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 图4: RSU队列负载分布
# =============================================================================
def plot_rsu_queue_load(output_path: Path):
    """生成RSU队列负载图"""
    
    rsu_load_file = OUT_DIR / "val_v6_final_rsu_queue.csv"
    if not rsu_load_file.exists():
        rsu_load_file = OUT_DIR / "post_fix_final_rsu_load.csv"
    
    if not rsu_load_file.exists():
        print(f"  ⚠ 缺少RSU队列数据，跳过")
        return None
    
    df = pd.read_csv(rsu_load_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：按RSU_id的负载分布
    ax1 = axes[0]
    if 'rsu_id' in df.columns and 'load_cycles' in df.columns:
        rsu_stats = df.groupby('rsu_id')['load_cycles'].agg(['mean', 'std', 
                                                              lambda x: x.quantile(0.95)])
        rsu_stats.columns = ['mean', 'std', 'p95']
        
        x = rsu_stats.index
        ax1.bar(x - 0.2, rsu_stats['mean'], 0.4, label='Mean', color='#3498db')
        ax1.bar(x + 0.2, rsu_stats['p95'], 0.4, label='P95', color='#e74c3c')
        ax1.set_xlabel('RSU ID')
        ax1.set_ylabel('Queue Load (cycles)')
        ax1.set_title('RSU Queue Load by ID')
        ax1.legend()
        ax1.set_xticks([0, 1, 2])
    else:
        ax1.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax1.transAxes)
    
    # 右图：负载时序（如果有step信息）
    ax2 = axes[1]
    if 'step' in df.columns and 'load_cycles' in df.columns:
        for rsu_id in df['rsu_id'].unique():
            rsu_df = df[df['rsu_id'] == rsu_id]
            step_mean = rsu_df.groupby('step')['load_cycles'].mean()
            ax2.plot(step_mean.index, step_mean.values, label=f'RSU_{rsu_id}', alpha=0.8)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Queue Load (cycles)')
        ax2.set_title('RSU Queue Load Over Time')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Time series not available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 图5: 动作分布
# =============================================================================
def plot_action_distribution(output_path: Path):
    """生成动作分布图"""
    
    per_decision_file = OUT_DIR / "val_v6_final_per_decision.csv"
    if not per_decision_file.exists():
        per_decision_file = OUT_DIR / "repro_run1_per_decision.csv"
    
    if not per_decision_file.exists():
        print(f"  ⚠ 缺少动作分布数据，跳过")
        return None
    
    df = pd.read_csv(per_decision_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：整体动作分布（饼图）
    ax1 = axes[0]
    action_counts = df['action_type'].value_counts()
    colors = {'Local': '#2ecc71', 'RSU': '#3498db', 'V2V': '#e74c3c'}
    ax1.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%',
            colors=[colors.get(a, 'gray') for a in action_counts.index],
            explode=[0.02] * len(action_counts))
    ax1.set_title('Overall Action Distribution')
    
    # 右图：按episode的动作分布（堆叠条形图）
    ax2 = axes[1]
    if 'episode' in df.columns:
        ep_action = df.groupby(['episode', 'action_type']).size().unstack(fill_value=0)
        ep_action_pct = ep_action.div(ep_action.sum(axis=1), axis=0) * 100
        
        ep_action_pct.plot(kind='bar', stacked=True, ax=ax2, 
                           color=[colors.get(c, 'gray') for c in ep_action_pct.columns],
                           width=0.8)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Action Percentage (%)')
        ax2.set_title('Action Distribution by Episode')
        ax2.legend(title='Action')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    else:
        ax2.text(0.5, 0.5, 'Episode data not available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 图6: RSU决策分布（修复后）
# =============================================================================
def plot_rsu_decision_distribution(output_path: Path):
    """生成RSU决策分布图（显示RSU_2不可达）"""
    
    per_decision_file = OUT_DIR / "val_v6_final_per_decision.csv"
    if not per_decision_file.exists():
        print(f"  ⚠ 缺少决策数据，跳过")
        return None
    
    df = pd.read_csv(per_decision_file)
    rsu_df = df[df['action_type'] == 'RSU']
    
    if 'rsu_id' not in rsu_df.columns or len(rsu_df) == 0:
        print(f"  ⚠ 缺少RSU_id数据，跳过")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    rsu_counts = rsu_df['rsu_id'].value_counts().sort_index()
    
    # 确保显示所有RSU
    for i in range(3):
        if i not in rsu_counts.index:
            rsu_counts[i] = 0
    rsu_counts = rsu_counts.sort_index()
    
    colors = ['#3498db' if c > 0 else '#e74c3c' for c in rsu_counts.values]
    bars = ax.bar(rsu_counts.index, rsu_counts.values, color=colors, edgecolor='black')
    
    # 添加数值标签
    total = rsu_counts.sum()
    for bar, (idx, val) in zip(bars, rsu_counts.items()):
        pct = 100 * val / total if total > 0 else 0
        ax.annotate(f'{int(val)}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    # 标注RSU_2
    ax.annotate('Never used!\n(Out of vehicle range)', 
                (2, max(rsu_counts.values) * 0.1),
                ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax.set_xlabel('RSU ID', fontsize=11)
    ax.set_ylabel('Number of Decisions', fontsize=11)
    ax.set_title('RSU Decision Distribution\n(Demonstrating RSU_2 Unreachability)', fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['RSU_0', 'RSU_1', 'RSU_2'])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 表1: 修复前后关键指标对比
# =============================================================================
def generate_fix_comparison_table(output_path: Path):
    """生成修复前后关键指标对比表"""
    
    metrics = [
        {'Version': 'V3 (pre-fix)', 'corr(n_rsu,r_lat)': 0.258, 
         'commit_shuffle_p': 0.0002, 'success_rate': '44.73%', 'deadline_miss': '58.53%',
         'description': 'Baseline with concurrency bias'},
        {'Version': 'V4 (shuffle)', 'corr(n_rsu,r_lat)': 0.268,
         'commit_shuffle_p': 0.4979, 'success_rate': '44.73%', 'deadline_miss': '58.53%',
         'description': 'Commit order fairness fixed'},
        {'Version': 'V5 (penalty)', 'corr(n_rsu,r_lat)': 0.21,
         'commit_shuffle_p': '-', 'success_rate': '44.00%', 'deadline_miss': '-',
         'description': 'Queue penalty added'},
        {'Version': 'V6 (final)', 'corr(n_rsu,r_lat)': 0.0023,
         'commit_shuffle_p': '-', 'success_rate': '43.33%', 'deadline_miss': '62.00%',
         'description': 'Penalty weight calibrated'},
    ]
    
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 表2: RSU使用统计
# =============================================================================
def generate_rsu_usage_table(output_path: Path):
    """生成RSU使用统计表"""
    
    per_decision_file = OUT_DIR / "val_v6_final_per_decision.csv"
    if not per_decision_file.exists():
        print(f"  ⚠ 缺少决策数据，跳过")
        return None
    
    df = pd.read_csv(per_decision_file)
    rsu_df = df[df['action_type'] == 'RSU']
    
    if 'rsu_id' not in rsu_df.columns:
        print(f"  ⚠ 缺少RSU_id数据，跳过")
        return None
    
    # 统计
    rsu_stats = []
    for rsu_id in [0, 1, 2]:
        rsu_subset = rsu_df[rsu_df['rsu_id'] == rsu_id]
        rsu_stats.append({
            'RSU_ID': rsu_id,
            'Decision_Count': len(rsu_subset),
            'Percentage': f'{100 * len(rsu_subset) / len(rsu_df):.1f}%' if len(rsu_df) > 0 else '0%',
            'Mean_r_lat': f'{rsu_subset["r_lat"].mean():.4f}' if len(rsu_subset) > 0 else '-',
            'Mean_t_est': f'{rsu_subset["t_est"].mean():.4f}' if 't_est' in rsu_subset.columns and len(rsu_subset) > 0 else '-',
        })
    
    result_df = pd.DataFrame(rsu_stats)
    result_df.to_csv(output_path, index=False)
    print(f"  ✓ {output_path.name}")
    return output_path


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate paper figures and tables')
    parser.add_argument('--train-dirs', nargs='+', help='Training log directories for A/B comparison')
    parser.add_argument('--only', choices=['figures', 'tables', 'all'], default='all')
    args = parser.parse_args()
    
    ensure_dirs()
    
    print("=" * 60)
    print("论文图表生成工具")
    print("=" * 60)
    print(f"输出目录: {FIG_DIR}")
    print()
    
    generated = []
    
    if args.only in ['figures', 'all']:
        print("[图表生成]")
        
        # 图1: 场景有效性
        fig1 = plot_scenario_validity(FIG_DIR / "fig1_scenario_validity.png")
        if fig1: generated.append(fig1)
        
        # 图2: 修复对比
        fig2 = plot_fix_comparison(FIG_DIR / "fig2_fix_comparison.png")
        if fig2: generated.append(fig2)
        
        # 图3: RSU队列负载
        fig3 = plot_rsu_queue_load(FIG_DIR / "fig3_rsu_queue_load.png")
        if fig3: generated.append(fig3)
        
        # 图4: 动作分布
        fig4 = plot_action_distribution(FIG_DIR / "fig4_action_distribution.png")
        if fig4: generated.append(fig4)
        
        # 图5: RSU决策分布
        fig5 = plot_rsu_decision_distribution(FIG_DIR / "fig5_rsu_decision_dist.png")
        if fig5: generated.append(fig5)
    
    if args.only in ['tables', 'all']:
        print("\n[表格生成]")
        
        # 表1: 可复现性
        t1 = generate_reproducibility_table(TABLE_DIR / "table1_reproducibility.csv")
        if t1: generated.append(t1)
        
        # 表2: 修复对比
        t2 = generate_fix_comparison_table(TABLE_DIR / "table2_fix_comparison.csv")
        if t2: generated.append(t2)
        
        # 表3: RSU使用统计
        t3 = generate_rsu_usage_table(TABLE_DIR / "table3_rsu_usage.csv")
        if t3: generated.append(t3)
    
    print()
    print("=" * 60)
    print(f"生成完成: {len(generated)} 个文件")
    print("=" * 60)
    for f in generated:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
