#!/usr/bin/env python
"""
生成训练数据的对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def create_comparison_charts():
    run_dir = Path(__file__).parent
    
    # 加载数据
    metrics_df = pd.read_csv(run_dir / 'logs' / 'metrics.csv')
    episode_log_df = pd.read_csv(run_dir / 'episode_log.csv')
    
    # 创建输出目录
    output_dir = run_dir / 'analysis_charts'
    output_dir.mkdir(exist_ok=True)
    
    # 1. 训练收敛曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('训练收敛性分析', fontsize=16, fontweight='bold')
    
    # 奖励曲线
    window = 50
    axes[0, 0].plot(metrics_df['episode'], metrics_df['r_total'], alpha=0.3, label='原始')
    axes[0, 0].plot(metrics_df['episode'], metrics_df['r_total'].rolling(window).mean(), 
                    linewidth=2, label=f'{window}-ep滑动平均')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('总奖励')
    axes[0, 0].set_title('总奖励收敛曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 任务成功率
    axes[0, 1].plot(metrics_df['episode'], metrics_df['task_success_rate'], alpha=0.3, label='原始')
    axes[0, 1].plot(metrics_df['episode'], metrics_df['task_success_rate'].rolling(window).mean(),
                    linewidth=2, label=f'{window}-ep滑动平均')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('任务成功率')
    axes[0, 1].set_title('任务成功率收敛曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Oracle匹配率
    axes[1, 0].plot(episode_log_df['episode'], episode_log_df['oracle_match_rate'], alpha=0.3, label='原始')
    axes[1, 0].plot(episode_log_df['episode'], episode_log_df['oracle_match_rate'].rolling(window).mean(),
                    linewidth=2, label=f'{window}-ep滑动平均')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Oracle匹配率')
    axes[1, 0].set_title('Oracle匹配率演变')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # 动作遗憾
    axes[1, 1].plot(episode_log_df['episode'], episode_log_df['action_regret_mean'], alpha=0.3, label='原始')
    axes[1, 1].plot(episode_log_df['episode'], episode_log_df['action_regret_mean'].rolling(window).mean(),
                    linewidth=2, label=f'{window}-ep滑动平均')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('动作遗憾')
    axes[1, 1].set_title('动作遗憾演变')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: convergence_analysis.png")
    plt.close()
    
    # 2. 决策分布演变
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('决策分布演变分析', fontsize=16, fontweight='bold')
    
    # 决策分布堆叠图
    axes[0, 0].fill_between(episode_log_df['episode'], 0, episode_log_df['decision_frac_local'], 
                            alpha=0.7, label='Local')
    axes[0, 0].fill_between(episode_log_df['episode'], episode_log_df['decision_frac_local'],
                            episode_log_df['decision_frac_local'] + episode_log_df['decision_frac_rsu'],
                            alpha=0.7, label='RSU')
    axes[0, 0].fill_between(episode_log_df['episode'], 
                            episode_log_df['decision_frac_local'] + episode_log_df['decision_frac_rsu'],
                            1.0, alpha=0.7, label='V2V')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('决策占比')
    axes[0, 0].set_title('决策分布堆叠图')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # 各决策类型趋势
    axes[0, 1].plot(episode_log_df['episode'], episode_log_df['decision_frac_local'].rolling(window).mean(),
                    linewidth=2, label='Local')
    axes[0, 1].plot(episode_log_df['episode'], episode_log_df['decision_frac_rsu'].rolling(window).mean(),
                    linewidth=2, label='RSU')
    axes[0, 1].plot(episode_log_df['episode'], episode_log_df['decision_frac_v2v'].rolling(window).mean(),
                    linewidth=2, label='V2V')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('决策占比')
    axes[0, 1].set_title(f'决策分布趋势 ({window}-ep滑动平均)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 分阶段决策分布
    phases = [(0, 250, '初期'), (250, 500, '中前期'), (500, 750, '中后期'), (750, 1000, '后期')]
    phase_data = []
    for start, end, name in phases:
        phase_log = episode_log_df.iloc[start:end]
        phase_data.append({
            '阶段': name,
            'Local': phase_log['decision_frac_local'].mean(),
            'RSU': phase_log['decision_frac_rsu'].mean(),
            'V2V': phase_log['decision_frac_v2v'].mean()
        })
    
    phase_df = pd.DataFrame(phase_data)
    x = np.arange(len(phase_df))
    width = 0.25
    
    axes[1, 0].bar(x - width, phase_df['Local'], width, label='Local', alpha=0.8)
    axes[1, 0].bar(x, phase_df['RSU'], width, label='RSU', alpha=0.8)
    axes[1, 0].bar(x + width, phase_df['V2V'], width, label='V2V', alpha=0.8)
    axes[1, 0].set_xlabel('训练阶段')
    axes[1, 0].set_ylabel('平均决策占比')
    axes[1, 0].set_title('分阶段决策分布对比')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(phase_df['阶段'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Oracle vs 实际决策对比
    oracle_data = {
        'Oracle建议RSU\n但选择V2V': episode_log_df['oracle_rsu_chosen_v2v_rate'].mean(),
        'Oracle建议V2V\n但选择RSU': episode_log_df['oracle_v2v_chosen_rsu_rate'].mean(),
        'Oracle建议V2V\n且选择V2V': episode_log_df['oracle_v2v_chosen_v2v_rate'].mean()
    }
    
    axes[1, 1].bar(oracle_data.keys(), oracle_data.values(), alpha=0.8)
    axes[1, 1].set_ylabel('比例')
    axes[1, 1].set_title('Oracle建议 vs 实际选择')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: decision_distribution_analysis.png")
    plt.close()
    
    # 3. 训练稳定性分析
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('训练稳定性分析', fontsize=16, fontweight='bold')
    
    # 梯度范数
    axes[0, 0].plot(metrics_df['episode'], metrics_df['grad_norm'], alpha=0.5)
    axes[0, 0].axhline(y=100, color='r', linestyle='--', label='阈值=100')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('梯度范数')
    axes[0, 0].set_title('梯度范数演变')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # KL散度
    axes[0, 1].plot(metrics_df['episode'], metrics_df['approx_kl'], alpha=0.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('KL散度')
    axes[0, 1].set_title('KL散度演变')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 策略熵
    axes[1, 0].plot(metrics_df['episode'], metrics_df['policy_entropy'], alpha=0.5)
    axes[1, 0].plot(metrics_df['episode'], metrics_df['policy_entropy'].rolling(window).mean(),
                    linewidth=2, label=f'{window}-ep滑动平均')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('策略熵')
    axes[1, 0].set_title('策略熵演变')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Clip Fraction
    axes[1, 1].plot(metrics_df['episode'], metrics_df['clip_frac'], alpha=0.5)
    axes[1, 1].plot(metrics_df['episode'], metrics_df['clip_frac'].rolling(window).mean(),
                    linewidth=2, label=f'{window}-ep滑动平均')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Clip Fraction')
    axes[1, 1].set_title('Clip Fraction演变')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_stability_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: training_stability_analysis.png")
    plt.close()
    
    # 4. 多智能体协作分析
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('多智能体协作分析', fontsize=16, fontweight='bold')
    
    # 公平性
    axes[0].plot(episode_log_df['episode'], episode_log_df['ma_fairness'], alpha=0.3)
    axes[0].plot(episode_log_df['episode'], episode_log_df['ma_fairness'].rolling(window).mean(),
                linewidth=2, label=f'{window}-ep滑动平均')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('公平性')
    axes[0].set_title('多智能体公平性')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # 奖励差距
    axes[1].plot(episode_log_df['episode'], episode_log_df['ma_reward_gap'], alpha=0.3)
    axes[1].plot(episode_log_df['episode'], episode_log_df['ma_reward_gap'].rolling(window).mean(),
                linewidth=2, label=f'{window}-ep滑动平均')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('奖励差距')
    axes[1].set_title('智能体间奖励差距')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 协作得分
    axes[2].plot(episode_log_df['episode'], episode_log_df['ma_collaboration'], alpha=0.3)
    axes[2].plot(episode_log_df['episode'], episode_log_df['ma_collaboration'].rolling(window).mean(),
                linewidth=2, label=f'{window}-ep滑动平均')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('协作得分')
    axes[2].set_title('多智能体协作得分')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_agent_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: multi_agent_analysis.png")
    plt.close()
    
    # 5. 综合对比表格
    summary_data = {
        '指标': [],
        '初期\n(0-250)': [],
        '中前期\n(250-500)': [],
        '中后期\n(500-750)': [],
        '后期\n(750-1000)': [],
        '整体': []
    }
    
    metrics_to_compare = [
        ('r_total', '平均奖励', metrics_df),
        ('task_success_rate', '任务成功率', metrics_df),
        ('oracle_match_rate', 'Oracle匹配率', episode_log_df),
        ('action_regret_mean', '动作遗憾', episode_log_df),
        ('decision_frac_rsu', 'RSU占比', episode_log_df),
    ]
    
    for col, name, df in metrics_to_compare:
        summary_data['指标'].append(name)
        for start, end, _ in phases:
            phase_data = df.iloc[start:end][col].mean()
            summary_data[f'{_}\n({start}-{end})'].append(f'{phase_data:.4f}')
        summary_data['整体'].append(f'{df[col].mean():.4f}')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=[summary_data[k] for k in summary_data.keys()][1:],
                    rowLabels=summary_data['指标'],
                    colLabels=list(summary_data.keys())[1:],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(summary_data.keys()) - 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('训练指标分阶段对比表', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: comparison_table.png")
    plt.close()
    
    print(f"\n✓ 所有图表已保存到: {output_dir}")
    print(f"  共生成 5 个图表文件")

if __name__ == '__main__':
    create_comparison_charts()
