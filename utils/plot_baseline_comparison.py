"""
基准策略对比可视化

生成对比图表：
1. 平均奖励对比（柱状图）
2. 任务成功率对比（分组柱状图）
3. 决策分布对比（堆叠柱状图）
4. 平均完成时间对比（柱状图）
5. 雷达图（多维度综合对比）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(json_path):
    """加载评估结果"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def plot_reward_comparison(results, save_dir):
    """绘制平均奖励对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    policies = [r['policy_name'] for r in results]
    rewards = [r['avg_reward'] for r in results]
    stds = [r['std_reward'] for r in results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(policies, rewards, yerr=stds, capsize=5, 
                   color=colors[:len(policies)], alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Average Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('Policy Performance Comparison - Average Reward', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上标注数值
    for bar, reward, std in zip(bars, rewards, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.2f}\n±{std:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reward_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_dir}/reward_comparison.png")


def plot_success_rate_comparison(results, save_dir):
    """绘制任务成功率对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    policies = [r['policy_name'] for r in results]
    vehicle_success = [r['avg_vehicle_success_rate'] for r in results]
    subtask_success = [r['avg_subtask_success_rate'] for r in results]
    
    x = np.arange(len(policies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, vehicle_success, width, label='Vehicle Success Rate',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, subtask_success, width, label='Subtask Success Rate',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Task Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/success_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_dir}/success_rate_comparison.png")


def plot_decision_distribution(results, save_dir):
    """绘制决策分布对比图（堆叠柱状图）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    policies = [r['policy_name'] for r in results]
    local_pcts = [r['decision_distribution']['local'] for r in results]
    rsu_pcts = [r['decision_distribution']['rsu'] for r in results]
    v2v_pcts = [r['decision_distribution']['v2v'] for r in results]
    
    x = np.arange(len(policies))
    width = 0.6
    
    bars1 = ax.bar(x, local_pcts, width, label='Local Execution',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, rsu_pcts, width, bottom=local_pcts, label='RSU Offloading',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x, v2v_pcts, width, 
                   bottom=np.array(local_pcts) + np.array(rsu_pcts),
                   label='V2V Offloading', color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Offloading Decision Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim([0, 100])
    
    # 标注百分比（只标注大于5%的）
    for i, policy in enumerate(policies):
        y_offset = 0
        for pct, color in zip([local_pcts[i], rsu_pcts[i], v2v_pcts[i]], 
                              ['white', 'white', 'black']):
            if pct > 5:
                ax.text(i, y_offset + pct/2, f'{pct:.1f}%',
                       ha='center', va='center', fontsize=10, 
                       fontweight='bold', color=color)
            y_offset += pct
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/decision_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_dir}/decision_distribution.png")


def plot_completion_time_comparison(results, save_dir):
    """绘制平均完成时间对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    policies = [r['policy_name'] for r in results]
    completion_times = [r['avg_completion_time'] for r in results]
    stds = [r['std_completion_time'] for r in results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(policies, completion_times, yerr=stds, capsize=5,
                   color=colors[:len(policies)], alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Average Completion Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Task Completion Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注数值
    for bar, ct, std in zip(bars, completion_times, stds):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ct:.2f}s\n±{std:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/completion_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_dir}/completion_time_comparison.png")


def plot_radar_chart(results, save_dir):
    """绘制雷达图（多维度综合对比）"""
    # 归一化指标到[0, 1]范围
    metrics = {
        'Reward': [r['avg_reward'] for r in results],
        'Vehicle\nSuccess': [r['avg_vehicle_success_rate'] / 100 for r in results],
        'Subtask\nSuccess': [r['avg_subtask_success_rate'] / 100 for r in results],
        'Completion\nTime': [1 / (r['avg_completion_time'] + 1e-6) if r['avg_completion_time'] > 0 else 0 
                            for r in results],  # 时间越短越好，取倒数
        'Queue\nLength': [1 / (r['avg_queue_length'] + 1) for r in results]  # 队列越短越好
    }
    
    # 归一化到[0, 1]
    for key in metrics:
        values = metrics[key]
        min_val, max_val = min(values), max(values)
        if max_val - min_val > 1e-6:
            metrics[key] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            metrics[key] = [0.5] * len(values)
    
    categories = list(metrics.keys())
    num_vars = len(categories)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, result in enumerate(results):
        values = [metrics[cat][i] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['policy_name'],
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Multi-Dimensional Performance Comparison\n(Normalized)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_dir}/radar_comparison.png")


def main():
    """主函数"""
    # 加载结果
    json_path = "eval_results/baseline_comparison.json"
    
    if not os.path.exists(json_path):
        print(f"错误: 未找到评估结果文件 {json_path}")
        print("请先运行 eval_baselines.py 生成评估结果")
        return
    
    results = load_results(json_path)
    
    # 创建输出目录
    save_dir = "eval_results/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("生成基准策略对比图表")
    print("="*60)
    
    # 生成各类图表
    plot_reward_comparison(results, save_dir)
    plot_success_rate_comparison(results, save_dir)
    plot_decision_distribution(results, save_dir)
    plot_completion_time_comparison(results, save_dir)
    plot_radar_chart(results, save_dir)
    
    print("\n" + "="*60)
    print(f"所有图表已保存到: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

