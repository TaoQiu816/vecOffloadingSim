#!/usr/bin/env python3
"""
深入分析 latency_cleanup_300ep_seed42 训练过程指标
重点关注收敛性、稳定性和学习动态
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 读取数据
data_dir = Path(__file__).parent
metrics_df = pd.read_csv(data_dir / 'logs' / 'metrics.csv')
episode_df = pd.read_csv(data_dir / 'episode_log.csv')

print("=" * 80)
print("训练过程深度分析报告")
print("=" * 80)
print(f"\n数据集: latency_cleanup_300ep_seed42")
print(f"总Episode数: {len(metrics_df)}")
print(f"训练步数范围: {metrics_df['steps'].min()} - {metrics_df['steps'].max()}")

# ============================================================================
# 1. 收敛性分析
# ============================================================================
print("\n" + "=" * 80)
print("1. 收敛性分析")
print("=" * 80)

# 1.1 总奖励收敛趋势
print("\n1.1 总奖励(r_total)收敛趋势:")
early = metrics_df.iloc[:100]
mid = metrics_df.iloc[100:200]
late = metrics_df.iloc[200:]

print(f"  早期(1-100)  均值: {early['r_total'].mean():.6f}, 标准差: {early['r_total'].std():.6f}")
print(f"  中期(101-200) 均值: {mid['r_total'].mean():.6f}, 标准差: {mid['r_total'].std():.6f}")
print(f"  后期(201-300) 均值: {late['r_total'].mean():.6f}, 标准差: {late['r_total'].std():.6f}")

# 计算趋势
reward_trend = np.polyfit(range(len(metrics_df)), metrics_df['r_total'].fillna(0), 1)
print(f"  线性趋势斜率: {reward_trend[0]:.8f} (正值表示上升趋势)")

# 检测收敛平台期
window = 50
rolling_mean = metrics_df['r_total'].rolling(window=window).mean()
rolling_std = metrics_df['r_total'].rolling(window=window).std()
convergence_threshold = 0.001  # 标准差阈值

stable_periods = rolling_std < convergence_threshold
if stable_periods.any():
    first_stable = stable_periods.idxmax()
    print(f"  首次进入稳定期(std<{convergence_threshold}): Episode {first_stable}")
else:
    print(f"  未检测到明显的收敛平台期(std<{convergence_threshold})")

# 1.2 成功率提升曲线
print("\n1.2 成功率提升曲线:")
for metric in ['task_success_rate', 'subtask_success_rate']:
    if metric in metrics_df.columns:
        early_val = early[metric].mean()
        mid_val = mid[metric].mean()
        late_val = late[metric].mean()
        
        print(f"\n  {metric}:")
        print(f"    早期: {early_val:.4f}")
        print(f"    中期: {mid_val:.4f}")
        print(f"    后期: {late_val:.4f}")
        print(f"    总提升: {late_val - early_val:.4f} ({(late_val - early_val)/early_val*100:.2f}%)")

# 1.3 Loss下降趋势
print("\n1.3 Loss下降趋势:")
for loss_metric in ['policy_loss', 'value_loss']:
    if loss_metric in metrics_df.columns:
        loss_data = metrics_df[loss_metric].dropna()
        if len(loss_data) > 0:
            early_loss = loss_data.iloc[:len(loss_data)//3].mean()
            late_loss = loss_data.iloc[-len(loss_data)//3:].mean()
            
            print(f"\n  {loss_metric}:")
            print(f"    早期均值: {early_loss:.4f}")
            print(f"    后期均值: {late_loss:.4f}")
            print(f"    下降幅度: {early_loss - late_loss:.4f} ({(early_loss - late_loss)/early_loss*100:.2f}%)")

# ============================================================================
# 2. 稳定性分析
# ============================================================================
print("\n" + "=" * 80)
print("2. 稳定性分析")
print("=" * 80)

# 2.1 后期指标波动性
print("\n2.1 后期(最后100个episode)关键指标波动性:")
late_100 = metrics_df.iloc[-100:]

key_metrics = ['r_total', 'task_success_rate', 'subtask_success_rate', 
               'mean_cft_est', 'energy_norm_mean']

for metric in key_metrics:
    if metric in late_100.columns:
        data = late_100[metric].dropna()
        if len(data) > 0:
            mean_val = data.mean()
            std_val = data.std()
            cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
            
            print(f"\n  {metric}:")
            print(f"    均值: {mean_val:.6f}")
            print(f"    标准差: {std_val:.6f}")
            print(f"    变异系数(CV): {cv:.4f}")
            
            # 判断稳定性
            if cv < 0.1:
                stability = "非常稳定"
            elif cv < 0.3:
                stability = "稳定"
            elif cv < 0.5:
                stability = "中等波动"
            else:
                stability = "波动较大"
            print(f"    稳定性评估: {stability}")

# 2.2 性能退化检测
print("\n2.2 性能退化检测:")
mid_50 = metrics_df.iloc[150:200]
late_50 = metrics_df.iloc[250:]

for metric in ['r_total', 'task_success_rate']:
    if metric in metrics_df.columns:
        mid_mean = mid_50[metric].mean()
        late_mean = late_50[metric].mean()
        degradation = mid_mean - late_mean
        
        print(f"\n  {metric}:")
        print(f"    中期(150-200): {mid_mean:.6f}")
        print(f"    后期(250-300): {late_mean:.6f}")
        print(f"    变化: {degradation:.6f}")
        
        if degradation > 0.01:
            print(f"    ⚠️ 检测到性能退化")
        elif degradation < -0.01:
            print(f"    ✓ 性能持续提升")
        else:
            print(f"    ✓ 性能保持稳定")

# ============================================================================
# 3. 学习动态分析
# ============================================================================
print("\n" + "=" * 80)
print("3. 学习动态分析")
print("=" * 80)

# 3.1 决策分布演化
print("\n3.1 决策分布演化(Local/RSU/V2V比例):")
decision_metrics = ['decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v']

for phase_name, phase_data in [("早期", early), ("中期", mid), ("后期", late)]:
    print(f"\n  {phase_name}:")
    for metric in decision_metrics:
        if metric in phase_data.columns:
            mean_val = phase_data[metric].mean()
            print(f"    {metric}: {mean_val:.4f}")

# 3.2 熵变化趋势
print("\n3.2 策略熵(entropy)变化趋势:")
if 'entropy' in metrics_df.columns:
    entropy_data = metrics_df['entropy'].dropna()
    if len(entropy_data) > 0:
        early_entropy = entropy_data.iloc[:len(entropy_data)//3].mean()
        late_entropy = entropy_data.iloc[-len(entropy_data)//3:].mean()
        
        print(f"  早期均值: {early_entropy:.4f}")
        print(f"  后期均值: {late_entropy:.4f}")
        print(f"  变化: {late_entropy - early_entropy:.4f}")
        
        if late_entropy < early_entropy:
            print(f"  解释: 熵下降表明策略逐渐收敛，决策更加确定")
        else:
            print(f"  解释: 熵上升表明策略保持探索性")

# 3.3 KL散度和裁剪率
print("\n3.3 KL散度(approx_kl)和裁剪率(clip_frac):")
for metric in ['approx_kl', 'clip_frac']:
    if metric in metrics_df.columns:
        data = metrics_df[metric].dropna()
        if len(data) > 0:
            print(f"\n  {metric}:")
            print(f"    均值: {data.mean():.6f}")
            print(f"    中位数: {data.median():.6f}")
            print(f"    最大值: {data.max():.6f}")
            
            if metric == 'approx_kl':
                if data.mean() < 0.01:
                    print(f"    评估: KL散度较小，策略更新保守")
                elif data.mean() < 0.05:
                    print(f"    评估: KL散度适中，策略更新稳健")
                else:
                    print(f"    评估: KL散度较大，策略更新激进")

# 3.4 梯度范数
print("\n3.4 梯度范数(grad_norm):")
if 'grad_norm' in metrics_df.columns:
    grad_data = metrics_df['grad_norm'].dropna()
    if len(grad_data) > 0:
        print(f"  均值: {grad_data.mean():.4f}")
        print(f"  标准差: {grad_data.std():.4f}")
        print(f"  最大值: {grad_data.max():.4f}")
        print(f"  最小值: {grad_data.min():.4f}")
        
        # 检测梯度爆炸
        if grad_data.max() > 100:
            print(f"  ⚠️ 检测到梯度爆炸风险(max > 100)")
        else:
            print(f"  ✓ 梯度范数正常")

# ============================================================================
# 4. 物理指标分析
# ============================================================================
print("\n" + "=" * 80)
print("4. 物理指标分析")
print("=" * 80)

# 4.1 延迟相关指标
print("\n4.1 延迟相关指标:")
latency_metrics = ['mean_cft_est', 'mean_cft_completed']

for metric in latency_metrics:
    if metric in metrics_df.columns:
        data = metrics_df[metric].dropna()
        if len(data) > 0:
            print(f"\n  {metric}:")
            print(f"    均值: {data.mean():.4f}")
            print(f"    中位数: {data.median():.4f}")
            print(f"    标准差: {data.std():.4f}")
            print(f"    范围: [{data.min():.4f}, {data.max():.4f}]")

# 4.2 能量/功率指标
print("\n4.2 能量/功率指标:")
energy_metrics = ['energy_norm_mean', 'avg_power', 'power_ratio_mean']

for metric in energy_metrics:
    if metric in metrics_df.columns:
        data = metrics_df[metric].dropna()
        if len(data) > 0:
            print(f"\n  {metric}:")
            print(f"    均值: {data.mean():.4f}")
            print(f"    标准差: {data.std():.4f}")
            
            # 分阶段分析
            early_val = early[metric].mean()
            late_val = late[metric].mean()
            print(f"    早期→后期: {early_val:.4f} → {late_val:.4f} (变化: {late_val-early_val:.4f})")

# 4.3 队列长度
print("\n4.3 队列长度(avg_queue_len):")
if 'avg_queue_len' in metrics_df.columns:
    queue_data = metrics_df['avg_queue_len'].dropna()
    if len(queue_data) > 0:
        print(f"  均值: {queue_data.mean():.4f}")
        print(f"  最大值: {queue_data.max():.4f}")
        print(f"  95分位数: {queue_data.quantile(0.95):.4f}")
        
        if queue_data.mean() < 0.1:
            print(f"  评估: 队列负载很低，系统资源充足")
        elif queue_data.mean() < 0.5:
            print(f"  评估: 队列负载适中")
        else:
            print(f"  评估: 队列负载较高，可能存在资源瓶颈")

# 4.4 干扰指标
print("\n4.4 干扰指标(I_caused_mean):")
if 'I_caused_mean' in metrics_df.columns:
    interf_data = metrics_df['I_caused_mean'].dropna()
    if len(interf_data) > 0:
        print(f"  均值: {interf_data.mean():.2e}")
        print(f"  标准差: {interf_data.std():.2e}")
        print(f"  最大值: {interf_data.max():.2e}")

# ============================================================================
# 5. 多智能体协作分析
# ============================================================================
print("\n" + "=" * 80)
print("5. 多智能体协作分析")
print("=" * 80)

ma_metrics = ['ma_fairness', 'ma_reward_gap', 'ma_collaboration']

for metric in ma_metrics:
    if metric in metrics_df.columns:
        data = metrics_df[metric].dropna()
        if len(data) > 0:
            print(f"\n{metric}:")
            print(f"  均值: {data.mean():.4f}")
            print(f"  标准差: {data.std():.4f}")
            print(f"  范围: [{data.min():.4f}, {data.max():.4f}]")
            
            # 分阶段分析
            early_val = early[metric].mean() if metric in early.columns else np.nan
            late_val = late[metric].mean() if metric in late.columns else np.nan
            
            if not np.isnan(early_val) and not np.isnan(late_val):
                print(f"  早期→后期: {early_val:.4f} → {late_val:.4f}")
                
                if metric == 'ma_fairness':
                    if late_val > 0.8:
                        print(f"  评估: 公平性很好")
                    elif late_val > 0.5:
                        print(f"  评估: 公平性中等")
                    else:
                        print(f"  评估: 公平性较差")

# ============================================================================
# 6. 分阶段统计汇总
# ============================================================================
print("\n" + "=" * 80)
print("6. 分阶段统计汇总")
print("=" * 80)

summary_metrics = [
    'r_total', 'task_success_rate', 'subtask_success_rate',
    'mean_cft_est', 'energy_norm_mean', 'avg_power',
    'decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v',
    'entropy', 'policy_loss', 'value_loss'
]

print("\n关键指标分阶段对比:")
print(f"{'指标':<30} {'早期(1-100)':<20} {'中期(101-200)':<20} {'后期(201-300)':<20}")
print("-" * 90)

for metric in summary_metrics:
    if metric in metrics_df.columns:
        early_val = early[metric].mean()
        mid_val = mid[metric].mean()
        late_val = late[metric].mean()
        
        print(f"{metric:<30} {early_val:<20.6f} {mid_val:<20.6f} {late_val:<20.6f}")

# ============================================================================
# 7. 训练健康度评估
# ============================================================================
print("\n" + "=" * 80)
print("7. 训练健康度评估")
print("=" * 80)

health_score = 0
max_score = 0

# 评估1: 奖励趋势
max_score += 20
if reward_trend[0] > 0:
    health_score += 20
    print("\n✓ 奖励趋势: 正向增长 (+20分)")
elif reward_trend[0] > -0.0001:
    health_score += 10
    print("\n△ 奖励趋势: 基本稳定 (+10分)")
else:
    print("\n✗ 奖励趋势: 下降趋势 (+0分)")

# 评估2: 成功率提升
max_score += 20
if 'task_success_rate' in metrics_df.columns:
    success_improvement = late['task_success_rate'].mean() - early['task_success_rate'].mean()
    if success_improvement > 0.1:
        health_score += 20
        print(f"✓ 成功率提升: 显著提升{success_improvement:.2%} (+20分)")
    elif success_improvement > 0:
        health_score += 10
        print(f"△ 成功率提升: 轻微提升{success_improvement:.2%} (+10分)")
    else:
        print(f"✗ 成功率提升: 无提升或下降 (+0分)")

# 评估3: 后期稳定性
max_score += 20
late_reward_cv = late['r_total'].std() / abs(late['r_total'].mean())
if late_reward_cv < 0.3:
    health_score += 20
    print(f"✓ 后期稳定性: 非常稳定(CV={late_reward_cv:.3f}) (+20分)")
elif late_reward_cv < 0.5:
    health_score += 10
    print(f"△ 后期稳定性: 中等稳定(CV={late_reward_cv:.3f}) (+10分)")
else:
    print(f"✗ 后期稳定性: 波动较大(CV={late_reward_cv:.3f}) (+0分)")

# 评估4: Loss下降
max_score += 20
if 'policy_loss' in metrics_df.columns:
    loss_data = metrics_df['policy_loss'].dropna()
    if len(loss_data) > 10:
        early_loss = loss_data.iloc[:len(loss_data)//3].mean()
        late_loss = loss_data.iloc[-len(loss_data)//3:].mean()
        loss_reduction = (early_loss - late_loss) / early_loss
        
        if loss_reduction > 0.2:
            health_score += 20
            print(f"✓ Loss下降: 显著下降{loss_reduction:.1%} (+20分)")
        elif loss_reduction > 0:
            health_score += 10
            print(f"△ Loss下降: 轻微下降{loss_reduction:.1%} (+10分)")
        else:
            print(f"✗ Loss下降: 未下降 (+0分)")

# 评估5: 梯度健康
max_score += 20
if 'grad_norm' in metrics_df.columns:
    grad_data = metrics_df['grad_norm'].dropna()
    if len(grad_data) > 0:
        if grad_data.max() < 100 and grad_data.mean() > 0.1:
            health_score += 20
            print(f"✓ 梯度健康: 正常范围 (+20分)")
        elif grad_data.max() < 200:
            health_score += 10
            print(f"△ 梯度健康: 基本正常 (+10分)")
        else:
            print(f"✗ 梯度健康: 存在异常 (+0分)")

print(f"\n{'='*80}")
print(f"训练健康度总分: {health_score}/{max_score} ({health_score/max_score*100:.1f}%)")

if health_score / max_score > 0.8:
    print("评级: 优秀 ⭐⭐⭐⭐⭐")
elif health_score / max_score > 0.6:
    print("评级: 良好 ⭐⭐⭐⭐")
elif health_score / max_score > 0.4:
    print("评级: 中等 ⭐⭐⭐")
else:
    print("评级: 需要改进 ⭐⭐")

print("=" * 80)
print("\n分析完成！")
