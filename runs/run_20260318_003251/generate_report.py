#!/usr/bin/env python
"""
生成1000 Episode训练运行的全面分析报告
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def generate_report():
    run_dir = Path(__file__).parent
    
    # 加载数据
    metrics_df = pd.read_csv(run_dir / 'logs' / 'metrics.csv')
    episode_log_df = pd.read_csv(run_dir / 'episode_log.csv')
    
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    with open(run_dir / 'run_meta.json', 'r') as f:
        meta = json.load(f)
    
    # 生成报告
    report = []
    
    report.append("# 1000 Episode训练运行全面分析报告")
    report.append("")
    report.append(f"**运行目录**: `{run_dir}`")
    report.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # 1. 执行摘要
    report.append("## 1. 执行摘要")
    report.append("")
    report.append(f"本次训练共完成 **1000 episodes**，训练时长约 **4小时9分钟**。")
    report.append("")
    
    # 关键指标
    last_100 = metrics_df.tail(100)
    report.append("### 关键指标 (最后100 episodes)")
    report.append("")
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| 平均总奖励 | {last_100['r_total'].mean():.4f} |")
    report.append(f"| 任务成功率 | {last_100['task_success_rate'].mean():.2%} |")
    report.append(f"| 子任务成功率 | {last_100['subtask_success_rate'].mean():.2%} |")
    report.append(f"| Oracle匹配率 | {episode_log_df.tail(100)['oracle_match_rate'].mean():.2%} |")
    report.append(f"| 动作遗憾 | {episode_log_df.tail(100)['action_regret_mean'].mean():.4f} |")
    report.append(f"| 归一化能耗 | {last_100['energy_norm_mean'].mean():.4f} |")
    report.append("")
    
    # 2. 训练配置
    report.append("## 2. 训练配置")
    report.append("")
    report.append("### 2.1 环境配置")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append(f"| 车辆数 | {config.get('NUM_VEHICLES', 'N/A')} |")
    report.append(f"| RSU数 | {config.get('NUM_RSU', 'N/A')} |")
    report.append(f"| 随机种子 | {config.get('SEED', 'N/A')} |")
    report.append(f"| 最大步数/episode | {config.get('MAX_STEPS', 'N/A')} |")
    report.append(f"| Trust启用 | {config.get('TRUST_ENABLED', 'N/A')} |")
    report.append(f"| 恶意车辆比例 | {config.get('MALICIOUS_RATIO', 'N/A')} |")
    report.append(f"| V2V动态K | {config.get('V2V_DYNAMIC_K', 'N/A')} |")
    report.append(f"| V2V候选门控 | {config.get('V2V_CAND_GATE_ENABLED', 'N/A')} |")
    report.append("")
    
    report.append("### 2.2 RL训练参数")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append(f"| Batch Size | {config.get('batch_size', 'N/A')} |")
    report.append(f"| Gamma | {config.get('gamma', 'N/A')} |")
    report.append(f"| GAE Lambda | {config.get('gae_lambda', 'N/A')} |")
    report.append(f"| Clip参数 | {config.get('clip_param', 'N/A')} |")
    report.append(f"| Entropy系数 | {config.get('entropy_coef', 'N/A')} |")
    report.append(f"| K Epochs | {config.get('k_epochs', 'N/A')} |")
    report.append("")
    
    # 3. 训练指标详细分析
    report.append("## 3. 训练指标详细分析")
    report.append("")
    
    report.append("### 3.1 整体统计")
    report.append("")
    key_metrics = {
        'r_total': '总奖励',
        'task_success_rate': '任务成功率',
        'subtask_success_rate': '子任务成功率',
        'energy_norm_mean': '归一化能耗',
    }
    
    report.append("| 指标 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |")
    report.append("|------|------|--------|--------|--------|--------|")
    for col, name in key_metrics.items():
        if col in metrics_df.columns:
            values = metrics_df[col].dropna()
            report.append(f"| {name} | {values.mean():.4f} | {values.median():.4f} | {values.std():.4f} | {values.min():.4f} | {values.max():.4f} |")
    report.append("")
    
    # 分阶段分析
    report.append("### 3.2 分阶段分析")
    report.append("")
    report.append("训练过程分为4个阶段，每阶段250 episodes：")
    report.append("")
    
    phases = [(0, 250, "初期"), (250, 500, "中前期"), (500, 750, "中后期"), (750, 1000, "后期")]
    
    report.append("| 阶段 | 平均奖励 | 任务成功率 | Oracle匹配率 | 动作遗憾 |")
    report.append("|------|----------|------------|--------------|----------|")
    for start, end, name in phases:
        phase_metrics = metrics_df.iloc[start:end]
        phase_log = episode_log_df.iloc[start:end]
        report.append(f"| {name} (ep {start}-{end}) | {phase_metrics['r_total'].mean():.4f} | {phase_metrics['task_success_rate'].mean():.2%} | {phase_log['oracle_match_rate'].mean():.2%} | {phase_log['action_regret_mean'].mean():.4f} |")
    report.append("")
    
    # 4. 决策分布分析
    report.append("## 4. 决策分布分析")
    report.append("")
    report.append("### 4.1 整体决策分布")
    report.append("")
    report.append("| 决策类型 | 整体占比 | 最后100ep占比 |")
    report.append("|----------|----------|---------------|")
    report.append(f"| Local | {metrics_df['decision_local_frac'].mean():.2%} | {last_100['decision_local_frac'].mean():.2%} |")
    report.append(f"| RSU | {metrics_df['decision_rsu_frac'].mean():.2%} | {last_100['decision_rsu_frac'].mean():.2%} |")
    report.append(f"| V2V | {metrics_df['decision_v2v_frac'].mean():.2%} | {last_100['decision_v2v_frac'].mean():.2%} |")
    report.append("")
    
    report.append("### 4.2 决策���变")
    report.append("")
    report.append("| 阶段 | Local | RSU | V2V |")
    report.append("|------|-------|-----|-----|")
    for start, end, name in phases:
        phase_log = episode_log_df.iloc[start:end]
        report.append(f"| {name} | {phase_log['decision_frac_local'].mean():.2%} | {phase_log['decision_frac_rsu'].mean():.2%} | {phase_log['decision_frac_v2v'].mean():.2%} |")
    report.append("")
    
    # 5. Oracle对比分析
    report.append("## 5. Oracle对比分析")
    report.append("")
    report.append(f"**Oracle匹配率**: {episode_log_df['oracle_match_rate'].mean():.2%}")
    report.append(f"**平均动作遗憾**: {episode_log_df['action_regret_mean'].mean():.4f}")
    report.append("")
    report.append("### Oracle建议 vs 实际选择")
    report.append("")
    report.append("| 情况 | 比例 |")
    report.append("|------|------|")
    report.append(f"| Oracle建议RSU但选择V2V | {episode_log_df['oracle_rsu_chosen_v2v_rate'].mean():.2%} |")
    report.append(f"| Oracle建议V2V但选择RSU | {episode_log_df['oracle_v2v_chosen_rsu_rate'].mean():.2%} |")
    report.append(f"| Oracle建议V2V且选择V2V | {episode_log_df['oracle_v2v_chosen_v2v_rate'].mean():.2%} |")
    report.append("")
    
    # 6. 训练稳定性分析
    report.append("## 6. 训练稳定性分析")
    report.append("")
    report.append("### 6.1 梯度范数")
    report.append("")
    report.append("| 阶段 | 平均梯度范数 | 最大梯度范数 | >100的比例 |")
    report.append("|------|--------------|--------------|------------|")
    for start, end, name in phases:
        phase_metrics = metrics_df.iloc[start:end]
        grad_norms = phase_metrics['grad_norm'].dropna()
        report.append(f"| {name} | {grad_norms.mean():.2f} | {grad_norms.max():.2f} | {(grad_norms > 100).mean():.2%} |")
    report.append("")
    
    report.append("### 6.2 其他稳定性指标")
    report.append("")
    kl_divs = metrics_df['approx_kl'].dropna()
    clip_fracs = metrics_df['clip_frac'].dropna()
    entropies = metrics_df['policy_entropy'].dropna()
    
    report.append("| 指标 | 均值 | 最大值 |")
    report.append("|------|------|--------|")
    report.append(f"| KL散度 | {kl_divs.mean():.6f} | {kl_divs.max():.6f} |")
    report.append(f"| Clip Fraction | {clip_fracs.mean():.4f} | {clip_fracs.max():.4f} |")
    report.append(f"| 策略熵 | {entropies.mean():.4f} | {entropies.max():.4f} |")
    report.append("")
    
    # 7. 多智能体协作分析
    report.append("## 7. 多智能体协作分析")
    report.append("")
    report.append("| 指标 | 均值 | 标准差 |")
    report.append("|------|------|--------|")
    report.append(f"| 公平性 (Fairness) | {episode_log_df['ma_fairness'].mean():.4f} | {episode_log_df['ma_fairness'].std():.4f} |")
    report.append(f"| 奖励差距 (Reward Gap) | {episode_log_df['ma_reward_gap'].mean():.4f} | {episode_log_df['ma_reward_gap'].std():.4f} |")
    report.append(f"| 协作得分 (Collaboration) | {episode_log_df['ma_collaboration'].mean():.2f} | {episode_log_df['ma_collaboration'].std():.2f} |")
    report.append("")
    
    # 8. 潜在问题识别
    report.append("## 8. 潜在问题识别与分析")
    report.append("")
    
    issues = []
    
    # 问题1: 奖励波动
    last_100_rewards = metrics_df.tail(100)['r_total'].dropna()
    cv = last_100_rewards.std() / abs(last_100_rewards.mean()) if last_100_rewards.mean() != 0 else float('inf')
    if cv > 0.5:
        issues.append({
            'level': '⚠️ 中等',
            'title': '奖励波动较大',
            'desc': f'最后100 episodes的奖励变异系数(CV)为 {cv:.2f}，表明训练可能未完全收敛。',
            'evidence': f'标准差: {last_100_rewards.std():.4f}, 均值: {last_100_rewards.mean():.4f}',
            'impact': '可能导致策略性能不稳定，在不同场景下表现差异较大。'
        })
    
    # 问题2: Oracle匹配率低
    last_100_oracle = episode_log_df.tail(100)['oracle_match_rate'].mean()
    if last_100_oracle < 0.5:
        issues.append({
            'level': '⚠️ 严重',
            'title': 'Oracle匹配率较低',
            'desc': f'最后100 episodes的Oracle匹配率仅为 {last_100_oracle:.2%}，远低于理想水平(>50%)。',
            'evidence': f'整体Oracle匹配率: {episode_log_df["oracle_match_rate"].mean():.2%}, 动作遗憾: {episode_log_df["action_regret_mean"].mean():.4f}',
            'impact': '策略决策质量不够优，存在较大的改进空间。'
        })
    
    # 问题3: 梯度爆炸
    max_grad = metrics_df['grad_norm'].max()
    if max_grad > 100:
        issues.append({
            'level': '⚠️ 严重',
            'title': '梯度爆炸',
            'desc': f'检测到梯度爆炸现象，最大梯度范数达到 {max_grad:.2f}。',
            'evidence': f'平均梯度范数: {metrics_df["grad_norm"].mean():.2f}, >100的比例: {(metrics_df["grad_norm"] > 100).mean():.2%}',
            'impact': '可能导致训练不稳定，影响收敛速度和最终性能。'
        })
    
    # 问题4: RSU过度依赖
    last_100_rsu = last_100['decision_rsu_frac'].mean()
    if last_100_rsu > 0.7:
        issues.append({
            'level': '⚠️ 中等',
            'title': 'RSU决策占比过高',
            'desc': f'最后100 episodes中RSU决策占比达到 {last_100_rsu:.2%}，可能过度依赖RSU。',
            'evidence': f'Local: {last_100["decision_local_frac"].mean():.2%}, RSU: {last_100_rsu:.2%}, V2V: {last_100["decision_v2v_frac"].mean():.2%}',
            'impact': '可能导致RSU资源过载，V2V协作潜力未充分利用。'
        })
    
    # 问题5: 策略熵低
    last_100_entropy = metrics_df.tail(100)['policy_entropy'].mean()
    if last_100_entropy < 0.5:
        issues.append({
            'level': '⚠️ 轻微',
            'title': '策略熵较低',
            'desc': f'最后100 episodes的策略熵为 {last_100_entropy:.4f}，可能过早收敛。',
            'evidence': f'整体策略熵: {metrics_df["policy_entropy"].mean():.4f}',
            'impact': '可能限制策略的探索能力，难以发现更优解。'
        })
    
    # 问题6: V2V利用率低
    v2v_frac = last_100['decision_v2v_frac'].mean()
    if v2v_frac < 0.05:
        issues.append({
            'level': '⚠️ 中等',
            'title': 'V2V决策占比过低',
            'desc': f'V2V决策占比仅为 {v2v_frac:.2%}，V2V协作潜力未充分利用。',
            'evidence': f'Oracle建议V2V但选择RSU的比例: {episode_log_df["oracle_v2v_chosen_rsu_rate"].mean():.2%}',
            'impact': '可能错失V2V协作带来的性能提升机会。'
        })
    
    if issues:
        for i, issue in enumerate(issues, 1):
            report.append(f"### 8.{i} {issue['level']} {issue['title']}")
            report.append("")
            report.append(f"**描述**: {issue['desc']}")
            report.append("")
            report.append(f"**证据**: {issue['evidence']}")
            report.append("")
            report.append(f"**影响**: {issue['impact']}")
            report.append("")
    else:
        report.append("✓ 未发现明显问题")
        report.append("")
    
    # 9. 修复建议
    report.append("## 9. 修复建议")
    report.append("")
    
    if issues:
        report.append("基于上述问题识别，提供以下修复建议：")
        report.append("")
        
        suggestions = []
        
        # 针对梯度爆炸
        if any('梯度爆炸' in issue['title'] for issue in issues):
            suggestions.append({
                'problem': '梯度爆炸',
                'suggestions': [
                    '**降低学习率**: 当前可能学习率过高，建议降低至原来的50%-70%',
                    '**增强梯度裁剪**: 检查并调整梯度裁剪阈值，建议设置为10-20',
                    '**调整奖励缩放**: 检查奖励函数的缩放，避免奖励值过大',
                    '**增加批次大小**: 考虑增加batch_size以稳定梯度估计'
                ],
                'priority': '高',
                'verification': '监控梯度范数，确保95%以上的更新步骤梯度范数<10'
            })
        
        # 针对Oracle匹配率低
        if any('Oracle匹配率' in issue['title'] for issue in issues):
            suggestions.append({
                'problem': 'Oracle匹配率低',
                'suggestions': [
                    '**增加训练时长**: 当前1000 episodes可能不足，建议延长至2000-3000 episodes',
                    '**调整奖励函数**: 检查奖励函数设计，确保与Oracle目标一致',
                    '**增强探索**: 提高entropy_coef (当前0.012)至0.02-0.05',
                    '**使用课程学习**: 从简单场景逐步过渡到复杂场景',
                    '**添加模仿学习**: 在训练初期使用Oracle策略进行预训练'
                ],
                'priority': '高',
                'verification': '目标: Oracle匹配率>50%, 动作遗憾<0.05'
            })
        
        # 针对RSU过度依赖
        if any('RSU决策占比过高' in issue['title'] for issue in issues):
            suggestions.append({
                'problem': 'RSU过度依赖',
                'suggestions': [
                    '**调整奖励权重**: 增加V2V协作的奖励权重',
                    '**添加负载均衡惩罚**: 对RSU过载情况增加惩罚',
                    '**改进V2V候选集**: 优化V2V候选集生成策略，提高V2V选项质量',
                    '**调整Trust参数**: 检查Trust机制是否过度惩罚V2V'
                ],
                'priority': '中',
                'verification': '目标: RSU占比<60%, V2V占比>10%'
            })
        
        # 针对奖励波动
        if any('奖励波动' in issue['title'] for issue in issues):
            suggestions.append({
                'problem': '奖励波动大',
                'suggestions': [
                    '**延长训练**: 继续训练至收敛',
                    '**调整GAE Lambda**: 当前0.95可能过高，尝试0.90-0.93',
                    '**增加值函数训练**: 提高值函数的训练频率或学习率',
                    '**使用奖励归一化**: 实施running mean/std归一化'
                ],
                'priority': '中',
                'verification': '目标: 最后100ep的CV<0.3'
            })
        
        for i, sug in enumerate(suggestions, 1):
            report.append(f"### 9.{i} {sug['problem']} (优先级: {sug['priority']})")
            report.append("")
            report.append("**建议措施**:")
            report.append("")
            for s in sug['suggestions']:
                report.append(f"- {s}")
            report.append("")
            report.append(f"**验证标准**: {sug['verification']}")
            report.append("")
    else:
        report.append("当前训练状态良好，建议继续监控以下指标：")
        report.append("")
        report.append("- Oracle匹配率保持>50%")
        report.append("- 梯度范数保持稳定")
        report.append("- 决策分布保持合理")
        report.append("")
    
    # 10. 下一步行动计划
    report.append("## 10. 下一步行动计划")
    report.append("")
    report.append("### 10.1 短期行动 (1-2天)")
    report.append("")
    report.append("1. **运行baseline评估**: 执行 `python run_baseline_evaluation.py` 获取baseline对比数据")
    report.append("2. **生成对比图表**: 创建MAPPO vs Baselines的全面对比可视化")
    report.append("3. **验证问题**: 针对识别的问题进行详细验证")
    report.append("")
    
    report.append("### 10.2 中期行动 (3-7天)")
    report.append("")
    report.append("1. **实施修复**: 根据优先级实施修复建议")
    report.append("2. **重新训练**: 使用调整后的参数重新训练")
    report.append("3. **对比分析**: 对比修复前后的性能差异")
    report.append("")
    
    report.append("### 10.3 长期行动 (1-2周)")
    report.append("")
    report.append("1. **扩展评估**: 在不同场景下评估策略鲁棒性")
    report.append("2. **消融实验**: 进行消融实验验证各组件贡献")
    report.append("3. **论文撰写**: 整理实验结果，撰写论文")
    report.append("")
    
    # 11. 结论
    report.append("## 11. 结论")
    report.append("")
    report.append(f"本次1000 episode训练取得了 **{last_100['task_success_rate'].mean():.1%}** 的任务成功率，")
    report.append(f"但Oracle匹配率仅为 **{last_100_oracle:.1%}**，存在较大改进空间。")
    report.append("")
    report.append("**主要发现**:")
    report.append("")
    report.append(f"- ✓ 任务成功率较高 ({last_100['task_success_rate'].mean():.1%})")
    report.append(f"- ✓ 子任务成功率优秀 ({last_100['subtask_success_rate'].mean():.1%})")
    report.append(f"- ⚠️ Oracle匹配率偏低 ({last_100_oracle:.1%})")
    report.append(f"- ⚠️ 梯度爆炸问题 (max={max_grad:.0f})")
    report.append(f"- ⚠️ RSU过度依赖 ({last_100_rsu:.1%})")
    report.append("")
    report.append("**建议**: 优先解决梯度爆炸和Oracle匹配率低的问题，然后优化决策分布。")
    report.append("")
    
    # 保存报告
    report_text = '\n'.join(report)
    output_file = run_dir / 'COMPREHENSIVE_ANALYSIS_REPORT.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 报告已生成: {output_file}")
    print(f"  总行数: {len(report)}")
    print(f"  识别问题数: {len(issues)}")
    
    return report_text

if __name__ == '__main__':
    report = generate_report()
