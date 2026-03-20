#!/usr/bin/env python
"""
1000 Episode Training Run Comprehensive Analysis
运行目录: runs/run_20260318_003251
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingAnalyzer:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.metrics_df = None
        self.episode_log_df = None
        self.config = None
        self.meta = None
        
    def load_data(self):
        """加载所有数据"""
        print("正在加载数据...")
        
        # 加载训练指标
        self.metrics_df = pd.read_csv(self.run_dir / 'logs' / 'metrics.csv')
        print(f"✓ 加载训练指标: {len(self.metrics_df)} episodes")
        
        # 加载episode log
        self.episode_log_df = pd.read_csv(self.run_dir / 'episode_log.csv')
        print(f"✓ 加载episode log: {len(self.episode_log_df)} episodes")
        
        # 加载配置
        with open(self.run_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        print(f"✓ 加载配置")
        
        # 加载元数据
        with open(self.run_dir / 'run_meta.json', 'r') as f:
            self.meta = json.load(f)
        print(f"✓ 加载元数据")
        print()
        
    def print_basic_info(self):
        """打印基本信息"""
        print("=" * 80)
        print("训练基本信息")
        print("=" * 80)
        print(f"运行目录: {self.run_dir}")
        print(f"开始时间: {self.meta.get('start_time', 'N/A')}")
        print(f"结束时间: {self.meta.get('end_time', 'N/A')}")
        print(f"训练时长: {self.meta.get('elapsed_time', 'N/A')}")
        print(f"总episodes: {len(self.metrics_df)}")
        print(f"随机种子: {self.config.get('seed', 'N/A')}")
        print(f"车辆数: {self.config.get('NUM_VEHICLES', 'N/A')}")
        print(f"RSU数: {self.config.get('NUM_RSUS', 'N/A')}")
        print(f"最大步数/episode: {self.config.get('MAX_STEPS', 'N/A')}")
        print(f"Trust启用: {self.config.get('TRUST_ENABLED', 'N/A')}")
        print(f"恶意车辆比例: {self.config.get('MALICIOUS_RATIO', 'N/A')}")
        print()
        
    def analyze_training_metrics(self):
        """分析训练指标"""
        print("=" * 80)
        print("训练指标统计")
        print("=" * 80)
        
        df = self.metrics_df
        
        # 关键指标
        key_metrics = {
            'r_total': '总奖励',
            'task_success_rate': '任务成功率',
            'subtask_success_rate': '子任务成功率',
            'mean_cft_est': '平均完成时间',
            'energy_norm_mean': '归一化能耗',
            'I_caused_mean': '平均干扰',
            'risk_penalty_mean': '风险惩罚',
            'oracle_match_rate': 'Oracle匹配率',
            'action_regret_mean': '动作遗憾',
        }
        
        for col, name in key_metrics.items():
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    print(f"\n{name} ({col}):")
                    print(f"  均值: {values.mean():.4f}")
                    print(f"  中位数: {values.median():.4f}")
                    print(f"  标准差: {values.std():.4f}")
                    print(f"  最小值: {values.min():.4f}")
                    print(f"  最大值: {values.max():.4f}")
                    
                    # 最后100 episodes
                    last_100 = df.tail(100)[col].dropna()
                    if len(last_100) > 0:
                        print(f"  最后100ep均值: {last_100.mean():.4f}")
        
        print()
        
    def analyze_by_phase(self):
        """分阶段分析"""
        print("=" * 80)
        print("分阶段分析 (每250 episodes)")
        print("=" * 80)
        
        df = self.metrics_df
        phases = [(0, 250), (250, 500), (500, 750), (750, 1000)]
        
        metrics_to_track = [
            ('r_total', '总奖励'),
            ('task_success_rate', '任务成功率'),
            ('subtask_success_rate', '子任务成功率'),
            ('oracle_match_rate', 'Oracle匹配率'),
            ('action_regret_mean', '动作遗憾'),
        ]
        
        for start, end in phases:
            phase_df = df.iloc[start:end]
            print(f"\nEpisodes {start}-{end}:")
            for col, name in metrics_to_track:
                if col in phase_df.columns:
                    values = phase_df[col].dropna()
                    if len(values) > 0:
                        print(f"  {name}: {values.mean():.4f} ± {values.std():.4f}")
        
        print()
        
    def analyze_decision_distribution(self):
        """分析决策分布"""
        print("=" * 80)
        print("决策分布分析")
        print("=" * 80)
        
        df = self.metrics_df
        
        # 决策分布
        if all(col in df.columns for col in ['decision_local_frac', 'decision_rsu_frac', 'decision_v2v_frac']):
            print("\n整体决策分布:")
            print(f"  Local: {df['decision_local_frac'].mean():.2%}")
            print(f"  RSU: {df['decision_rsu_frac'].mean():.2%}")
            print(f"  V2V: {df['decision_v2v_frac'].mean():.2%}")
            
            print("\n最后100 episodes决策分布:")
            last_100 = df.tail(100)
            print(f"  Local: {last_100['decision_local_frac'].mean():.2%}")
            print(f"  RSU: {last_100['decision_rsu_frac'].mean():.2%}")
            print(f"  V2V: {last_100['decision_v2v_frac'].mean():.2%}")
        
        print()
        
    def analyze_training_stability(self):
        """分析训练稳定性"""
        print("=" * 80)
        print("训练稳定性分析")
        print("=" * 80)
        
        df = self.metrics_df
        
        # 检查梯度范数
        if 'grad_norm' in df.columns:
            grad_norms = df['grad_norm'].dropna()
            print(f"\n梯度范数:")
            print(f"  均值: {grad_norms.mean():.4f}")
            print(f"  最大值: {grad_norms.max():.4f}")
            print(f"  >10的比例: {(grad_norms > 10).mean():.2%}")
            print(f"  >100的比例: {(grad_norms > 100).mean():.2%}")
        
        # 检查KL散度
        if 'approx_kl' in df.columns:
            kl_divs = df['approx_kl'].dropna()
            print(f"\nKL散度:")
            print(f"  均值: {kl_divs.mean():.6f}")
            print(f"  最大值: {kl_divs.max():.6f}")
        
        # 检查clip fraction
        if 'clip_frac' in df.columns:
            clip_fracs = df['clip_frac'].dropna()
            print(f"\nClip Fraction:")
            print(f"  均值: {clip_fracs.mean():.4f}")
        
        # 检查熵
        if 'policy_entropy' in df.columns:
            entropies = df['policy_entropy'].dropna()
            print(f"\n策略熵:")
            print(f"  均值: {entropies.mean():.4f}")
            print(f"  最小值: {entropies.min():.4f}")
            print(f"  最大值: {entropies.max():.4f}")
        
        print()
        
    def analyze_convergence(self):
        """分析收敛性"""
        print("=" * 80)
        print("收敛性分析")
        print("=" * 80)
        
        df = self.metrics_df
        
        # 计算滑动平均
        window = 50
        if 'r_total' in df.columns:
            rewards = df['r_total'].dropna()
            if len(rewards) >= window:
                ma = rewards.rolling(window=window).mean()
                print(f"\n总奖励 (50-episode滑动平均):")
                print(f"  前50ep: {ma.iloc[window-1]:.4f}")
                print(f"  中间50ep (ep {len(df)//2-25}-{len(df)//2+25}): {ma.iloc[len(df)//2]:.4f}")
                print(f"  最后50ep: {ma.iloc[-1]:.4f}")
                
                # 检查是否收敛
                last_100_std = rewards.tail(100).std()
                print(f"  最后100ep标准差: {last_100_std:.4f}")
                
        if 'task_success_rate' in df.columns:
            success_rates = df['task_success_rate'].dropna()
            if len(success_rates) >= window:
                ma = success_rates.rolling(window=window).mean()
                print(f"\n任务成功率 (50-episode滑动平均):")
                print(f"  前50ep: {ma.iloc[window-1]:.4f}")
                print(f"  中间50ep: {ma.iloc[len(df)//2]:.4f}")
                print(f"  最后50ep: {ma.iloc[-1]:.4f}")
        
        print()
        
    def identify_issues(self):
        """识别潜在问题"""
        print("=" * 80)
        print("潜在问题识别")
        print("=" * 80)
        
        df = self.metrics_df
        issues = []
        
        # 1. 检查奖励是否收敛
        if 'r_total' in df.columns:
            last_100_rewards = df.tail(100)['r_total'].dropna()
            if len(last_100_rewards) > 0:
                std = last_100_rewards.std()
                mean = last_100_rewards.mean()
                cv = std / abs(mean) if mean != 0 else float('inf')
                if cv > 0.5:
                    issues.append(f"⚠️  奖励波动较大 (CV={cv:.2f}), 可能未完全收敛")
        
        # 2. 检查成功率
        if 'task_success_rate' in df.columns:
            last_100_success = df.tail(100)['task_success_rate'].dropna()
            if len(last_100_success) > 0:
                mean_success = last_100_success.mean()
                if mean_success < 0.9:
                    issues.append(f"⚠️  最后100ep平均任务成功率较低 ({mean_success:.2%})")
        
        # 3. 检查Oracle匹配率
        if 'oracle_match_rate' in df.columns:
            last_100_oracle = df.tail(100)['oracle_match_rate'].dropna()
            if len(last_100_oracle) > 0:
                mean_oracle = last_100_oracle.mean()
                if mean_oracle < 0.5:
                    issues.append(f"⚠️  Oracle匹配率较低 ({mean_oracle:.2%}), 策略可能不够优")
        
        # 4. 检查梯度爆炸
        if 'grad_norm' in df.columns:
            grad_norms = df['grad_norm'].dropna()
            if len(grad_norms) > 0:
                max_grad = grad_norms.max()
                if max_grad > 100:
                    issues.append(f"⚠️  检测到梯度爆炸 (max={max_grad:.2f})")
        
        # 5. 检查决策分布
        if all(col in df.columns for col in ['decision_local_frac', 'decision_rsu_frac', 'decision_v2v_frac']):
            last_100 = df.tail(100)
            rsu_frac = last_100['decision_rsu_frac'].mean()
            if rsu_frac > 0.8:
                issues.append(f"⚠️  RSU决策占比过高 ({rsu_frac:.2%}), 可能过度依赖RSU")
        
        # 6. 检查熵
        if 'policy_entropy' in df.columns:
            last_100_entropy = df.tail(100)['policy_entropy'].dropna()
            if len(last_100_entropy) > 0:
                mean_entropy = last_100_entropy.mean()
                if mean_entropy < 0.5:
                    issues.append(f"⚠️  策略熵较低 ({mean_entropy:.4f}), 可能过早收敛")
        
        # 7. 检查非法动作率
        if 'illegal_action_rate' in df.columns:
            illegal_rate = df['illegal_action_rate'].mean()
            if illegal_rate > 0.05:
                issues.append(f"⚠️  非法动作率较高 ({illegal_rate:.2%})")
        
        if issues:
            print("\n发现以下潜在问题:\n")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")
        else:
            print("\n✓ 未发现明显问题")
        
        print()
        return issues
        
    def generate_summary_table(self):
        """生成汇总表格"""
        print("=" * 80)
        print("训练结果汇总表")
        print("=" * 80)
        
        df = self.metrics_df
        
        # 计算关键指标
        metrics = {}
        
        # 整体指标
        if 'r_total' in df.columns:
            metrics['平均总奖励'] = df['r_total'].mean()
        if 'task_success_rate' in df.columns:
            metrics['平均任务成功率'] = df['task_success_rate'].mean()
        if 'subtask_success_rate' in df.columns:
            metrics['平均子任务成功率'] = df['subtask_success_rate'].mean()
        if 'mean_cft_est' in df.columns:
            metrics['平均完成时间'] = df['mean_cft_est'].mean()
        if 'energy_norm_mean' in df.columns:
            metrics['平均归一化能耗'] = df['energy_norm_mean'].mean()
        if 'oracle_match_rate' in df.columns:
            metrics['Oracle匹配率'] = df['oracle_match_rate'].mean()
        if 'action_regret_mean' in df.columns:
            metrics['平均动作遗憾'] = df['action_regret_mean'].mean()
        
        # 最后100 episodes
        last_100 = df.tail(100)
        if 'r_total' in last_100.columns:
            metrics['最后100ep平均奖励'] = last_100['r_total'].mean()
        if 'task_success_rate' in last_100.columns:
            metrics['最后100ep任务成功率'] = last_100['task_success_rate'].mean()
        if 'oracle_match_rate' in last_100.columns:
            metrics['最后100ep Oracle匹配率'] = last_100['oracle_match_rate'].mean()
        
        print()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.4f}")
            else:
                print(f"{key:.<40} {value}")
        print()
        
        return metrics
        
    def run_full_analysis(self):
        """运行完整分析"""
        self.load_data()
        self.print_basic_info()
        self.analyze_training_metrics()
        self.analyze_by_phase()
        self.analyze_decision_distribution()
        self.analyze_training_stability()
        self.analyze_convergence()
        issues = self.identify_issues()
        summary = self.generate_summary_table()
        
        return {
            'issues': issues,
            'summary': summary
        }

if __name__ == '__main__':
    run_dir = Path(__file__).parent
    analyzer = TrainingAnalyzer(run_dir)
    results = analyzer.run_full_analysis()
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
