#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据全面分析脚本
分析 pre500ep_500ep_seed42 训练运行的所有数据
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TrainingDataAnalyzer:
    """训练数据分析器"""
    
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.report_lines = []
        
        # 加载数据
        self.episode_log = pd.read_csv(self.run_dir / "episode_log.csv")
        self.metrics = pd.read_csv(self.run_dir / "logs/metrics.csv")
        self.training_stats = pd.read_csv(self.run_dir / "logs/training_stats.csv")
        
        # 加载配置
        with open(self.run_dir / "config.json", 'r') as f:
            self.config = json.load(f)
        with open(self.run_dir / "run_meta.json", 'r') as f:
            self.run_meta = json.load(f)
    
    def add_section(self, title, level=1):
        """添加章节标题"""
        prefix = "#" * level
        self.report_lines.append(f"\n{prefix} {title}\n")
    
    def add_text(self, text):
        """添加文本"""
        self.report_lines.append(text + "\n")
    
    def add_table(self, df, caption=""):
        """添加表格"""
        if caption:
            self.report_lines.append(f"\n**{caption}**\n")
        self.report_lines.append(df.to_markdown(index=False) + "\n")
    
    def analyze_basic_stats(self):
        """基础统计分析"""
        self.add_section("基础统计", 2)
        
        total_episodes = len(self.episode_log)
        total_duration = self.episode_log['duration'].sum()
        avg_duration = self.episode_log['duration'].mean()
        
        stats = {
            "指标": ["训练总轮数", "训练总时长 (秒)", "平均每轮时长 (秒)", 
                    "最短轮时长 (秒)", "最长轮时长 (秒)"],
            "数值": [
                total_episodes,
                f"{total_duration:.2f}",
                f"{avg_duration:.2f}",
                f"{self.episode_log['duration'].min():.2f}",
                f"{self.episode_log['duration'].max():.2f}"
            ]
        }
        
        self.add_table(pd.DataFrame(stats), "训练基础统计")
        
        # 数据完整性检查
        self.add_section("数据完整性检查", 3)
        
        missing_episode = self.episode_log.isnull().sum()
        missing_metrics = self.metrics.isnull().sum()
        missing_training = self.training_stats.isnull().sum()
        
        if missing_episode.sum() == 0 and missing_metrics.sum() == 0 and missing_training.sum() == 0:
            self.add_text("✅ 所有数据文件完整，无缺失值")
        else:
            self.add_text("⚠️ 发现数据缺失：")
            if missing_episode.sum() > 0:
                self.add_text(f"  - episode_log.csv: {missing_episode[missing_episode > 0].to_dict()}")
            if missing_metrics.sum() > 0:
                self.add_text(f"  - metrics.csv: {missing_metrics[missing_metrics > 0].to_dict()}")
            if missing_training.sum() > 0:
                self.add_text(f"  - training_stats.csv: {missing_training[missing_training > 0].to_dict()}")
    
    def analyze_convergence(self):
        """收敛性分析"""
        self.add_section("收敛性分析", 2)
        
        # 奖励曲线分析
        rewards = self.episode_log['total_reward'].values
        
        # 移动平均
        window_sizes = [10, 50, 100]
        ma_stats = []
        for w in window_sizes:
            ma = pd.Series(rewards).rolling(window=w, min_periods=1).mean()
            ma_stats.append({
                "窗口大小": w,
                "最终值": f"{ma.iloc[-1]:.4f}",
                "最大值": f"{ma.max():.4f}",
                "最小值": f"{ma.min():.4f}"
            })
        
        self.add_table(pd.DataFrame(ma_stats), "奖励移动平均统计")
        
        # 收敛判断：比较前100轮和后100轮
        first_100 = rewards[:100]
        last_100 = rewards[-100:]
        
        improvement = last_100.mean() - first_100.mean()
        improvement_pct = (improvement / abs(first_100.mean())) * 100 if first_100.mean() != 0 else 0
        
        convergence_stats = {
            "阶段": ["前100轮", "后100轮", "改善"],
            "平均奖励": [
                f"{first_100.mean():.4f}",
                f"{last_100.mean():.4f}",
                f"{improvement:.4f} ({improvement_pct:+.2f}%)"
            ],
            "标准差": [
                f"{first_100.std():.4f}",
                f"{last_100.std():.4f}",
                f"{last_100.std() - first_100.std():.4f}"
            ]
        }
        
        self.add_table(pd.DataFrame(convergence_stats), "收敛性对比")
        
        # 收敛判断
        if improvement > 0 and last_100.std() < first_100.std():
            self.add_text("✅ **训练已收敛**：后期奖励提升且波动减小")
        elif improvement > 0:
            self.add_text("⚠️ **部分收敛**：奖励有提升但波动仍较大")
        else:
            self.add_text("❌ **未收敛**：后期奖励未见明显提升")
        
        # 训练稳定性
        self.add_section("训练稳定性", 3)
        
        # 计算滚动方差
        rolling_std = pd.Series(rewards).rolling(window=50, min_periods=1).std()
        
        stability_stats = {
            "指标": ["整体方差", "前期方差 (1-100)", "中期方差 (200-300)", "后期方差 (400-500)"],
            "数值": [
                f"{rewards.std():.4f}",
                f"{rewards[:100].std():.4f}",
                f"{rewards[200:300].std():.4f}",
                f"{rewards[400:500].std():.4f}"
            ]
        }
        
        self.add_table(pd.DataFrame(stability_stats), "训练稳定性统计")
        
        if rolling_std.iloc[-50:].mean() < rolling_std.iloc[:50].mean():
            self.add_text("✅ 训练稳定性良好：后期波动明显减小")
        else:
            self.add_text("⚠️ 训练稳定性一般：波动未见明显减小")
    
    def analyze_performance_metrics(self):
        """性能指标分析"""
        self.add_section("性能指标分析", 2)
        
        # 任务成功率
        self.add_section("任务成功率", 3)
        
        task_sr = self.metrics['task_success_rate'].values
        subtask_sr = self.metrics['subtask_success_rate'].values
        
        sr_stats = {
            "指标": ["任务成功率", "子任务成功率"],
            "平均值": [f"{task_sr.mean():.4f}", f"{subtask_sr.mean():.4f}"],
            "最大值": [f"{task_sr.max():.4f}", f"{subtask_sr.max():.4f}"],
            "最小值": [f"{task_sr.min():.4f}", f"{subtask_sr.min():.4f}"],
            "最终100轮均值": [f"{task_sr[-100:].mean():.4f}", f"{subtask_sr[-100:].mean():.4f}"]
        }
        
        self.add_table(pd.DataFrame(sr_stats), "成功率统计")
        
        # 延迟指标
        self.add_section("延迟指标", 3)
        
        mean_cft = self.metrics['mean_cft_est'].values
        chain_p95 = self.metrics['chain_p95_mean'].values
        
        latency_stats = {
            "指标": ["平均完成时间 (mean_cft_est)", "链路P95延迟 (chain_p95_mean)"],
            "平均值": [f"{np.nanmean(mean_cft):.4f}", f"{np.nanmean(chain_p95):.4f}"],
            "中位数": [f"{np.nanmedian(mean_cft):.4f}", f"{np.nanmedian(chain_p95):.4f}"],
            "P95": [f"{np.nanpercentile(mean_cft, 95):.4f}", f"{np.nanpercentile(chain_p95, 95):.4f}"]
        }
        
        self.add_table(pd.DataFrame(latency_stats), "延迟指标统计")
        
        # 能耗指标
        self.add_section("能耗指标", 3)
        
        energy_norm = self.metrics['energy_norm_mean'].values
        power_ratio = self.metrics['power_ratio_mean'].values
        
        energy_stats = {
            "指标": ["归一化能耗 (energy_norm_mean)", "功率比 (power_ratio_mean)"],
            "平均值": [f"{np.nanmean(energy_norm):.4f}", f"{np.nanmean(power_ratio):.4f}"],
            "最大值": [f"{np.nanmax(energy_norm):.4f}", f"{np.nanmax(power_ratio):.4f}"],
            "最小值": [f"{np.nanmin(energy_norm):.4f}", f"{np.nanmin(power_ratio):.4f}"]
        }
        
        self.add_table(pd.DataFrame(energy_stats), "能耗指标统计")
        
        # 风险指标
        self.add_section("风险指标", 3)
        
        risk_penalty = self.metrics['risk_penalty_mean'].values
        I_total_p95 = self.metrics['I_total_p95'].values
        
        risk_stats = {
            "指标": ["风险惩罚 (risk_penalty_mean)", "总干扰P95 (I_total_p95)"],
            "平均值": [f"{np.nanmean(risk_penalty):.4f}", f"{np.nanmean(I_total_p95):.4f}"],
            "最大值": [f"{np.nanmax(risk_penalty):.4f}", f"{np.nanmax(I_total_p95):.4f}"],
            "非零率": [
                f"{(risk_penalty != 0).sum() / len(risk_penalty):.2%}",
                f"{(I_total_p95 != 0).sum() / len(I_total_p95):.2%}"
            ]
        }
        
        self.add_table(pd.DataFrame(risk_stats), "风险指标统计")
        
        # 信任相关指标
        self.add_section("信任相关指标", 3)
        
        trust_failure_rate = self.metrics['trust_failure_rate'].values
        rho_selected_p10 = self.metrics['rho_selected_p10'].values
        
        trust_stats = {
            "指标": ["信任失败率 (trust_failure_rate)", "信任度P10 (rho_selected_p10)"],
            "平均值": [f"{np.nanmean(trust_failure_rate):.4f}", f"{np.nanmean(rho_selected_p10):.4f}"],
            "最大值": [f"{np.nanmax(trust_failure_rate):.4f}", f"{np.nanmax(rho_selected_p10):.4f}"],
            "非零率": [
                f"{(trust_failure_rate != 0).sum() / len(trust_failure_rate):.2%}",
                f"{(rho_selected_p10 != 0).sum() / len(rho_selected_p10):.2%}"
            ]
        }
        
        self.add_table(pd.DataFrame(trust_stats), "信任指标统计")
    
    def analyze_training_health(self):
        """训练健康度分析"""
        self.add_section("训练健康度", 2)
        
        # Loss 曲线
        self.add_section("Loss 分析", 3)
        
        actor_loss = self.training_stats['actor_loss'].values
        critic_loss = self.training_stats['critic_loss'].values
        
        loss_stats = {
            "Loss类型": ["Actor Loss", "Critic Loss"],
            "平均值": [f"{actor_loss.mean():.4f}", f"{critic_loss.mean():.4f}"],
            "最大值": [f"{actor_loss.max():.4f}", f"{critic_loss.max():.4f}"],
            "最小值": [f"{actor_loss.min():.4f}", f"{critic_loss.min():.4f}"],
            "最终100轮均值": [f"{actor_loss[-100:].mean():.4f}", f"{critic_loss[-100:].mean():.4f}"]
        }
        
        self.add_table(pd.DataFrame(loss_stats), "Loss 统计")
        
        # 检查 loss 趋势
        first_100_actor = actor_loss[:100].mean()
        last_100_actor = actor_loss[-100:].mean()
        first_100_critic = critic_loss[:100].mean()
        last_100_critic = critic_loss[-100:].mean()
        
        if last_100_actor < first_100_actor and last_100_critic < first_100_critic:
            self.add_text("✅ Loss 下降趋势良好")
        elif last_100_actor < first_100_actor or last_100_critic < first_100_critic:
            self.add_text("⚠️ Loss 部分下降")
        else:
            self.add_text("❌ Loss 未见下降趋势")
        
        # 梯度统计
        self.add_section("梯度统计", 3)
        
        grad_norm = self.training_stats['grad_norm'].values
        
        grad_stats = {
            "指标": ["梯度范数"],
            "平均值": [f"{grad_norm.mean():.4f}"],
            "最大值": [f"{grad_norm.max():.4f}"],
            "最小值": [f"{grad_norm.min():.4f}"],
            "P95": [f"{np.percentile(grad_norm, 95):.4f}"]
        }
        
        self.add_table(pd.DataFrame(grad_stats), "梯度统计")
        
        # 检查梯度爆炸/消失
        if grad_norm.max() > 100:
            self.add_text("⚠️ 检测到梯度过大（可能梯度爆炸）")
        elif grad_norm.mean() < 0.01:
            self.add_text("⚠️ 检测到梯度过小（可能梯度消失）")
        else:
            self.add_text("✅ 梯度范围正常")
        
        # 熵值变化
        self.add_section("熵值分析", 3)
        
        entropy = self.training_stats['entropy'].values
        
        entropy_stats = {
            "指标": ["策略熵"],
            "平均值": [f"{entropy.mean():.4f}"],
            "前100轮均值": [f"{entropy[:100].mean():.4f}"],
            "后100轮均值": [f"{entropy[-100:].mean():.4f}"],
            "变化": [f"{entropy[-100:].mean() - entropy[:100].mean():.4f}"]
        }
        
        self.add_table(pd.DataFrame(entropy_stats), "熵值统计")
        
        if entropy[-100:].mean() < entropy[:100].mean():
            self.add_text("✅ 策略熵下降，策略逐渐确定")
        else:
            self.add_text("⚠️ 策略熵未下降，探索可能过多")
        
        # KL散度
        self.add_section("KL散度", 3)
        
        approx_kl = self.training_stats['approx_kl'].values
        
        kl_stats = {
            "指标": ["近似KL散度"],
            "平均值": [f"{approx_kl.mean():.6f}"],
            "最大值": [f"{approx_kl.max():.6f}"],
            "P95": [f"{np.percentile(approx_kl, 95):.6f}"]
        }
        
        self.add_table(pd.DataFrame(kl_stats), "KL散度统计")
        
        if approx_kl.max() > 0.1:
            self.add_text("⚠️ KL散度过大，策略更新步长可能过大")
        else:
            self.add_text("✅ KL散度在合理范围内")
    
    def analyze_reward_decomposition(self):
        """奖励分解分析"""
        self.add_section("奖励分解分析", 2)
        
        # 奖励组件
        reward_components = {
            'r_term': '终止奖励',
            'r_energy': '能耗奖励',
            'r_risk': '风险奖励',
            'r_pbrs': 'PBRS奖励',
            'r_interf': '干扰奖励'
        }
        
        component_stats = []
        for col, name in reward_components.items():
            if col in self.metrics.columns:
                values = self.metrics[col].values
                component_stats.append({
                    "奖励组件": name,
                    "平均值": f"{np.nanmean(values):.4f}",
                    "标准差": f"{np.nanstd(values):.4f}",
                    "最大值": f"{np.nanmax(values):.4f}",
                    "最小值": f"{np.nanmin(values):.4f}"
                })
        
        if component_stats:
            self.add_table(pd.DataFrame(component_stats), "奖励组件统计")
        
        # 奖励组件贡献度
        self.add_section("奖励组件贡献度", 3)
        
        abs_ratio_cols = [col for col in self.metrics.columns if col.startswith('abs_ratio_')]
        if abs_ratio_cols:
            ratio_stats = []
            for col in abs_ratio_cols:
                values = pd.to_numeric(self.metrics[col], errors='coerce').values
                if not np.all(np.isnan(values)):
                    ratio_stats.append({
                        "组件": col.replace('abs_ratio_', ''),
                        "平均贡献": f"{np.nanmean(values):.4f}",
                        "最大贡献": f"{np.nanmax(values):.4f}"
                    })
            
            if ratio_stats:
                self.add_table(pd.DataFrame(ratio_stats), "奖励组件贡献度")
    
    def analyze_multi_agent(self):
        """多智能体协作分析"""
        self.add_section("多智能体协作", 2)
        
        if 'ma_fairness' in self.episode_log.columns:
            ma_fairness = self.episode_log['ma_fairness'].values
            ma_reward_gap = self.episode_log['ma_reward_gap'].values
            ma_collaboration = self.episode_log['ma_collaboration'].values
            
            ma_stats = {
                "指标": ["公平性 (ma_fairness)", "奖励差距 (ma_reward_gap)", "协作度 (ma_collaboration)"],
                "平均值": [
                    f"{np.nanmean(ma_fairness):.4f}",
                    f"{np.nanmean(ma_reward_gap):.4f}",
                    f"{np.nanmean(ma_collaboration):.4f}"
                ],
                "标准差": [
                    f"{np.nanstd(ma_fairness):.4f}",
                    f"{np.nanstd(ma_reward_gap):.4f}",
                    f"{np.nanstd(ma_collaboration):.4f}"
                ]
            }
            
            self.add_table(pd.DataFrame(ma_stats), "多智能体协作统计")
            
            # 评估协作质量
            if np.nanmean(ma_fairness) > 0.7:
                self.add_text("✅ 智能体间公平性良好")
            else:
                self.add_text("⚠️ 智能体间公平性较差")
    
    def detect_anomalies(self):
        """异常检测"""
        self.add_section("异常检测", 2)
        
        anomalies = []
        
        # 检查奖励突降
        rewards = self.episode_log['total_reward'].values
        reward_diff = np.diff(rewards)
        sudden_drops = np.where(reward_diff < -1.0)[0]
        
        if len(sudden_drops) > 0:
            anomalies.append(f"⚠️ 检测到 {len(sudden_drops)} 次奖励突降（降幅 > 1.0）")
            anomalies.append(f"   突降位置: {sudden_drops[:10].tolist()}")
        
        # 检查成功率骤降
        if 'task_success_rate' in self.metrics.columns:
            sr = self.metrics['task_success_rate'].values
            sr_diff = np.diff(sr)
            sr_drops = np.where(sr_diff < -0.3)[0]
            
            if len(sr_drops) > 0:
                anomalies.append(f"⚠️ 检测到 {len(sr_drops)} 次成功率骤降（降幅 > 0.3）")
                anomalies.append(f"   骤降位置: {sr_drops[:10].tolist()}")
        
        # 检查 NaN 或 Inf
        for df, name in [(self.episode_log, "episode_log"), 
                         (self.metrics, "metrics"), 
                         (self.training_stats, "training_stats")]:
            nan_count = df.isnull().sum().sum()
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            
            if nan_count > 0:
                anomalies.append(f"⚠️ {name} 包含 {nan_count} 个 NaN 值")
            if inf_count > 0:
                anomalies.append(f"⚠️ {name} 包含 {inf_count} 个 Inf 值")
        
        # 检查数值范围
        if 'total_reward' in self.episode_log.columns:
            reward_min = self.episode_log['total_reward'].min()
            reward_max = self.episode_log['total_reward'].max()
            
            if reward_min < -100 or reward_max > 100:
                anomalies.append(f"⚠️ 奖励范围异常: [{reward_min:.2f}, {reward_max:.2f}]")
        
        if anomalies:
            for anomaly in anomalies:
                self.add_text(anomaly)
        else:
            self.add_text("✅ 未检测到明显异常")
    
    def generate_summary(self):
        """生成总结"""
        self.add_section("总结与建议", 2)
        
        # 收集关键指标
        final_reward = self.episode_log['total_reward'].iloc[-100:].mean()
        final_sr = self.metrics['task_success_rate'].iloc[-100:].mean()
        final_subtask_sr = self.metrics['subtask_success_rate'].iloc[-100:].mean()
        
        improvement = self.episode_log['total_reward'].iloc[-100:].mean() - \
                     self.episode_log['total_reward'].iloc[:100].mean()
        
        summary = []
        summary.append(f"**训练完成情况：**")
        summary.append(f"- 总轮数: {len(self.episode_log)}")
        summary.append(f"- 最终100轮平均奖励: {final_reward:.4f}")
        summary.append(f"- 最终100轮任务成功率: {final_sr:.2%}")
        summary.append(f"- 最终100轮子任务成功率: {final_subtask_sr:.2%}")
        summary.append(f"- 相比前100轮奖励改善: {improvement:.4f}")
        summary.append("")
        
        # 评估训练质量
        summary.append("**训练质量评估：**")
        
        if improvement > 0 and final_sr > 0.7:
            summary.append("✅ 训练效果良好，模型性能显著提升")
        elif improvement > 0:
            summary.append("⚠️ 训练有改善但成功率仍需提高")
        else:
            summary.append("❌ 训练效果不佳，建议调整超参数")
        
        for line in summary:
            self.add_text(line)
    
    def run_analysis(self):
        """运行完整分析"""
        self.add_section("训练数据全面分析报告", 1)
        self.add_text(f"**运行ID:** {self.run_meta['run_id']}")
        self.add_text(f"**分析时间:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_text(f"**训练状态:** {self.run_meta['status']}")
        self.add_text("")
        
        # 执行各项分析
        self.analyze_basic_stats()
        self.analyze_convergence()
        self.analyze_performance_metrics()
        self.analyze_training_health()
        self.analyze_reward_decomposition()
        self.analyze_multi_agent()
        self.detect_anomalies()
        self.generate_summary()
        
        # 保存报告
        report_path = self.run_dir / "TRAINING_DATA_ANALYSIS.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(self.report_lines)
        
        print(f"✅ 分析报告已生成: {report_path}")
        return report_path


def main():
    """主函数"""
    run_dir = Path(__file__).parent
    print(f"开始分析训练数据: {run_dir}")
    
    analyzer = TrainingDataAnalyzer(run_dir)
    report_path = analyzer.run_analysis()
    
    print(f"\n分析完成！报告路径: {report_path}")


if __name__ == "__main__":
    main()
