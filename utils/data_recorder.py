import os
import json
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    if os.environ.get("TB_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
        print("[Warning] TensorBoard not available. Install with: pip install tensorboard")


# --- 自定义编码器，用于解决 TypeError: Object of type ndarray is not JSON serializable ---
class NumpyEncoder(json.JSONEncoder):
    """
    [工具类] 专门处理 NumPy 数据类型的 JSON 编码器。
    防止在保存 Config 时因为 numpy.float32 或 numpy.ndarray 报错。
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # [新增] 兼容处理 PyTorch Tensor
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DataRecorder:
    """
    [数据记录器]
    负责实验数据的存储、模型 Checkpoint 保存以及自动绘图。

    增强特性:
    - 支持多智能体指标可视化 (公平性、协作率、个体差异)。
    - 自动处理 Matplotlib 绘图风格和字体兼容性。
    """

    def __init__(self, experiment_name="Default_Exp", base_dir=None, quiet=False):
        """
        初始化文件结构
        """
        self.quiet = bool(quiet)
        if base_dir:
            self.exp_dir = os.path.abspath(base_dir)
            os.makedirs(self.exp_dir, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        else:
            # 1. 创建 data 根目录
            if not os.path.exists("./data"):
                os.makedirs("./data")

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_dir = os.path.join("./data", f"{experiment_name}_{timestamp}")

            if not os.path.exists(self.exp_dir):
                os.makedirs(self.exp_dir)

        self.model_dir = os.path.join(self.exp_dir, "models")
        self.plot_dir = os.path.join(self.exp_dir, "plots")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.step_log_path = os.path.join(self.exp_dir, "step_log.csv")
        self.episode_log_path = os.path.join(self.exp_dir, "episode_log.csv")

        self.step_header_written = False
        self.episode_header_written = False

        # [新增] 初始化 TensorBoard Writer
        # log_dir 使用相对路径（兼容本地和AutoDL环境）
        if TENSORBOARD_AVAILABLE:
            # 优先使用 /root/tf-logs（AutoDL环境），否则使用项目目录下的 logs 文件夹
            if base_dir:
                tb_log_dir = os.path.join(self.exp_dir, "logs", "tb")
                os.makedirs(tb_log_dir, exist_ok=True)
            elif os.path.exists('/root') and os.access('/root', os.W_OK):
                tb_log_dir = f"/root/tf-logs/{experiment_name}_{timestamp}"
            else:
                # 本地环境使用项目目录
                tb_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", f"{experiment_name}_{timestamp}")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            if not self.quiet:
                print(f"[TensorBoard] Log dir: {tb_log_dir}")
        else:
            self.writer = None

    def save_config(self, config_dict):
        """
        保存超参数配置到 config.json，便于实验复现。
        """
        config_path = os.path.join(self.exp_dir, "config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            print(f"[Error] Failed to save config: {e}")

    def log_step(self, step_data_list):
        """
        批量写入 Step 级日志 (train.py 中每个 Episode 结束时调用一次)
        """
        if not step_data_list: return
        keys = step_data_list[0].keys()
        try:
            # 使用 'a' 模式追加
            with open(self.step_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not self.step_header_written:
                    writer.writeheader()
                    self.step_header_written = True
                writer.writerows(step_data_list)
        except Exception as e:
            print(f"[Error] Failed to log step: {e}")

    def log_episode(self, episode_data):
        """
        写入 Episode 级汇总日志
        """
        if not episode_data: return
        keys = episode_data.keys()
        try:
            with open(self.episode_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not self.episode_header_written:
                    writer.writeheader()
                    self.episode_header_written = True
                writer.writerow(episode_data)
            # [新增] 将数据写入 TensorBoard
            step = episode_data['episode']
            if self.writer is not None:
                # 区分训练数据和baseline数据
                policy_name = episode_data.get('policy', None)
                if policy_name:
                    # baseline数据：使用Baseline/{policy_name}/前缀
                    for key, value in episode_data.items():
                        if key not in ['episode', 'policy'] and isinstance(value, (int, float)):
                            self.writer.add_scalar(f'Baseline/{policy_name}/{key}', value, step)
                else:
                    # 训练数据：直接写入（保持向后兼容）
                    for key, value in episode_data.items():
                        if key not in ['episode', 'policy'] and isinstance(value, (int, float)):
                            self.writer.add_scalar(key, value, step)
        except Exception as e:
            print(f"[Error] Failed to log episode: {e}")

        # =========================================================================
        # [关键修改 1] 完善模型保存策略 (Three-Tier Saving Strategy)
        # =========================================================================
    def save_model(self, agent, episode, is_best=False):
        """
        保存模型权重 (State Dict)。
        """
        save_dict = {
            'episode': episode,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            # 必须保存这个，否则连续动作的探索方差会丢失
            'power_log_std': getattr(agent, 'power_log_std', None),
            # 这里的属性名必须与 MAPPOAgent 中的 self.optimizer 一致
            'optimizer_state_dict': agent.optimizer.state_dict()
        }

        # 1. 保存 Best (覆盖)
        if is_best:
            torch.save(save_dict, os.path.join(self.model_dir, "best_model.pth"))

        # 2. 保存 Checkpoint (每 500 轮留底一个，用于回溯分析)
        if episode > 0 and episode % 500 == 0:
            torch.save(save_dict, os.path.join(self.model_dir, f"model_ep{episode}.pth"))

        # 3. 始终更新 Latest (覆盖)，用于断点续训
        torch.save(save_dict, os.path.join(self.model_dir, "latest_model.pth"))

    def auto_plot(self, baseline_results=None, baseline_history=None):
        """
        [可视化核心] 读取 CSV 并绘制全面的分析图表。
        
        Args:
            baseline_results: 基准策略结果字典（已废弃，保留用于兼容性）
            baseline_history: baseline历史记录字典 {'Random': [metrics_dict, ...], 'Local-Only': [...], 'Greedy': [...]}
        """
        if not os.path.exists(self.episode_log_path):
            print("[DataRecorder] No episode log found to plot.")
            return

        try:
            # 1. 读取数据
            df = pd.read_csv(self.episode_log_path)
            if df.empty: return
            
            # 分离训练数据和baseline数据
            if 'policy' in df.columns:
                df_train = df[df['policy'].isna() | (df['policy'] == '')].copy()
                df_baseline = df[df['policy'].notna() & (df['policy'] != '')].copy()
            else:
                df_train = df.copy()
                df_baseline = pd.DataFrame()

            # 2. 设置绘图风格 (兼容性设置)
            # 尝试使用 seaborn 风格，如果不可用则回退
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except OSError:
                plt.style.use('seaborn-whitegrid')  # 旧版本 matplotlib

            # 字体兼容性: 优先使用支持中文或通用符号的字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

            # 3. 辅助绘图函数 (带平滑处理和baseline时间序列)
            def save_plot(x, y_dict, title, ylabel, filename, window=20, baseline_dict=None, 
                         df_baseline=None, metric_key=None):
                """
                x: 横坐标数据
                y_dict: {Label: Data} 字典
                window: 滑动平均窗口大小
                baseline_dict: 基准策略结果字典（已废弃，保留用于兼容性）
                df_baseline: baseline数据DataFrame（包含policy列和metric_key列）
                metric_key: 要绘制的指标列名（如'total_reward'）
                """
                plt.figure(figsize=(12, 7))
                # 定义一组对比度高的颜色
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                for idx, (label, y) in enumerate(y_dict.items()):
                    if y is None or len(y) == 0: continue
                    color = colors[idx % len(colors)]

                    # 绘制原始数据的淡色背景
                    plt.plot(x, y, color=color, alpha=0.15)

                    # 绘制平滑曲线
                    if len(y) > window:
                        y_smooth = y.rolling(window=window, min_periods=1).mean()
                        plt.plot(x, y_smooth, color=color, linewidth=2.5, label=label, zorder=5)
                    else:
                        plt.plot(x, y, color=color, linewidth=2.5, label=label, zorder=5)
                
                # 绘制baseline时间序列（如果提供了数据）
                # 与训练曲线保持一致的绘图风格：平滑曲线，无marker
                if df_baseline is not None and metric_key is not None and not df_baseline.empty:
                    baseline_colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                    baseline_styles = {'Random': '--', 'Local-Only': '-.', 'Greedy': ':'}
                    baseline_linewidths = {'Random': 2.0, 'Local-Only': 2.0, 'Greedy': 2.0}
                    
                    for policy_name in ['Random', 'Local-Only', 'Greedy']:
                        policy_data = df_baseline[df_baseline['policy'] == policy_name].copy()
                        if not policy_data.empty and metric_key in policy_data.columns:
                            x_baseline = policy_data['episode'].values
                            y_baseline = policy_data[metric_key].values
                            
                            # 与训练曲线一致：应用相同的平滑处理
                            if window > 1 and len(y_baseline) > window:
                                y_smooth = pd.Series(y_baseline).rolling(window=window, min_periods=1).mean().values
                            else:
                                y_smooth = y_baseline
                            
                            color = baseline_colors.get(policy_name, '#7f8c8d')
                            style = baseline_styles.get(policy_name, '--')
                            linewidth = baseline_linewidths.get(policy_name, 2.0)
                            # 绘制平滑曲线，无marker，与训练曲线风格一致
                            plt.plot(x_baseline, y_smooth, color=color, linestyle=style, 
                                   linewidth=linewidth, label=f'{policy_name}', 
                                   alpha=0.85, zorder=4)
                
                # 兼容性：绘制水平线（如果提供了baseline_dict但没有df_baseline）
                elif baseline_dict is not None:
                    baseline_colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                    baseline_styles = {'Random': '--', 'Local-Only': '-.', 'Greedy': ':'}
                    for baseline_name, baseline_value in baseline_dict.items():
                        color = baseline_colors.get(baseline_name, '#7f8c8d')
                        style = baseline_styles.get(baseline_name, '--')
                        plt.axhline(y=baseline_value, color=color, linestyle=style, 
                                   linewidth=2, label=f'{baseline_name} Baseline', alpha=0.8)

                plt.title(title, fontsize=14, fontweight='bold')
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.legend(fontsize=10, loc='best', framealpha=0.9)
                plt.grid(True, linestyle='--', alpha=0.7)

                save_path = os.path.join(self.plot_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

            # 4. 生成各类图表
            episodes = df_train['episode']

            #
            # (A) 奖励曲线 (最核心指标) - 包含基准策略对比
            if 'total_reward' in df_train.columns:
                save_plot(episodes, {'MAPPO': df_train['total_reward']},
                          'Training Convergence - Reward Comparison with Baselines', 
                          'Average Reward', 'reward_curve_with_baselines.png',
                          baseline_dict=baseline_results, df_baseline=df_baseline, metric_key='total_reward')

            # (B) Loss 曲线
            if 'loss' in df_train.columns:
                save_plot(episodes, {'Loss': df_train['loss']},
                          'Training Loss (Actor+Critic)', 'Loss Value', 'loss_curve.png')

            # (C) 成功率 (%) - Veh%
            if 'veh_success_rate' in df_train.columns:
                df_baseline_scaled = df_baseline.copy()
                if 'veh_success_rate' in df_baseline_scaled.columns:
                    df_baseline_scaled['veh_success_rate'] = df_baseline_scaled['veh_success_rate'] * 100
                save_plot(episodes, {'MAPPO': df_train['veh_success_rate'] * 100},
                          'Vehicle Success Rate (DAG Completion %)', 'Rate (%)', 'veh_success_rate_with_baselines.png',
                          df_baseline=df_baseline_scaled, metric_key='veh_success_rate')
            
            # (C2) 子任务成功率 (%)
            if 'subtask_success_rate' in df_train.columns:
                df_baseline_scaled = df_baseline.copy()
                if 'subtask_success_rate' in df_baseline_scaled.columns:
                    df_baseline_scaled['subtask_success_rate'] = df_baseline_scaled['subtask_success_rate'] * 100
                save_plot(episodes, {'MAPPO': df_train['subtask_success_rate'] * 100},
                          'Subtask Success Rate (%)', 'Rate (%)', 'subtask_success_rate_with_baselines.png',
                          df_baseline=df_baseline_scaled, metric_key='subtask_success_rate')

            #
            # (D) 卸载决策分布 (%)
            offload_data = {}
            for k, label in zip(['decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v'], ['Local', 'RSU', 'V2V']):
                if k in df_train.columns:
                    offload_data[label] = df_train[k] * 100
            if offload_data:
                save_plot(episodes, offload_data, 'Offloading Decision Distribution', 'Proportion (%)',
                          'offloading_ratio.png')
                
                # 为每个baseline策略绘制卸载决策分布
                if not df_baseline.empty:
                    for policy_name in ['Random', 'Local-Only', 'Greedy']:
                        policy_data = df_baseline[df_baseline['policy'] == policy_name]
                        if not policy_data.empty:
                            baseline_offload = {}
                            for k, label in zip(['decision_frac_local', 'decision_frac_rsu', 'decision_frac_v2v'], ['Local', 'RSU', 'V2V']):
                                if k in policy_data.columns:
                                    baseline_offload[f'{policy_name} {label}'] = policy_data[k] * 100
                            if baseline_offload:
                                x_baseline = policy_data['episode']
                                save_plot(x_baseline, baseline_offload, 
                                         f'Offloading Decision Distribution - {policy_name}', 
                                         'Proportion (%)', f'offloading_ratio_{policy_name.lower().replace("-", "_")}.png')

            # (E) [多智能体] 公平性指标 (Jain's Fairness Index)
            if 'ma_fairness' in df_train.columns:
                save_plot(episodes, {'Fairness Index': df_train['ma_fairness']},
                          'System Fairness (Jain Index)', 'Index [0-1]', 'ma_fairness.png')

            # (F) [多智能体] 协作活跃度 (V2V Collaboration Rate)
            if 'ma_collaboration' in df_train.columns:
                save_plot(episodes, {'MAPPO': df_train['ma_collaboration']},
                          'Agent Collaboration (V2V) Rate', 'Rate (%)', 'ma_collaboration_with_baselines.png',
                          df_baseline=df_baseline, metric_key='ma_collaboration')

            # (G) [多智能体] 奖励差距 (Reward Gap)
            if 'ma_reward_gap' in df_train.columns:
                save_plot(episodes, {'Reward Gap': df_train['ma_reward_gap']},
                          'Individual Reward Gap (Max - Min)', 'Gap Value', 'ma_reward_gap.png')

            # (H) 队列拥堵情况
            queue_data = {}
            if 'avg_veh_queue' in df_train.columns: queue_data['Vehicle Avg'] = df_train['avg_veh_queue']
            if 'avg_rsu_queue' in df_train.columns: queue_data['RSU Avg'] = df_train['avg_rsu_queue']
            if queue_data:
                save_plot(episodes, queue_data, 'Average Queue Length', 'Num Tasks', 'queue_len.png')

            # (I) 算力分配效率
            if 'avg_assigned_cpu_ghz' in df_train.columns:
                save_plot(episodes, {'Assigned CPU': df_train['avg_assigned_cpu_ghz']},
                          'Average Assigned Computing Power', 'GHz', 'cpu_efficiency.png')

            # (J) [新增] 智能体个体奖励分布分析
            self.plot_agent_reward_distribution()
            
            # (K) [新增] 高级分析图表
            self.plot_latency_energy_tradeoff(df_train, df_baseline)
            self.plot_performance_radar(df_train, df_baseline)
            self.plot_resource_utilization(df_train)
            self.plot_training_stability(df_train)
            self.plot_completion_time_distribution(df_train, df_baseline)
            self.plot_rsu_load_balance(df_train)
            self.plot_episode_duration_analysis(df_train, df_baseline)
            self.plot_reward_decomposition(df_train)
            self.plot_success_rate_comparison(df_train, df_baseline)

            print(f"[DataRecorder] All plots generated in: {self.plot_dir}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error] Failed to plot data: {e}")

    def plot_agent_reward_distribution(self):
        """
        绘制智能体奖励分布。
        优化: 避免使用 pd.read_csv 读取整个几 GB 的 step_log.csv。
        """
        if not os.path.exists(self.step_log_path):
            return

        try:
            # 方案: 只读取最后 N 行 (Tail)，而不是全部读取
            # 估算: 20 agents * 100 steps * 20 episodes = 40000 行
            # 使用 chunksize 分块读取，只保留最后一块

            chunk_size = 50000
            last_chunk = None

            # 分块读取，迭代到最后
            for chunk in pd.read_csv(self.step_log_path, chunksize=chunk_size):
                last_chunk = chunk

            if last_chunk is None or last_chunk.empty:
                return

            df = last_chunk

            # 再次筛选，确保只取最后 20 个 Episode (因为 chunk 可能包含更多)
            last_eps = df['episode'].unique()[-20:]
            df_recent = df[df['episode'].isin(last_eps)]

            if df_recent.empty: return

            # 确保数据类型正确
            df_recent.loc[:, 'reward'] = pd.to_numeric(df_recent['reward'], errors='coerce')

            # 绘图逻辑不变
            agent_rewards = df_recent.groupby(['episode', 'veh_id'])['reward'].sum().unstack()

            plt.figure(figsize=(12, 6))
            agent_rewards.boxplot()
            plt.title('Agent Reward Distribution (Last 20 Eps)', fontsize=14)
            plt.xlabel('Vehicle ID')
            plt.ylabel('Reward')
            plt.grid(True, linestyle='--', alpha=0.5)

            plt.savefig(os.path.join(self.plot_dir, 'agent_reward_boxplot.png'), dpi=100)
            plt.close()

        except Exception as e:
            print(f"[Warning] Agent distribution plot skipped due to error: {e}")
    
    def plot_latency_energy_tradeoff(self, df_train, df_baseline):
        """绘制时延-能耗权衡图（Pareto前沿分析）"""
        try:
            # 检查必要列
            if 'avg_latency' not in df_train.columns or 'avg_energy' not in df_train.columns:
                return
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 训练数据：取最后50个episode
            df_recent = df_train.tail(50)
            ax.scatter(df_recent['avg_latency'], df_recent['avg_energy'], 
                      c=df_recent['episode'], cmap='viridis', s=100, alpha=0.6,
                      label='MAPPO (Recent 50 Eps)', edgecolors='black', linewidth=0.5)
            
            # 添加colorbar
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Episode', rotation=270, labelpad=20)
            
            # Baseline数据
            if df_baseline is not None and not df_baseline.empty:
                baseline_colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                baseline_markers = {'Random': 'x', 'Local-Only': 's', 'Greedy': '^'}
                
                for policy_name in ['Random', 'Local-Only', 'Greedy']:
                    policy_data = df_baseline[df_baseline['policy'] == policy_name]
                    if not policy_data.empty and 'avg_latency' in policy_data.columns:
                        avg_lat = policy_data['avg_latency'].mean()
                        avg_eng = policy_data['avg_energy'].mean()
                        ax.scatter(avg_lat, avg_eng, 
                                 marker=baseline_markers.get(policy_name, 'o'),
                                 s=300, color=baseline_colors.get(policy_name, '#7f8c8d'),
                                 label=f'{policy_name}', edgecolors='black', linewidth=2,
                                 zorder=10)
            
            ax.set_xlabel('Average Latency (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Energy (J)', fontsize=12, fontweight='bold')
            ax.set_title('Latency-Energy Tradeoff Analysis', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'latency_energy_tradeoff.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Latency-energy tradeoff plot skipped: {e}")
    
    def plot_performance_radar(self, df_train, df_baseline):
        """绘制多维性能雷达图"""
        try:
            from math import pi
            
            # 准备数据：取最后20个episode的平均值
            df_recent = df_train.tail(20)
            
            metrics = {
                'Task Success\nRate': df_recent['veh_success_rate'].mean() if 'veh_success_rate' in df_recent.columns else 0,
                'Subtask Success\nRate': df_recent['subtask_success_rate'].mean() if 'subtask_success_rate' in df_recent.columns else 0,
                'Avg Reward\n(Normalized)': (df_recent['total_reward'].mean() + 10) / 20 if 'total_reward' in df_recent.columns else 0,  # 归一化到[0,1]
                'Resource\nUtilization': df_recent['avg_assigned_cpu_ghz'].mean() / 12 if 'avg_assigned_cpu_ghz' in df_recent.columns else 0,  # 假设最大12GHz
                'Queue\nEfficiency': max(0, 1 - df_recent['avg_rsu_queue'].mean() / 10) if 'avg_rsu_queue' in df_recent.columns else 0,  # 反向指标
            }
            
            # 确保所有值在[0,1]范围内
            metrics = {k: max(0, min(1, v)) for k, v in metrics.items()}
            
            categories = list(metrics.keys())
            values = list(metrics.values())
            N = len(categories)
            
            # 计算角度
            angles = [n / float(N) * 2 * pi for n in range(N)]
            values += values[:1]  # 闭合
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 绘制MAPPO
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4', label='MAPPO')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            
            # 绘制Baseline（如果有）
            if df_baseline is not None and not df_baseline.empty:
                baseline_colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                for policy_name in ['Random', 'Local-Only', 'Greedy']:
                    policy_data = df_baseline[df_baseline['policy'] == policy_name]
                    if not policy_data.empty:
                        baseline_values = [
                            policy_data['veh_success_rate'].mean() if 'veh_success_rate' in policy_data.columns else 0,
                            policy_data['subtask_success_rate'].mean() if 'subtask_success_rate' in policy_data.columns else 0,
                            (policy_data['total_reward'].mean() + 10) / 20 if 'total_reward' in policy_data.columns else 0,
                            policy_data['avg_assigned_cpu_ghz'].mean() / 12 if 'avg_assigned_cpu_ghz' in policy_data.columns else 0,
                            max(0, 1 - policy_data['avg_rsu_queue'].mean() / 10) if 'avg_rsu_queue' in policy_data.columns else 0,
                        ]
                        baseline_values = [max(0, min(1, v)) for v in baseline_values]
                        baseline_values += baseline_values[:1]
                        ax.plot(angles, baseline_values, 'o--', linewidth=1.5, 
                               color=baseline_colors.get(policy_name), label=policy_name, alpha=0.7)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
            ax.set_title('Multi-Dimensional Performance Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Performance radar plot skipped: {e}")
    
    def plot_resource_utilization(self, df_train):
        """绘制资源利用率时序图"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            episodes = df_train['episode']
            window = 20
            
            # CPU利用率
            if 'avg_assigned_cpu_ghz' in df_train.columns:
                ax = axes[0]
                ax.plot(episodes, df_train['avg_assigned_cpu_ghz'], alpha=0.3, color='steelblue')
                ax.plot(episodes, df_train['avg_assigned_cpu_ghz'].rolling(window, min_periods=1).mean(),
                       linewidth=2, color='darkblue', label='CPU Utilization')
                ax.set_ylabel('CPU (GHz)', fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
                ax.set_title('Resource Utilization Over Training', fontsize=14, fontweight='bold')
            
            # RSU队列长度
            if 'avg_rsu_queue' in df_train.columns:
                ax = axes[1]
                ax.plot(episodes, df_train['avg_rsu_queue'], alpha=0.3, color='orange')
                ax.plot(episodes, df_train['avg_rsu_queue'].rolling(window, min_periods=1).mean(),
                       linewidth=2, color='darkorange', label='RSU Queue Length')
                ax.set_ylabel('Queue Length', fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
            
            # Vehicle队列长度
            if 'avg_veh_queue' in df_train.columns:
                ax = axes[2]
                ax.plot(episodes, df_train['avg_veh_queue'], alpha=0.3, color='green')
                ax.plot(episodes, df_train['avg_veh_queue'].rolling(window, min_periods=1).mean(),
                       linewidth=2, color='darkgreen', label='Vehicle Queue Length')
                ax.set_xlabel('Episode', fontweight='bold')
                ax.set_ylabel('Queue Length', fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'resource_utilization.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Resource utilization plot skipped: {e}")
    
    def plot_training_stability(self, df_train):
        """绘制训练稳定性指标（方差、波动率）"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            window = 50
            
            # Reward标准差（滚动窗口）
            if 'total_reward' in df_train.columns:
                ax = axes[0]
                reward_std = df_train['total_reward'].rolling(window, min_periods=1).std()
                ax.plot(df_train['episode'], reward_std, linewidth=2, color='purple', label='Reward Std Dev')
                ax.set_ylabel('Reward Std Dev', fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
                ax.set_title('Training Stability Analysis', fontsize=14, fontweight='bold')
            
            # Success Rate方差
            if 'veh_success_rate' in df_train.columns:
                ax = axes[1]
                sr_std = df_train['veh_success_rate'].rolling(window, min_periods=1).std()
                ax.plot(df_train['episode'], sr_std, linewidth=2, color='teal', label='Success Rate Std Dev')
                ax.set_xlabel('Episode', fontweight='bold')
                ax.set_ylabel('Success Rate Std Dev', fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Training stability plot skipped: {e}")
    
    def plot_completion_time_distribution(self, df_train, df_baseline):
        """绘制任务完成时间分布（CDF）"""
        try:
            if 'avg_completion_time' not in df_train.columns:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # MAPPO的CDF（最后50个episode）
            df_recent = df_train.tail(50)
            completion_times = df_recent['avg_completion_time'].dropna().sort_values()
            cdf = np.arange(1, len(completion_times) + 1) / len(completion_times)
            ax.plot(completion_times, cdf, linewidth=2.5, color='#1f77b4', label='MAPPO', marker='o', markersize=3)
            
            # Baseline的CDF
            if df_baseline is not None and not df_baseline.empty:
                baseline_colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                baseline_styles = {'Random': '--', 'Local-Only': '-.', 'Greedy': ':'}
                
                for policy_name in ['Random', 'Local-Only', 'Greedy']:
                    policy_data = df_baseline[df_baseline['policy'] == policy_name]
                    if not policy_data.empty and 'avg_completion_time' in policy_data.columns:
                        times = policy_data['avg_completion_time'].dropna().sort_values()
                        cdf_bl = np.arange(1, len(times) + 1) / len(times)
                        ax.plot(times, cdf_bl, linewidth=2, 
                               color=baseline_colors.get(policy_name),
                               linestyle=baseline_styles.get(policy_name),
                               label=policy_name, marker='s', markersize=3, alpha=0.8)
            
            ax.set_xlabel('Task Completion Time (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('CDF', fontsize=12, fontweight='bold')
            ax.set_title('Task Completion Time Distribution (CDF)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'completion_time_cdf.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Completion time CDF plot skipped: {e}")
    
    def plot_rsu_load_balance(self, df_train):
        """绘制RSU负载均衡分析"""
        try:
            # 使用avg_queue_len列（实际可用的列）
            if 'avg_queue_len' not in df_train.columns:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            window = 20
            episodes = df_train['episode']
            queue_smooth = df_train['avg_queue_len'].rolling(window, min_periods=1).mean()
            
            ax.fill_between(episodes, 0, queue_smooth, alpha=0.3, color='orange', label='Avg Queue Length')
            ax.plot(episodes, queue_smooth, linewidth=2, color='darkorange')
            
            # 添加负载阈值线
            ax.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='High Load Threshold', alpha=0.7)
            ax.axhline(y=2, color='green', linestyle='--', linewidth=1.5, label='Low Load Threshold', alpha=0.7)
            
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Queue Length', fontsize=12, fontweight='bold')
            ax.set_title('Queue Load Balance Analysis', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'queue_load_balance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Queue load balance plot skipped: {e}")
    
    def plot_episode_duration_analysis(self, df_train, df_baseline):
        """绘制Episode时长分析"""
        try:
            if 'duration' not in df_train.columns:
                return
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 时长时序图
            ax = axes[0]
            episodes = df_train['episode']
            window = 20
            ax.plot(episodes, df_train['duration'], alpha=0.3, color='steelblue', label='Raw')
            ax.plot(episodes, df_train['duration'].rolling(window, min_periods=1).mean(),
                   linewidth=2, color='darkblue', label=f'Smoothed ({window}-ep)')
            ax.set_ylabel('Episode Duration (s)', fontweight='bold')
            ax.set_title('Episode Duration Analysis', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5)
            
            # 时长分布直方图
            ax = axes[1]
            ax.hist(df_train['duration'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(df_train['duration'].mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {df_train["duration"].mean():.2f}s')
            ax.axvline(df_train['duration'].median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {df_train["duration"].median():.2f}s')
            ax.set_xlabel('Duration (s)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'episode_duration.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Episode duration plot skipped: {e}")
    
    def plot_reward_decomposition(self, df_train):
        """绘制Reward组成分解（与其他指标的相关性）"""
        try:
            if 'total_reward' not in df_train.columns:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Reward vs Success Rate
            if 'veh_success_rate' in df_train.columns:
                ax = axes[0, 0]
                scatter = ax.scatter(df_train['veh_success_rate'] * 100, df_train['total_reward'],
                                   c=df_train['episode'], cmap='viridis', s=50, alpha=0.6)
                ax.set_xlabel('Vehicle Success Rate (%)', fontweight='bold')
                ax.set_ylabel('Total Reward', fontweight='bold')
                ax.set_title('Reward vs Success Rate', fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Episode')
                ax.grid(True, alpha=0.3)
            
            # Reward vs Decision Distribution
            if 'decision_frac_rsu' in df_train.columns:
                ax = axes[0, 1]
                scatter = ax.scatter(df_train['decision_frac_rsu'] * 100, df_train['total_reward'],
                                   c=df_train['episode'], cmap='plasma', s=50, alpha=0.6)
                ax.set_xlabel('RSU Offloading Ratio (%)', fontweight='bold')
                ax.set_ylabel('Total Reward', fontweight='bold')
                ax.set_title('Reward vs RSU Usage', fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Episode')
                ax.grid(True, alpha=0.3)
            
            # Reward vs Queue Length
            if 'avg_queue_len' in df_train.columns:
                ax = axes[1, 0]
                scatter = ax.scatter(df_train['avg_queue_len'], df_train['total_reward'],
                                   c=df_train['episode'], cmap='coolwarm', s=50, alpha=0.6)
                ax.set_xlabel('Average Queue Length', fontweight='bold')
                ax.set_ylabel('Total Reward', fontweight='bold')
                ax.set_title('Reward vs Queue Congestion', fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Episode')
                ax.grid(True, alpha=0.3)
            
            # Reward vs CPU Allocation
            if 'avg_assigned_cpu_ghz' in df_train.columns:
                ax = axes[1, 1]
                scatter = ax.scatter(df_train['avg_assigned_cpu_ghz'], df_train['total_reward'],
                                   c=df_train['episode'], cmap='viridis', s=50, alpha=0.6)
                ax.set_xlabel('Avg Assigned CPU (GHz)', fontweight='bold')
                ax.set_ylabel('Total Reward', fontweight='bold')
                ax.set_title('Reward vs Resource Allocation', fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Episode')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'reward_decomposition.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Reward decomposition plot skipped: {e}")
    
    def plot_success_rate_comparison(self, df_train, df_baseline):
        """绘制多维成功率对比（任务/子任务/V2V）"""
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            episodes = df_train['episode']
            window = 20
            
            # Task Success Rate
            if 'task_success_rate' in df_train.columns:
                task_sr_smooth = df_train['task_success_rate'].rolling(window, min_periods=1).mean() * 100
                ax.plot(episodes, task_sr_smooth, linewidth=2.5, color='#1f77b4', 
                       label='Task Success Rate', marker='o', markersize=2, markevery=10)
            
            # Subtask Success Rate
            if 'subtask_success_rate' in df_train.columns:
                subtask_sr_smooth = df_train['subtask_success_rate'].rolling(window, min_periods=1).mean() * 100
                ax.plot(episodes, subtask_sr_smooth, linewidth=2.5, color='#ff7f0e',
                       label='Subtask Success Rate', marker='s', markersize=2, markevery=10)
            
            # V2V Subtask Success Rate
            if 'v2v_subtask_success_rate' in df_train.columns:
                v2v_sr_smooth = df_train['v2v_subtask_success_rate'].rolling(window, min_periods=1).mean() * 100
                ax.plot(episodes, v2v_sr_smooth, linewidth=2.5, color='#2ca02c',
                       label='V2V Subtask Success Rate', marker='^', markersize=2, markevery=10)
            
            # 添加目标线
            ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (80%)')
            
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
            ax.set_title('Multi-Level Success Rate Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim([0, 105])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'success_rate_multilevel.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[Warning] Success rate comparison plot skipped: {e}")

    def plot_training_stats(self, training_stats_csv, baseline_csv=None):
        """
        从training_stats.csv绘制完整的训练分析图表

        Args:
            training_stats_csv: 训练统计CSV文件路径
            baseline_csv: baseline统计CSV文件路径（可选）
        """
        if not os.path.exists(training_stats_csv):
            print(f"[DataRecorder] training_stats.csv not found: {training_stats_csv}")
            return

        try:
            df = pd.read_csv(training_stats_csv)
            if df.empty:
                return

            # 读取baseline数据（如果存在），并扩展为完整曲线
            df_baseline = None
            if baseline_csv and os.path.exists(baseline_csv):
                df_baseline_raw = pd.read_csv(baseline_csv)
                # 将baseline扩展为完整episode范围（使用forward fill插值）
                if not df_baseline_raw.empty:
                    max_ep = df['episode'].max()
                    expanded_rows = []
                    for policy in df_baseline_raw['policy'].unique():
                        policy_data = df_baseline_raw[df_baseline_raw['policy'] == policy].copy()
                        policy_data = policy_data.set_index('episode')
                        # 创建完整episode范围的索引
                        full_idx = pd.Index(range(1, max_ep + 1), name='episode')
                        # 重新索引并forward fill
                        policy_expanded = policy_data.reindex(full_idx).ffill().bfill()
                        policy_expanded['policy'] = policy
                        policy_expanded = policy_expanded.reset_index()
                        expanded_rows.append(policy_expanded)
                    df_baseline = pd.concat(expanded_rows, ignore_index=True)

            # 设置绘图风格
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except OSError:
                plt.style.use('seaborn-whitegrid')
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

            window = 20  # 平滑窗口
            episodes = df['episode']

            # ================================================================
            # 1. 综合训练曲线（Reward + Success Rate + Baseline对比）
            # ================================================================
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1.1 Reward曲线
            ax = axes[0, 0]
            if 'reward_mean' in df.columns:
                ax.plot(episodes, df['reward_mean'], alpha=0.2, color='steelblue')
                ax.plot(episodes, df['reward_mean'].rolling(window, min_periods=1).mean(),
                       linewidth=2.5, color='darkblue', label='MAPPO (Mean/Step)')
            if df_baseline is not None and 'reward_mean' in df_baseline.columns:
                for policy in ['Random', 'Local-Only', 'Greedy']:
                    policy_data = df_baseline[df_baseline['policy'] == policy].sort_values('episode')
                    if not policy_data.empty:
                        colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                        styles = {'Random': '--', 'Local-Only': '-.', 'Greedy': ':'}
                        # 使用drawstyle='steps-post'绘制阶梯曲线，确保baseline完整显示
                        ax.plot(policy_data['episode'], policy_data['reward_mean'],
                               color=colors.get(policy, 'gray'), linestyle=styles.get(policy, '--'),
                               linewidth=2, label=f'{policy}', alpha=0.8, drawstyle='steps-post')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward (per step)')
            ax.set_title('Reward Convergence with Baseline Comparison', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5)

            # 1.2 成功率曲线
            ax = axes[0, 1]
            sr_cols = {'vehicle_sr': ('V_SR (DAG)', '#1f77b4'),
                      'task_sr': ('T_SR', '#ff7f0e'),
                      'subtask_sr': ('S_SR', '#2ca02c')}
            for col, (label, color) in sr_cols.items():
                if col in df.columns:
                    ax.plot(episodes, df[col].rolling(window, min_periods=1).mean() * 100,
                           linewidth=2, color=color, label=label)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Success Rate Curves', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5)
            ax.set_ylim([0, 105])

            # 1.3 卸载决策分布
            ax = axes[1, 0]
            if 'ratio_local' in df.columns:
                ax.stackplot(episodes,
                            df['ratio_local'].rolling(window, min_periods=1).mean() * 100,
                            df['ratio_rsu'].rolling(window, min_periods=1).mean() * 100,
                            df['ratio_v2v'].rolling(window, min_periods=1).mean() * 100,
                            labels=['Local', 'RSU', 'V2V'],
                            colors=['#2ca02c', '#1f77b4', '#ff7f0e'],
                            alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Ratio (%)')
            ax.set_title('Offloading Decision Distribution', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.5)
            ax.set_ylim([0, 100])

            # 1.4 训练诊断（Loss + Entropy）
            ax = axes[1, 1]
            ax2 = ax.twinx()
            if 'actor_loss' in df.columns:
                valid_loss = df['actor_loss'].dropna()
                if len(valid_loss) > 0:
                    ax.plot(df.loc[valid_loss.index, 'episode'], valid_loss,
                           linewidth=1.5, color='red', alpha=0.7, label='Actor Loss')
            if 'critic_loss' in df.columns:
                valid_loss = df['critic_loss'].dropna()
                if len(valid_loss) > 0:
                    ax.plot(df.loc[valid_loss.index, 'episode'], valid_loss,
                           linewidth=1.5, color='blue', alpha=0.7, label='Critic Loss')
            if 'entropy' in df.columns:
                valid_ent = df['entropy'].dropna()
                if len(valid_ent) > 0:
                    ax2.plot(df.loc[valid_ent.index, 'episode'], valid_ent,
                            linewidth=2, color='green', label='Entropy')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss', color='red')
            ax2.set_ylabel('Entropy', color='green')
            ax.set_title('Training Diagnostics', fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            ax.grid(True, alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # ================================================================
            # 2. 物理性能指标
            # ================================================================
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 2.1 任务完成时间
            ax = axes[0, 0]
            if 'task_duration_mean' in df.columns:
                ax.plot(episodes, df['task_duration_mean'], alpha=0.3, color='steelblue')
                ax.plot(episodes, df['task_duration_mean'].rolling(window, min_periods=1).mean(),
                       linewidth=2.5, color='darkblue', label='Mean Duration')
            if 'task_duration_p95' in df.columns:
                ax.plot(episodes, df['task_duration_p95'].rolling(window, min_periods=1).mean(),
                       linewidth=2, color='orange', linestyle='--', label='P95 Duration')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Task Duration (s)')
            ax.set_title('Task Completion Time', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5)

            # 2.2 能耗
            ax = axes[0, 1]
            if 'energy_mean' in df.columns:
                ax.plot(episodes, df['energy_mean'], alpha=0.3, color='red')
                ax.plot(episodes, df['energy_mean'].rolling(window, min_periods=1).mean(),
                       linewidth=2.5, color='darkred', label='Energy (Normalized)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Energy')
            ax.set_title('Energy Consumption', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5)

            # 2.3 服务率和空闲率
            ax = axes[1, 0]
            if 'service_rate_ghz' in df.columns:
                ax.plot(episodes, df['service_rate_ghz'].rolling(window, min_periods=1).mean(),
                       linewidth=2.5, color='green', label='Service Rate (GHz)')
            ax2 = ax.twinx()
            if 'idle_fraction' in df.columns:
                ax2.plot(episodes, df['idle_fraction'].rolling(window, min_periods=1).mean() * 100,
                        linewidth=2, color='orange', linestyle='--', label='Idle Fraction (%)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Service Rate (GHz)', color='green')
            ax2.set_ylabel('Idle Fraction (%)', color='orange')
            ax.set_title('Resource Utilization', fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            ax.grid(True, alpha=0.5)

            # 2.4 Deadline Miss和传输统计
            ax = axes[1, 1]
            if 'deadline_misses' in df.columns:
                ax.bar(episodes, df['deadline_misses'], alpha=0.5, color='red', label='Deadline Misses')
            ax2 = ax.twinx()
            if 'tx_created' in df.columns:
                ax2.plot(episodes, df['tx_created'].rolling(window, min_periods=1).mean(),
                        linewidth=2, color='blue', label='TX Created')
            if 'same_node_no_tx' in df.columns:
                ax2.plot(episodes, df['same_node_no_tx'].rolling(window, min_periods=1).mean(),
                        linewidth=2, color='green', linestyle='--', label='Same Node (No TX)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Deadline Misses', color='red')
            ax2.set_ylabel('Count', color='blue')
            ax.set_title('Deadline and Transmission Statistics', fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            ax.grid(True, alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'physical_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # ================================================================
            # 3. Bias退火曲线
            # ================================================================
            if 'bias_rsu' in df.columns and 'bias_local' in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(episodes, df['bias_rsu'], linewidth=2.5, color='blue', label='Bias_RSU')
                ax.plot(episodes, df['bias_local'], linewidth=2.5, color='green', label='Bias_Local')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Bias Value')
                ax.set_title('Logit Bias Annealing', fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_dir, 'bias_annealing.png'), dpi=300, bbox_inches='tight')
                plt.close()

            print(f"[DataRecorder] Training stats plots saved to: {self.plot_dir}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error] Failed to plot training stats: {e}")

    def close(self):
        """
        关闭 TensorBoard Writer，确保所有缓冲数据都已写入磁盘。
        """
        if hasattr(self, 'writer'):
            self.writer.close()
