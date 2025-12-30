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

    def __init__(self, experiment_name="Default_Exp"):
        """
        初始化文件结构
        """
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
            if os.path.exists('/root') and os.access('/root', os.W_OK):
                tb_log_dir = f"/root/tf-logs/{experiment_name}_{timestamp}"
            else:
                # 本地环境使用项目目录
                tb_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", f"{experiment_name}_{timestamp}")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
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
                if df_baseline is not None and metric_key is not None and not df_baseline.empty:
                    baseline_colors = {'Random': '#e74c3c', 'Local-Only': '#95a5a6', 'Greedy': '#f39c12'}
                    baseline_styles = {'Random': '--', 'Local-Only': '-.', 'Greedy': ':'}
                    baseline_linewidths = {'Random': 2.0, 'Local-Only': 2.0, 'Greedy': 2.0}
                    
                    for policy_name in ['Random', 'Local-Only', 'Greedy']:
                        policy_data = df_baseline[df_baseline['policy'] == policy_name]
                        if not policy_data.empty and metric_key in policy_data.columns:
                            x_baseline = policy_data['episode']
                            y_baseline = policy_data[metric_key]
                            color = baseline_colors.get(policy_name, '#7f8c8d')
                            style = baseline_styles.get(policy_name, '--')
                            linewidth = baseline_linewidths.get(policy_name, 2.0)
                            plt.plot(x_baseline, y_baseline, color=color, linestyle=style, 
                                   linewidth=linewidth, label=f'{policy_name} Baseline', 
                                   alpha=0.8, zorder=4, marker='o', markersize=4)
                
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

    def close(self):
        """
        关闭 TensorBoard Writer，确保所有缓冲数据都已写入磁盘。
        """
        if hasattr(self, 'writer'):
            self.writer.close()
