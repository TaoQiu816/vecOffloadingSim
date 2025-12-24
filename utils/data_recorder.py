import os
import json
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


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

        # 2. 创建带时间戳的实验目录 (e.g., data/MAPPO_N12-16_Veh20_2023-10-01_12-00-00)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.exp_dir = os.path.join("./data", f"{experiment_name}_{timestamp}")

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # 3. 创建子目录
        self.model_dir = os.path.join(self.exp_dir, "models")  # 存放模型权重
        self.plot_dir = os.path.join(self.exp_dir, "plots")  # 存放生成的曲线图

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        print(f"[DataRecorder] Experiment data will be saved to: {self.exp_dir}")

        # 4. 初始化 CSV 文件路径
        self.step_log_path = os.path.join(self.exp_dir, "step_log.csv")  # 详细到每一步的日志
        self.episode_log_path = os.path.join(self.exp_dir, "episode_log.csv")  # 每个 Episode 的汇总日志

        # 标记是否已经写入了 CSV 表头 (避免追加模式下重复写入表头)
        self.step_header_written = False
        self.episode_header_written = False

    def save_config(self, config_dict):
        """
        保存超参数配置到 config.json，便于实验复现。
        """
        config_path = os.path.join(self.exp_dir, "config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            print(f"[DataRecorder] Config saved to {config_path}")
        except Exception as e:
            print(f"[Error] Failed to save config: {e}")

    def log_step(self, step_data_list):
        """
        批量写入 Step 级日志 (train.py 中每个 Episode 结束时调用一次)
        """
        if not step_data_list:
            return

        keys = step_data_list[0].keys()
        file_exists = os.path.isfile(self.step_log_path)

        try:
            with open(self.step_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists or not self.step_header_written:
                    writer.writeheader()
                    self.step_header_written = True
                writer.writerows(step_data_list)
        except Exception as e:
            print(f"[Error] Failed to log step data: {e}")

    def log_episode(self, episode_data):
        """
        写入 Episode 级汇总日志
        """
        if not episode_data:
            return

        keys = episode_data.keys()
        file_exists = os.path.isfile(self.episode_log_path)

        try:
            with open(self.episode_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists or not self.episode_header_written:
                    writer.writeheader()
                    self.episode_header_written = True
                writer.writerow(episode_data)
        except Exception as e:
            print(f"[Error] Failed to log episode data: {e}")

    def save_model(self, agent, episode, is_best=False):
        """
        保存模型权重 (Checkpoint)
        [关键] 包含了 power_log_std 的保存，这是连续动作的关键参数。
        """
        name = "best_model.pth" if is_best else f"model_ep{episode}.pth"
        save_path = os.path.join(self.model_dir, name)

        try:
            checkpoint = {
                'episode': episode,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'power_log_std': agent.power_log_std.data  # 保存可学习的方差参数
            }
            torch.save(checkpoint, save_path)
        except Exception as e:
            print(f"[Error] Failed to save model: {e}")

    def auto_plot(self):
        """
        [可视化核心] 读取 CSV 并绘制全面的分析图表。
        """
        if not os.path.exists(self.episode_log_path):
            print("[DataRecorder] No episode log found to plot.")
            return

        try:
            # 1. 读取数据
            df = pd.read_csv(self.episode_log_path)
            if df.empty: return

            # 2. 设置绘图风格 (兼容性设置)
            # 尝试使用 seaborn 风格，如果不可用则回退
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except OSError:
                plt.style.use('seaborn-whitegrid')  # 旧版本 matplotlib

            # 字体兼容性: 优先使用支持中文或通用符号的字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

            # 3. 辅助绘图函数 (带平滑处理)
            def save_plot(x, y_dict, title, ylabel, filename, window=20):
                """
                x: 横坐标数据
                y_dict: {Label: Data} 字典
                window: 滑动平均窗口大小
                """
                plt.figure(figsize=(10, 6))
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
                        plt.plot(x, y_smooth, color=color, linewidth=2, label=label)
                    else:
                        plt.plot(x, y, color=color, linewidth=2, label=label)

                plt.title(title, fontsize=14, fontweight='bold')
                plt.xlabel('Episode', fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.legend(fontsize=10, loc='best')
                plt.grid(True, linestyle='--', alpha=0.7)

                save_path = os.path.join(self.plot_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

            # 4. 生成各类图表
            episodes = df['episode']

            #
            # (A) 奖励曲线 (最核心指标)
            if 'total_reward' in df.columns:
                save_plot(episodes, {'Total Reward': df['total_reward']},
                          'Total Reward per Episode', 'Reward', 'reward_curve.png')

            # (B) Loss 曲线
            if 'loss' in df.columns:
                save_plot(episodes, {'Loss': df['loss']},
                          'Training Loss (Actor+Critic)', 'Loss Value', 'loss_curve.png')

            # (C) 成功率 (%)
            if 'success_rate' in df.columns:
                save_plot(episodes, {'Success Rate': df['success_rate']},
                          'System Success Rate (%)', 'Rate (%)', 'success_rate.png')

            #
            # (D) 卸载决策分布 (%)
            offload_data = {}
            for k, label in zip(['pct_local', 'pct_rsu', 'pct_v2v'], ['Local', 'RSU', 'V2V']):
                if k in df.columns: offload_data[label] = df[k]
            if offload_data:
                save_plot(episodes, offload_data, 'Offloading Decision Distribution', 'Proportion (%)',
                          'offloading_ratio.png')

            # (E) [多智能体] 公平性指标 (Jain's Fairness Index)
            if 'ma_fairness' in df.columns:
                save_plot(episodes, {'Fairness Index': df['ma_fairness']},
                          'System Fairness (Jain Index)', 'Index [0-1]', 'ma_fairness.png')

            # (F) [多智能体] 协作活跃度 (V2V Collaboration Rate)
            if 'ma_collaboration' in df.columns:
                save_plot(episodes, {'Collaboration Rate': df['ma_collaboration']},
                          'Agent Collaboration (V2V) Rate', 'Rate (%)', 'ma_collaboration.png')

            # (G) [多智能体] 奖励差距 (Reward Gap)
            if 'ma_reward_gap' in df.columns:
                save_plot(episodes, {'Reward Gap': df['ma_reward_gap']},
                          'Individual Reward Gap (Max - Min)', 'Gap Value', 'ma_reward_gap.png')

            # (H) 队列拥堵情况
            queue_data = {}
            if 'avg_veh_queue' in df.columns: queue_data['Vehicle Avg'] = df['avg_veh_queue']
            if 'avg_rsu_queue' in df.columns: queue_data['RSU Avg'] = df['avg_rsu_queue']
            if queue_data:
                save_plot(episodes, queue_data, 'Average Queue Length', 'Num Tasks', 'queue_len.png')

            # (I) 算力分配效率
            if 'avg_assigned_cpu_ghz' in df.columns:
                save_plot(episodes, {'Assigned CPU': df['avg_assigned_cpu_ghz']},
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
        [多智能体分析] 绘制每个智能体的累积奖励分布 (Boxplot)。
        用于观察是否存在某些车辆因为位置偏僻或算力低而长期获得低分。
        """
        if not os.path.exists(self.step_log_path):
            return

        try:
            # 读取 step log (只取最后 20 个 episode 进行统计，避免数据量过大)
            # 优化: 使用 chunksize 读取或者只读部分行，这里简单起见读全部然后 filter
            df = pd.read_csv(self.step_log_path)

            # [鲁棒性] 确保 reward 是数字类型 (csv 中可能是 string)
            df['reward'] = pd.to_numeric(df['reward'], errors='coerce')

            # 筛选最后 20 个 Episode
            last_eps = df['episode'].unique()[-20:]
            df_recent = df[df['episode'].isin(last_eps)]

            # 按 Episode 和 Veh_ID 分组求奖励和 (Sum Reward per Episode per Agent)
            agent_rewards = df_recent.groupby(['episode', 'veh_id'])['reward'].sum().unstack()

            # 绘图
            plt.figure(figsize=(12, 6))
            # Boxplot 能显示中位数、四分位数和异常值
            agent_rewards.boxplot()

            plt.title('Individual Agent Reward Distribution (Last 20 Episodes)', fontsize=14, fontweight='bold')
            plt.xlabel('Vehicle ID', fontsize=12)
            plt.ylabel('Episode Total Reward', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            save_path = os.path.join(self.plot_dir, 'agent_reward_boxplot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[DataRecorder] Agent distribution plot saved.")

        except Exception as e:
            # 这里的报错不应中断主流程
            print(f"[Warning] Could not generate agent distribution plot: {e}")