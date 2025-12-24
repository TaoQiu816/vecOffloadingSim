import time
import numpy as np
import torch
import os
import sys

# --- 导入自定义模块 ---
# 将当前目录添加到系统路径，防止在不同目录下运行时的导入错误
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC  # [引入训练配置]
from envs.vec_offloading_env import VecOffloadingEnv
from agents.mappo_agent import MAPPOAgent
from agents.buffer import RolloutBuffer
from utils.data_recorder import DataRecorder
from utils.graph_builder import GraphBuilder


def main():
    # 开启 CuDNN 加速，提升卷积/矩阵运算效率
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # =========================================================================
    # 1. 初始化配置与日志记录器 (Config & Logger Init)
    # =========================================================================
    # 实验命名: 包含 DAG 节点数范围和车辆数，便于在 Tensorboard/Logs 中区分实验版本
    exp_name = f"MAPPO_DAG_N{Cfg.MIN_NODES}-{Cfg.MAX_NODES}_Veh{Cfg.NUM_VEHICLES}"

    # 初始化数据记录器 (自动创建 logs 文件夹，处理 CSV 和 Tensorboard)
    recorder = DataRecorder(experiment_name=exp_name)

    # --- 构建配置字典用于保存 (Reproducibility) ---
    config_dict = {}

    # 1.1 保存 SystemConfig (物理环境参数: 带宽, 算力, 任务大小等)
    for k, v in Cfg.__dict__.items():
        # 过滤掉内置属性和方法
        if k.startswith('__') or isinstance(v, (staticmethod, classmethod)) or callable(v): continue
        config_dict[k] = v

    # 1.2 保存 TrainConfig (RL 超参数: LR, BatchSize, PPO Clip 等)
    device = TC.DEVICE_NAME if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()  # 清理显存

    hyperparams = {
        "lr_actor": TC.LR_ACTOR,
        "lr_critic": TC.LR_CRITIC,
        "gamma": TC.GAMMA,
        "gae_lambda": TC.GAE_LAMBDA,
        "clip_param": TC.CLIP_PARAM,
        "batch_size": TC.MINI_BATCH_SIZE,
        "k_epochs": TC.PPO_EPOCH,
        "entropy_coef": TC.ENTROPY_COEF,
        "max_episodes": TC.MAX_EPISODES,
        "max_steps_per_ep": TC.MAX_STEPS,
        "device": device
    }
    config_dict.update(hyperparams)

    # 保存配置到 JSON/YAML (由 Recorder 处理)
    recorder.save_config(config_dict)

    # 打印实验信息
    print(f"{'=' * 60}")
    print(f" Experiment: {exp_name}")
    print(f" Device:     {hyperparams['device']}")
    print(f" Max Eps:    {hyperparams['max_episodes']}")
    print(f" LR Actor:   {hyperparams['lr_actor']}")
    print(f"{'=' * 60}")

    # =========================================================================
    # 2. 初始化环境与智能体 (Env & Agent Init)
    # =========================================================================
    # 初始化 Gym 环境
    env = VecOffloadingEnv()

    # 获取输入维度 (适配 Encoder)
    TASK_DIM = TC.TASK_INPUT_DIM
    VEH_DIM = TC.VEH_INPUT_DIM
    RSU_DIM = TC.RSU_INPUT_DIM

    # 初始化 MAPPO Agent (包含 Actor-Critic 网络)
    #
    agent = MAPPOAgent(
        task_dim=TASK_DIM,
        veh_dim=VEH_DIM,
        rsu_dim=RSU_DIM,
        device=hyperparams['device']
    )

    # 初始化经验回放池 (用于存储 Trajectories)
    buffer = RolloutBuffer(
        num_agents=Cfg.NUM_VEHICLES,
        gamma=TC.GAMMA,
        gae_lambda=TC.GAE_LAMBDA
    )

    best_reward = -float('inf')

    # =========================================================================
    # 3. 训练主循环 (Training Loop)
    #
    # =========================================================================
    print("[Info] Start Training...")

    for episode in range(1, hyperparams['max_episodes'] + 1):

        # 学习率衰减策略
        if TC.USE_LR_DECAY and episode > 0 and episode % TC.LR_DECAY_STEPS == 0:
            agent.decay_lr(decay_rate=0.9)
            print(f"[Info] LR Decayed at Episode {episode}")

        # 重置环境，获取初始观测
        raw_obs_list, _ = env.reset()

        ep_reward = 0
        ep_start_time = time.time()
        step_logs_buffer = []

        # [统计容器] 用于计算多智能体指标 (公平性、协作率等)
        stats = {
            "power_sum": 0.0,
            "local_cnt": 0,
            "rsu_cnt": 0,
            "neighbor_cnt": 0,
            "queue_len_sum": 0,
            "rsu_queue_sum": 0,
            "assigned_cpu_sum": 0.0,
            # 每个智能体的独立累积奖励 (用于计算 Jain's Fairness Index)
            "agent_rewards": np.zeros(Cfg.NUM_VEHICLES),
            # 协作关系矩阵 [Source, Target] (用于分析谁卸载给了谁)
            "v2v_matrix": np.zeros((Cfg.NUM_VEHICLES, Cfg.NUM_VEHICLES))
        }

        # --- Rollout Loop (采集轨迹: 1个 Episode 包含 Max_Steps) ---
        for step in range(hyperparams['max_steps_per_ep']):
            current_time = env.time
            dag_graph_list = []

            # A. 构建 DAG 图数据 (含 Task Mask 和 Target Mask)
            for i, v_obs in enumerate(raw_obs_list):
                # 从 Obs 中提取 Target Mask (物理约束)
                t_mask = v_obs.get('target_mask', None)
                # 使用 GraphBuilder 将原始 Obs 转换为 PyG Data
                dag_data = GraphBuilder.get_dag_graph(env.vehicles[i].task_dag, current_time, target_mask=t_mask)
                dag_graph_list.append(dag_data)

            # B. 构建拓扑图数据 (含 Edge Attributes)
            # 拓扑图对所有智能体是共享的 (Shared View)，但为了 Batch 处理，复制 N 份
            topo_data = GraphBuilder.get_topology_graph(env.vehicles, env.rsu_queue_curr)
            topo_graph_list = [topo_data] * Cfg.NUM_VEHICLES

            # C. 智能体决策 (Select Action)
            # 输入: List[Data], List[HeteroData]
            # 输出: Dict {'action_d', 'action_p', 'logprob_d', ...}
            action_dict = agent.select_action(dag_graph_list, topo_graph_list)

            # D. 动作解码 (Decode Actions)
            # Neural Output (Flatten Index) -> Env Input (Subtask ID, Target ID)
            # Num_Targets = Local(1) + RSU(1) + Neighbors(N-1) = N + 1
            num_targets_decode = Cfg.NUM_VEHICLES + 1
            env_actions = agent.decode_actions(action_dict['action_d'], action_dict['power_val'], num_targets_decode)

            # E. 环境步进 (Env Step)
            next_raw_obs_list, rewards, done, info = env.step(env_actions)

            # [统计] 累加每个智能体的实时奖励
            stats["agent_rewards"] += np.array(rewards)

            # F. 获取当前状态价值 (Value) 用于 GAE
            values = agent.get_value(dag_graph_list, topo_graph_list)

            # G. 存入 Buffer
            # 注意: logprob 应该是 离散+连续 的联合概率对数
            buffer.add(
                dag_list=dag_graph_list,
                topo_list=topo_graph_list,
                act_d=action_dict['action_d'],
                act_c=action_dict['action_p'],  # 存原始采样值 (未截断)
                logprob=action_dict['logprob_d'] + action_dict['logprob_p'],
                val=values,
                rew=rewards,
                done=done
            )

            # H. 过程统计 (Statistics)
            # 这里的逻辑是将所有车辆的奖励求和再取平均，这已经是"全体平均奖励"的计算方式
            step_r = sum(rewards) / Cfg.NUM_VEHICLES
            ep_reward += step_r

            for i, act in enumerate(env_actions):
                tgt = act['target']

                # 1. 统计功率和队列
                stats['power_sum'] += act['power']
                stats['queue_len_sum'] += env.vehicles[i].task_queue_len

                # 2. 分类统计逻辑 (Local / RSU / Neighbor)
                if tgt == 0:
                    # --- 本地计算 (Local) ---
                    stats['local_cnt'] += 1
                    stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq

                elif tgt == 1:
                    # --- RSU 卸载 ---
                    stats['rsu_cnt'] += 1
                    stats['assigned_cpu_sum'] += Cfg.F_RSU

                else:
                    # --- V2V 卸载 (Neighbor) ---
                    stats['neighbor_cnt'] += 1

                    # 映射逻辑：Env Target ID (2+) -> Neighbor List Index (0+)
                    # 这里的索引对应 stats["v2v_matrix"] 的列
                    neighbor_idx = tgt - 2

                    # 更新协作矩阵
                    if neighbor_idx < stats["v2v_matrix"].shape[1]:
                        stats["v2v_matrix"][i, neighbor_idx] += 1

                    # 算力统计：查找对应的邻居车辆
                    # candidate_vehs 是 [0, 1, ... i-1, i+1, ...] (排除自己)
                    candidate_vehs = [v for v in env.vehicles if v.id != i]

                    if 0 <= neighbor_idx < len(candidate_vehs):
                        target_veh = candidate_vehs[neighbor_idx]
                        stats['assigned_cpu_sum'] += target_veh.cpu_freq
                    else:
                        # 兜底：如果索引异常，计入本地算力
                        stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq

            stats['rsu_queue_sum'] += env.rsu_queue_curr

            # 记录详细日志 (用于 CSV)
            for i, act in enumerate(env_actions):
                step_logs_buffer.append({
                    "episode": episode, "step": step, "veh_id": i,
                    "subtask": act['subtask'], "target": act['target'],
                    "power": f"{act['power']:.3f}", "reward": f"{rewards[i]:.3f}",
                    "q_len": env.vehicles[i].task_queue_len
                })

            # 更新观测
            raw_obs_list = next_raw_obs_list
            if done: break

        # =========================================================================
        # 4. Episode 结束后的分析与更新 (Post-Episode)
        # =========================================================================
        total_steps = step + 1
        total_decisions = total_steps * Cfg.NUM_VEHICLES

        # --- Metric 1: Jain's Fairness Index (公平性指数) ---
        # 衡量系统资源分配或奖励分配的公平程度 (1.0 为绝对公平)
        sum_agent_rew = np.sum(stats["agent_rewards"])
        sum_sq_agent_rew = np.sum(stats["agent_rewards"] ** 2)
        fairness_index = (sum_agent_rew ** 2) / (Cfg.NUM_VEHICLES * sum_sq_agent_rew + 1e-9)

        # --- Metric 2: 个体奖励差异 (Gap) ---
        reward_gap = np.max(stats["agent_rewards"]) - np.min(stats["agent_rewards"])

        # --- Metric 3: 协作率 ---
        total_v2v_actions = np.sum(stats["v2v_matrix"])
        collaboration_rate = (total_v2v_actions / total_decisions) * 100

        # --- PPO 更新 (Update) ---
        # 1. 计算最后一个状态的 Value (用于 GAE 的截断)
        last_dag_list = []
        for i, v_obs in enumerate(raw_obs_list):
            t_mask = v_obs.get('target_mask', None)
            d = GraphBuilder.get_dag_graph(env.vehicles[i].task_dag, env.time, target_mask=t_mask)
            last_dag_list.append(d)

        last_topo_list = [GraphBuilder.get_topology_graph(env.vehicles, env.rsu_queue_curr)] * Cfg.NUM_VEHICLES
        last_val = agent.get_value(last_dag_list, last_topo_list)

        # 2. 计算 Advantage & Return
        buffer.compute_returns_and_advantages(last_val, done)

        # 3. 执行梯度下降更新
        update_loss = agent.update(buffer, batch_size=TC.MINI_BATCH_SIZE)

        # 4. 清空 Buffer
        buffer.reset()

        # 显存清理
        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # --- 汇总本 Episode 数据 ---
        duration = time.time() - ep_start_time
        avg_assigned_cpu = stats['assigned_cpu_sum'] / total_decisions

        # [修改] 增加平均奖励记录
        avg_step_reward = ep_reward / total_steps

        avg_power = stats['power_sum'] / total_decisions
        avg_veh_queue = stats['queue_len_sum'] / total_decisions
        avg_rsu_queue = stats['rsu_queue_sum'] / total_steps

        pct_local = (stats['local_cnt'] / total_decisions) * 100
        pct_rsu = (stats['rsu_cnt'] / total_decisions) * 100
        pct_v2v = (stats['neighbor_cnt'] / total_decisions) * 100

        # --- 成功率统计 (Success Rates) ---

        # 1. 车辆级成功率 (Vehicle Success Rate)
        success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished])
        veh_success_rate = (success_count / Cfg.NUM_VEHICLES) * 100

        # 2. [新增] 子任务级成功率 (Subtask Success Rate)
        total_subtasks = 0
        completed_subtasks = 0
        for v in env.vehicles:
            total_subtasks += v.task_dag.num_subtasks
            # 统计状态为 3 (DONE) 的节点数。假设 3 代表完成状态
            # 如果您的环境中状态码不同，请调整此处的 '== 3'
            completed_subtasks += np.sum(v.task_dag.status == 3)

        subtask_success_rate = (completed_subtasks / total_subtasks * 100) if total_subtasks > 0 else 0

        # --- 控制台输出 (Console Log) ---
        # [修改] 增加 Sub% (子任务成功率), AvgQ (平均队列), Pwr (平均功率)
        if episode == 1 or episode % 10 == 0:
            header = (
                f"{'Ep':<5} | {'Reward':<8} | {'Veh%':<5} {'Sub%':<5} | {'MA_F':<4} | "
                f"{'Loc%':<4} {'RSU%':<4} {'V2V%':<4} | {'AvgQ':<5} {'Pwr':<5} | {'Loss':<6} | {'Time':<4}"
            )
            print("-" * len(header))
            print(header)
            print("-" * len(header))

        print(f"{episode:<5d} | "
              f"{ep_reward:<8.2f} | "
              f"{veh_success_rate:<5.1f} {subtask_success_rate:<5.1f} | "
              f"{fairness_index:<4.2f} | "
              f"{pct_local:<4.1f} {pct_rsu:<4.1f} {pct_v2v:<4.1f} | "
              f"{avg_veh_queue:<5.2f} {avg_power:<5.2f} | "
              f"{update_loss:<6.3f} | "
              f"{duration:<4.1f}")

        # --- 记录到 Tensorboard / CSV ---
        recorder.log_step(step_logs_buffer)

        # 记录多维度指标
        episode_metrics = {
            "episode": episode,
            "total_reward": ep_reward,
            "avg_step_reward": avg_step_reward,  # [新增] 每步平均奖励
            "loss": update_loss,
            "veh_success_rate": veh_success_rate,
            "subtask_success_rate": subtask_success_rate,  # [新增] 子任务成功率
            "pct_local": pct_local,
            "pct_rsu": pct_rsu,
            "pct_v2v": pct_v2v,
            "avg_power": avg_power,
            "avg_queue_len": avg_veh_queue,  # [新增] 平均队列长度
            "ma_fairness": fairness_index,  # 公平性
            "ma_reward_gap": reward_gap,  # 贫富差距
            "ma_collaboration": collaboration_rate,  # 协作程度
            "max_agent_reward": np.max(stats["agent_rewards"]),
            "min_agent_reward": np.min(stats["agent_rewards"]),
            "avg_assigned_cpu_ghz": avg_assigned_cpu / 1e9,
            "duration": duration
        }
        recorder.log_episode(episode_metrics)

        # --- 模型保存 (Checkpoint) ---
        # 保存最佳模型
        if ep_reward > best_reward:
            best_reward = ep_reward
            recorder.save_model(agent, episode, is_best=True)

        # 定期保存
        if episode % TC.SAVE_INTERVAL == 0:
            recorder.save_model(agent, episode, is_best=False)

    # =========================================================================
    # 5. 训练结束与绘图 (Finalize)
    # =========================================================================
    print("\n[Info] Training Finished.")
    print("[Info] Generating plots...")

    # 自动生成训练曲线图 (Reward, Loss, Metrics)
    recorder.auto_plot()
    print(f"[Info] Plots saved to {recorder.plot_dir}")


if __name__ == "__main__":
    main()