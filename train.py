import time
import numpy as np
import torch
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from utils.data_recorder import DataRecorder
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy


def split_batch_to_list(dag_batch, topo_batch):
    """
    将Batch对象拆分为列表，用于Buffer存储
    """
    # DAG Batch可以直接使用to_data_list
    dag_list = dag_batch.to_data_list()
    
    # Topology Batch (HeteroData Batch) 也支持to_data_list
    topo_list = topo_batch.to_data_list()
    
    return dag_list, topo_list


def evaluate_baselines(env, num_episodes=10):
    """
    评估基准策略性能
    
    Args:
        env: 环境实例
        num_episodes: 评估回合数
    
    Returns:
        baseline_results: 字典，包含各基准策略的平均奖励
    """
    baseline_results = {}
    
    # 1. 随机策略
    random_policy = RandomPolicy(seed=42)
    random_rewards = []
    for _ in range(num_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0
        for step in range(TC.MAX_STEPS):
            actions = random_policy.select_action(obs_list)
            obs_list, rewards, done, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done:
                break
        random_rewards.append(ep_reward)
    baseline_results['Random'] = np.mean(random_rewards)
    
    # 2. 全本地执行
    local_policy = LocalOnlyPolicy()
    local_rewards = []
    for _ in range(num_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0
        for step in range(TC.MAX_STEPS):
            actions = local_policy.select_action(obs_list)
            obs_list, rewards, done, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done:
                break
        local_rewards.append(ep_reward)
    baseline_results['Local-Only'] = np.mean(local_rewards)
    
    # 3. 贪婪策略
    greedy_policy = GreedyPolicy(env)
    greedy_rewards = []
    for _ in range(num_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0
        for step in range(TC.MAX_STEPS):
            actions = greedy_policy.select_action(obs_list)
            obs_list, rewards, done, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done:
                break
        greedy_rewards.append(ep_reward)
    baseline_results['Greedy'] = np.mean(greedy_rewards)
    
    return baseline_results


def main():
    # 开启 CuDNN 加速，提升卷积/矩阵运算效率
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 初始化配置和日志记录器
    exp_name = f"MAPPO_DAG_N{Cfg.MIN_NODES}-{Cfg.MAX_NODES}_Veh{Cfg.NUM_VEHICLES}"
    recorder = DataRecorder(experiment_name=exp_name)

    # 构建配置字典用于保存（便于实验复现）
    config_dict = {}
    for k, v in Cfg.__dict__.items():
        if k.startswith('__') or isinstance(v, (staticmethod, classmethod)) or callable(v):
            continue
        config_dict[k] = v
    device = TC.DEVICE_NAME if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

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

    # 初始化环境
    env = VecOffloadingEnv()

    # 初始化网络
    network = OffloadingPolicyNetwork(
        d_model=TC.EMBED_DIM,
        num_heads=TC.NUM_HEADS,
        num_layers=TC.NUM_LAYERS
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam([
        {'params': network.parameters(), 'lr': TC.LR_ACTOR}
    ])

    best_reward = -float('inf')
    
    # 评估基准策略（仅在训练开始时评估一次）
    print("\n[Info] 评估基准策略...")
    baseline_results = evaluate_baselines(env, num_episodes=10)
    print(f"基准策略性能:")
    for policy_name, avg_reward in baseline_results.items():
        print(f"  {policy_name}: {avg_reward:.2f}")
        recorder.writer.add_scalar(f'Baseline/{policy_name}', avg_reward, 0)

    # =========================================================================
    # 3. 训练主循环 (Training Loop)
    #
    # =========================================================================
    print("\n[Info] Start Training...")

    for episode in range(1, hyperparams['max_episodes'] + 1):

        # 学习率衰减
        if TC.USE_LR_DECAY and episode > 0 and episode % TC.LR_DECAY_STEPS == 0:
            agent.decay_lr(decay_rate=TC.LR_DECAY_RATE)
            print(f"[Info] LR Decayed at Episode {episode}")

        # 重置环境，获取初始观测
        raw_obs_list, _ = env.reset()

        ep_reward = 0
        ep_start_time = time.time()
        step_logs_buffer = []

        # 统计容器：用于计算多智能体指标（公平性、协作率等）
        stats = {
            "power_sum": 0.0,
            "local_cnt": 0,
            "rsu_cnt": 0,
            "neighbor_cnt": 0,
            "queue_len_sum": 0,
            "rsu_queue_sum": 0,
            "assigned_cpu_sum": 0.0,
            "agent_rewards": np.zeros(Cfg.NUM_VEHICLES),  # 每个智能体的累积奖励（用于计算公平性）
            "v2v_matrix": np.zeros((Cfg.NUM_VEHICLES, Cfg.NUM_VEHICLES))  # 协作关系矩阵
        }

        # Rollout循环：采集轨迹
        for step in range(hyperparams['max_steps_per_ep']):
            # 构建图数据（统一通过data_utils处理）
            dag_batch, local_topo_batch, global_topo = process_env_obs(raw_obs_list, agent.device)

            # 智能体决策
            action_dict = agent.select_action(dag_batch, local_topo_batch)

            # 动作解码：将策略输出转换为环境动作格式
            num_targets_decode = Cfg.NUM_VEHICLES + 1
            env_actions = agent.decode_actions(action_dict['action_d'], action_dict['power_val'], num_targets_decode)

            # 环境步进
            next_raw_obs_list, rewards, terminated, truncated, info = env.step(env_actions)
            done = terminated or truncated

            # 统计：累加每个智能体的实时奖励
            stats["agent_rewards"] += np.array(rewards)

            # 获取状态价值（用于GAE计算）
            values = agent.get_value(dag_batch, local_topo_batch)

            # 存入Buffer（将batch拆分为列表）
            dag_graph_list, topo_graph_list = split_batch_to_list(dag_batch, local_topo_batch)
            buffer.add(
                dag_list=dag_graph_list,
                topo_list=topo_graph_list,
                act_d=action_dict['action_d'],
                act_c=action_dict['action_p'],
                logprob=action_dict['logprob_total'],
                val=values,
                rew=rewards,
                done=done
            )

            # 过程统计
            step_r = sum(rewards) / Cfg.NUM_VEHICLES
            ep_reward += step_r

            for i, act in enumerate(env_actions):
                tgt = act['target']

                # 统计功率和队列
                stats['power_sum'] += act['power']
                stats['queue_len_sum'] += env.vehicles[i].task_queue_len

                # 分类统计：Local / RSU / Neighbor
                if tgt == 0:
                    stats['local_cnt'] += 1
                    stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq

                elif tgt == 1:
                    # --- RSU 卸载 ---
                    stats['rsu_cnt'] += 1
                    stats['assigned_cpu_sum'] += Cfg.F_RSU

                else:
                    # --- V2V 卸载 (Neighbor) ---
                    stats['neighbor_cnt'] += 1

                    # 环境目标ID(2+)转换为邻居列表索引(0+)
                    neighbor_idx = tgt - 2
                    if neighbor_idx < stats["v2v_matrix"].shape[1]:
                        stats["v2v_matrix"][i, neighbor_idx] += 1

                    # 算力统计：查找对应的邻居车辆
                    candidate_vehs = [v for v in env.vehicles if v.id != i]
                    if 0 <= neighbor_idx < len(candidate_vehs):
                        target_veh = candidate_vehs[neighbor_idx]
                        stats['assigned_cpu_sum'] += target_veh.cpu_freq
                    else:
                        stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq

            stats['rsu_queue_sum'] += env.rsu_queue_curr

            # 记录详细日志
            for i, act in enumerate(env_actions):
                step_logs_buffer.append({
                    "episode": episode, "step": step, "veh_id": i,
                    "subtask": act['subtask'], "target": act['target'],
                    "power": f"{act['power']:.3f}", "reward": f"{rewards[i]:.3f}",
                    "q_len": env.vehicles[i].task_queue_len
                })

            raw_obs_list = next_raw_obs_list
            if done:
                break

        # Episode结束后的分析与更新
        total_steps = step + 1
        total_decisions = total_steps * Cfg.NUM_VEHICLES

        # 计算公平性指数（Jain's Fairness Index，1.0为绝对公平）
        sum_agent_rew = np.sum(stats["agent_rewards"])
        sum_sq_agent_rew = np.sum(stats["agent_rewards"] ** 2)
        fairness_index = (sum_agent_rew ** 2) / (Cfg.NUM_VEHICLES * sum_sq_agent_rew + 1e-9)

        # 个体奖励差异
        reward_gap = np.max(stats["agent_rewards"]) - np.min(stats["agent_rewards"])

        # 协作率
        total_v2v_actions = np.sum(stats["v2v_matrix"])
        collaboration_rate = (total_v2v_actions / total_decisions) * 100

        # PPO更新
        last_dag_batch, last_topo_batch, _ = process_env_obs(raw_obs_list, agent.device)
        last_val = agent.get_value(last_dag_batch, last_topo_batch)
        buffer.compute_returns_and_advantages(last_val, done)
        update_loss = agent.update(buffer, batch_size=TC.MINI_BATCH_SIZE)
        buffer.clear()

        # 显存清理
        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # 汇总Episode数据
        duration = time.time() - ep_start_time
        avg_assigned_cpu = stats['assigned_cpu_sum'] / total_decisions
        avg_step_reward = ep_reward / total_steps
        avg_power = stats['power_sum'] / total_decisions
        avg_veh_queue = stats['queue_len_sum'] / total_decisions
        avg_rsu_queue = stats['rsu_queue_sum'] / total_steps

        pct_local = (stats['local_cnt'] / total_decisions) * 100
        pct_rsu = (stats['rsu_cnt'] / total_decisions) * 100
        pct_v2v = (stats['neighbor_cnt'] / total_decisions) * 100

        # 成功率统计

        # 车辆级成功率
        success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished])
        veh_success_rate = (success_count / Cfg.NUM_VEHICLES) * 100

        # 子任务级成功率
        total_subtasks = 0
        completed_subtasks = 0
        for v in env.vehicles:
            total_subtasks += v.task_dag.num_subtasks
            # 统计状态为 3 (DONE) 的节点数。假设 3 代表完成状态
            # 如果您的环境中状态码不同，请调整此处的 '== 3'
            completed_subtasks += np.sum(v.task_dag.status == 3)

        subtask_success_rate = (completed_subtasks / total_subtasks * 100) if total_subtasks > 0 else 0

        # --- 控制台输出 (Console Log) ---
        # 打印训练进度
        if episode == 1 or episode % 10 == 0:
            header = (
                f"{'Ep':<5} | {'Reward':<9} {'AvgR':<7} | {'Veh%':<5} {'Sub%':<5} | {'MA_F':<4} {'CPU':<4} | "
                f"{'Loc%':<5} {'RSU%':<5} {'V2V%':<5} | {'AvgQ':<6} {'Pwr':<5} | {'Loss':<8} | {'Time':<4}"
            )
            print("-" * len(header))
            print(header)
            print("-" * len(header))

        print(f"{episode:<5d} | "
              f"{ep_reward:<9.2f} {avg_step_reward:<7.3f} | "
              f"{veh_success_rate:<5.1f} {subtask_success_rate:<5.1f} | "
              f"{fairness_index:<4.2f} {avg_assigned_cpu/1e9:<4.2f} | "
              f"{pct_local:<5.1f} {pct_rsu:<5.1f} {pct_v2v:<5.1f} | "
              f"{avg_veh_queue:<6.2f} {avg_power:<5.2f} | "
              f"{update_loss:<8.3f} | "
              f"{duration:<4.1f}")

        # 记录到Tensorboard/CSV
        recorder.log_step(step_logs_buffer)

        # 记录多维度指标
        episode_metrics = {
            "episode": episode,
            "total_reward": ep_reward,
            "avg_step_reward": avg_step_reward,
            "loss": update_loss,
            "veh_success_rate": veh_success_rate,
            "subtask_success_rate": subtask_success_rate,
            "pct_local": pct_local,
            "pct_rsu": pct_rsu,
            "pct_v2v": pct_v2v,
            "avg_power": avg_power,
            "avg_queue_len": avg_veh_queue,
            "ma_fairness": fairness_index,
            "ma_reward_gap": reward_gap,
            "ma_collaboration": collaboration_rate,
            "max_agent_reward": np.max(stats["agent_rewards"]),
            "min_agent_reward": np.min(stats["agent_rewards"]),
            "avg_assigned_cpu_ghz": avg_assigned_cpu / 1e9,
            "duration": duration
        }
        recorder.log_episode(episode_metrics)

        # 模型保存
        if ep_reward > best_reward:
            best_reward = ep_reward
            recorder.save_model(agent, episode, is_best=True)

        if episode % TC.SAVE_INTERVAL == 0:
            recorder.save_model(agent, episode, is_best=False)

    # 训练结束与绘图
    print("\n[Info] Training Finished.")
    print("[Info] Generating plots...")
    recorder.auto_plot(baseline_results=baseline_results)
    print(f"[Info] Plots saved to {recorder.plot_dir}")


if __name__ == "__main__":
    main()