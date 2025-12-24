import time
import numpy as np
import torch
import os
import sys

# --- 导入自定义模块 ---
# 将当前目录添加到系统路径，防止导入错误
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC  # [引入训练配置]
from envs.vec_offloading_env import VecOffloadingEnv
from agents.mappo_agent import MAPPOAgent
from agents.buffer import RolloutBuffer
from utils.data_recorder import DataRecorder
from utils.graph_builder import GraphBuilder


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    # =========================================================================
    # 1. 初始化配置与日志记录器 (Config & Logger Init)
    # =========================================================================
    # 实验命名: 包含 DAG 节点数范围和车辆数，便于区分实验版本
    exp_name = f"MAPPO_DAG_N{Cfg.MIN_NODES}-{Cfg.MAX_NODES}_Veh{Cfg.NUM_VEHICLES}"

    # 初始化数据记录器 (自动创建 logs 文件夹)
    recorder = DataRecorder(experiment_name=exp_name)

    # --- 构建配置字典用于保存 ---
    config_dict = {}

    # 1.1 保存 SystemConfig (物理环境参数)
    for k, v in Cfg.__dict__.items():
        if k.startswith('__') or isinstance(v, (staticmethod, classmethod)) or callable(v): continue
        config_dict[k] = v

    # 1.2 保存 TrainConfig (RL 超参数)
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

    recorder.save_config(config_dict)

    print(f"{'=' * 60}")
    print(f" Experiment: {exp_name}")
    print(f" Device:     {hyperparams['device']}")
    print(f" Max Eps:    {hyperparams['max_episodes']}")
    print(f" LR Actor:   {hyperparams['lr_actor']}")
    print(f"{'=' * 60}")

    # =========================================================================
    # 2. 初始化环境与智能体 (Env & Agent Init)
    # =========================================================================
    env = VecOffloadingEnv()

    TASK_DIM = TC.TASK_INPUT_DIM
    VEH_DIM = TC.VEH_INPUT_DIM
    RSU_DIM = TC.RSU_INPUT_DIM

    agent = MAPPOAgent(
        task_dim=TASK_DIM,
        veh_dim=VEH_DIM,
        rsu_dim=RSU_DIM,
        device=hyperparams['device']
    )

    buffer = RolloutBuffer(
        num_agents=Cfg.NUM_VEHICLES,
        gamma=TC.GAMMA,
        gae_lambda=TC.GAE_LAMBDA
    )

    best_reward = -float('inf')

    # =========================================================================
    # 3. 训练主循环 (Training Loop)
    # =========================================================================
    print("[Info] Start Training...")

    for episode in range(1, hyperparams['max_episodes'] + 1):

        if TC.USE_LR_DECAY and episode > 0 and episode % TC.LR_DECAY_STEPS == 0:
            agent.decay_lr(decay_rate=0.9)

        raw_obs_list, _ = env.reset()

        ep_reward = 0
        ep_start_time = time.time()
        step_logs_buffer = []

        # [修改] 详细统计累加器：增加多智能体数据容器
        stats = {
            "power_sum": 0.0,
            "local_cnt": 0,
            "rsu_cnt": 0,
            "neighbor_cnt": 0,
            "queue_len_sum": 0,
            "rsu_queue_sum": 0,
            "assigned_cpu_sum": 0.0,
            # --- [新增多智能体统计] ---
            "agent_rewards": np.zeros(Cfg.NUM_VEHICLES),  # 每个智能体的独立累积奖励
            "v2v_matrix": np.zeros((Cfg.NUM_VEHICLES, Cfg.NUM_VEHICLES))  # 协作关系矩阵
        }

        # --- Rollout Loop (采集轨迹) ---
        for step in range(hyperparams['max_steps_per_ep']):
            current_time = env.time
            dag_graph_list = []

            for i, v_obs in enumerate(raw_obs_list):
                t_mask = v_obs.get('target_mask', None)
                dag_data = GraphBuilder.get_dag_graph(env.vehicles[i].task_dag, current_time, target_mask=t_mask)
                dag_graph_list.append(dag_data)

            topo_data = GraphBuilder.get_topology_graph(env.vehicles, env.rsu_queue_curr)
            topo_graph_list = [topo_data] * Cfg.NUM_VEHICLES

            action_dict = agent.select_action(dag_graph_list, topo_graph_list)
            num_targets_decode = Cfg.NUM_VEHICLES + 1
            env_actions = agent.decode_actions(action_dict['action_d'], action_dict['power_val'], num_targets_decode)

            # Step 环境
            next_raw_obs_list, rewards, done, info = env.step(env_actions)

            # [新增] 累加每个智能体的实时奖励
            stats["agent_rewards"] += np.array(rewards)

            values = agent.get_value(dag_graph_list, topo_graph_list)

            buffer.add(
                dag_list=dag_graph_list,
                topo_list=topo_graph_list,
                act_d=action_dict['action_d'],
                act_c=action_dict['action_p'],
                logprob=action_dict['logprob_d'] + action_dict['logprob_p'],
                val=values,
                rew=rewards,
                done=done
            )

            # 统计
            step_r = sum(rewards) / Cfg.NUM_VEHICLES
            ep_reward += step_r

            for i, act in enumerate(env_actions):
                tgt = act['target']
                # [新增] 统计 V2V 协作矩阵：源车辆 i -> 目标邻居 (tgt-2)
                if tgt >= 2:
                    neighbor_idx = tgt - 2
                    stats["v2v_matrix"][i, neighbor_idx] += 1

                # 算力统计
                    # 映射逻辑：动作索引 tgt 对应的邻居索引是 tgt - 2
                    neighbor_idx = tgt - 2

                    # 在 _get_obs 中，邻居是按 self.vehicles 中除去自己以外的顺序排列的
                    # 这里最稳妥的方法是获取该车辆对应的“邻居列表”
                    candidate_vehs = [v for v in env.vehicles if v.id != i]

                    if neighbor_idx < len(candidate_vehs):
                        target_veh = candidate_vehs[neighbor_idx]
                        stats['assigned_cpu_sum'] += target_veh.cpu_freq
                    else:
                        # 容错：如果索引依然越界，按本地算力统计
                        stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq

                # 动作分布与功率
                if tgt == 0:
                    stats['local_cnt'] += 1
                elif tgt == 1:
                    stats['rsu_cnt'] += 1
                else:
                    stats['neighbor_cnt'] += 1

                stats['power_sum'] += act['power']
                stats['queue_len_sum'] += env.vehicles[i].task_queue_len

            stats['rsu_queue_sum'] += env.rsu_queue_curr

            for i, act in enumerate(env_actions):
                step_logs_buffer.append({
                    "episode": episode, "step": step, "veh_id": i,
                    "subtask": act['subtask'], "target": act['target'],
                    "power": f"{act['power']:.3f}", "reward": f"{rewards[i]:.3f}",
                    "q_len": env.vehicles[i].task_queue_len
                })

            raw_obs_list = next_raw_obs_list
            if done: break

        # =========================================================================
        # --- [新增] Episode 结束后的多智能体分析 (移出step循环以修正报错) ---
        # =========================================================================
        total_steps = step + 1
        total_decisions = total_steps * Cfg.NUM_VEHICLES

        # 1. 计算 Jain's Fairness Index (公平性指数)
        sum_agent_rew = np.sum(stats["agent_rewards"])
        sum_sq_agent_rew = np.sum(stats["agent_rewards"] ** 2)
        fairness_index = (sum_agent_rew ** 2) / (Cfg.NUM_VEHICLES * sum_sq_agent_rew + 1e-9)

        # 2. 计算个体奖励差异 (贫富差距)
        reward_gap = np.max(stats["agent_rewards"]) - np.min(stats["agent_rewards"])

        # 3. 统计协作率
        total_v2v_actions = np.sum(stats["v2v_matrix"])
        collaboration_rate = (total_v2v_actions / total_decisions) * 100

        # --- Update Loop (PPO 更新) ---
        last_dag_list = []
        for i, v_obs in enumerate(raw_obs_list):
            t_mask = v_obs.get('target_mask', None)
            d = GraphBuilder.get_dag_graph(env.vehicles[i].task_dag, env.time, target_mask=t_mask)
            last_dag_list.append(d)

        last_topo_list = [GraphBuilder.get_topology_graph(env.vehicles, env.rsu_queue_curr)] * Cfg.NUM_VEHICLES
        last_val = agent.get_value(last_dag_list, last_topo_list)

        buffer.compute_returns_and_advantages(last_val, done)
        update_loss = agent.update(buffer, batch_size=TC.MINI_BATCH_SIZE)
        buffer.reset()

        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # --- Episode Summary Calculation ---
        duration = time.time() - ep_start_time
        avg_assigned_cpu = stats['assigned_cpu_sum'] / total_decisions
        avg_step_reward = ep_reward / total_steps
        avg_power = stats['power_sum'] / total_decisions
        avg_veh_queue = stats['queue_len_sum'] / total_decisions
        avg_rsu_queue = stats['rsu_queue_sum'] / total_steps

        pct_local = (stats['local_cnt'] / total_decisions) * 100
        pct_rsu = (stats['rsu_cnt'] / total_decisions) * 100
        pct_v2v = (stats['neighbor_cnt'] / total_decisions) * 100

        success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished])
        success_rate = (success_count / Cfg.NUM_VEHICLES) * 100

        # --- [修正] 控制台表格输出：增加公平性 (MA_F) 和 奖励差距 (Gap) ---
        if episode == 1 or episode % 10 == 0:
            header = (
                f"{'Ep':<5} | {'Reward':<8} | {'Succ%':<6} | {'MA_F':<5} | "
                f"{'Loc%':<5} {'RSU%':<5} {'V2V%':<5} | {'Gap':<6} | {'Time':<5}"
            )
            print("-" * len(header))
            print(header)
            print("-" * len(header))

        print(f"{episode:<5d} | "
              f"{ep_reward:<8.2f} | "
              f"{success_rate:<6.1f} | "
              f"{fairness_index:<5.2f} | "
              f"{pct_local:<5.1f} {pct_rsu:<5.1f} {pct_v2v:<5.1f} | "
              f"{reward_gap:<6.1f} | "
              f"{duration:<5.1f}")

        # --- Save Logs ---
        recorder.log_step(step_logs_buffer)

        # [修改] 记录多智能体 Metrics
        episode_metrics = {
            "episode": episode,
            "total_reward": ep_reward,
            "avg_step_reward": avg_step_reward,
            "loss": update_loss,
            "success_rate": success_rate,
            "pct_local": pct_local,
            "pct_rsu": pct_rsu,
            "pct_v2v": pct_v2v,
            "avg_power": avg_power,
            "ma_fairness": fairness_index,  # [新增]
            "ma_reward_gap": reward_gap,  # [新增]
            "ma_collaboration": collaboration_rate,  # [新增]
            "max_agent_reward": np.max(stats["agent_rewards"]),
            "min_agent_reward": np.min(stats["agent_rewards"]),
            "avg_assigned_cpu_ghz": avg_assigned_cpu / 1e9,
            "duration": duration
        }
        recorder.log_episode(episode_metrics)

        # --- Model Saving ---
        if ep_reward > best_reward:
            best_reward = ep_reward
            recorder.save_model(agent, episode, is_best=True)

        if episode % TC.SAVE_INTERVAL == 0:
            recorder.save_model(agent, episode, is_best=False)

    print("\n[Info] Training Finished.")
    print("[Info] Generating plots...")
    recorder.auto_plot()
    print(f"[Info] Plots saved to {recorder.plot_dir}")


if __name__ == "__main__":
    main()