import time
import numpy as np
import torch
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.mappo_agent import MAPPOAgent
from agents.rollout_buffer import RolloutBuffer
from utils.data_recorder import DataRecorder
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy


def evaluate_baselines(env, num_episodes=10):
    """评估基准策略性能（多次episode取平均）"""
    baseline_results = {}
    
    # 1. 随机策略
    random_policy = RandomPolicy(seed=42)
    random_rewards = []
    for _ in range(num_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0
        for step in range(TC.MAX_STEPS):
            actions = random_policy.select_action(obs_list)
            obs_list, rewards, done, truncated, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done or truncated:
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
            obs_list, rewards, done, truncated, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done or truncated:
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
            obs_list, rewards, done, truncated, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done or truncated:
                break
        greedy_rewards.append(ep_reward)
    baseline_results['Greedy'] = np.mean(greedy_rewards)
    
    return baseline_results


def evaluate_single_baseline_episode(env, policy_name):
    """评估单个baseline策略的一个episode，返回完整的指标（与训练指标一致）"""
    if policy_name == 'Random':
        policy = RandomPolicy(seed=int(time.time()))
    elif policy_name == 'Local-Only':
        policy = LocalOnlyPolicy()
    elif policy_name == 'Greedy':
        policy = GreedyPolicy(env)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    obs_list, _ = env.reset()
    ep_reward = 0
    total_steps = 0
    
    # 统计容器（与训练循环一致）
    stats = {
        "power_sum": 0.0,
        "local_cnt": 0,
        "rsu_cnt": 0,
        "neighbor_cnt": 0,
        "queue_len_sum": 0,
        "rsu_queue_sum": 0,
        "assigned_cpu_sum": 0.0,
    }
    
    for step in range(TC.MAX_STEPS):
        actions = policy.select_action(obs_list)
        obs_list, rewards, done, truncated, _ = env.step(actions)
        ep_reward += sum(rewards) / len(rewards)
        total_steps += 1
        
        # 统计决策分布
        for i, act in enumerate(actions):
            if act['target'] == 0:
                stats['local_cnt'] += 1
            elif act['target'] == 1:
                stats['rsu_cnt'] += 1
            else:
                stats['neighbor_cnt'] += 1
            
            stats['power_sum'] += act.get('power', 0.0)
            stats['queue_len_sum'] += env.vehicles[i].task_queue_len if i < len(env.vehicles) else 0
        
        stats['rsu_queue_sum'] += sum([rsu.queue_length for rsu in env.rsus])
        
        if done or truncated:
            break
    
    avg_step_reward = ep_reward / total_steps if total_steps > 0 else 0
    total_decisions = stats['local_cnt'] + stats['rsu_cnt'] + stats['neighbor_cnt']
    
    # 成功率统计（与训练循环一致）
    episode_vehicle_count = len(env.vehicles)
    success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished])
    veh_success_rate = success_count / max(episode_vehicle_count, 1)
    task_success_rate = success_count / max(episode_vehicle_count, 1)
    
    total_subtasks = 0
    completed_subtasks = 0
    v2v_subtasks_attempted = 0
    v2v_subtasks_completed = 0
    for v in env.vehicles:
        total_subtasks += v.task_dag.num_subtasks
        completed_subtasks += np.sum(v.task_dag.status == 3)
        
        # 统计V2V子任务
        if hasattr(v, 'exec_locations'):
            for i, loc in enumerate(v.exec_locations):
                if isinstance(loc, int):  # V2V卸载
                    v2v_subtasks_attempted += 1
                    if v.task_dag.status[i] == 3:  # 已完成
                        v2v_subtasks_completed += 1
    
    subtask_success_rate = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
    v2v_subtask_success_rate = (v2v_subtasks_completed / v2v_subtasks_attempted) if v2v_subtasks_attempted > 0 else 0.0
    
    # 计算平均指标
    pct_local = (stats['local_cnt'] / total_decisions * 100) if total_decisions > 0 else 0
    pct_rsu = (stats['rsu_cnt'] / total_decisions * 100) if total_decisions > 0 else 0
    pct_v2v = (stats['neighbor_cnt'] / total_decisions * 100) if total_decisions > 0 else 0
    avg_power = stats['power_sum'] / total_decisions if total_decisions > 0 else 0
    avg_veh_queue = stats['queue_len_sum'] / total_decisions if total_decisions > 0 else 0
    avg_rsu_queue = stats['rsu_queue_sum'] / total_steps if total_steps > 0 else 0
    
    return {
        'total_reward': ep_reward,
        'avg_step_reward': avg_step_reward,
        'veh_success_rate': veh_success_rate,
        'vehicle_success_rate': veh_success_rate,
        'task_success_rate': task_success_rate,
        'subtask_success_rate': subtask_success_rate,
        'v2v_subtask_success_rate': v2v_subtask_success_rate,
        'pct_local': pct_local,
        'pct_rsu': pct_rsu,
        'pct_v2v': pct_v2v,
        'avg_power': avg_power,
        'avg_queue_len': avg_veh_queue,
        'avg_rsu_queue': avg_rsu_queue,
    }


def main():
    disable_baseline_eval = False
    env_reward_mode = os.environ.get("REWARD_MODE")
    env_bonus_mode = os.environ.get("BONUS_MODE")
    env_seed = os.environ.get("SEED")
    env_max_episodes = os.environ.get("MAX_EPISODES")
    env_max_steps = os.environ.get("MAX_STEPS")
    env_eval_interval = os.environ.get("EVAL_INTERVAL")
    env_save_interval = os.environ.get("SAVE_INTERVAL")
    env_disable_baseline = os.environ.get("DISABLE_BASELINE_EVAL")
    env_use_lr_decay = os.environ.get("USE_LR_DECAY")
    env_device = os.environ.get("DEVICE_NAME")

    if env_reward_mode:
        Cfg.REWARD_MODE = env_reward_mode
    if env_bonus_mode:
        Cfg.BONUS_MODE = env_bonus_mode
    if env_max_episodes:
        TC.MAX_EPISODES = int(env_max_episodes)
    if env_max_steps:
        TC.MAX_STEPS = int(env_max_steps)
    if env_eval_interval:
        TC.EVAL_INTERVAL = int(env_eval_interval)
    if env_save_interval:
        TC.SAVE_INTERVAL = int(env_save_interval)
    if env_use_lr_decay is not None:
        TC.USE_LR_DECAY = env_use_lr_decay.lower() in ("1", "true", "yes")
    if env_device:
        TC.DEVICE_NAME = env_device
    if env_disable_baseline:
        disable_baseline_eval = env_disable_baseline.lower() in ("1", "true", "yes")

    if env_seed is not None:
        seed = int(env_seed)
        Cfg.SEED = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 开启 CuDNN 加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 初始化配置和日志记录器
    exp_name = f"MAPPO_DAG_N{Cfg.MIN_NODES}-{Cfg.MAX_NODES}_Veh{Cfg.NUM_VEHICLES}"
    recorder = DataRecorder(experiment_name=exp_name)

    # 构建配置字典
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
    recorder.save_config(config_dict)

    # 打印实验信息
    print(f"{'=' * 60}")
    print(f" Experiment: {exp_name}")
    print(f" Device:     {device}")
    print(f" Max Eps:    {hyperparams['max_episodes']}")
    print(f" LR Actor:   {hyperparams['lr_actor']}")
    print(f"{'=' * 60}")

    # 初始化环境
    env = VecOffloadingEnv()

    # 初始化网络和智能体
    network = OffloadingPolicyNetwork(
        d_model=TC.EMBED_DIM,
        num_heads=TC.NUM_HEADS,
        num_layers=TC.NUM_LAYERS
    )
    agent = MAPPOAgent(network, device=device)

    # 初始化经验缓冲区
    buffer = RolloutBuffer(gamma=TC.GAMMA, gae_lambda=TC.GAE_LAMBDA)

    best_reward = -float('inf')
    
    # Baseline策略列表
    baseline_policies = ['Random', 'Local-Only', 'Greedy']
    
    # 存储baseline的episode级指标（用于绘图）
    baseline_history = {policy: [] for policy in baseline_policies}

    print("\n[Info] Start Training...")

    for episode in range(1, hyperparams['max_episodes'] + 1):

        # 学习率衰减
        if TC.USE_LR_DECAY and episode > 0 and episode % TC.LR_DECAY_STEPS == 0:
            agent.decay_lr()
            print(f"[Info] LR Decayed at Episode {episode}")

        # 重置环境
        obs_list, _ = env.reset()

        ep_reward = 0
        ep_start_time = time.time()
        step_logs_buffer = []

        # 统计容器
        stats = {
            "power_sum": 0.0,
            "local_cnt": 0,
            "rsu_cnt": 0,
            "neighbor_cnt": 0,
            "queue_len_sum": 0,
            "rsu_queue_sum": 0,
            "assigned_cpu_sum": 0.0,
            "agent_rewards_sum": 0.0,
            "agent_rewards_count": 0,
            "v2v_count": 0
        }

        # Rollout循环
        for step in range(hyperparams['max_steps_per_ep']):
            # 智能体决策
            action_dict = agent.select_action(obs_list, deterministic=False)
            actions = action_dict['actions']
            log_probs = action_dict['log_probs']
            values = action_dict['values']

            # 环境步进
            next_obs_list, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # 统计
            stats["agent_rewards_sum"] += sum(rewards)
            stats["agent_rewards_count"] += len(rewards)

            # 存入Buffer
            buffer.add(obs_list, actions, rewards, values, log_probs, done)

            # 过程统计
            num_agents = len(rewards) if len(rewards) > 0 else 1
            step_r = sum(rewards) / num_agents
            ep_reward += step_r

            num_vehs = len(env.vehicles)
            for i, act in enumerate(actions):
                if i >= num_vehs:
                    break
                tgt = act['target']
                stats['power_sum'] += act['power']
                stats['queue_len_sum'] += env.vehicles[i].task_queue_len

                if tgt == 0:
                    stats['local_cnt'] += 1
                    stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq
                elif tgt == 1:
                    stats['rsu_cnt'] += 1
                    stats['assigned_cpu_sum'] += Cfg.F_RSU
                else:
                    stats['neighbor_cnt'] += 1
                    stats['v2v_count'] += 1
                    neighbor_idx = tgt - 2
                    candidate_vehs = [v for v in env.vehicles if v.id != env.vehicles[i].id]
                    if 0 <= neighbor_idx < len(candidate_vehs):
                        target_veh = candidate_vehs[neighbor_idx]
                        stats['assigned_cpu_sum'] += target_veh.cpu_freq
                    else:
                        stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq

            stats['rsu_queue_sum'] += sum(q.get_total_load() for q in env.rsus[0].queue_manager.processor_queues) if env.rsus else 0

            # 记录详细日志
            for i, act in enumerate(actions):
                step_logs_buffer.append({
                    "episode": episode, "step": step, "veh_id": i,
                    "target": act['target'],
                    "power": f"{act['power']:.3f}",
                    "reward": f"{rewards[i]:.3f}",
                    "q_len": env.vehicles[i].task_queue_len
                })

            obs_list = next_obs_list
            if done:
                break

        # Episode结束后的分析与更新
        total_steps = step + 1
        total_decisions = stats["agent_rewards_count"] if stats["agent_rewards_count"] > 0 else 1

        # 简化的公平性指数（使用平均值，因为我们不再跟踪每个agent的单独奖励）
        avg_agent_reward = stats["agent_rewards_sum"] / total_decisions if total_decisions > 0 else 0
        fairness_index = 1.0  # 简化处理

        # 个体奖励差异（简化处理）
        reward_gap = 0.0

        # 协作率
        collaboration_rate = (stats['v2v_count'] / total_decisions) * 100 if total_decisions > 0 else 0

        # PPO更新
        last_value = agent.get_value(obs_list)
        buffer.compute_returns_and_advantages(last_value)
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

        # 成功率统计（存储为0-1，展示时再乘100）
        episode_vehicle_count = len(env.vehicles)
        success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished])
        veh_success_rate = success_count / max(episode_vehicle_count, 1)
        task_success_rate = success_count / max(episode_vehicle_count, 1)

        total_subtasks = 0
        completed_subtasks = 0
        v2v_subtasks_attempted = 0
        v2v_subtasks_completed = 0
        for v in env.vehicles:
            total_subtasks += v.task_dag.num_subtasks
            completed_subtasks += np.sum(v.task_dag.status == 3)
            
            # 统计V2V子任务（通过检查task_locations）
            if hasattr(v, 'exec_locations'):
                for i, loc in enumerate(v.exec_locations):
                    if isinstance(loc, int):  # V2V卸载
                        v2v_subtasks_attempted += 1
                        if v.task_dag.status[i] == 3:  # 已完成
                            v2v_subtasks_completed += 1

        subtask_success_rate = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
        v2v_subtask_success_rate = (v2v_subtasks_completed / v2v_subtasks_attempted) if v2v_subtasks_attempted > 0 else 0.0

        # 控制台输出
        if episode == 1 or episode % 10 == 0:
            header = (
                f"{'Ep':<5} | {'Reward':<9} {'AvgR':<7} | {'Veh%':<5} {'Sub%':<5} {'V2VS%':<6} | {'MA_F':<4} {'CPU':<4} | "
                f"{'Loc%':<5} {'RSU%':<5} {'V2V%':<5} | {'AvgQ':<6} {'Pwr':<5} | {'Loss':<8} | {'Time':<4}"
            )
            print("-" * len(header))
            print(header)
            print("-" * len(header))

        print(f"{episode:<5d} | "
              f"{ep_reward:<9.2f} {avg_step_reward:<7.3f} | "
              f"{veh_success_rate*100:<5.1f} {subtask_success_rate*100:<5.1f} {v2v_subtask_success_rate*100:<6.1f} | "
              f"{fairness_index:<4.2f} {avg_assigned_cpu/1e9:<4.2f} | "
              f"{pct_local:<5.1f} {pct_rsu:<5.1f} {pct_v2v:<5.1f} | "
              f"{avg_veh_queue:<6.2f} {avg_power:<5.2f} | "
              f"{update_loss:<8.3f} | "
              f"{duration:<4.1f}")

        # 记录到Tensorboard/CSV
        recorder.log_step(step_logs_buffer)

        episode_metrics = {
            "episode": episode,
            "total_reward": ep_reward,
            "avg_step_reward": avg_step_reward,
            "loss": update_loss,
            "veh_success_rate": veh_success_rate,
            "vehicle_success_rate": veh_success_rate,
            "task_success_rate": task_success_rate,
            "subtask_success_rate": subtask_success_rate,
            "v2v_subtask_success_rate": v2v_subtask_success_rate,
            "pct_local": pct_local,
            "pct_rsu": pct_rsu,
            "pct_v2v": pct_v2v,
            "avg_power": avg_power,
            "avg_queue_len": avg_veh_queue,
            "ma_fairness": fairness_index,
            "ma_reward_gap": reward_gap,
            "ma_collaboration": collaboration_rate,
            "max_agent_reward": avg_agent_reward,
            "min_agent_reward": avg_agent_reward,
            "avg_assigned_cpu_ghz": avg_assigned_cpu / 1e9,
            "episode_vehicle_count": episode_vehicle_count,
            "episode_task_count": episode_vehicle_count,
            "duration": duration
        }
        recorder.log_episode(episode_metrics)

        # 定期评估baseline策略（与当前训练方法对比）
        if (not disable_baseline_eval) and (episode % TC.EVAL_INTERVAL == 0 or episode == 1):
            for policy_name in baseline_policies:
                baseline_metrics = evaluate_single_baseline_episode(env, policy_name)
                baseline_metrics['episode'] = episode
                baseline_metrics['policy'] = policy_name
                
                # 存储历史记录
                baseline_history[policy_name].append(baseline_metrics)
                
                # 记录到TensorBoard
                if recorder.writer is not None:
                    recorder.writer.add_scalar(f'Baseline/{policy_name}/total_reward', 
                                              baseline_metrics['total_reward'], episode)
                    recorder.writer.add_scalar(f'Baseline/{policy_name}/veh_success_rate', 
                                              baseline_metrics['veh_success_rate'], episode)
                    recorder.writer.add_scalar(f'Baseline/{policy_name}/subtask_success_rate', 
                                              baseline_metrics['subtask_success_rate'], episode)
                    recorder.writer.add_scalar(f'Baseline/{policy_name}/v2v_subtask_success_rate', 
                                              baseline_metrics['v2v_subtask_success_rate'], episode)
                
                # 记录到CSV（使用log_episode，但添加policy字段）
                baseline_episode_dict = {
                    "episode": episode,
                    "total_reward": baseline_metrics['total_reward'],
                    "avg_step_reward": baseline_metrics['avg_step_reward'],
                    "veh_success_rate": baseline_metrics['veh_success_rate'],
                    "vehicle_success_rate": baseline_metrics['veh_success_rate'],
                    "task_success_rate": baseline_metrics.get('task_success_rate', baseline_metrics['veh_success_rate']),
                    "subtask_success_rate": baseline_metrics['subtask_success_rate'],
                    "v2v_subtask_success_rate": baseline_metrics['v2v_subtask_success_rate'],
                    "pct_local": baseline_metrics['pct_local'],
                    "pct_rsu": baseline_metrics['pct_rsu'],
                    "pct_v2v": baseline_metrics['pct_v2v'],
                    "avg_power": baseline_metrics['avg_power'],
                    "avg_queue_len": baseline_metrics['avg_queue_len'],
                    "ma_fairness": 1.0,  # baseline无公平性概念，设为1.0
                    "ma_reward_gap": 0.0,
                    "ma_collaboration": baseline_metrics['pct_v2v'],
                    "max_agent_reward": baseline_metrics['total_reward'],
                    "min_agent_reward": baseline_metrics['total_reward'],
                    "avg_assigned_cpu_ghz": 0.0,  # baseline无此指标
                    "duration": 0.0,
                    "loss": 0.0,
                    "policy": policy_name  # 标识这是baseline
                }
                recorder.log_episode(baseline_episode_dict)

        # 模型保存
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(recorder.model_dir, "best_model.pth"))

        if episode % TC.SAVE_INTERVAL == 0:
            agent.save(os.path.join(recorder.model_dir, f"model_ep{episode}.pth"))

    # 训练结束与绘图
    print("\n[Info] Training Finished.")
    print("[Info] Generating plots...")
    recorder.auto_plot(baseline_history=baseline_history)
    print(f"[Info] Plots saved to {recorder.plot_dir}")


if __name__ == "__main__":
    main()
