import time
import json
import csv
import numpy as np
import torch
import os
import sys
import random
import re
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.mappo_agent import MAPPOAgent
from agents.rollout_buffer import RolloutBuffer
from utils.data_recorder import DataRecorder
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _read_last_jsonl(path):
    last = None
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


def _format_table_header(columns):
    parts = []
    for col in columns:
        label, width = col
        parts.append(str(label).ljust(width))
    return " ".join(parts)


def _format_table_row(values, columns):
    parts = []
    for col in columns:
        key, width = col
        val = values.get(key)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            cell = "-"
        elif isinstance(val, (int, np.integer)):
            cell = str(val)
        elif isinstance(val, float):
            cell = f"{val:.3f}"
        else:
            cell = str(val)
        parts.append(cell.rjust(width))
    return " ".join(parts)


def _collect_obs_stats(obs_list):
    if not obs_list:
        return {}
    def _stack(key):
        arrs = [obs.get(key) for obs in obs_list if obs.get(key) is not None]
        if not arrs:
            return None
        return np.stack(arrs)

    stats = {}
    node_x = _stack("node_x")
    if node_x is not None:
        stats["obs/node_x_mean"] = float(np.mean(node_x))
        stats["obs/node_x_std"] = float(np.std(node_x))
    self_info = _stack("self_info")
    if self_info is not None:
        stats["obs/self_info_mean"] = float(np.mean(self_info))
        stats["obs/self_info_std"] = float(np.std(self_info))
    neighbors = _stack("neighbors")
    if neighbors is not None:
        stats["obs/neighbors_mean"] = float(np.mean(neighbors))
        stats["obs/neighbors_std"] = float(np.std(neighbors))
    resource_raw = _stack("resource_raw")
    if resource_raw is not None:
        stats["obs/resource_raw_mean"] = float(np.mean(resource_raw))
        stats["obs/resource_raw_std"] = float(np.std(resource_raw))
    action_mask = _stack("action_mask")
    if action_mask is not None:
        stats["obs/action_mask_true_frac"] = float(np.mean(action_mask))
    target_mask = _stack("target_mask")
    if target_mask is not None:
        stats["obs/target_mask_true_frac"] = float(np.mean(target_mask))
    return stats


def _inject_obs_stamp(obs_list, actions):
    for i, act in enumerate(actions):
        if act is None:
            continue
        if i < len(obs_list) and "obs_stamp" in obs_list[i] and "obs_stamp" not in act:
            act["obs_stamp"] = int(obs_list[i]["obs_stamp"])


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return str(obj)


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
            _inject_obs_stamp(obs_list, actions)
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
            _inject_obs_stamp(obs_list, actions)
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
            _inject_obs_stamp(obs_list, actions)
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
        _inject_obs_stamp(obs_list, actions)
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
    
    # 计算平均指标（内部用比例）
    dec_den = total_decisions if total_decisions > 0 else 1
    frac_local = (stats['local_cnt'] / dec_den) if total_decisions > 0 else 0.0
    frac_rsu = (stats['rsu_cnt'] / dec_den) if total_decisions > 0 else 0.0
    frac_v2v = (stats['neighbor_cnt'] / dec_den) if total_decisions > 0 else 0.0
    avg_power = stats['power_sum'] / dec_den if total_decisions > 0 else 0.0
    avg_veh_queue = stats['queue_len_sum'] / dec_den if total_decisions > 0 else 0.0
    avg_rsu_queue = stats['rsu_queue_sum'] / total_steps if total_steps > 0 else 0.0
    
    return {
        'total_reward': ep_reward,
        'avg_step_reward': avg_step_reward,
        'veh_success_rate': veh_success_rate,
        'vehicle_success_rate': veh_success_rate,
        'task_success_rate': task_success_rate,
        'subtask_success_rate': subtask_success_rate,
        'v2v_subtask_success_rate': v2v_subtask_success_rate,
        'decision_frac_local': frac_local,
        'decision_frac_rsu': frac_rsu,
        'decision_frac_v2v': frac_v2v,
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
    env_log_interval = os.environ.get("LOG_INTERVAL")
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
    if env_log_interval:
        TC.LOG_INTERVAL = int(env_log_interval)
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

    start_ts = time.strftime("%Y%m%d_%H%M%S")
    def _has_ts(name):
        return bool(re.search(r"\d{8}_\d{6}$", name))
    run_dir_env = os.environ.get("RUN_DIR")
    run_id_env = os.environ.get("RUN_ID")
    if run_dir_env:
        run_dir = run_dir_env
        base = os.path.basename(run_dir.rstrip(os.sep))
        if not _has_ts(base):
            run_dir = f"{run_dir_env}_{start_ts}"
            base = os.path.basename(run_dir)
        run_id = run_id_env or base
        if not _has_ts(run_id):
            run_id = f"{run_id}_{start_ts}"
    else:
        run_id = run_id_env or f"run_{start_ts}"
        if not _has_ts(run_id):
            run_id = f"{run_id}_{start_ts}"
        run_dir = os.path.join("runs", run_id)
    run_dir = os.path.abspath(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    metrics_dir = os.path.join(run_dir, "metrics")
    plots_dir = os.path.join(run_dir, "plots")
    models_dir = os.path.join(run_dir, "models")
    _ensure_dir(run_dir)
    _ensure_dir(logs_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)
    _ensure_dir(models_dir)
    os.environ["RUN_ID"] = run_id
    os.environ["RUN_DIR"] = run_dir
    os.environ["MAX_EPISODES"] = str(TC.MAX_EPISODES)
    os.environ["REWARD_MODE"] = str(Cfg.REWARD_MODE)
    os.environ["SEED"] = str(Cfg.SEED)

    reward_jsonl_path = os.environ.get("REWARD_JSONL_PATH")
    if not reward_jsonl_path:
        reward_jsonl_path = os.path.join(logs_dir, "env_reward.jsonl")
        os.environ["REWARD_JSONL_PATH"] = reward_jsonl_path

    tb_log_obs = os.environ.get("TB_LOG_OBS")
    log_obs_stats = True
    if tb_log_obs is not None:
        log_obs_stats = tb_log_obs.lower() in ("1", "true", "yes")

    # 初始化配置和日志记录器
    exp_name = f"MAPPO_DAG_N{Cfg.MIN_NODES}-{Cfg.MAX_NODES}_Veh{Cfg.NUM_VEHICLES}"
    recorder = DataRecorder(experiment_name=exp_name, base_dir=run_dir)

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

    train_config_dict = {}
    for k, v in TC.__dict__.items():
        if k.startswith('__') or isinstance(v, (staticmethod, classmethod)) or callable(v):
            continue
        train_config_dict[k] = v

    env_snapshot = {
        "CFG_PROFILE": os.environ.get("CFG_PROFILE"),
        "REWARD_MODE": os.environ.get("REWARD_MODE"),
        "BONUS_MODE": os.environ.get("BONUS_MODE"),
        "SEED": os.environ.get("SEED"),
        "RUN_ID": run_id,
        "RUN_DIR": run_dir,
        "REWARD_JSONL_PATH": reward_jsonl_path,
        "DEVICE_NAME": device,
    }
    snapshot = {
        "system_config": config_dict,
        "train_config": train_config_dict,
        "env": env_snapshot,
    }
    with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=True, indent=2, default=_json_default)

    # 打印实验信息
    print(f"{'=' * 60}")
    print(f" Experiment: {exp_name}")
    print(f" Device:     {device}")
    print(f" Run Dir:    {run_dir}")
    print(f" Reward:     {Cfg.REWARD_MODE}")
    print(f" Seed:       {Cfg.SEED}")
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
    metrics_csv_path = os.path.join(metrics_dir, "train_metrics.csv")
    metrics_jsonl_path = os.path.join(metrics_dir, "train_metrics.jsonl")
    metrics_header_written = os.path.exists(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0
    disable_auto_plot = os.environ.get("DISABLE_AUTO_PLOT", "").lower() in ("1", "true", "yes")
    table_columns = [
        ("ep", 6),
        ("steps", 7),
        ("reward", 10),
        ("entropy", 8),
        ("clip", 6),
        ("v_loss", 8),
        ("p_loss", 8),
        ("succ", 6),
        ("miss", 6),
        ("L", 5),
        ("R", 5),
        ("V", 5),
        ("mean_cft", 9),
        ("delta_cft", 9),
        ("elapsed", 8),
    ]
    table_row_count = 0

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
        terminated = False
        truncated = False

        # Rollout循环
        for step in range(hyperparams['max_steps_per_ep']):
            # 智能体决策
            action_dict = agent.select_action(obs_list, deterministic=False)
            actions = action_dict['actions']
            log_probs = action_dict['log_probs']
            values = action_dict['values']

            # 环境步进
            _inject_obs_stamp(obs_list, actions)
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
                    resource_ids = obs_list[i].get('resource_ids') if i < len(obs_list) else None
                    if resource_ids is not None and 0 <= tgt < len(resource_ids):
                        token = int(resource_ids[tgt])
                        if token >= 3:
                            neighbor_id = token - 3
                            target_veh = next((v for v in env.vehicles if v.id == neighbor_id), None)
                            stats['assigned_cpu_sum'] += target_veh.cpu_freq if target_veh else env.vehicles[i].cpu_freq
                        else:
                            stats['assigned_cpu_sum'] += env.vehicles[i].cpu_freq
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

        if not (terminated or truncated):
            env._log_episode_stats(False, True)

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

        frac_local = (stats['local_cnt'] / total_decisions)
        frac_rsu = (stats['rsu_cnt'] / total_decisions)
        frac_v2v = (stats['neighbor_cnt'] / total_decisions)

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

        if Cfg.DEBUG_ASSERT_METRICS:
            for name, val in [
                ("veh_success_rate", veh_success_rate),
                ("task_success_rate", task_success_rate),
                ("subtask_success_rate", subtask_success_rate),
                ("v2v_subtask_success_rate", v2v_subtask_success_rate),
                ("decision_frac_local", frac_local),
                ("decision_frac_rsu", frac_rsu),
                ("decision_frac_v2v", frac_v2v),
            ]:
                assert 0.0 <= val <= 1.0 + 1e-6, f"{name} out of range: {val}"
            assert abs((frac_local + frac_rsu + frac_v2v) - 1.0) <= 1e-3 or total_decisions == 0, \
                f"decision fractions not summing to 1: {frac_local + frac_rsu + frac_v2v}"

        env_stats = _read_last_jsonl(reward_jsonl_path)
        env_metrics = {}
        if env_stats and isinstance(env_stats.get("metrics"), dict):
            for key, stat in env_stats["metrics"].items():
                if isinstance(stat, dict):
                    env_metrics[f"{key}.mean"] = stat.get("mean")
                    env_metrics[f"{key}.p95"] = stat.get("p95")

        reward_abs_mean = env_metrics.get("reward_abs.mean")
        reward_display = reward_abs_mean if reward_abs_mean is not None else avg_step_reward
        success_rate_end = env_stats.get("success_rate_end") if env_stats else veh_success_rate
        subtask_success = env_stats.get("subtask_success_rate") if env_stats else subtask_success_rate
        deadline_miss_rate = env_stats.get("deadline_miss_rate") if env_stats else 0.0
        mean_cft = env_stats.get("mean_cft") if env_stats else None
        frac_local = env_stats.get("decision_frac_local", frac_local) if env_stats else frac_local
        frac_rsu = env_stats.get("decision_frac_rsu", frac_rsu) if env_stats else frac_rsu
        frac_v2v = env_stats.get("decision_frac_v2v", frac_v2v) if env_stats else frac_v2v
        clip_hit_ratio = env_stats.get("clip_hit_ratio") if env_stats else None
        delta_cft_mean = env_metrics.get("delta_cft.mean")

        # 控制台输出（每 LOG_INTERVAL 一行）
        if episode == 1 or episode % TC.LOG_INTERVAL == 0:
            if table_row_count % 20 == 0:
                print(_format_table_header(table_columns), flush=True)
            table_row = {
                "ep": episode,
                "steps": env_stats.get("episode_steps", total_steps) if env_stats else total_steps,
                "reward": reward_display,
                "entropy": None,
                "clip": clip_hit_ratio,
                "v_loss": None,
                "p_loss": None,
                "succ": success_rate_end,
                "miss": deadline_miss_rate,
                "L": frac_local,
                "R": frac_rsu,
                "V": frac_v2v,
                "mean_cft": mean_cft,
                "delta_cft": delta_cft_mean,
                "elapsed": duration,
            }
            print(_format_table_row(table_row, table_columns), flush=True)
            table_row_count += 1

        metrics_row = {
            "episode": episode,
            "episode_steps": env_stats.get("episode_steps", total_steps) if env_stats else total_steps,
            "reward_mode": env_stats.get("reward_mode", Cfg.REWARD_MODE) if env_stats else Cfg.REWARD_MODE,
            "seed": env_stats.get("seed", Cfg.SEED) if env_stats else Cfg.SEED,
            "reward_mean": reward_display,
            "entropy": None,
            "clip_fraction": clip_hit_ratio,
            "value_loss": None,
            "policy_loss": None,
            "mean_cft": mean_cft,
            "delta_cft_mean": delta_cft_mean,
            "success_rate_end": success_rate_end,
            "deadline_miss_rate": deadline_miss_rate,
            "subtask_success_rate": subtask_success,
            "decision_frac_local": frac_local,
            "decision_frac_rsu": frac_rsu,
            "decision_frac_v2v": frac_v2v,
            "clip_hit_ratio": clip_hit_ratio,
            "avg_step_reward": avg_step_reward,
            "update_loss": update_loss,
            "elapsed_time": duration,
        }
        metrics_row.update(env_metrics)

        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row, ensure_ascii=True, default=_json_default) + "\n")

        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
            if not metrics_header_written:
                writer.writeheader()
                metrics_header_written = True
            writer.writerow(metrics_row)

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
            "decision_frac_local": frac_local,
            "decision_frac_rsu": frac_rsu,
            "decision_frac_v2v": frac_v2v,
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
        if log_obs_stats and recorder.writer is not None:
            obs_stats = _collect_obs_stats(obs_list)
            for key, val in obs_stats.items():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    recorder.writer.add_scalar(key, val, episode)

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
                    "decision_frac_local": baseline_metrics['decision_frac_local'],
                    "decision_frac_rsu": baseline_metrics['decision_frac_rsu'],
                    "decision_frac_v2v": baseline_metrics['decision_frac_v2v'],
                    "avg_power": baseline_metrics['avg_power'],
                    "avg_queue_len": baseline_metrics['avg_queue_len'],
                    "ma_fairness": 1.0,  # baseline无公平性概念，设为1.0
                    "ma_reward_gap": 0.0,
                    "ma_collaboration": baseline_metrics['decision_frac_v2v'],
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
    if not disable_auto_plot:
        print("[Info] Generating plots...")
        recorder.auto_plot(baseline_history=baseline_history)
        print(f"[Info] Plots saved to {recorder.plot_dir}")
    else:
        print("[Info] Auto-plot disabled.")


if __name__ == "__main__":
    main()
