import time
import json
import csv
import numpy as np
import torch
import os
import sys
import random
import re
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg, apply_profile
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
        parts.append(str(label).center(width))
    return "| " + " | ".join(parts) + " |"


def _format_table_divider(columns):
    parts = []
    for _, width in columns:
        parts.append("-" * width)
    return "+-" + "-+-".join(parts) + "-+"


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
    return "| " + " | ".join(parts) + " |"


def _compute_time_limit_penalty(mode, remaining_time, deadline, base_penalty, k, ratio_clip):
    """
    Compute scaled time-limit penalty.
    mode: "fixed" or "scaled"
    remaining_time: seconds remaining to finish
    deadline: deadline seconds
    """
    if mode == "scaled":
        if remaining_time is None or not np.isfinite(remaining_time):
            return 0.0, 0.0
        denom = max(deadline if deadline is not None and np.isfinite(deadline) else 1.0, 1e-6)
        ratio = np.clip(remaining_time / denom, 0.0, ratio_clip)
        penalty = -float(k) * float(ratio)
        return penalty, ratio
    else:
        return float(base_penalty), 0.0


def _parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO offloading policy.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--bonus-mode", type=str, default=None)
    parser.add_argument("--cfg-profile", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--step-metrics", action="store_true", default=False)
    parser.add_argument("--no-step-metrics", action="store_true", default=False)
    return parser.parse_args()


def _env_int(name):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_str(name):
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = str(raw).strip()
    return raw if raw else None


def _bool_env(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


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


def _snapshot_reward_stats(env):
    snapshot = {}
    if not hasattr(env, "_reward_stats"):
        return snapshot
    for name, bucket in env._reward_stats.metrics.items():
        snapshot[name] = (bucket.sum, bucket.count)
    return snapshot


def _delta_mean(snapshot, env, key):
    if not hasattr(env, "_reward_stats"):
        return None
    bucket = env._reward_stats.metrics.get(key)
    if bucket is None:
        return None
    prev_sum, prev_count = snapshot.get(key, (0.0, 0))
    delta_sum = bucket.sum - prev_sum
    delta_count = bucket.count - prev_count
    if delta_count <= 0:
        return None
    return delta_sum / delta_count


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
    args = _parse_args()
    disable_baseline_eval = False
    if args.cfg_profile:
        os.environ["CFG_PROFILE"] = args.cfg_profile
        apply_profile(args.cfg_profile)

    if os.environ.get("EPISODE_JSONL_STDOUT") is None:
        Cfg.EPISODE_JSONL_STDOUT = False

    env_reward_mode = _env_str("REWARD_MODE")
    env_bonus_mode = _env_str("BONUS_MODE")
    env_seed = _env_int("SEED")
    env_max_episodes = _env_int("MAX_EPISODES")
    env_max_steps = _env_int("MAX_STEPS")
    env_log_interval = _env_int("LOG_INTERVAL")
    env_eval_interval = _env_int("EVAL_INTERVAL")
    env_save_interval = _env_int("SAVE_INTERVAL")
    env_disable_baseline = _env_str("DISABLE_BASELINE_EVAL")
    env_use_lr_decay = _env_str("USE_LR_DECAY")
    env_device = _env_str("DEVICE_NAME")

    reward_mode = args.reward_mode or env_reward_mode or Cfg.REWARD_MODE
    bonus_mode = args.bonus_mode or env_bonus_mode or Cfg.BONUS_MODE
    if reward_mode:
        Cfg.REWARD_MODE = reward_mode
    if bonus_mode:
        Cfg.BONUS_MODE = bonus_mode

    if args.max_episodes is not None:
        TC.MAX_EPISODES = int(args.max_episodes)
    elif env_max_episodes is not None:
        TC.MAX_EPISODES = int(env_max_episodes)
    else:
        profile_max = getattr(Cfg, "MAX_EPISODES", None)
        if profile_max is not None:
            TC.MAX_EPISODES = int(profile_max)

    if args.max_steps is not None:
        TC.MAX_STEPS = int(args.max_steps)
    elif env_max_steps is not None:
        TC.MAX_STEPS = int(env_max_steps)
    else:
        # Respect SystemConfig overrides (e.g., CFG_PROFILE) if MAX_STEPS not explicitly set.
        try:
            if int(TC.MAX_STEPS) != int(Cfg.MAX_STEPS):
                TC.MAX_STEPS = int(Cfg.MAX_STEPS)
        except Exception:
            pass

    if args.log_interval is not None:
        TC.LOG_INTERVAL = int(args.log_interval)
    elif env_log_interval is not None:
        TC.LOG_INTERVAL = int(env_log_interval)

    if args.eval_interval is not None:
        TC.EVAL_INTERVAL = int(args.eval_interval)
    elif env_eval_interval is not None:
        TC.EVAL_INTERVAL = int(env_eval_interval)

    if args.save_interval is not None:
        TC.SAVE_INTERVAL = int(args.save_interval)
    elif env_save_interval is not None:
        TC.SAVE_INTERVAL = int(env_save_interval)

    if env_use_lr_decay is not None:
        TC.USE_LR_DECAY = env_use_lr_decay.lower() in ("1", "true", "yes")
    if args.device:
        TC.DEVICE_NAME = args.device
    elif env_device:
        TC.DEVICE_NAME = env_device

    if env_disable_baseline:
        disable_baseline_eval = env_disable_baseline.lower() in ("1", "true", "yes")

    seed = args.seed if args.seed is not None else env_seed
    if seed is not None:
        Cfg.SEED = int(seed)
        np.random.seed(int(seed))
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    log_step_metrics = _bool_env("LOG_STEP_METRICS", False)
    if args.step_metrics:
        log_step_metrics = True
    if args.no_step_metrics:
        log_step_metrics = False

    # 开启 CuDNN 加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    start_ts = time.strftime("%Y%m%d_%H%M%S")
    def _has_ts(name):
        return bool(re.search(r"\d{8}_\d{6}$", name))
    run_dir_env = args.run_dir or os.environ.get("RUN_DIR")
    run_id_env = args.run_id or os.environ.get("RUN_ID")
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
    recorder = DataRecorder(experiment_name=exp_name, base_dir=run_dir, quiet=True)

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
    config_snapshot_path = os.path.join(logs_dir, "config_snapshot.json")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=True, indent=2, default=_json_default)

    # 打印生效配置表（stdout仅保留这一张表 + 训练表格行）
    info_rows = [
        ("REWARD_MODE", Cfg.REWARD_MODE),
        ("SEED", Cfg.SEED),
        ("DT", Cfg.DT),
        ("MAX_STEPS", TC.MAX_STEPS),
        ("MAX_EPISODES", TC.MAX_EPISODES),
        ("CFG_PROFILE", os.environ.get("CFG_PROFILE")),
        ("BW_V2I", Cfg.BW_V2I),
        ("BW_V2V", Cfg.BW_V2V),
        ("V2V_RANGE", Cfg.V2V_RANGE),
        ("RSU_RANGE", Cfg.RSU_RANGE),
        ("MIN_CPU", Cfg.MIN_VEHICLE_CPU_FREQ),
        ("MAX_CPU", Cfg.MAX_VEHICLE_CPU_FREQ),
        ("F_RSU", Cfg.F_RSU),
        ("MIN_NODES", Cfg.MIN_NODES),
        ("MAX_NODES", Cfg.MAX_NODES),
        ("MIN_COMP", Cfg.MIN_COMP),
        ("MAX_COMP", Cfg.MAX_COMP),
        ("MIN_DATA", Cfg.MIN_DATA),
        ("MAX_DATA", Cfg.MAX_DATA),
    ]
    info_cols = [("key", 18), ("value", 32)]
    print(_format_table_divider(info_cols), flush=True)
    print(_format_table_header(info_cols), flush=True)
    print(_format_table_divider(info_cols), flush=True)
    for key, val in info_rows:
        print(_format_table_row({"key": key, "value": val}, info_cols), flush=True)
    print(_format_table_divider(info_cols), flush=True)

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
    metrics_csv_path = os.path.join(logs_dir, "metrics.csv")
    metrics_jsonl_path = os.path.join(logs_dir, "metrics.jsonl")
    legacy_metrics_csv_path = os.path.join(metrics_dir, "metrics.csv")
    legacy_metrics_jsonl_path = os.path.join(metrics_dir, "metrics.jsonl")
    legacy_train_csv_path = os.path.join(metrics_dir, "train_metrics.csv")
    legacy_train_jsonl_path = os.path.join(metrics_dir, "train_metrics.jsonl")
    _ensure_dir(metrics_dir)
    metrics_header_written = os.path.exists(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0
    legacy_metrics_header_written = os.path.exists(legacy_metrics_csv_path) and os.path.getsize(legacy_metrics_csv_path) > 0
    legacy_train_header_written = os.path.exists(legacy_train_csv_path) and os.path.getsize(legacy_train_csv_path) > 0
    disable_auto_plot = os.environ.get("DISABLE_AUTO_PLOT", "").lower() in ("1", "true", "yes")
    step_metrics_csv_path = os.path.join(logs_dir, "step_metrics.csv")
    step_metrics_header_written = os.path.exists(step_metrics_csv_path) and os.path.getsize(step_metrics_csv_path) > 0
    metrics_fields = [
        "episode",
        "steps",
        "elapsed_sec",
        "reward_mode",
        "seed",
        "terminated",
        "truncated",
        "termination_reason",
        "time_limit_rate",
        "episode_time_seconds",
        "mean_cft_est",
        "mean_cft_completed",
        "vehicle_cft_count",
        "cft_est_valid",
        "deadline_gamma",
        "deadline_seconds",
        "critical_path_cycles",
        "episode_vehicle_count",
        "episode_task_count",
        "total_subtasks",
        # reward: signed per-step mean/p95; abs_mean optional
        "reward_mean",
        "reward_p50",
        "reward_p95",
        "reward_min",
        "reward_max",
        "reward_abs_mean",
        # CFT metrics: mean_cft is absolute mean; delta_cft_rem is remaining-time delta
        "mean_cft",
        "delta_cft_rem_mean",
        "delta_cft_rem_p95",
        "mean_cft_rem",
        # success/safety
        "success_rate_end",
        "task_success_rate",
        "subtask_success_rate",
        "deadline_miss_rate",
        "illegal_action_rate",
        "hard_trigger_rate",
        # decisions
        "decision_local_frac",
        "decision_rsu_frac",
        "decision_v2v_frac",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
        # action power ratio
        "power_ratio_mean",
        "power_ratio_p95",
        # PPO diagnostics
        "entropy",
        "approx_kl",
        "clip_frac",
        "policy_loss",
        "value_loss",
        "total_loss",
        "grad_norm",
        "policy_entropy",
        "entropy_loss",
        # Diagnostics
        "avail_L",
        "avail_R",
        "avail_V",
        "neighbor_count_mean",
        "best_v2v_rate_mean",
        "best_v2v_valid_rate",
        "v2v_beats_rsu_rate",
        "mean_cost_gap_v2v_minus_rsu",
        "mean_cost_rsu",
        "mean_cost_v2v",
        "time_limit_penalty_applied",
        "time_limit_penalty_value",
        "remaining_time_seconds_used",
        "remaining_ratio_used",
    ]
    step_metrics_fields = [
        "episode",
        "step",
        "reward_mean",
        "delta_cft_mean",
        "cft_prev_rem_mean",
        "cft_curr_rem_mean",
        "cft_rem_ratio",
        "energy_norm_mean",
        "delay_norm_mean",
    ]
    table_columns = [
        ("ep", 4),
        ("steps", 6),
        ("r_mean", 8),
        ("r_p95", 8),
        ("ent", 7),
        ("kl", 8),
        ("clip_frac", 9),
        ("p_loss", 8),
        ("v_loss", 8),
        ("total_loss", 10),
        ("succ", 6),
        ("task", 6),
        ("sub", 6),
        ("miss", 6),
        ("ill", 6),
        ("hard", 6),
        ("L", 5),
        ("R", 5),
        ("V", 5),
        ("mean_cft", 9),
        ("dCFT", 9),
        ("power", 7),
        ("elapsed", 7),
    ]
    table_row_count = 0

    for episode in range(1, hyperparams['max_episodes'] + 1):

        # 学习率衰减
        if TC.USE_LR_DECAY and episode > 0 and episode % TC.LR_DECAY_STEPS == 0:
            agent.decay_lr()

        # 重置环境
        obs_list, _ = env.reset()

        ep_reward = 0
        ep_start_time = time.time()
        step_logs_buffer = []
        ep_step_rewards = []
        step_metrics_rows = []

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
            reward_snapshot = _snapshot_reward_stats(env) if log_step_metrics else {}
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
            ep_step_rewards.append(step_r)

            if log_step_metrics:
                delta_cft_step = _delta_mean(reward_snapshot, env, "delta_cft")
                cft_prev_rem_step = _delta_mean(reward_snapshot, env, "cft_prev_rem")
                cft_curr_rem_step = _delta_mean(reward_snapshot, env, "cft_curr_rem")
                energy_step = _delta_mean(reward_snapshot, env, "energy_norm")
                delay_step = _delta_mean(reward_snapshot, env, "delay_norm")
                cft_ratio = None
                if cft_curr_rem_step is not None and cft_curr_rem_step > 0:
                    cft_ratio = (cft_prev_rem_step or 0.0) / cft_curr_rem_step
                step_metrics_rows.append({
                    "episode": episode,
                    "step": step,
                    "reward_mean": step_r,
                    "delta_cft_mean": delta_cft_step,
                    "cft_prev_rem_mean": cft_prev_rem_step,
                    "cft_curr_rem_mean": cft_curr_rem_step,
                    "cft_rem_ratio": cft_ratio,
                    "energy_norm_mean": energy_step,
                    "delay_norm_mean": delay_step,
                })

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

        # 末尾截断惩罚（time_limit且未完成）
        time_limit_penalty_applied = False
        time_limit_penalty_value = 0.0
        remaining_time_used = None
        remaining_ratio_used = None
        if truncated and not terminated and buffer.rewards_buffer:
            # 估算剩余时间：优先用 env_metrics 的 cft_curr_rem.mean
            remaining_time_used = env_metrics.get("cft_curr_rem.mean")
            if remaining_time_used is None:
                remaining_time_used = mean_cft_rem
            deadline_used = deadline_seconds if deadline_seconds is not None else None
            penalty, ratio = _compute_time_limit_penalty(
                getattr(Cfg, "TIME_LIMIT_PENALTY_MODE", "fixed"),
                remaining_time_used if remaining_time_used is not None else 0.0,
                deadline_used if deadline_used is not None else episode_time_seconds,
                getattr(Cfg, "TIME_LIMIT_PENALTY", -1.0),
                getattr(Cfg, "TIME_LIMIT_PENALTY_K", 2.0),
                getattr(Cfg, "TIME_LIMIT_PENALTY_RATIO_CLIP", 3.0),
            )
            remaining_ratio_used = ratio
            buffer.rewards_buffer[-1] = buffer.rewards_buffer[-1] + penalty
            if ep_step_rewards:
                ep_step_rewards[-1] += penalty
            ep_reward += penalty
            time_limit_penalty_applied = True
            time_limit_penalty_value = penalty

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
        episode_task_count = episode_vehicle_count
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

        step_rewards = np.array(ep_step_rewards, dtype=np.float32) if ep_step_rewards else np.array([0.0], dtype=np.float32)
        reward_mean = float(np.mean(step_rewards))
        reward_p50 = float(np.percentile(step_rewards, 50))
        reward_p95 = float(np.percentile(step_rewards, 95))
        reward_min = float(np.min(step_rewards))
        reward_max = float(np.max(step_rewards))

        reward_abs_mean = env_metrics.get("reward_abs.mean")
        if reward_abs_mean is None:
            reward_abs_mean = float(np.mean(np.abs(step_rewards)))
        terminated_flag = bool(env_stats.get("terminated")) if env_stats else bool(terminated)
        truncated_flag = bool(env_stats.get("truncated")) if env_stats else bool(truncated)
        if terminated_flag:
            termination_reason = "all_finished"
        elif truncated_flag:
            termination_reason = "time_limit"
        else:
            termination_reason = "other"
        success_rate_end = env_stats.get("success_rate_end") if env_stats else veh_success_rate
        task_success_rate = env_stats.get("task_success_rate", task_success_rate) if env_stats else task_success_rate
        subtask_success = env_stats.get("subtask_success_rate") if env_stats else subtask_success_rate
        deadline_miss_rate = env_stats.get("deadline_miss_rate") if env_stats else 0.0
        illegal_action_rate = env_stats.get("illegal_action_rate") if env_stats else None
        hard_trigger_rate = env_stats.get("hard_trigger_rate") if env_stats else None
        time_limit_rate = env_stats.get("time_limit_rate") if env_stats else (1.0 if (truncated and not terminated) else 0.0)
        mean_cft = env_stats.get("mean_cft") if env_stats else None
        frac_local = env_stats.get("decision_frac_local", frac_local) if env_stats else frac_local
        frac_rsu = env_stats.get("decision_frac_rsu", frac_rsu) if env_stats else frac_rsu
        frac_v2v = env_stats.get("decision_frac_v2v", frac_v2v) if env_stats else frac_v2v
        clip_hit_ratio = env_stats.get("clip_hit_ratio") if env_stats else None
        delta_cft_rem_mean = env_metrics.get("delta_cft_rem.mean")
        delta_cft_rem_p95 = env_metrics.get("delta_cft_rem.p95")
        if delta_cft_rem_mean is None:
            delta_cft_rem_mean = env_metrics.get("delta_cft.mean")
        if delta_cft_rem_p95 is None:
            delta_cft_rem_p95 = env_metrics.get("delta_cft.p95")
        episode_time_seconds = env_stats.get("episode_time_seconds") if env_stats else (env.time if env else None)
        mean_cft_est = env_stats.get("mean_cft_est") if env_stats else None
        mean_cft_completed = env_stats.get("mean_cft_completed") if env_stats else None
        vehicle_cft_count = env_stats.get("vehicle_cft_count") if env_stats else 0
        cft_est_valid = env_stats.get("cft_est_valid") if env_stats else False
        mean_cft = episode_time_seconds  # 保持旧列但语义为episode时长
        mean_cft_rem = env_metrics.get("cft_curr_rem.mean")
        if mean_cft_rem is None and episode_time_seconds is not None:
            mean_cft_rem = max(episode_time_seconds - env.time, 0.0)
        power_ratio_mean = env_metrics.get("power_ratio.mean")
        power_ratio_p95 = env_metrics.get("power_ratio.p95")
        deadline_gamma = env_stats.get("deadline_gamma_mean") if env_stats else None
        deadline_seconds = env_stats.get("deadline_seconds_mean") if env_stats else None
        critical_path_cycles = env_stats.get("critical_path_cycles_mean") if env_stats else None
        avail_L = env_stats.get("avail_L") if env_stats else None
        avail_R = env_stats.get("avail_R") if env_stats else None
        avail_V = env_stats.get("avail_V") if env_stats else None
        neighbor_count_mean = env_stats.get("neighbor_count_mean") if env_stats else None
        best_v2v_rate_mean = env_stats.get("best_v2v_rate_mean") if env_stats else None
        best_v2v_valid_rate = env_stats.get("best_v2v_valid_rate") if env_stats else None
        if avail_L is None: avail_L = 0.0
        if avail_R is None: avail_R = 0.0
        if avail_V is None: avail_V = 0.0
        if neighbor_count_mean is None: neighbor_count_mean = 0.0
        if best_v2v_valid_rate is None or not (np.isfinite(best_v2v_valid_rate)): best_v2v_valid_rate = 0.0
        if best_v2v_rate_mean is None or (isinstance(best_v2v_rate_mean, float) and not np.isfinite(best_v2v_rate_mean)):
            best_v2v_rate_mean = float("nan")
        for name, val in (("avail_L", avail_L), ("avail_R", avail_R), ("avail_V", avail_V), ("neighbor_count_mean", neighbor_count_mean)):
            if not np.isfinite(val):
                if name == "neighbor_count_mean":
                    neighbor_count_mean = 0.0
                elif name == "avail_L":
                    avail_L = 0.0
                elif name == "avail_R":
                    avail_R = 0.0
                elif name == "avail_V":
                    avail_V = 0.0
        episode_vehicle_count = env_stats.get("episode_vehicle_count", episode_vehicle_count) if env_stats else episode_vehicle_count
        episode_task_count = env_stats.get("episode_task_count", episode_task_count) if env_stats else episode_task_count
        total_subtasks_metric = env_stats.get("total_subtasks", total_subtasks) if env_stats else total_subtasks
        update_stats = getattr(agent, "last_update_stats", {}) or {}
        policy_entropy_val = update_stats.get("policy_entropy", update_stats.get("entropy"))
        if policy_entropy_val is None:
            policy_entropy_val = 0.0
        entropy_loss_val = update_stats.get("entropy_loss")
        if entropy_loss_val is None and policy_entropy_val is not None:
            entropy_loss_val = -policy_entropy_val
        v2v_beats_rsu_rate = env_stats.get("v2v_beats_rsu_rate", 0.0) if env_stats else 0.0
        mean_cost_gap = env_stats.get("mean_cost_gap_v2v_minus_rsu") if env_stats else float("nan")
        mean_cost_rsu = env_stats.get("mean_cost_rsu") if env_stats else float("nan")
        mean_cost_v2v = env_stats.get("mean_cost_v2v") if env_stats else float("nan")

        # 控制台输出（每 LOG_INTERVAL 一行）
        if episode == 1 or episode % TC.LOG_INTERVAL == 0:
            if table_row_count % 20 == 0:
                divider_line = _format_table_divider(table_columns)
                header_line = _format_table_header(table_columns)
                print(divider_line, flush=True)
                print(header_line, flush=True)
                print(divider_line, flush=True)
            table_row = {
                "ep": episode,
                "steps": env_stats.get("episode_steps", total_steps) if env_stats else total_steps,
                "r_mean": reward_mean,
                "r_p95": reward_p95,
                "ent": None,
                "kl": None,
                "clip_frac": clip_hit_ratio,
                "p_loss": None,
                "v_loss": None,
                "total_loss": update_loss,
                "succ": success_rate_end,
                "task": task_success_rate,
                "sub": subtask_success,
                "miss": deadline_miss_rate,
                "ill": illegal_action_rate,
                "hard": hard_trigger_rate,
                "L": frac_local,
                "R": frac_rsu,
                "V": frac_v2v,
                "mean_cft": mean_cft,
                "dCFT": delta_cft_rem_mean,
                "power": power_ratio_mean,
                "elapsed": duration,
            }
            update_stats = getattr(agent, "last_update_stats", {}) or {}
            table_row["ent"] = policy_entropy_val
            table_row["p_loss"] = update_stats.get("policy_loss")
            table_row["v_loss"] = update_stats.get("value_loss")
            table_row["kl"] = update_stats.get("approx_kl")
            table_row["clip_frac"] = update_stats.get("clip_fraction", clip_hit_ratio)
            table_row["total_loss"] = update_stats.get("loss", update_loss)
            print(_format_table_row(table_row, table_columns), flush=True)
            table_row_count += 1

        update_stats = getattr(agent, "last_update_stats", {}) or {}
        policy_entropy_val = update_stats.get("policy_entropy", update_stats.get("entropy"))
        if policy_entropy_val is None:
            policy_entropy_val = 0.0
        entropy_loss_val = update_stats.get("entropy_loss")
        if entropy_loss_val is None and policy_entropy_val is not None:
            entropy_loss_val = -policy_entropy_val
        metrics_row = {
            # episode metadata
            "episode": episode,
            "steps": env_stats.get("episode_steps", total_steps) if env_stats else total_steps,
            "elapsed_sec": duration,
            "reward_mode": env_stats.get("reward_mode", Cfg.REWARD_MODE) if env_stats else Cfg.REWARD_MODE,
            "seed": env_stats.get("seed", Cfg.SEED) if env_stats else Cfg.SEED,
            "terminated": terminated_flag,
            "truncated": truncated_flag,
            "termination_reason": termination_reason,
            "time_limit_rate": time_limit_rate,
            "episode_time_seconds": episode_time_seconds,
            "mean_cft_est": mean_cft_est,
            "mean_cft_completed": mean_cft_completed,
            "vehicle_cft_count": vehicle_cft_count,
            "cft_est_valid": cft_est_valid,
            "time_limit_penalty_applied": time_limit_penalty_applied,
            "time_limit_penalty_value": time_limit_penalty_value,
            "remaining_time_seconds_used": remaining_time_used if remaining_time_used is not None else 0.0,
            "remaining_ratio_used": remaining_ratio_used if remaining_ratio_used is not None else 0.0,
            "deadline_gamma": deadline_gamma,
            "deadline_seconds": deadline_seconds,
            "critical_path_cycles": critical_path_cycles,
            "episode_vehicle_count": episode_vehicle_count,
            "episode_task_count": episode_task_count,
            "total_subtasks": total_subtasks_metric,
            # reward: signed per-step mean/p95 (avoid reward_abs for policy quality)
            "reward_mean": reward_mean,
            "reward_p50": reward_p50,
            "reward_p95": reward_p95,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "reward_abs_mean": reward_abs_mean,
            # CFT: absolute mean and remaining-time delta (delta_cft_rem)
            "mean_cft": mean_cft,
            "delta_cft_rem_mean": delta_cft_rem_mean,
            "delta_cft_rem_p95": delta_cft_rem_p95,
            "mean_cft_rem": mean_cft_rem,
            # success/safety
            "success_rate_end": success_rate_end,
            "task_success_rate": task_success_rate,
            "subtask_success_rate": subtask_success,
            "deadline_miss_rate": deadline_miss_rate,
            "illegal_action_rate": illegal_action_rate if illegal_action_rate is not None else 0.0,
            "hard_trigger_rate": hard_trigger_rate if hard_trigger_rate is not None else 0.0,
            # decisions
            "decision_local_frac": frac_local,
            "decision_rsu_frac": frac_rsu,
            "decision_v2v_frac": frac_v2v,
            "decision_frac_local": frac_local,
            "decision_frac_rsu": frac_rsu,
            "decision_frac_v2v": frac_v2v,
            # action power: normalized power ratio stats
            "power_ratio_mean": power_ratio_mean,
            "power_ratio_p95": power_ratio_p95,
            # PPO diagnostics
            "entropy": policy_entropy_val,
            "policy_entropy": policy_entropy_val,
            "entropy_loss": entropy_loss_val,
            "approx_kl": update_stats.get("approx_kl"),
            "clip_frac": update_stats.get("clip_fraction", clip_hit_ratio),
            "policy_loss": update_stats.get("policy_loss"),
            "value_loss": update_stats.get("value_loss"),
            "total_loss": update_stats.get("loss", update_loss),
            "grad_norm": update_stats.get("grad_norm"),
            # diagnostics
            "avail_L": avail_L,
            "avail_R": avail_R,
            "avail_V": avail_V,
            "neighbor_count_mean": neighbor_count_mean,
            "best_v2v_rate_mean": best_v2v_rate_mean,
            "best_v2v_valid_rate": best_v2v_valid_rate,
            "v2v_beats_rsu_rate": v2v_beats_rsu_rate,
            "mean_cost_gap_v2v_minus_rsu": mean_cost_gap,
            "mean_cost_rsu": mean_cost_rsu,
            "mean_cost_v2v": mean_cost_v2v,
        }
        metrics_row_full = dict(metrics_row)
        metrics_row_full.update(env_metrics)

        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row_full, ensure_ascii=True, default=_json_default) + "\n")
        with open(legacy_metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row_full, ensure_ascii=True, default=_json_default) + "\n")
        with open(legacy_train_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_row_full, ensure_ascii=True, default=_json_default) + "\n")

        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields, extrasaction="ignore")
            if not metrics_header_written:
                writer.writeheader()
                metrics_header_written = True
            writer.writerow(metrics_row)
        with open(legacy_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields, extrasaction="ignore")
            if not legacy_metrics_header_written:
                writer.writeheader()
                legacy_metrics_header_written = True
            writer.writerow(metrics_row)
        with open(legacy_train_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields, extrasaction="ignore")
            if not legacy_train_header_written:
                writer.writeheader()
                legacy_train_header_written = True
            writer.writerow(metrics_row)

        if recorder.writer is not None:
            tb = recorder.writer
            # reward
            tb.add_scalar("reward/mean", reward_mean, episode)
            tb.add_scalar("reward/p95", reward_p95, episode)
            if reward_abs_mean is not None:
                tb.add_scalar("reward/abs_mean", reward_abs_mean, episode)
            # CFT
            if mean_cft_est is not None:
                tb.add_scalar("cft/mean_est", mean_cft_est, episode)
            if mean_cft_completed is not None:
                tb.add_scalar("cft/mean_completed", mean_cft_completed, episode)
            if delta_cft_rem_mean is not None:
                tb.add_scalar("cft/delta_cft_rem_mean", delta_cft_rem_mean, episode)
            if episode_time_seconds is not None:
                tb.add_scalar("time/episode_time_seconds", episode_time_seconds, episode)
            # success
            tb.add_scalar("success/success_rate_end", success_rate_end, episode)
            tb.add_scalar("success/task_success_rate", task_success_rate, episode)
            tb.add_scalar("success/subtask_success_rate", subtask_success, episode)
            tb.add_scalar("success/deadline_miss_rate", deadline_miss_rate, episode)
            # safety
            tb.add_scalar("constraint/illegal_rate", illegal_action_rate or 0.0, episode)
            tb.add_scalar("constraint/hard_trigger_rate", hard_trigger_rate or 0.0, episode)
            # decision
            tb.add_scalar("decision/local_frac", frac_local, episode)
            tb.add_scalar("decision/rsu_frac", frac_rsu, episode)
            tb.add_scalar("decision/v2v_frac", frac_v2v, episode)
            # PPO
            if policy_entropy_val is not None:
                tb.add_scalar("ppo/policy_entropy", policy_entropy_val, episode)
            if update_stats.get("approx_kl") is not None:
                tb.add_scalar("ppo/approx_kl", update_stats.get("approx_kl"), episode)
            if update_stats.get("clip_fraction") is not None:
                tb.add_scalar("ppo/clip_frac", update_stats.get("clip_fraction"), episode)
            if update_stats.get("policy_loss") is not None:
                tb.add_scalar("ppo/p_loss", update_stats.get("policy_loss"), episode)
            if update_stats.get("value_loss") is not None:
                tb.add_scalar("ppo/v_loss", update_stats.get("value_loss"), episode)
            if update_stats.get("loss") is not None:
                tb.add_scalar("ppo/total_loss", update_stats.get("loss"), episode)
            # action power
            if power_ratio_mean is not None:
                tb.add_scalar("action/power_ratio_mean", power_ratio_mean, episode)
            # diagnostics
            tb.add_scalar("diag/avail_L", avail_L, episode)
            tb.add_scalar("diag/avail_R", avail_R, episode)
            tb.add_scalar("diag/avail_V", avail_V, episode)
            tb.add_scalar("diag/neighbor_count_mean", neighbor_count_mean, episode)
            if best_v2v_rate_mean is not None:
                tb.add_scalar("diag/best_v2v_rate_mean", best_v2v_rate_mean, episode)
            if best_v2v_valid_rate is not None:
                tb.add_scalar("diag/best_v2v_valid_rate", best_v2v_valid_rate, episode)
            tb.add_scalar("diag/v2v_beats_rsu_rate", v2v_beats_rsu_rate, episode)
            if mean_cost_gap is not None and np.isfinite(mean_cost_gap):
                tb.add_scalar("diag/mean_cost_gap_v2v_minus_rsu", mean_cost_gap, episode)
            if mean_cost_rsu is not None and np.isfinite(mean_cost_rsu):
                tb.add_scalar("diag/mean_cost_rsu", mean_cost_rsu, episode)
            if mean_cost_v2v is not None and np.isfinite(mean_cost_v2v):
                tb.add_scalar("diag/mean_cost_v2v", mean_cost_v2v, episode)
            tb.add_scalar("constraint/time_limit_penalty", time_limit_penalty_value, episode)
            tb.add_scalar("constraint/remaining_ratio_used", remaining_ratio_used or 0.0, episode)

        if log_step_metrics and step_metrics_rows:
            with open(step_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=step_metrics_fields, extrasaction="ignore")
                if not step_metrics_header_written:
                    writer.writeheader()
                    step_metrics_header_written = True
                writer.writerows(step_metrics_rows)

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
    if not disable_auto_plot:
        recorder.auto_plot(baseline_history=baseline_history)


if __name__ == "__main__":
    main()
