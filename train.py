"""
[训练主脚本] train.py
MAPPO Training Script for VEC Task Offloading

作用 (Purpose):
    使用MAPPO算法训练DAG任务卸载策略，支持动态车联网环境中的多智能体协作决策。
    Trains DAG task offloading policy using MAPPO algorithm for multi-agent collaborative 
    decision-making in dynamic vehicular edge computing environments.

核心功能 (Core Features):
    1. 参数自检 - 启动时验证关键配置参数（RESOURCE_RAW_DIM, DEADLINE, LOGIT_BIAS等）
    2. 全指标记录 - 记录训练过程的所有关键指标到CSV（reward, success_rate, loss等）
    3. 最佳模型保存 - 基于成功率（50-ep滑动平均）保存最佳模型
    4. 自动可视化 - 训练结束后自动调用plot_results.py生成图表
    5. Baseline对比 - 定期评估Random/LocalOnly/Greedy策略作为基准

使用方法 (Usage):
    python train.py --max-episodes 5000 --device cuda --seed 42
    python train.py --max-episodes 1000 --log-interval 10 --save-interval 100

输出文件 (Output Files):
    - runs/run_XXX/logs/training_stats.csv - 训练指标（用于绘图）
    - runs/run_XXX/logs/metrics.csv - 详细指标（包含物理量和诊断信息）
    - runs/run_XXX/models/best_model.pth - 最佳模型（基于成功率）
    - runs/run_XXX/plots/*.png - 自动生成的可视化图表

参考文献 (References):
    - PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - MAPPO: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
"""

import time
import json
import csv
from collections import deque
import numpy as np
import torch
import os
import sys
import random
import re
import argparse
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.mappo_agent import MAPPOAgent
from agents.rollout_buffer import RolloutBuffer
from utils.data_recorder import DataRecorder
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy, StaticPolicy, EFTPPolicy
from utils.train_helpers import (
    ensure_dir as _ensure_dir,
    read_last_jsonl as _read_last_jsonl,
    format_table_header as _format_table_header,
    format_table_divider as _format_table_divider,
    format_table_row as _format_table_row,
    compute_time_limit_penalty as _compute_time_limit_penalty,
    env_int as _env_int,
    env_float as _env_float,
    env_bool as _env_bool,
    env_str as _env_str,
    bool_env as _bool_env,
    json_default as _json_default,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO offloading policy.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--cfg-profile", type=str, default=None, help="[DEPRECATED] Config profiles removed")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--step-metrics", action="store_true", default=False)
    parser.add_argument("--no-step-metrics", action="store_true", default=False)
    return parser.parse_args()


def apply_env_overrides():
    """Apply env var overrides to SystemConfig (Cfg) and TrainConfig (TC)."""
    # System / environment knobs
    overrides_float = {
        "VEHICLE_ARRIVAL_RATE": "VEHICLE_ARRIVAL_RATE",
        "BW_V2V": "BW_V2V",
        "MIN_CPU": "MIN_VEHICLE_CPU_FREQ",
        "MAX_CPU": "MAX_VEHICLE_CPU_FREQ",
    }
    overrides_int = {
        "RSU_NUM_PROCESSORS": "RSU_NUM_PROCESSORS",
    }
    for env_key, cfg_attr in overrides_float.items():
        val = _env_float(env_key)
        if val is not None:
            setattr(Cfg, cfg_attr, val)
    for env_key, cfg_attr in overrides_int.items():
        val = _env_int(env_key)
        if val is not None:
            setattr(Cfg, cfg_attr, val)

    # Train / PPO knobs
    tc_float = {
        "GAMMA": "GAMMA",
        "CLIP_PARAM": "CLIP_PARAM",
        "ENTROPY_COEF": "ENTROPY_COEF",
        "LR_ACTOR": "LR_ACTOR",
        "LR_CRITIC": "LR_CRITIC",
        "LOGIT_BIAS_LOCAL": "LOGIT_BIAS_LOCAL",
        "LOGIT_BIAS_RSU": "LOGIT_BIAS_RSU",
        "VALUE_CLIP_RANGE": "VALUE_CLIP_RANGE",
    }
    tc_int = {
        "MINI_BATCH_SIZE": "MINI_BATCH_SIZE",
        "MIN_ACTIVE_SAMPLES": "MIN_ACTIVE_SAMPLES",
    }
    for env_key, attr in tc_float.items():
        val = _env_float(env_key)
        if val is not None:
            setattr(TC, attr, val)
    for env_key, attr in tc_int.items():
        val = _env_int(env_key)
        if val is not None:
            setattr(TC, attr, val)
    use_logit_bias = _env_bool("USE_LOGIT_BIAS")
    if use_logit_bias is not None:
        TC.USE_LOGIT_BIAS = use_logit_bias
    use_value_clip = _env_bool("USE_VALUE_CLIP")
    if use_value_clip is not None:
        TC.USE_VALUE_CLIP = use_value_clip
    use_value_target_norm = _env_bool("USE_VALUE_TARGET_NORM")
    if use_value_target_norm is not None:
        TC.USE_VALUE_TARGET_NORM = use_value_target_norm


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
        greedy_policy.reset()
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

    # 4. EFT策略
    eft_policy = EFTPPolicy(env)
    eft_rewards = []
    for _ in range(num_episodes):
        obs_list, _ = env.reset()
        eft_policy.reset()
        ep_reward = 0
        for step in range(TC.MAX_STEPS):
            actions = eft_policy.select_action(obs_list)
            _inject_obs_stamp(obs_list, actions)
            obs_list, rewards, done, truncated, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done or truncated:
                break
        eft_rewards.append(ep_reward)
    baseline_results['EFT'] = np.mean(eft_rewards)

    # 5. 静态策略
    static_policy = StaticPolicy()
    static_rewards = []
    for _ in range(num_episodes):
        obs_list, _ = env.reset()
        static_policy.reset()
        ep_reward = 0
        for step in range(TC.MAX_STEPS):
            actions = static_policy.select_action(obs_list)
            _inject_obs_stamp(obs_list, actions)
            obs_list, rewards, done, truncated, _ = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            if done or truncated:
                break
        static_rewards.append(ep_reward)
    baseline_results['Static'] = np.mean(static_rewards)
    
    return baseline_results


def evaluate_single_baseline_episode(env, policy_name):
    """评估单个baseline策略的一个episode，返回完整的指标（与训练指标一致）"""
    if policy_name == 'Random':
        policy = RandomPolicy(seed=int(time.time()))
    elif policy_name == 'Local-Only':
        policy = LocalOnlyPolicy()
    elif policy_name == 'Greedy':
        policy = GreedyPolicy(env)
    elif policy_name == 'EFT':
        policy = EFTPPolicy(env)
    elif policy_name == 'Static':
        policy = StaticPolicy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    obs_list, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
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
    
    last_info = None
    for step in range(TC.MAX_STEPS):
        actions = policy.select_action(obs_list)
        _inject_obs_stamp(obs_list, actions)
        obs_list, rewards, done, truncated, info = env.step(actions)
        last_info = info
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
        
        # RSU队列长度（任务数），与训练侧口径一致：使用env.rsu_cpu_q
        if env.rsus:
            rsu_queue_len = 0
            for rsu in env.rsus:
                proc_dict = env.rsu_cpu_q.get(rsu.id, {})
                rsu_queue_len += sum(len(q) for q in proc_dict.values())
            stats['rsu_queue_sum'] += rsu_queue_len
        
        if done or truncated:
            break
    
    avg_step_reward = ep_reward / total_steps if total_steps > 0 else 0
    total_decisions = stats['local_cnt'] + stats['rsu_cnt'] + stats['neighbor_cnt']

    # 成功率统计（与训练循环一致）
    episode_vehicle_count = len(env.vehicles)
    success_count = sum([1 for v in env.vehicles 
                         if v.task_dag.is_finished and not v.task_dag.is_failed])
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
        if hasattr(v.task_dag, 'exec_locations'):
            for i, loc in enumerate(v.task_dag.exec_locations):
                if isinstance(loc, int):  # V2V卸载
                    v2v_subtasks_attempted += 1
                    if v.task_dag.status[i] == 3:  # 已完成
                        v2v_subtasks_completed += 1
    
    subtask_success_rate = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
    v2v_subtask_success_rate = (v2v_subtasks_completed / v2v_subtasks_attempted) if v2v_subtasks_attempted > 0 else 0.0
    
    episode_metrics = last_info.get("episode_metrics", {}) if last_info else {}
    collab_gain_mean = episode_metrics.get("v2v_gain_mean")
    collab_gain_pos_rate = episode_metrics.get("v2v_gain_pos_rate")
    collab_gain_pos_mean = episode_metrics.get("v2v_gain_pos_mean")

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
        'episode_vehicle_count': episode_vehicle_count,
        'episode_task_count': episode_vehicle_count,  # 每辆车一个任务
        'v2v_gain_mean': collab_gain_mean if collab_gain_mean is not None else 0.0,
        'v2v_gain_pos_rate': collab_gain_pos_rate if collab_gain_pos_rate is not None else 0.0,
        'v2v_gain_pos_mean': collab_gain_pos_mean if collab_gain_pos_mean is not None else 0.0,
    }


def main():
    args = _parse_args()
    disable_baseline_eval = False
    
    # CFG_PROFILE已移除，如果传入则忽略并警告
    if args.cfg_profile:
        print(f"⚠ Warning: --cfg-profile is deprecated and will be ignored.", file=sys.stderr)

    if os.environ.get("EPISODE_JSONL_STDOUT") is None:
        Cfg.EPISODE_JSONL_STDOUT = False

    env_seed = _env_int("SEED")
    env_max_episodes = _env_int("MAX_EPISODES")
    env_max_steps = _env_int("MAX_STEPS")
    env_log_interval = _env_int("LOG_INTERVAL")
    env_eval_interval = _env_int("EVAL_INTERVAL")
    env_save_interval = _env_int("SAVE_INTERVAL")
    env_disable_baseline = _env_str("DISABLE_BASELINE_EVAL")
    env_use_lr_decay = _env_str("USE_LR_DECAY")
    env_device = _env_str("DEVICE_NAME")
    env_time_penalty_mode = _env_str("TIME_LIMIT_PENALTY_MODE")
    env_time_penalty = _env_float("TIME_LIMIT_PENALTY")
    env_time_penalty_k = _env_float("TIME_LIMIT_PENALTY_K")
    env_time_penalty_clip = _env_float("TIME_LIMIT_PENALTY_RATIO_CLIP")

    # Env/train overrides from environment variables (after profile/reward selection)
    apply_env_overrides()

    if Cfg.REWARD_SCHEME == "PBRS_KP":
        print(f"[PBRS] reward_gamma={Cfg.REWARD_GAMMA} train_gamma={TC.GAMMA}")
        if abs(Cfg.REWARD_GAMMA - TC.GAMMA) > 1e-9:
            print("[PBRS] Warning: reward_gamma != train_gamma, aligning reward_gamma to train_gamma.")
            Cfg.REWARD_GAMMA = float(TC.GAMMA)

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

    if env_time_penalty_mode:
        Cfg.TIME_LIMIT_PENALTY_MODE = env_time_penalty_mode
    if env_time_penalty is not None:
        Cfg.TIME_LIMIT_PENALTY = env_time_penalty
    if env_time_penalty_k is not None:
        Cfg.TIME_LIMIT_PENALTY_K = env_time_penalty_k
    if env_time_penalty_clip is not None:
        Cfg.TIME_LIMIT_PENALTY_RATIO_CLIP = env_time_penalty_clip

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
    os.environ["SEED"] = str(Cfg.SEED)

    reward_jsonl_path = os.environ.get("REWARD_JSONL_PATH")
    if not reward_jsonl_path:
        reward_jsonl_path = os.path.join(logs_dir, "env_reward.jsonl")
        os.environ["REWARD_JSONL_PATH"] = reward_jsonl_path
    # ensure jsonl file exists for downstream tooling/tests
    _ensure_dir(os.path.dirname(reward_jsonl_path))
    if not os.path.exists(reward_jsonl_path):
        with open(reward_jsonl_path, "w", encoding="utf-8") as f:
            f.write("{}\n")
    run_jsonl_path = os.path.join(logs_dir, "run.jsonl")
    if not os.path.exists(run_jsonl_path):
        with open(run_jsonl_path, "w", encoding="utf-8") as f:
            f.write("{}\n")

    tb_log_obs = os.environ.get("TB_LOG_OBS")
    log_obs_stats = True
    if tb_log_obs is not None:
        log_obs_stats = tb_log_obs.lower() in ("1", "true", "yes")

    # 确定训练设备
    device = TC.DEVICE_NAME if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # =========================================================================
    # 启动参数自检 (Startup Parameter Verification)
    # =========================================================================
    print("\n" + "="*80)
    print("  STARTUP PARAMETER VERIFICATION")
    print("="*80)
    print(f"  RESOURCE_RAW_DIM:             {Cfg.RESOURCE_RAW_DIM} (Expected: 14)")
    print(f"  DEADLINE_TIGHTENING_MIN:      {Cfg.DEADLINE_TIGHTENING_MIN:.2f} (Expected: 0.70)")
    print(f"  DEADLINE_TIGHTENING_MAX:      {Cfg.DEADLINE_TIGHTENING_MAX:.2f} (Expected: 0.80)")
    print(f"  LOGIT_BIAS_LOCAL:             {TC.LOGIT_BIAS_LOCAL:.1f} (Expected: 1.0)")
    print(f"  LOGIT_BIAS_RSU:               {TC.LOGIT_BIAS_RSU:.1f} (Expected: 2.0)")
    print(f"  F_RSU:                        {Cfg.F_RSU/1e9:.1f} GHz (Expected: 12.0 GHz)")
    print(f"  Device:                       {device}")
    print("="*80 + "\n")

    # 初始化配置和日志记录器
    exp_name = f"MAPPO_DAG_N{Cfg.MIN_NODES}-{Cfg.MAX_NODES}_Veh{Cfg.NUM_VEHICLES}"
    recorder = DataRecorder(experiment_name=exp_name, base_dir=run_dir, quiet=True)

    # 构建配置字典
    config_dict = {}
    for k, v in Cfg.__dict__.items():
        if k.startswith('__') or isinstance(v, (staticmethod, classmethod)) or callable(v):
            continue
        if k == "REWARD_MODE":
            # 单一奖励方案，无模式选择；避免在快照中暴露已废弃字段
            continue
        config_dict[k] = v

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
        ("SEED", Cfg.SEED),
        ("DT", Cfg.DT),
        ("MAX_STEPS", TC.MAX_STEPS),
        ("MAX_EPISODES", TC.MAX_EPISODES),
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
    best_success_rate = 0.0  # 用于保存最佳模型
    recent_success_rates = deque(maxlen=50)  # 最近50轮的成功率
    
    # Baseline策略列表
    baseline_policies = ['Random', 'Local-Only', 'Greedy', 'EFT', 'Static']
    
    # 存储baseline的episode级指标（用于绘图）
    baseline_history = {policy: [] for policy in baseline_policies}
    
    # =========================================================================
    # training_stats.csv (用于plot_results.py)
    # =========================================================================
    training_stats_csv = os.path.join(logs_dir, "training_stats.csv")
    training_stats_header_written = os.path.exists(training_stats_csv) and os.path.getsize(training_stats_csv) > 0
    training_stats_fields = [
        # 基本信息
        "episode", "steps", "wall_time", "sim_time",
        # 奖励指标（与控制台打印一致）
        "reward_mean", "reward_total", "reward_p95",
        # 成功率指标（0-1范围，绘图时转换为百分比）
        "vehicle_sr", "task_sr", "subtask_sr",
        # 物理性能指标
        "task_duration_mean", "task_duration_p95", "completed_tasks",
        "energy_mean", "deadline_misses",
        # 卸载决策分布（0-1范围，绘图时转换为百分比）
        "ratio_local", "ratio_rsu", "ratio_v2v",
        # 服务指标
        "tx_created", "same_node_no_tx", "service_rate_ghz", "idle_fraction",
        # 训练诊断指标
        "actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac",
        # Bias状态
        "bias_rsu", "bias_local",
    ]
    
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
        "dT_mean",
        "cft_prev_rem_mean",
        "cft_curr_rem_mean",
        "dCFT_abs_mean",
        "dCFT_abs_p95",
        "dCFT_rem_mean",
        "dCFT_rem_p95",
        "dt_used_mean",
        "implied_dt_mean",
        "dT_eff_mean",
        "dT_eff_p95",
        "energy_norm_mean",
        "energy_norm_p95",
        "t_tx_mean",
        "reward_step_p95",
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
        "active_ratio",
        "active_samples",
        "total_samples",
        "adv_mean",
        "adv_std",
        "value_target_mean",
        "value_target_std",
        "value_pred_mean",
        "value_pred_std",
        "value_clip_fraction",
        "skipped_update_count",
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
        ("ep", 4), ("steps", 6), ("term", 7), ("tlr", 5),
        ("t_ep", 7),
        ("r_mean", 8), ("r_sum", 8), ("r_p95", 8),
        ("succ", 6), ("miss", 6),
        ("L", 5), ("R", 5), ("V", 5),
        ("v2v_win", 8), ("gap", 7), ("c_rsu", 7), ("c_v2v", 7),
        ("ent", 6), ("kl", 7), ("clip", 6), ("v_loss", 8),
        ("rSum15", 8), ("succ15", 7), ("miss15", 7), ("tl15", 6), ("V15", 6),
    ]
    table_row_count = 0
    roll_rsum = deque(maxlen=15)
    roll_succ = deque(maxlen=15)
    roll_miss = deque(maxlen=15)
    roll_tl = deque(maxlen=15)
    roll_v = deque(maxlen=15)

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
            "v2v_count": 0,
            "agent_rewards_per_veh": {},  # 追踪每个Agent的累计奖励
            "active_sum": 0.0,
            "active_total": 0.0,
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
            active_mask = info.get("active_agent_mask")
            if not active_mask or len(active_mask) != len(rewards):
                active_mask = [1] * len(rewards)
            stats["active_sum"] += float(np.sum(active_mask))
            stats["active_total"] += float(len(active_mask))
            # 追踪每个Agent的累计奖励
            for agent_idx, r in enumerate(rewards):
                if agent_idx not in stats["agent_rewards_per_veh"]:
                    stats["agent_rewards_per_veh"][agent_idx] = 0.0
                stats["agent_rewards_per_veh"][agent_idx] += r

            # [修复] 存入Buffer时分离terminated和truncated
            buffer.add(
                obs_list,
                actions,
                rewards,
                values,
                log_probs,
                done,
                terminated=terminated,
                truncated=truncated,
                active_masks=active_mask,
            )

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

            # RSU队列长度（任务数），用于可视化
            if env.rsus:
                rsu_queue_len = 0
                for rsu in env.rsus:
                    proc_dict = env.rsu_cpu_q.get(rsu.id, {})
                    rsu_queue_len += sum(len(q) for q in proc_dict.values())
                stats['rsu_queue_sum'] += rsu_queue_len

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

        # 计算每个Agent的累计奖励统计
        agent_rewards_list = list(stats["agent_rewards_per_veh"].values())
        avg_agent_reward = stats["agent_rewards_sum"] / total_decisions if total_decisions > 0 else 0
        
        if len(agent_rewards_list) > 0:
            max_agent_r = max(agent_rewards_list)
            min_agent_r = min(agent_rewards_list)
            # 个体奖励差异 (Max - Min)
            reward_gap = max_agent_r - min_agent_r
            # Jain's Fairness Index: (sum(x))^2 / (n * sum(x^2))
            sum_r = sum(agent_rewards_list)
            sum_r2 = sum(r**2 for r in agent_rewards_list)
            n = len(agent_rewards_list)
            fairness_index = (sum_r ** 2) / (n * sum_r2) if sum_r2 > 0 else 1.0
        else:
            max_agent_r = 0.0
            min_agent_r = 0.0
            reward_gap = 0.0
            fairness_index = 1.0

        # 协作率
        collaboration_rate = (stats['v2v_count'] / total_decisions) * 100 if total_decisions > 0 else 0

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
        # [关键修复] 成功 = 完成且未超时失败
        episode_vehicle_count = len(env.vehicles)
        episode_task_count = episode_vehicle_count
        
        success_count = sum([1 for v in env.vehicles 
                             if v.task_dag.is_finished and not v.task_dag.is_failed])
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
            if hasattr(v.task_dag, 'exec_locations'):
                for i, loc in enumerate(v.task_dag.exec_locations):
                    if isinstance(loc, int):  # V2V卸载
                        v2v_subtasks_attempted += 1
                        if v.task_dag.status[i] == 3:  # 已完成
                            v2v_subtasks_completed += 1

        subtask_success_rate = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
        v2v_subtask_success_rate = (v2v_subtasks_completed / v2v_subtasks_attempted) if v2v_subtasks_attempted > 0 else 0.0
        
        # 更新成功率历史（用于最佳模型保存）
        recent_success_rates.append(task_success_rate)

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
        if log_step_metrics and step_metrics_rows:
            # 如有需要，可汇总step级指标；当前使用env_stats为主
            pass
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
        dT_mean = env_stats.get("dT_mean") if env_stats else env_metrics.get("delta_cft.mean")
        cft_prev_rem_mean = env_stats.get("cft_prev_rem_mean") if env_stats else env_metrics.get("cft_prev_rem.mean")
        cft_curr_rem_mean = env_stats.get("cft_curr_rem_mean") if env_stats else env_metrics.get("cft_curr_rem.mean")
        dT_eff_mean = env_stats.get("dT_eff_mean") if env_stats else env_metrics.get("dT_eff.mean")
        dT_eff_p95 = env_stats.get("dT_eff_p95") if env_stats else env_metrics.get("dT_eff.p95")
        energy_norm_mean = env_stats.get("energy_norm_mean") if env_stats else env_metrics.get("energy_norm.mean")
        energy_norm_p95 = env_stats.get("energy_norm_p95") if env_stats else env_metrics.get("energy_norm.p95")
        
        # [新增] 真实任务完成时间（物理指标，给人类看的）
        task_duration_mean = env_stats.get("task_duration_mean") if env_stats else 0.0
        task_duration_p95 = env_stats.get("task_duration_p95") if env_stats else 0.0
        completed_tasks_count = env_stats.get("completed_tasks_count") if env_stats else 0
        t_tx_mean = env_stats.get("t_tx_mean") if env_stats else env_metrics.get("t_tx.mean")
        dt_used_mean = env_stats.get("dt_used_mean") if env_stats else env_metrics.get("dt_used.mean")
        implied_dt_mean = env_stats.get("implied_dt_mean")
        dCFT_abs_mean = env_stats.get("dCFT_abs_mean") if env_stats else env_metrics.get("delta_cft_abs.mean")
        dCFT_abs_p95 = env_stats.get("dCFT_abs_p95") if env_stats else env_metrics.get("delta_cft_abs.p95")
        dCFT_rem_mean = env_stats.get("dCFT_rem_mean") if env_stats else env_metrics.get("delta_cft_rem.mean")
        dCFT_rem_p95 = env_stats.get("dCFT_rem_p95") if env_stats else env_metrics.get("delta_cft_rem.p95")
        reward_step_p95 = env_stats.get("reward_step_p95") if env_stats else env_metrics.get("reward_step.p95")
        episode_time_seconds = env_stats.get("episode_time_seconds") if env_stats else (env.time if env else None)
        mean_cft_est = env_stats.get("mean_cft_est") if env_stats else None
        mean_cft_completed = env_stats.get("mean_cft_completed") if env_stats else None
        vehicle_cft_count = env_stats.get("vehicle_cft_count") if env_stats else 0
        cft_est_valid = env_stats.get("cft_est_valid") if env_stats else False
        power_ratio_mean = env_metrics.get("power_ratio.mean")
        power_ratio_p95 = env_metrics.get("power_ratio.p95")
        deadline_gamma = env_stats.get("deadline_gamma_mean") if env_stats else None
        deadline_seconds = env_stats.get("deadline_seconds_mean") if env_stats else None
        # mean_cft_rem: 优先使用env_stats，其次用env_metrics，最后fallback到deadline剩余时间
        mean_cft_rem = env_stats.get("mean_cft_rem") if env_stats else None
        if mean_cft_rem is None:
            mean_cft_rem = env_metrics.get("cft_curr_rem.mean")
        if mean_cft_rem is None and deadline_seconds is not None and episode_time_seconds is not None:
            mean_cft_rem = max(deadline_seconds - episode_time_seconds, 0.0)
        # mean_cft: 优先使用env_stats；若仅有剩余时间，则还原绝对CFT
        if mean_cft is None and mean_cft_rem is not None and episode_time_seconds is not None:
            mean_cft = mean_cft_rem + episode_time_seconds
        critical_path_cycles = env_stats.get("critical_path_cycles_mean") if env_stats else None
        avail_L = env_stats.get("avail_L") if env_stats else None
        avail_R = env_stats.get("avail_R") if env_stats else None
        avail_V = env_stats.get("avail_V") if env_stats else None
        neighbor_count_mean = env_stats.get("neighbor_count_mean") if env_stats else None
        best_v2v_rate_mean = env_stats.get("best_v2v_rate_mean") if env_stats else None
        best_v2v_valid_rate = env_stats.get("best_v2v_valid_rate") if env_stats else None
        collab_gain_mean = env_stats.get("v2v_gain_mean") if env_stats else None
        collab_gain_pos_rate = env_stats.get("v2v_gain_pos_rate") if env_stats else None
        collab_gain_pos_mean = env_stats.get("v2v_gain_pos_mean") if env_stats else None
        if avail_L is None: avail_L = 0.0
        if avail_R is None: avail_R = 0.0
        if avail_V is None: avail_V = 0.0
        if neighbor_count_mean is None: neighbor_count_mean = 0.0
        if best_v2v_valid_rate is None or not (np.isfinite(best_v2v_valid_rate)): best_v2v_valid_rate = 0.0
        if best_v2v_rate_mean is None or (isinstance(best_v2v_rate_mean, float) and not np.isfinite(best_v2v_rate_mean)):
            best_v2v_rate_mean = float("nan")
        if collab_gain_pos_rate is not None:
            collaboration_rate = collab_gain_pos_rate * 100.0
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
        # 末尾截断惩罚默认值
        time_limit_penalty_applied = False
        time_limit_penalty_value = 0.0
        remaining_time_used = None
        remaining_ratio_used = None
        should_apply_tl_penalty = (
            truncated_flag
            and termination_reason == "time_limit"
            and (success_rate_end is None or success_rate_end < 1.0)
            and buffer.rewards_buffer
        )
        if should_apply_tl_penalty:
            remaining_time_used = env_metrics.get("cft_curr_rem.mean")
            if remaining_time_used is None:
                remaining_time_used = mean_cft_rem
            if remaining_time_used is None and env_stats:
                remaining_time_used = env_stats.get("mean_cft_rem")
            deadline_used = deadline_seconds if deadline_seconds is not None else episode_time_seconds
            penalty, ratio = _compute_time_limit_penalty(
                getattr(Cfg, "TIME_LIMIT_PENALTY_MODE", "fixed"),
                remaining_time_used if remaining_time_used is not None else 0.0,
                deadline_used if deadline_used is not None else 1.0,
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

        # PPO更新（在末步惩罚后计算，以确保惩罚参与梯度更新）
        last_value = agent.get_value(obs_list)
        buffer.compute_returns_and_advantages(last_value)
        update_loss = agent.update(buffer, batch_size=TC.MINI_BATCH_SIZE)
        buffer.clear()

        # 显存清理
        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # =====================================================================
        # Bias退火逻辑 (Bias Annealing)
        # [P18修复] 网络从TC直接读取bias值，无需额外同步
        # =====================================================================
        if TC.USE_LOGIT_BIAS and (episode % TC.BIAS_DECAY_EVERY_EP == 0 and episode > 0):
            TC.LOGIT_BIAS_RSU = max(TC.BIAS_MIN_RSU, TC.LOGIT_BIAS_RSU - TC.BIAS_DECAY_RSU)
            TC.LOGIT_BIAS_LOCAL = max(TC.BIAS_MIN_LOCAL, TC.LOGIT_BIAS_LOCAL - TC.BIAS_DECAY_LOCAL)

        # =====================================================================
        # 控制台输出（精简且全面的单行格式 - 每个episode都打印）
        # =====================================================================
        if True:  # 每个episode都打印
            # [物理指标] 使用真实任务完成时间（current_time - arrival_time）
            # 这是给人类看的物理指标，必然 > 0，有直观意义
            # dT_eff_mean是给RL看的奖励信号，对人类不直观
            avg_latency = task_duration_mean if task_duration_mean is not None else 0.0
            avg_energy = energy_norm_mean if energy_norm_mean is not None else 0.0
            
            # 获取训练诊断指标
            actor_loss = update_stats.get("policy_loss", 0.0) if update_stats.get("policy_loss") is not None else 0.0
            critic_loss = update_stats.get("value_loss", 0.0) if update_stats.get("value_loss") is not None else 0.0
            entropy_val = update_stats.get("entropy", 0.0) if update_stats.get("entropy") is not None else 0.0
            
            # 获取关键健康指标（从info中提取）
            deadlock_count = env_stats.get('deadlock_vehicle_count', 0) if env_stats else 0
            deadline_misses = env_stats.get('audit_deadline_misses', 0) if env_stats else 0
            tx_created = env_stats.get('tx_tasks_created_count', 0) if env_stats else 0
            same_node_no_tx = env_stats.get('same_node_no_tx_count', 0) if env_stats else 0
            vehicle_sr = env_stats.get('vehicle_success_rate', veh_success_rate) if env_stats else veh_success_rate
            episode_all_success = env_stats.get('episode_all_success', 0.0) if env_stats else 0.0
            service_rate_active = env_stats.get('service_rate_when_active', 0.0) if env_stats else 0.0
            idle_fraction = env_stats.get('idle_fraction', 0.0) if env_stats else 0.0
            
            # 计算仿真时间
            sim_time = total_steps * Cfg.DT
            
            # 打印表头（每个episode都打印，但只在episode 1或每20个episode打印表头）
            if episode == 1 or episode % 20 == 1:
                print("\n" + "="*180)
                print(f"| {'Ep':>5} | {'Time':>6} | {'Reward':>7} | {'V_SR':>5} | {'T_SR':>5} | {'Deci(L/R/V)':>14} | {'P_Loss':>7} | {'V_Loss':>7} | {'Ent':>6} | {'KL':>6} | {'Clip':>5} | {'GNorm':>6} |")
                print("="*180)
            
            # 获取PPO诊断指标
            approx_kl = update_stats.get("approx_kl", 0.0) if update_stats.get("approx_kl") is not None else 0.0
            clip_frac = update_stats.get("clip_fraction", 0.0) if update_stats.get("clip_fraction") is not None else 0.0
            grad_norm_val = update_stats.get("grad_norm", 0.0) if update_stats.get("grad_norm") is not None else 0.0
            
            # 打印数据行（每个episode都打印）
            deci_str = f"{frac_local:4.1%}/{frac_rsu:4.1%}/{frac_v2v:4.1%}"
            print(f"| {episode:5d} | {duration:6.1f}s | {reward_mean:7.4f} | {vehicle_sr:5.1%} | {task_success_rate:5.1%} | {deci_str:>14} | {actor_loss:7.4f} | {critic_loss:7.2f} | {entropy_val:6.3f} | {approx_kl:6.4f} | {clip_frac*100:5.1f}% | {grad_norm_val:6.3f} |", flush=True)

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
            "dT_mean": dT_mean if dT_mean is not None else 0.0,
            "cft_prev_rem_mean": cft_prev_rem_mean if cft_prev_rem_mean is not None else 0.0,
            "cft_curr_rem_mean": cft_curr_rem_mean if cft_curr_rem_mean is not None else 0.0,
            "dCFT_abs_mean": dCFT_abs_mean if dCFT_abs_mean is not None else 0.0,
            "dCFT_abs_p95": dCFT_abs_p95 if dCFT_abs_p95 is not None else 0.0,
            "dCFT_rem_mean": dCFT_rem_mean if dCFT_rem_mean is not None else 0.0,
            "dCFT_rem_p95": dCFT_rem_p95 if dCFT_rem_p95 is not None else 0.0,
            "dt_used_mean": dt_used_mean if dt_used_mean is not None else 0.0,
            "implied_dt_mean": implied_dt_mean if implied_dt_mean is not None else ( (dT_mean if dT_mean is not None else 0.0) - (dT_eff_mean if dT_eff_mean is not None else 0.0)),
            "dT_eff_mean": dT_eff_mean if dT_eff_mean is not None else 0.0,
            "dT_eff_p95": dT_eff_p95 if dT_eff_p95 is not None else 0.0,
            "energy_norm_mean": energy_norm_mean if energy_norm_mean is not None else 0.0,
            "energy_norm_p95": energy_norm_p95 if energy_norm_p95 is not None else 0.0,
            "t_tx_mean": t_tx_mean if t_tx_mean is not None else 0.0,
            "reward_step_p95": reward_step_p95 if reward_step_p95 is not None else 0.0,
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
            "active_ratio": (stats["active_sum"] / stats["active_total"]) if stats["active_total"] > 0 else 0.0,
            "active_samples": update_stats.get("active_samples"),
            "total_samples": update_stats.get("total_samples"),
            "adv_mean": update_stats.get("adv_mean"),
            "adv_std": update_stats.get("adv_std"),
            "value_target_mean": update_stats.get("value_target_mean"),
            "value_target_std": update_stats.get("value_target_std"),
            "value_pred_mean": update_stats.get("value_pred_mean"),
            "value_pred_std": update_stats.get("value_pred_std"),
            "value_clip_fraction": update_stats.get("value_clip_fraction"),
            "skipped_update_count": update_stats.get("skipped_update_count"),
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
        
        # =====================================================================
        # 写入 training_stats.csv (用于 plot_results.py)
        # 【关键】确保字段与控制台打印一致，便于对照检查
        # =====================================================================
        update_stats = getattr(agent, "last_update_stats", {}) or {}

        # 获取环境统计数据
        deadline_misses = env_stats.get('audit_deadline_misses', 0) if env_stats else 0
        tx_created = env_stats.get('tx_tasks_created_count', 0) if env_stats else 0
        same_node_no_tx = env_stats.get('same_node_no_tx_count', 0) if env_stats else 0
        service_rate_active = env_stats.get('service_rate_when_active', 0.0) if env_stats else 0.0
        idle_fraction = env_stats.get('idle_fraction', 0.0) if env_stats else 0.0

        training_stats_row = {
            # 基本信息
            "episode": episode,
            "steps": total_steps,
            "wall_time": duration,
            "sim_time": total_steps * Cfg.DT,
            # 奖励指标（与控制台打印一致）
            "reward_mean": reward_mean,  # 每步平均奖励（控制台显示的Reward）
            "reward_total": ep_reward,   # episode总奖励
            "reward_p95": reward_p95,
            # 成功率指标（0-1范围）
            "vehicle_sr": veh_success_rate,  # V_SR
            "task_sr": task_success_rate,    # T_SR
            "subtask_sr": subtask_success,   # S_SR
            # 物理性能指标
            "task_duration_mean": task_duration_mean if task_duration_mean is not None else 0.0,
            "task_duration_p95": task_duration_p95 if task_duration_p95 is not None else 0.0,
            "completed_tasks": completed_tasks_count if completed_tasks_count is not None else 0,
            "energy_mean": energy_norm_mean if energy_norm_mean is not None else 0.0,
            "deadline_misses": deadline_misses,  # D_Miss
            # 卸载决策分布（0-1范围）
            "ratio_local": frac_local,  # Local
            "ratio_rsu": frac_rsu,      # RSU
            "ratio_v2v": frac_v2v,      # V2V
            # 服务指标
            "tx_created": tx_created,             # TX
            "same_node_no_tx": same_node_no_tx,   # NoTX
            "service_rate_ghz": service_rate_active / 1e9,  # SvcRate (GHz)
            "idle_fraction": idle_fraction,       # Idle
            # 训练诊断指标
            "actor_loss": update_stats.get("policy_loss"),
            "critic_loss": update_stats.get("value_loss"),
            "entropy": update_stats.get("policy_entropy", update_stats.get("entropy")),
            "approx_kl": update_stats.get("approx_kl"),
            "clip_frac": update_stats.get("clip_fraction"),
            # Bias状态
            "bias_rsu": TC.LOGIT_BIAS_RSU,
            "bias_local": TC.LOGIT_BIAS_LOCAL,
        }
        with open(training_stats_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=training_stats_fields, extrasaction="ignore")
            if not training_stats_header_written:
                writer.writeheader()
                training_stats_header_written = True
            writer.writerow(training_stats_row)

        if recorder.writer is not None:
            tb = recorder.writer
            # reward
            tb.add_scalar("reward/mean", reward_mean, episode)
            tb.add_scalar("reward/p95", reward_p95, episode)
            if reward_abs_mean is not None:
                tb.add_scalar("reward/abs_mean", reward_abs_mean, episode)
            if dT_mean is not None:
                tb.add_scalar("reward/dT_mean", dT_mean, episode)
            if dT_eff_mean is not None:
                tb.add_scalar("reward/dT_eff_mean", dT_eff_mean, episode)
            if dt_used_mean is not None:
                tb.add_scalar("reward/dt_used_mean", dt_used_mean, episode)
            if implied_dt_mean is not None:
                tb.add_scalar("reward/implied_dt_mean", implied_dt_mean, episode)
            if energy_norm_mean is not None:
                tb.add_scalar("energy/energy_norm_mean", energy_norm_mean, episode)
            if t_tx_mean is not None:
                tb.add_scalar("tx/t_tx_mean", t_tx_mean, episode)
            # CFT
            if mean_cft_est is not None:
                tb.add_scalar("cft/mean_est", mean_cft_est, episode)
            if mean_cft_completed is not None:
                tb.add_scalar("cft/mean_completed", mean_cft_completed, episode)
            if cft_prev_rem_mean is not None:
                tb.add_scalar("cft/prev_rem_mean", cft_prev_rem_mean, episode)
            if cft_curr_rem_mean is not None:
                tb.add_scalar("cft/curr_rem_mean", cft_curr_rem_mean, episode)
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
            if update_stats.get("active_ratio") is not None:
                tb.add_scalar("ppo/active_ratio", update_stats.get("active_ratio"), episode)
            if update_stats.get("active_samples") is not None:
                tb.add_scalar("ppo/active_samples", update_stats.get("active_samples"), episode)
            if update_stats.get("adv_std") is not None:
                tb.add_scalar("ppo/adv_std", update_stats.get("adv_std"), episode)
            if update_stats.get("value_clip_fraction") is not None:
                tb.add_scalar("ppo/value_clip_frac", update_stats.get("value_clip_fraction"), episode)
            if update_stats.get("skipped_update_count") is not None:
                tb.add_scalar("ppo/skipped_updates", update_stats.get("skipped_update_count"), episode)
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
            "policy": "",
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
            "avg_veh_queue": avg_veh_queue,
            "avg_rsu_queue": avg_rsu_queue,
            "ma_fairness": fairness_index,
            "ma_reward_gap": reward_gap,
            "ma_collaboration": collaboration_rate,
            "v2v_gain_mean": collab_gain_mean if collab_gain_mean is not None else 0.0,
            "v2v_gain_pos_rate": collab_gain_pos_rate if collab_gain_pos_rate is not None else (collaboration_rate / 100.0),
            "v2v_gain_pos_mean": collab_gain_pos_mean if collab_gain_pos_mean is not None else 0.0,
            "max_agent_reward": max_agent_r,
            "min_agent_reward": min_agent_r,
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
            # baseline统计CSV路径
            baseline_stats_csv = os.path.join(logs_dir, "baseline_stats.csv")
            baseline_stats_fields = [
                "episode", "policy", "reward_mean", "reward_total",
                "vehicle_sr", "task_sr", "subtask_sr",
                "ratio_local", "ratio_rsu", "ratio_v2v",
            ]
            baseline_header_written = os.path.exists(baseline_stats_csv) and os.path.getsize(baseline_stats_csv) > 0

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

                # 保存到baseline_stats.csv（用于绘图对比）
                baseline_stats_row = {
                    "episode": episode,
                    "policy": policy_name,
                    "reward_mean": baseline_metrics['avg_step_reward'],
                    "reward_total": baseline_metrics['total_reward'],
                    "vehicle_sr": baseline_metrics['veh_success_rate'],
                    "task_sr": baseline_metrics.get('task_success_rate', baseline_metrics['veh_success_rate']),
                    "subtask_sr": baseline_metrics['subtask_success_rate'],
                    "ratio_local": baseline_metrics['decision_frac_local'],
                    "ratio_rsu": baseline_metrics['decision_frac_rsu'],
                    "ratio_v2v": baseline_metrics['decision_frac_v2v'],
                }
                with open(baseline_stats_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=baseline_stats_fields, extrasaction="ignore")
                    if not baseline_header_written:
                        writer.writeheader()
                        baseline_header_written = True
                    writer.writerow(baseline_stats_row)

                # 记录到CSV（使用log_episode，但添加policy字段）
                # 注意：字段顺序和数量必须与训练数据一致，避免CSV列错位
                baseline_episode_dict = {
                    "episode": episode,
                    "policy": policy_name,
                    "total_reward": baseline_metrics['total_reward'],
                    "avg_step_reward": baseline_metrics['avg_step_reward'],
                    "loss": 0.0,  # baseline无loss
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
                    "avg_veh_queue": baseline_metrics['avg_queue_len'],
                    "avg_rsu_queue": baseline_metrics.get('avg_rsu_queue', 0.0),
                    "ma_fairness": 1.0,  # baseline无公平性概念，设为1.0
                    "ma_reward_gap": 0.0,
                    "ma_collaboration": baseline_metrics.get('v2v_gain_pos_rate', baseline_metrics['decision_frac_v2v']) * 100.0,
                    "v2v_gain_mean": baseline_metrics.get('v2v_gain_mean', 0.0),
                    "v2v_gain_pos_rate": baseline_metrics.get('v2v_gain_pos_rate', 0.0),
                    "v2v_gain_pos_mean": baseline_metrics.get('v2v_gain_pos_mean', 0.0),
                    "max_agent_reward": baseline_metrics['total_reward'],
                    "min_agent_reward": baseline_metrics['total_reward'],
                    "avg_assigned_cpu_ghz": 0.0,  # baseline无此指标
                    "episode_vehicle_count": baseline_metrics.get('episode_vehicle_count', 20),
                    "episode_task_count": baseline_metrics.get('episode_task_count', 20),
                    "duration": 0.0
                }
                recorder.log_episode(baseline_episode_dict)

        # =====================================================================
        # 模型保存 (Best Model Based on Success Rate)
        # =====================================================================
        # 保存基于reward的最佳模型（保留原逻辑）
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(recorder.model_dir, "best_model_reward.pth"))
        
        # 保存基于task_success_rate的最佳模型（新增）
        avg_recent_sr = np.mean(recent_success_rates) if recent_success_rates else 0.0
        if avg_recent_sr > best_success_rate:
            best_success_rate = avg_recent_sr
            agent.save(os.path.join(recorder.model_dir, "best_model.pth"))
            if episode == 1 or episode % TC.LOG_INTERVAL == 0:
                print(f"  → Best model saved: Success Rate = {best_success_rate:.3f} (50-ep avg)")

        if episode % TC.SAVE_INTERVAL == 0:
            agent.save(os.path.join(recorder.model_dir, f"model_ep{episode}.pth"))

    # =========================================================================
    # 训练结束：自动绘图 (Auto Plotting)
    # =========================================================================
    if not disable_auto_plot:
        # 1. 使用DataRecorder的新绘图方法绘制完整训练分析图
        print("\n[Auto Plotting] Generating comprehensive training plots...")
        baseline_stats_csv = os.path.join(logs_dir, "baseline_stats.csv")
        recorder.plot_training_stats(training_stats_csv, baseline_stats_csv)

        # 2. 调用plot_results.py生成额外图表
        plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "plot_results.py")
        if os.path.exists(training_stats_csv):
            print("[Auto Plotting] Generating additional plots from training_stats.csv...")
        try:
            subprocess.run(
                [sys.executable, plot_script, "--log-file", training_stats_csv, "--output-dir", plots_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"✓ Plots saved to: {plots_dir}")
        except subprocess.CalledProcessError as e:
            print(f"⚠ plot_results.py failed: {e.stdout}")
        except Exception as e:
            print(f"⚠ plot_results.py failed: {e}")

        # 3. 调用DataRecorder的auto_plot保持兼容性
        print("[Auto Plotting] Generating legacy episode_log plots...")
        recorder.auto_plot(baseline_history=baseline_history)

        # 4. 保留旧的绘图脚本调用（兼容性）
        legacy_plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "plot_key_metrics_v4.py")
        if os.path.exists(legacy_plot_script):
            try:
                subprocess.run(
                    [sys.executable, legacy_plot_script, "--run-dir", run_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                pass  # 静默失败，不影响新绘图


if __name__ == "__main__":
    main()
