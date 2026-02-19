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
    3. 模型保存策略 - 仅保存 best_model 与 last_model（覆盖写）
    4. 自动可视化 - 训练结束后自动调用plot_results.py生成图表
    5. Baseline对比 - 默认关闭（由独立脚本运行）

使用方法 (Usage):
    python train.py --max-episodes 5000 --device cuda --seed 42
    python train.py --max-episodes 1000 --log-interval 10 --save-interval 100

输出文件 (Output Files):
    - runs/run_XXX/logs/training_stats.csv - 训练指标（用于绘图）
    - runs/run_XXX/logs/metrics.csv - 详细指标（包含物理量和诊断信息）
    - runs/run_XXX/models/best_model.pth - 最佳模型（基于成功率）
    - runs/run_XXX/models/last_model.pth - 最后模型（训练末轮）
    - runs/run_XXX/plots/*.png - 自动生成的可视化图表

参考文献 (References):
    - PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - MAPPO: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
"""

import time
import json
import csv
import hashlib
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
import traceback
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.mappo_agent import MAPPOAgent
from agents.rollout_buffer import RolloutBuffer
from utils.data_recorder import DataRecorder
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy, StaticPolicy, EFTPPolicy, LBGreedyPolicy, OracleMinPolicy
from baselines.cp_first_eft_policy import CPFirstEFTPolicy
from utils.train_helpers import (
    ensure_dir as _ensure_dir,
    read_last_jsonl as _read_last_jsonl,
    compute_time_limit_penalty as _compute_time_limit_penalty,
    env_int as _env_int,
    env_float as _env_float,
    env_bool as _env_bool,
    env_str as _env_str,
    bool_env as _bool_env,
    json_default as _json_default,
)


BASELINE_POLICIES = ["Random", "Local-Only", "Greedy", "Oracle-Min", "EFT", "CP-EFT", "LB-Greedy", "Static"]

TRAINING_STATS_FIELDS = [
    "episode", "steps", "wall_time", "sim_time",
    "reward_mean", "reward_total", "reward_p95", "reward_abs_mean",
    "vehicle_sr", "task_sr", "subtask_sr",
    "task_duration_mean", "task_duration_p95", "completed_tasks",
    "mean_cft_est", "episode_time_seconds",
    "energy_mean", "energy_p95", "t_tx_mean", "dT_eff_mean",
    "deadline_misses", "deadline_miss_rate",
    "ratio_local", "ratio_rsu", "ratio_v2v",
    "decision_frac_local", "decision_frac_rsu", "decision_frac_v2v",
    "avg_power", "avg_rsu_queue", "rsu_queue_p95", "power_ratio_mean", "power_ratio_p95",
    "I_total_p50", "I_total_p95", "I_caused_mean", "I_caused_p95",
    "trust_failure_rate", "rho_selected_p10", "uncertainty_selected_p90",
    "tx_created", "same_node_no_tx", "service_rate_ghz", "idle_fraction",
    "time_limit_rate", "illegal_action_rate", "no_task_rate", "on_task_rate", "has_task_available_rate",
    "unified_illegal_trigger_rate", "hard_trigger_rate",
    "actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac",
    "grad_norm", "active_ratio", "value_clip_fraction", "skipped_update_count", "early_stop", "lr",
    "bias_rsu", "bias_local",
]

REQUIRED_COMPARE_COLUMNS = [
    "illegal_action_rate",
    "no_task_rate",
    "on_task_rate",
    "has_task_available_rate",
    "unified_illegal_trigger_rate",
    "decision_frac_local",
    "decision_frac_rsu",
    "decision_frac_v2v",
    "mean_cft_est",
    "episode_time_seconds",
    "deadline_miss_rate",
    "time_limit_rate",
    "power_ratio_mean",
    "power_ratio_p95",
    "avg_power",
    "I_total_p50",
    "I_total_p95",
    "I_caused_mean",
    "I_caused_p95",
    "trust_failure_rate",
    "rho_selected_p10",
    "uncertainty_selected_p90",
]

BASELINE_STATS_FIELDS = [
    "episode", "policy", "reward_mean", "reward_total",
    "vehicle_sr", "task_sr", "subtask_sr", "v2v_subtask_sr",
    "ratio_local", "ratio_rsu", "ratio_v2v",
    "decision_frac_local", "decision_frac_rsu", "decision_frac_v2v",
    "avg_power", "power_ratio_mean", "power_ratio_p95",
    "episode_time_seconds", "mean_cft_est", "mean_cft_completed",
    "task_duration_mean", "task_duration_p95",
    "deadline_miss_rate", "time_limit_rate",
    "illegal_action_rate", "no_task_rate", "on_task_rate", "has_task_available_rate", "unified_illegal_trigger_rate",
    "I_total_mean", "I_total_p50", "I_total_p95", "I_caused_mean", "I_caused_p95",
    "rho_selected_mean", "rho_selected_p10", "risk_penalty_mean",
    "rho_selected_p50", "rho_selected_p95", "rho_selected_lt_0p6_rate", "rho_selected_lt_0p7_rate",
    "uncertainty_selected_mean", "uncertainty_selected_p90",
    "chain_tx_total", "chain_p95_mean", "chain_pfail_mean", "chain_risk_cost_total",
    "trust_attempts", "trust_failures", "trust_failure_rate", "trust_retry_count",
    "avg_queue_len", "avg_rsu_queue",
]


def get_required_compare_columns() -> List[str]:
    return list(REQUIRED_COMPARE_COLUMNS)


def get_training_stats_fields() -> List[str]:
    return list(TRAINING_STATS_FIELDS)


def get_baseline_stats_fields() -> List[str]:
    return list(BASELINE_STATS_FIELDS)


def _get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return "unknown"


def _stable_config_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _LagrangianController:
    """Lightweight PPO-Lagrangian controller (episode-level multiplier update)."""

    def __init__(self):
        self.enabled = bool(getattr(TC, "CMDP_ENABLE", False))
        self.lr = float(getattr(TC, "CMDP_LAMBDA_LR", 0.02))
        self.lam_max = float(getattr(TC, "CMDP_LAMBDA_MAX", 5.0))
        self.warmup_episodes = int(max(getattr(TC, "CMDP_WARMUP_EPISODES", 0), 0))
        self.lam = {
            "energy": float(getattr(TC, "CMDP_LAMBDA_ENERGY_INIT", 0.0)),
            "interf": float(getattr(TC, "CMDP_LAMBDA_INTERF_INIT", 0.0)),
            "risk": float(getattr(TC, "CMDP_LAMBDA_RISK_INIT", 0.0)),
        }
        self.budget = {
            "energy": float(getattr(TC, "CMDP_BUDGET_ENERGY", 0.20)),
            "interf": float(getattr(TC, "CMDP_BUDGET_INTERF", 0.05)),
            "risk": float(getattr(TC, "CMDP_BUDGET_RISK", 0.35)),
        }

    def penalize_step(self, rewards: List[float], actions: List[Dict], obs_list: List[Dict], info: Dict) -> tuple:
        if not self.enabled or not rewards:
            zeros = {"energy": 0.0, "interf": 0.0, "risk": 0.0}
            return rewards, zeros

        i_ref = float(max(getattr(Cfg, "I_REF_MIN_UNIFIED", 1e-8), 1e-12))
        i_clip = float(max(getattr(Cfg, "INTERF_RATIO_CLIP_UNIFIED", 20.0), 1e-6))
        i_mean = max(
            float(
                (info or {}).get(
                    "interf_i_total_all_mean",
                    (info or {}).get("v2v_i_total_mean", 0.0),
                )
                or 0.0
            ),
            0.0,
        )
        interf_norm = float(min(i_mean / i_ref, i_clip)) / i_clip

        p_min = float(getattr(Cfg, "P_MIN_WATT", Cfg.dbm2watt(getattr(Cfg, "TX_POWER_MIN_DBM", 13.0))))
        p_max = float(getattr(Cfg, "P_MAX_WATT", Cfg.dbm2watt(getattr(Cfg, "TX_POWER_MAX_DBM", 23.0))))
        dt = float(getattr(Cfg, "DT", 0.1))
        e_ref = float(max(getattr(Cfg, "E_REF_UNIFIED", 1.0), 1e-9))

        out_rewards = []
        e_vals, i_vals, r_vals = [], [], []
        for i, r in enumerate(rewards):
            act = actions[i] if i < len(actions) else {}
            obs = obs_list[i] if i < len(obs_list) else {}
            tgt_idx = int(act.get("target", 0))
            a_power = float(np.clip(act.get("power", 0.0), 0.0, 1.0))

            ctype = 0
            c_rho = 1.0
            ctypes = obs.get("candidate_types")
            res = obs.get("resource_raw")
            if ctypes is not None and 0 <= tgt_idx < len(ctypes):
                ctype = int(ctypes[tgt_idx])
            if res is not None and isinstance(res, np.ndarray) and res.ndim == 2 and 0 <= tgt_idx < res.shape[0] and res.shape[1] >= 13:
                c_rho = float(np.clip(res[tgt_idx, 12], 0.0, 1.0))

            is_remote = (ctype in (2, 3))
            is_v2v = (ctype == 3)
            p_w = p_min * ((p_max / max(p_min, 1e-12)) ** a_power)
            c_energy = float((p_w * dt) / e_ref) if is_remote else 0.0
            c_interf = float(interf_norm) if is_v2v else 0.0
            c_risk = float(1.0 - c_rho) if is_remote else 0.0

            penalty = (
                self.lam["energy"] * c_energy
                + self.lam["interf"] * c_interf
                + self.lam["risk"] * c_risk
            )
            out_rewards.append(float(r) - float(penalty))
            e_vals.append(c_energy)
            i_vals.append(c_interf)
            r_vals.append(c_risk)

        step_cost = {
            "energy": float(np.mean(e_vals)) if e_vals else 0.0,
            "interf": float(np.mean(i_vals)) if i_vals else 0.0,
            "risk": float(np.mean(r_vals)) if r_vals else 0.0,
        }
        return out_rewards, step_cost

    def update_episode(self, ep_cost_mean: Dict[str, float], episode: int) -> Dict[str, float]:
        if not self.enabled:
            return dict(self.lam)
        if int(episode) <= self.warmup_episodes:
            return dict(self.lam)
        for k in ("energy", "interf", "risk"):
            c = float(ep_cost_mean.get(k, 0.0))
            b = float(self.budget[k])
            self.lam[k] = float(np.clip(self.lam[k] + self.lr * (c - b), 0.0, self.lam_max))
        return dict(self.lam)


def _build_ctde_global_state(env: VecOffloadingEnv, obs_list: List[Dict], step_info: Dict = None) -> np.ndarray:
    """Build fixed-dim centralized critic summary (paper version CTDE)."""
    num_veh = max(len(getattr(env, "vehicles", [])), 1)
    num_rsu = max(len(getattr(env, "rsus", [])), 1)
    rsu_q_lens = []
    rsu_load_ratio = []
    rsu_user_counts = [0.0 for _ in range(num_rsu)]
    per_proc_limit = float(max(getattr(Cfg, "RSU_QUEUE_CYCLES_LIMIT", 1.0), 1.0)) / float(
        max(getattr(Cfg, "RSU_NUM_PROCESSORS", 1), 1)
    )
    # Count V2I users per RSU from current tx queues (for RSU occupancy summary).
    for _, q in getattr(env, "txq_v2i", {}).items():
        if not q:
            continue
        dst = getattr(q[0], "dst_node", None)
        if isinstance(dst, tuple) and len(dst) >= 2 and dst[0] == "RSU":
            rid = int(dst[1])
            if 0 <= rid < num_rsu:
                rsu_user_counts[rid] += 1.0
    for rsu in getattr(env, "rsus", []):
        proc_dict = getattr(env, "rsu_cpu_q", {}).get(rsu.id, {})
        q_len = 0
        loads = []
        for q in proc_dict.values():
            q_len += len(q)
            loads.append(sum(getattr(j, "rem_cycles", 0.0) for j in q))
        rsu_q_lens.append(float(q_len))
        if loads:
            rsu_load_ratio.append(float(np.mean(loads) / max(per_proc_limit, 1e-9)))
    rsu_q_mean = float(np.mean(rsu_q_lens)) if rsu_q_lens else 0.0
    rsu_q_min = float(np.min(rsu_q_lens)) if rsu_q_lens else 0.0
    rsu_q_p95 = float(np.percentile(rsu_q_lens, 95)) if rsu_q_lens else 0.0
    rsu_q_max = float(np.max(rsu_q_lens)) if rsu_q_lens else 0.0
    rsu_load_mean = float(np.mean(rsu_load_ratio)) if rsu_load_ratio else 0.0
    rsu_load_p95 = float(np.percentile(rsu_load_ratio, 95)) if rsu_load_ratio else 0.0
    rsu_users_mean = float(np.mean(rsu_user_counts)) if rsu_user_counts else 0.0
    rsu_users_p95 = float(np.percentile(rsu_user_counts, 95)) if rsu_user_counts else 0.0

    active_v2i = sum(1 for tx, q in getattr(env, "txq_v2i", {}).items() if tx[0] == "VEH" and len(q) > 0)
    active_v2v = sum(1 for tx, q in getattr(env, "txq_v2v", {}).items() if tx[0] == "VEH" and len(q) > 0)

    rb_occ = getattr(getattr(env, "channel", None), "last_v2v_stats", {}).get(
        "rb_occupancy", np.zeros(max(getattr(Cfg, "V2V_NUM_RB", 1), 1))
    )
    rb_occ = np.asarray(rb_occ, dtype=float).reshape(-1) if rb_occ is not None else np.zeros(1, dtype=float)
    rb_use_ratio = float(np.mean(rb_occ > 0)) if rb_occ.size > 0 else 0.0
    rb_occ_mean = float(np.mean(rb_occ)) if rb_occ.size > 0 else 0.0
    rb_occ_p95 = float(np.percentile(rb_occ, 95)) if rb_occ.size > 0 else 0.0
    rb_concurrency = float(np.mean(rb_occ[rb_occ > 0])) if np.any(rb_occ > 0) else 0.0

    step_info = step_info or {}
    i_ref = float(max(getattr(Cfg, "I_REF_MIN_UNIFIED", 1e-8), 1e-12))
    i_clip = float(max(getattr(Cfg, "INTERF_RATIO_CLIP_UNIFIED", 20.0), 1e-6))
    i_mean = float(
        step_info.get(
            "interf_i_total_all_mean",
            step_info.get("v2v_i_total_mean", 0.0),
        )
        or 0.0
    )
    i_p95 = float(
        step_info.get(
            "interf_i_total_all_p95",
            step_info.get("v2v_i_total_p95", i_mean),
        )
        or i_mean
    )
    i_mean_norm = float(min(max(i_mean, 0.0) / i_ref, i_clip)) / i_clip
    i_p95_norm = float(min(max(i_p95, 0.0) / i_ref, i_clip)) / i_clip
    sinr_p50 = float(step_info.get("v2v_sinr_p50", 0.0) or 0.0)
    sinr_p10 = float(step_info.get("v2v_sinr_p10", sinr_p50) or sinr_p50)
    sinr_norm = float(np.log1p(max(sinr_p50, 0.0)) / np.log1p(100.0))
    sinr_p10_norm = float(np.log1p(max(sinr_p10, 0.0)) / np.log1p(100.0))

    on_task_rate = 0.0
    slack_mean = 0.0
    slack_p10 = 0.0
    slack_p90 = 0.0
    v2v_avail_ratio = 0.0
    rsu_avail_ratio = 0.0
    remote_avail_ratio = 0.0
    rho_mean = 1.0
    rho_p10 = 1.0
    unc_mean = 0.0
    unc_p90 = 0.0
    if obs_list:
        on_task_rate = float(np.mean([1.0 if int(obs.get("subtask_index", -1)) >= 0 else 0.0 for obs in obs_list]))
        slack_vals = []
        v2v_vals = []
        rsu_vals = []
        remote_vals = []
        rho_vals = []
        unc_vals = []
        for obs in obs_list:
            rr = obs.get("resource_raw")
            am = obs.get("action_mask")
            ct = obs.get("candidate_types")
            if isinstance(rr, np.ndarray) and rr.ndim == 2 and rr.shape[0] > 0 and rr.shape[1] > 9:
                slack_vals.append(float(np.clip(rr[0, 9], 0.0, 1.0)))
            if isinstance(am, np.ndarray) and isinstance(ct, np.ndarray) and len(am) == len(ct):
                valid = (am > 0)
                denom = float(np.sum(valid))
                if denom > 0:
                    v2v_vals.append(float(np.sum((ct == 3) & valid) / denom))
                    rsu_vals.append(float(np.sum((ct == 2) & valid) / denom))
                    remote_mask = ((ct == 2) | (ct == 3)) & valid
                    remote_vals.append(float(np.sum(remote_mask) / denom))
                    if isinstance(rr, np.ndarray) and rr.ndim == 2 and rr.shape[0] == len(ct) and rr.shape[1] >= 14:
                        if np.any(remote_mask):
                            rho_remote = rr[remote_mask, 12]
                            unc_remote = rr[remote_mask, 13]
                            rho_vals.extend(np.clip(rho_remote, 0.0, 1.0).tolist())
                            unc_vals.extend(np.clip(unc_remote, 0.0, 1.0).tolist())
        if slack_vals:
            slack_mean = float(np.mean(slack_vals))
            slack_p10 = float(np.percentile(slack_vals, 10))
            slack_p90 = float(np.percentile(slack_vals, 90))
        v2v_avail_ratio = float(np.mean(v2v_vals)) if v2v_vals else 0.0
        rsu_avail_ratio = float(np.mean(rsu_vals)) if rsu_vals else 0.0
        remote_avail_ratio = float(np.mean(remote_vals)) if remote_vals else 0.0
        if rho_vals:
            rho_mean = float(np.mean(rho_vals))
            rho_p10 = float(np.percentile(rho_vals, 10))
        if unc_vals:
            unc_mean = float(np.mean(unc_vals))
            unc_p90 = float(np.percentile(unc_vals, 90))

    no_task_rate = 1.0 - on_task_rate
    g = np.array([
        # RSU side
        np.clip(rsu_q_mean / 20.0, 0.0, 1.0),
        np.clip(rsu_q_p95 / 50.0, 0.0, 1.0),
        np.clip(rsu_q_min / 20.0, 0.0, 1.0),
        np.clip(rsu_q_max / 50.0, 0.0, 1.0),
        np.clip(rsu_load_mean / 2.0, 0.0, 1.0),
        np.clip(rsu_load_p95 / 2.0, 0.0, 1.0),
        np.clip(rsu_users_mean / float(num_veh), 0.0, 1.0),
        np.clip(rsu_users_p95 / float(num_veh), 0.0, 1.0),
        # Link activity / V2V interference
        np.clip(active_v2i / float(num_veh), 0.0, 1.0),
        np.clip(active_v2v / float(num_veh), 0.0, 1.0),
        np.clip(rb_use_ratio, 0.0, 1.0),
        np.clip(rb_occ_mean / float(num_veh), 0.0, 1.0),
        np.clip(rb_occ_p95 / float(num_veh), 0.0, 1.0),
        np.clip(rb_concurrency / float(num_veh), 0.0, 1.0),
        np.clip(i_mean_norm, 0.0, 1.0),
        np.clip(i_p95_norm, 0.0, 1.0),
        np.clip(sinr_norm, 0.0, 1.0),
        np.clip(sinr_p10_norm, 0.0, 1.0),
        # Deadline/task pressure
        np.clip(on_task_rate, 0.0, 1.0),
        np.clip(no_task_rate, 0.0, 1.0),
        np.clip(slack_mean, 0.0, 1.0),
        np.clip(slack_p10, 0.0, 1.0),
        np.clip(slack_p90, 0.0, 1.0),
        np.clip(v2v_avail_ratio, 0.0, 1.0),
        np.clip(rsu_avail_ratio, 0.0, 1.0),
        np.clip(remote_avail_ratio, 0.0, 1.0),
        # Trust summary
        np.clip(rho_mean, 0.0, 1.0),
        np.clip(rho_p10, 0.0, 1.0),
        np.clip(unc_mean, 0.0, 1.0),
        np.clip(unc_p90, 0.0, 1.0),
    ], dtype=np.float32)
    gdim = int(getattr(TC, "CTDE_GLOBAL_DIM", g.shape[0]))
    if g.shape[0] < gdim:
        g = np.pad(g, (0, gdim - g.shape[0]))
    elif g.shape[0] > gdim:
        g = g[:gdim]
    return g.astype(np.float32)


def _attach_global_state(obs_list: List[Dict], global_state: np.ndarray) -> None:
    if not obs_list:
        return
    g = np.asarray(global_state, dtype=np.float32).reshape(-1)
    for obs in obs_list:
        obs["global_state"] = g.copy()


def _print_unified_reward_scale_check():
    """Lightweight numeric sanity check for unified reward scales."""
    td_candidates = []
    for k in ("TASK_A_DEADLINE_MIN", "TASK_B_DEADLINE_MIN", "DEADLINE_FIXED_MIN"):
        v = getattr(Cfg, k, None)
        if v is not None:
            try:
                td_candidates.append(float(v))
            except Exception:
                pass
    td_min = min([x for x in td_candidates if x > 0.0], default=max(float(getattr(Cfg, "DT", 0.1)) * 4.0, 0.4))
    dt = float(getattr(Cfg, "DT", 0.1))
    max_steps = int(getattr(TC, "MAX_STEPS", getattr(Cfg, "MAX_STEPS", 200)))

    w_t = float(getattr(Cfg, "W_TIME", 0.0))
    w_e = float(getattr(Cfg, "W_ENERGY", 0.0))
    w_i = float(getattr(Cfg, "W_INTERF", 0.0))
    w_r = float(getattr(Cfg, "W_RISK", 0.0))
    p_e = float(getattr(Cfg, "P_ENERGY", 1.0))
    p_i = float(getattr(Cfg, "P_INTERF", 1.0))
    e_ratio_cap = float(getattr(Cfg, "REWARD_SCALE_E_RATIO_CAP", 1.5))
    i_ratio_cap = float(max(getattr(Cfg, "INTERF_RATIO_CLIP_UNIFIED", 3.0), 1e-6))

    r_time_max = w_t * (dt / max(td_min, 1e-6))
    r_energy_max = w_e * (e_ratio_cap ** p_e)
    r_interf_max = w_i * (i_ratio_cap ** p_i)
    r_risk_max = w_r
    step_legal_cap = r_time_max + r_energy_max + r_interf_max + r_risk_max
    ep_legal_cap = step_legal_cap * max_steps
    term_cap = max(float(getattr(Cfg, "R_SUCC", 10.0)), float(getattr(Cfg, "R_FAIL", 10.0)))
    ratio = ep_legal_cap / max(term_cap, 1e-6)
    print(
        "[ScaleCheck] UNIFIED cap:"
        f" step<=~{step_legal_cap:.3f}"
        f" ep<=~{ep_legal_cap:.1f}"
        f" term={term_cap:.1f}"
        f" ep/term={ratio:.2f}"
        f" (Td_min={td_min:.2f}s, DT={dt:.2f}s, i_cap={i_ratio_cap:g}, e_cap={e_ratio_cap:g})"
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
    parser.add_argument(
        "--exact-run-dir",
        action="store_true",
        default=False,
        help="Use --run-dir exactly as provided (no timestamp suffix).",
    )
    parser.add_argument("--step-metrics", action="store_true", default=False)
    parser.add_argument("--no-step-metrics", action="store_true", default=False)
    parser.add_argument("--step-logs", action="store_true", default=False)
    parser.add_argument("--no-step-logs", action="store_true", default=False)
    parser.add_argument("--enable-baseline-eval", action="store_true", default=False)
    parser.add_argument("--disable-baseline-eval", action="store_true", default=False)
    return parser.parse_args()


def apply_env_overrides():
    """Apply env var overrides to SystemConfig (Cfg) and TrainConfig (TC)."""
    # System / environment knobs
    overrides_float = {
        "VEHICLE_ARRIVAL_RATE": "VEHICLE_ARRIVAL_RATE",
        "BW_V2V": "BW_V2V",
        "BW_V2I": "BW_V2I",
        "V2I_RB_BW_HZ": "V2I_RB_BW_HZ",
        "MIN_DATA": "MIN_DATA",
        "MAX_DATA": "MAX_DATA",
        "MIN_EDGE_DATA": "MIN_EDGE_DATA",
        "MAX_EDGE_DATA": "MAX_EDGE_DATA",
        "MIN_CPU": "MIN_VEHICLE_CPU_FREQ",
        "MAX_CPU": "MAX_VEHICLE_CPU_FREQ",
        "TIME_QUEUE_PENALTY_WEIGHT": "TIME_QUEUE_PENALTY_WEIGHT",
        # Experiment config overrides
        "MAP_SIZE": "MAP_SIZE",
        "RSU_RANGE": "RSU_RANGE",
        "V2V_RANGE": "V2V_RANGE",
        "MIN_COMP": "MIN_COMP",
        "MAX_COMP": "MAX_COMP",
        "NORM_MAX_COMP": "NORM_MAX_COMP",
        "DEADLINE_TIGHTENING_MIN": "DEADLINE_TIGHTENING_MIN",
        "DEADLINE_TIGHTENING_MAX": "DEADLINE_TIGHTENING_MAX",
        "RSU_QUEUE_CYCLES_LIMIT": "RSU_QUEUE_CYCLES_LIMIT",
        "VEHICLE_QUEUE_CYCLES_LIMIT": "VEHICLE_QUEUE_CYCLES_LIMIT",
        "VEHICLE_SPAWN_X_MAX": "VEHICLE_SPAWN_X_MAX",
        # Chain proxy parameters
        "CHAIN_RISK_WEIGHT_DEPOSIT": "CHAIN_RISK_WEIGHT_DEPOSIT",
        "CHAIN_RISK_WEIGHT_FAIL": "CHAIN_RISK_WEIGHT_FAIL",
        "CHAIN_P50_LOW": "CHAIN_P50_LOW",
        "CHAIN_P95_LOW": "CHAIN_P95_LOW",
        "CHAIN_PFAIL_LOW": "CHAIN_PFAIL_LOW",
        "CHAIN_P50_HIGH": "CHAIN_P50_HIGH",
        "CHAIN_P95_HIGH": "CHAIN_P95_HIGH",
        "CHAIN_PFAIL_HIGH": "CHAIN_PFAIL_HIGH",
        "CHAIN_NOISE_STD": "CHAIN_NOISE_STD",
    }
    overrides_int = {
        "RSU_NUM_PROCESSORS": "RSU_NUM_PROCESSORS",
        "NUM_VEHICLES": "NUM_VEHICLES",
        "NUM_RSU": "NUM_RSU",
        "V2V_NUM_RB": "V2V_NUM_RB",
        "V2V_TOP_K": "V2V_TOP_K",
        "V2I_NUM_RB": "V2I_NUM_RB",
        "V2I_FREQ_REUSE_FACTOR": "V2I_FREQ_REUSE_FACTOR",
        "MIN_NODES": "MIN_NODES",
        "MAX_NODES": "MAX_NODES",
        # Chain proxy parameters
        "CHAIN_SWITCH_PERIOD_STEPS": "CHAIN_SWITCH_PERIOD_STEPS",
        "CHAIN_TRUST_DELAY_BASE_STEPS": "CHAIN_TRUST_DELAY_BASE_STEPS",
        "CHAIN_TRUST_DELAY_MIN_STEPS": "CHAIN_TRUST_DELAY_MIN_STEPS",
        "CHAIN_TRUST_DELAY_MAX_STEPS": "CHAIN_TRUST_DELAY_MAX_STEPS",
    }
    overrides_str = {
        "CHAIN_MODE": "CHAIN_MODE",
        "V2I_RATE_MODEL": "V2I_RATE_MODEL",
        "DEADLINE_MODE": "DEADLINE_MODE",
    }
    overrides_bool = {
        "CHAIN_ENABLED": "CHAIN_ENABLED",
        "CHAIN_TRUST_DELAY_COUPLED": "CHAIN_TRUST_DELAY_COUPLED",
        "TRUST_ENABLED": "TRUST_ENABLED",
        "V2I_ICI_ENABLED": "V2I_ICI_ENABLED",
    }
    for env_key, cfg_attr in overrides_float.items():
        val = _env_float(env_key)
        if val is not None:
            setattr(Cfg, cfg_attr, val)
    for env_key, cfg_attr in overrides_int.items():
        val = _env_int(env_key)
        if val is not None:
            setattr(Cfg, cfg_attr, val)
    for env_key, cfg_attr in overrides_str.items():
        val = _env_str(env_key)
        if val:
            setattr(Cfg, cfg_attr, val)
    for env_key, cfg_attr in overrides_bool.items():
        val = _env_bool(env_key)
        if val is not None:
            setattr(Cfg, cfg_attr, bool(val))

    # Train / PPO knobs
    tc_float = {
        "GAMMA": "GAMMA",
        "CLIP_PARAM": "CLIP_PARAM",
        "TARGET_KL": "TARGET_KL",
        "TARGET_KL_STOP_MULT": "TARGET_KL_STOP_MULT",
        "MAX_GRAD_NORM": "MAX_GRAD_NORM",
        "ENTROPY_COEF": "ENTROPY_COEF",
        "ENTROPY_COEF_START": "ENTROPY_COEF_START",
        "ENTROPY_COEF_END": "ENTROPY_COEF_END",
        "LR_ACTOR": "LR_ACTOR",
        "LR_CRITIC": "LR_CRITIC",
        "BIAS_ANNEAL_FRAC": "BIAS_ANNEAL_FRAC",
        "LOGIT_BIAS_LOCAL_INIT": "LOGIT_BIAS_LOCAL_INIT",
        "LOGIT_BIAS_LOCAL_END": "LOGIT_BIAS_LOCAL_END",
        "LOGIT_BIAS_RSU_INIT": "LOGIT_BIAS_RSU_INIT",
        "LOGIT_BIAS_RSU_END": "LOGIT_BIAS_RSU_END",
        "LOGIT_BIAS_LOCAL": "LOGIT_BIAS_LOCAL",
        "LOGIT_BIAS_RSU": "LOGIT_BIAS_RSU",
        "VALUE_CLIP_RANGE": "VALUE_CLIP_RANGE",
        "LR_DECAY_RATE": "LR_DECAY_RATE",
        "CMDP_LAMBDA_LR": "CMDP_LAMBDA_LR",
        "CMDP_LAMBDA_MAX": "CMDP_LAMBDA_MAX",
        "CMDP_LAMBDA_ENERGY_INIT": "CMDP_LAMBDA_ENERGY_INIT",
        "CMDP_LAMBDA_INTERF_INIT": "CMDP_LAMBDA_INTERF_INIT",
        "CMDP_LAMBDA_RISK_INIT": "CMDP_LAMBDA_RISK_INIT",
        "CMDP_BUDGET_ENERGY": "CMDP_BUDGET_ENERGY",
        "CMDP_BUDGET_INTERF": "CMDP_BUDGET_INTERF",
        "CMDP_BUDGET_RISK": "CMDP_BUDGET_RISK",
        "LOGIT_BIAS_V2V_INIT": "LOGIT_BIAS_V2V_INIT",
        "LOGIT_BIAS_V2V_END": "LOGIT_BIAS_V2V_END",
    }
    tc_int = {
        "PPO_EPOCH": "PPO_EPOCH",
        "MINI_BATCH_SIZE": "MINI_BATCH_SIZE",
        "MIN_ACTIVE_SAMPLES": "MIN_ACTIVE_SAMPLES",
        "LR_DECAY_STEPS": "LR_DECAY_STEPS",
        "ENTROPY_ANNEAL_STEPS": "ENTROPY_ANNEAL_STEPS",
        "SAVE_INTERVAL": "SAVE_INTERVAL",
        "CTDE_GLOBAL_DIM": "CTDE_GLOBAL_DIM",
        "LOGIT_BIAS_V2V_ANNEAL_STEPS": "LOGIT_BIAS_V2V_ANNEAL_STEPS",
        "CMDP_WARMUP_EPISODES": "CMDP_WARMUP_EPISODES",
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
    use_lr_decay = _env_bool("USE_LR_DECAY")
    if use_lr_decay is not None:
        TC.USE_LR_DECAY = use_lr_decay
    use_value_clip = _env_bool("USE_VALUE_CLIP")
    if use_value_clip is not None:
        TC.USE_VALUE_CLIP = use_value_clip
    use_value_target_norm = _env_bool("USE_VALUE_TARGET_NORM")
    if use_value_target_norm is not None:
        TC.USE_VALUE_TARGET_NORM = use_value_target_norm
    use_rank_bias = _env_bool("USE_RANK_BIAS")
    if use_rank_bias is not None:
        TC.USE_RANK_BIAS = use_rank_bias

    # Algorithm ablation overrides
    use_edge_bias = _env_bool("USE_EDGE_BIAS")
    if use_edge_bias is not None:
        TC.USE_EDGE_BIAS = use_edge_bias
    use_spatial_bias = _env_bool("USE_SPATIAL_BIAS")
    if use_spatial_bias is not None:
        TC.USE_SPATIAL_BIAS = use_spatial_bias
    use_physics_bias = _env_bool("USE_PHYSICS_BIAS")
    if use_physics_bias is not None:
        TC.USE_PHYSICS_BIAS = use_physics_bias
    use_fixed_power = _env_bool("USE_FIXED_POWER")
    if use_fixed_power is not None:
        TC.USE_FIXED_POWER = use_fixed_power
    use_cmdp = _env_bool("CMDP_ENABLE")
    if use_cmdp is not None:
        TC.CMDP_ENABLE = use_cmdp
    # Ablation: Transformer layers
    num_layers = _env_int("NUM_LAYERS")
    if num_layers is not None:
        TC.NUM_LAYERS = num_layers

    # Ablation study overrides (SystemConfig)
    reward_beta = _env_float("REWARD_BETA")
    if reward_beta is not None:
        Cfg.REWARD_BETA = reward_beta

    # String overrides (CANDIDATE_MODE等)
    cand_mode = os.environ.get("CANDIDATE_MODE")
    if cand_mode:
        Cfg.CANDIDATE_MODE = cand_mode.upper()
    topk_k = _env_int("TOPK_K")
    if topk_k is not None:
        Cfg.TOPK_K = topk_k
    randomk_k = _env_int("RANDOMK_K")
    if randomk_k is not None:
        Cfg.RANDOMK_K = randomk_k

    # Recalculate derived values after overrides.
    # Keep V2V bandwidth split self-consistent whenever V2V_NUM_RB / BW_V2V is overridden.
    if int(getattr(Cfg, "V2V_NUM_RB", 0)) <= 0:
        raise ValueError(f"V2V_NUM_RB must be positive, got {getattr(Cfg, 'V2V_NUM_RB', None)}")
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    bw_rb_lhs = float(Cfg.V2V_BW_PER_RB) * float(Cfg.V2V_NUM_RB)
    bw_rb_rhs = float(Cfg.BW_V2V)
    if abs(bw_rb_lhs - bw_rb_rhs) > 1e-6:
        raise RuntimeError(
            "Inconsistent V2V bandwidth split: "
            f"V2V_BW_PER_RB*V2V_NUM_RB={bw_rb_lhs:.12f}, BW_V2V={bw_rb_rhs:.12f}"
        )
    # V2I RB参数（RB_SINR模式会使用；SHARE模式忽略）
    v2i_rb_bw = float(max(getattr(Cfg, "V2I_RB_BW_HZ", 180e3), 1.0))
    if int(getattr(Cfg, "V2I_NUM_RB", 0)) <= 0:
        Cfg.V2I_NUM_RB = max(int(round(float(Cfg.BW_V2I) / v2i_rb_bw)), 1)

    Cfg.ALL_FEASIBLE = (str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL")
    Cfg.MAX_NEIGHBORS = (Cfg.NUM_VEHICLES - 1) if Cfg.ALL_FEASIBLE else max(0, min(Cfg.NUM_VEHICLES - 1, Cfg.V2V_TOP_K))
    Cfg.MAX_TARGETS = (1 + Cfg.NUM_RSU + Cfg.MAX_NEIGHBORS) if Cfg.ENABLE_RSU_SELECTION else (2 + Cfg.MAX_NEIGHBORS)


def _compute_unified_nominal_scale_check() -> Dict[str, float]:
    """Compute nominal unified-reward component scales without stepping the env."""
    # Fixed nominal values requested for static magnitude checks.
    w_cycles = 1.0e9
    f_hz = 3.0e9
    kappa = 1.0e-27
    d_bits = 1.0e6
    p_watt = 0.1
    snr_linear = 10.0 ** (20.0 / 10.0)
    dt = 0.1
    Td = 10.0
    rho = 0.7

    v2v_num_rb = max(float(getattr(Cfg, "V2V_NUM_RB", 1)), 1.0)
    b_rb = float(getattr(Cfg, "BW_V2V", 0.0)) / v2v_num_rb
    rate_rb = b_rb * np.log2(1.0 + snr_linear)
    t_tx = d_bits / max(rate_rb, 1e-12)

    e_loc = kappa * w_cycles * (f_hz ** 2)
    e_tx = p_watt * t_tx

    w_time = float(getattr(Cfg, "W_TIME", 0.0))
    w_energy = float(getattr(Cfg, "W_ENERGY", 0.0))
    p_energy = float(getattr(Cfg, "P_ENERGY", 1.0))
    w_interf = float(getattr(Cfg, "W_INTERF", 0.0))
    p_interf = float(getattr(Cfg, "P_INTERF", 1.0))
    w_risk = float(getattr(Cfg, "W_RISK", 0.0))
    p_risk = float(getattr(Cfg, "P_RISK", 1.0))
    e_ref = max(float(getattr(Cfg, "E_REF_UNIFIED", 1.0)), 1e-12)

    r_time = -w_time * (dt / max(Td, 1e-12))
    r_energy = -w_energy * ((max(e_tx, 0.0) / e_ref) ** p_energy)
    # Nominal check requirement: assume I_caused = I_ref => normalized interference = 1.
    r_interf = -w_interf * (1.0 ** p_interf)
    r_risk = -w_risk * ((1.0 - rho) ** p_risk)

    return {
        "w_cycles": w_cycles,
        "f_hz": f_hz,
        "kappa": kappa,
        "d_bits": d_bits,
        "p_watt": p_watt,
        "B_RB": b_rb,
        "SNR_linear": snr_linear,
        "dt": dt,
        "Td": Td,
        "rho": rho,
        "t_tx": float(t_tx),
        "E_loc": float(e_loc),
        "E_tx": float(e_tx),
        "r_time_step": float(r_time),
        "r_energy_step": float(r_energy),
        "r_interf_step": float(r_interf),
        "r_risk_step": float(r_risk),
    }


def _print_unified_nominal_scale_check():
    vals = _compute_unified_nominal_scale_check()
    print(
        "[ScaleCheck] UNIFIED nominal: "
        f"E_loc={vals['E_loc']:.6g}J E_tx={vals['E_tx']:.6g}J "
        f"r_time={vals['r_time_step']:.6g} r_energy={vals['r_energy_step']:.6g} "
        f"r_interf={vals['r_interf_step']:.6g} r_risk={vals['r_risk_step']:.6g} "
        f"(B_RB={vals['B_RB']:.6g}Hz, SNR=20dB, dt=0.1, Td=10, rho=0.7)",
        flush=True,
    )

    abs_parts = {
        "r_time_step": abs(vals["r_time_step"]),
        "r_energy_step": abs(vals["r_energy_step"]),
        "r_interf_step": abs(vals["r_interf_step"]),
        "r_risk_step": abs(vals["r_risk_step"]),
    }
    warned = False
    for name, mag in abs_parts.items():
        others = [v for k, v in abs_parts.items() if k != name and v > 1e-12]
        if not others:
            continue
        if mag > 10.0 * max(others):
            warned = True
            print(
                f"[ScaleCheck][Warn] {name} nominal abs magnitude ({mag:.6g}) exceeds others by >10x.",
                flush=True,
            )
    if warned:
        print(
            "[ScaleCheck][Suggest] Consider rebalancing reward scales; default suggestion: set E_REF_UNIFIED=2.0 (J).",
            flush=True,
        )


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


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _fmt_float(val, precision=3, fallback="-"):
    f = _safe_float(val)
    if f is None:
        return fallback
    # Avoid printing tiny non-zero values as 0.000, which hides signal (e.g. energy_norm ~ 1e-6).
    if f != 0.0 and abs(f) < (10.0 ** (-precision)):
        return f"{f:.2e}"
    return f"{f:.{precision}f}"


def _fmt_pct(val, precision=1, fallback="-"):
    f = _safe_float(val)
    if f is None:
        return fallback
    return f"{f * 100.0:.{precision}f}%"


def _print_startup_summary(
    run_id: str,
    run_dir: str,
    logs_dir: str,
    plots_dir: str,
    models_dir: str,
    device: str,
    baseline_eval_enabled: bool,
    log_step_metrics: bool,
    log_step_logs: bool,
):
    p_min_dbm = getattr(Cfg, "TX_POWER_MIN_DBM", None)
    p_max_dbm = getattr(Cfg, "TX_POWER_MAX_DBM", None)
    p_min_w = getattr(Cfg, "P_MIN_WATT", None)
    p_max_w = getattr(Cfg, "P_MAX_WATT", None)
    if p_min_w is None and p_min_dbm is not None:
        p_min_w = 10 ** ((float(p_min_dbm) - 30.0) / 10.0)
    if p_max_w is None and p_max_dbm is not None:
        p_max_w = 10 ** ((float(p_max_dbm) - 30.0) / 10.0)

    print("\n" + "=" * 118, flush=True)
    print("[Train] MAPPO VEC training started", flush=True)
    print("=" * 118, flush=True)
    print(
        f"[Run] id={run_id} seed={Cfg.SEED} device={device} reward={Cfg.REWARD_SCHEME} "
        f"candidate={getattr(Cfg, 'CANDIDATE_MODE', 'N/A')} "
        f"(all_feasible={bool(getattr(Cfg, 'ALL_FEASIBLE', False))})",
        flush=True,
    )
    print(
        f"[Train] episodes={TC.MAX_EPISODES} steps/ep={TC.MAX_STEPS} eval_interval={TC.EVAL_INTERVAL} "
        f"save_interval={TC.SAVE_INTERVAL} baseline_eval={'on' if baseline_eval_enabled else 'off'}",
        flush=True,
    )
    print(
        f"[Scenario] vehicles={Cfg.NUM_VEHICLES} rsu={Cfg.NUM_RSU} map={Cfg.MAP_SIZE}m lanes={Cfg.NUM_LANES} "
        f"v2v_range={Cfg.V2V_RANGE}m rsu_range={Cfg.RSU_RANGE}m",
        flush=True,
    )
    print(
        f"[Physics] BW_V2I={Cfg.BW_V2I/1e6:.2f}MHz BW_V2V={Cfg.BW_V2V/1e6:.2f}MHz RB={Cfg.V2V_NUM_RB} "
        f"Ptx=[{_fmt_float(p_min_w, 4)},{_fmt_float(p_max_w, 4)}]W "
        f"({_fmt_float(p_min_dbm, 1)},{_fmt_float(p_max_dbm, 1)} dBm) DT={Cfg.DT:.3f}s",
        flush=True,
    )
    _print_unified_nominal_scale_check()
    print(
        f"[Compute] f_local=[{Cfg.MIN_VEHICLE_CPU_FREQ/1e9:.2f},{Cfg.MAX_VEHICLE_CPU_FREQ/1e9:.2f}]GHz "
        f"f_rsu={Cfg.F_RSU/1e9:.2f}GHz",
        flush=True,
    )
    print(
        f"[PPO] lr={TC.LR_ACTOR:.2e} gamma={TC.GAMMA:.3f} gae={TC.GAE_LAMBDA:.3f} clip={TC.CLIP_PARAM:.3f} "
        f"ent={TC.ENTROPY_COEF:.3f} batch={TC.MINI_BATCH_SIZE} epochs={TC.PPO_EPOCH} d_model={TC.EMBED_DIM} layers={TC.NUM_LAYERS}",
        flush=True,
    )
    print(
        f"[Logging] step_metrics={'on' if log_step_metrics else 'off'} "
        f"step_logs={'on' if log_step_logs else 'off'} tb_obs={'on' if _bool_env('TB_LOG_OBS', True) else 'off'}",
        flush=True,
    )
    print(f"[Paths] run={run_dir}", flush=True)
    print(f"[Paths] logs={logs_dir}  models={models_dir}  plots={plots_dir}", flush=True)
    print("=" * 118 + "\n", flush=True)


def _collect_plot_manifest(plots_dir: str) -> List[Dict[str, Any]]:
    plot_root = Path(plots_dir)
    if not plot_root.exists():
        return []
    figures: List[Dict[str, Any]] = []
    for fig_path in sorted(plot_root.glob("*.png")):
        name = fig_path.name.lower()
        tags = []
        if "reward" in name:
            tags.append("reward")
        if "success" in name:
            tags.append("success")
        if "policy" in name or "offloading" in name or "decision" in name:
            tags.append("policy")
        if "diagnostic" in name or "stability" in name or "loss" in name or "convergence" in name:
            tags.append("rl_diagnostics")
        if "physical" in name or "resource" in name or "queue" in name or "power" in name:
            tags.append("simulation")
        if "baseline" in name or "comparison" in name:
            tags.append("baseline")
        if "fairness" in name or "collaboration" in name or "multi_agent" in name:
            tags.append("multi_agent")
        stat = fig_path.stat()
        figures.append(
            {
                "file": fig_path.name,
                "path": str(fig_path),
                "size_bytes": int(stat.st_size),
                "tags": tags,
            }
        )
    return figures


def _write_plot_manifest(plots_dir: str, jobs: List[Dict[str, Any]]) -> str:
    figures = _collect_plot_manifest(plots_dir)
    tag_counts: Dict[str, int] = {}
    for fig in figures:
        for tag in fig.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plots_dir": str(Path(plots_dir).resolve()),
        "plot_count": len(figures),
        "tag_counts": tag_counts,
        "jobs": jobs,
        "figures": figures,
    }
    manifest_path = Path(plots_dir) / "plot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return str(manifest_path)


def _run_plot_job(label: str, cmd: List[str]) -> Dict[str, Any]:
    print(f"[Auto Plot] {label} ...", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        dt = time.time() - t0
        out = (proc.stdout or "").strip()
        out_lines = [line for line in out.splitlines() if line.strip()]
        tail = out_lines[-3:] if out_lines else []
        if tail:
            print("[Auto Plot] " + " | ".join(tail), flush=True)
        print(f"[Auto Plot] {label} done in {dt:.1f}s", flush=True)
        return {"job": label, "ok": True, "seconds": dt, "tail": tail}
    except subprocess.CalledProcessError as e:
        dt = time.time() - t0
        out = (e.stdout or "").strip()
        out_lines = [line for line in out.splitlines() if line.strip()]
        tail = out_lines[-6:] if out_lines else [str(e)]
        print(f"[Auto Plot] {label} failed in {dt:.1f}s", flush=True)
        print("[Auto Plot] " + " | ".join(tail), flush=True)
        return {"job": label, "ok": False, "seconds": dt, "tail": tail}
    except Exception as e:
        dt = time.time() - t0
        print(f"[Auto Plot] {label} failed in {dt:.1f}s: {e}", flush=True)
        return {"job": label, "ok": False, "seconds": dt, "tail": [str(e)]}


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


def evaluate_single_baseline_episode(env, policy_name, episode_seed=None):
    """评估单个baseline策略的一个episode，返回完整的指标（与训练指标一致）"""
    if episode_seed is None:
        base_seed = int(getattr(Cfg, "SEED", 0))
        episode_seed = base_seed + int(getattr(env, "episode_count", 0))
    episode_seed = int(episode_seed)

    if policy_name == 'Random':
        policy = RandomPolicy(seed=episode_seed)
    elif policy_name == 'Local-Only':
        policy = LocalOnlyPolicy()
    elif policy_name == 'Greedy':
        policy = GreedyPolicy(env)
    elif policy_name == 'Oracle-Min':
        policy = OracleMinPolicy(env)
    elif policy_name == 'EFT':
        policy = EFTPPolicy(env)
    elif policy_name == 'CP-EFT':
        policy = CPFirstEFTPolicy(env)
    elif policy_name == 'LB-Greedy':
        policy = LBGreedyPolicy(env)
    elif policy_name == 'Static':
        policy = StaticPolicy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    obs_list, _ = env.reset(seed=episode_seed)
    if hasattr(policy, "reset"):
        policy.reset()
    ep_reward = 0
    total_steps = 0
    
    # 统计容器（与训练循环一致）
    stats = {
        "power_sum": 0.0,
        "power_values": [],
        "local_cnt": 0,
        "rsu_cnt": 0,
        "neighbor_cnt": 0,
        "queue_len_sum": 0,
        "rsu_queue_sum": 0,
        "assigned_cpu_sum": 0.0,
    }
    
    last_info = None
    for step in range(TC.MAX_STEPS):
        current_obs_list = obs_list  # classify decision types based on the obs used to select actions
        actions = policy.select_action(obs_list)
        _inject_obs_stamp(obs_list, actions)
        obs_list, rewards, done, truncated, info = env.step(actions)
        last_info = info
        ep_reward += sum(rewards) / len(rewards)
        total_steps += 1
        
        # 统计决策分布
        for i, act in enumerate(actions):
            target = int(act.get("target", 0))

            # Prefer candidate_types metadata for correct multi-RSU accounting.
            obs_i = current_obs_list[i] if i < len(current_obs_list) else {}
            candidate_types = obs_i.get("candidate_types") if isinstance(obs_i, dict) else None
            if candidate_types is not None and 0 <= target < len(candidate_types):
                ctype = int(candidate_types[target])
                if ctype == 1:
                    stats["local_cnt"] += 1
                elif ctype == 2:
                    stats["rsu_cnt"] += 1
                else:
                    stats["neighbor_cnt"] += 1
            else:
                # Fallback: target index convention
                if target == 0:
                    stats["local_cnt"] += 1
                elif getattr(Cfg, "ENABLE_RSU_SELECTION", False) and 1 <= target <= int(getattr(Cfg, "NUM_RSU", 1)):
                    stats["rsu_cnt"] += 1
                else:
                    stats["neighbor_cnt"] += 1
            
            p = float(act.get("power", 0.0))
            stats['power_sum'] += p
            if target != 0:
                # Only count tx power for remote decisions.
                stats["power_values"].append(p)
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
    power_ratio_mean = float(np.mean(stats["power_values"])) if stats["power_values"] else 0.0
    power_ratio_p95 = float(np.percentile(stats["power_values"], 95)) if stats["power_values"] else 0.0
    avg_veh_queue = stats['queue_len_sum'] / dec_den if total_decisions > 0 else 0.0
    avg_rsu_queue = stats['rsu_queue_sum'] / total_steps if total_steps > 0 else 0.0

    epm = last_info.get("episode_metrics", {}) if last_info else {}
    episode_time_seconds = epm.get("episode_time_seconds")
    time_limit_rate = epm.get("time_limit_rate")
    deadline_miss_rate = epm.get("deadline_miss_rate")
    mean_cft_est = epm.get("mean_cft_est")
    mean_cft_completed = epm.get("mean_cft_completed")
    task_duration_mean = epm.get("task_duration_mean")
    task_duration_p95 = epm.get("task_duration_p95")
    I_total_mean = epm.get("I_total_mean")
    I_total_p50 = epm.get("I_total_p50")
    I_total_p95 = epm.get("I_total_p95")
    I_caused_mean = epm.get("I_caused_mean")
    I_caused_p95 = epm.get("I_caused_p95")
    rho_selected_mean = epm.get("rho_selected_mean")
    rho_selected_p10 = epm.get("rho_selected_p10")
    uncertainty_selected_mean = epm.get("uncertainty_selected_mean")
    uncertainty_selected_p90 = epm.get("uncertainty_selected_p90")
    risk_penalty_mean = epm.get("risk_penalty_mean")
    chain_tx_total = epm.get("chain_tx_total")
    chain_p95_mean = epm.get("chain_p95_mean")
    chain_pfail_mean = epm.get("chain_pfail_mean")
    chain_risk_cost_total = epm.get("chain_risk_cost_total")
    trust_attempts = epm.get("trust_attempts")
    trust_failures = epm.get("trust_failures")
    trust_failure_rate = epm.get("trust_failure_rate")
    trust_retry_count = epm.get("trust_retry_count")
    illegal_action_rate = epm.get("illegal_action_rate")
    no_task_rate = epm.get("no_task_rate")
    on_task_rate = epm.get("on_task_rate")
    has_task_available_rate = epm.get("has_task_available_rate")
    unified_illegal_trigger_rate = epm.get("unified_illegal_trigger_rate")
    
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
        'power_ratio_mean': power_ratio_mean,
        'power_ratio_p95': power_ratio_p95,
        'avg_queue_len': avg_veh_queue,
        'avg_rsu_queue': avg_rsu_queue,
        'episode_time_seconds': float(episode_time_seconds) if episode_time_seconds is not None else None,
        'time_limit_rate': float(time_limit_rate) if time_limit_rate is not None else None,
        'deadline_miss_rate': float(deadline_miss_rate) if deadline_miss_rate is not None else None,
        'mean_cft_est': float(mean_cft_est) if mean_cft_est is not None else None,
        'mean_cft_completed': float(mean_cft_completed) if mean_cft_completed is not None else None,
        'task_duration_mean': float(task_duration_mean) if task_duration_mean is not None else None,
        'task_duration_p95': float(task_duration_p95) if task_duration_p95 is not None else None,
        'I_total_mean': float(I_total_mean) if I_total_mean is not None else None,
        'I_total_p50': float(I_total_p50) if I_total_p50 is not None else None,
        'I_total_p95': float(I_total_p95) if I_total_p95 is not None else None,
        'I_caused_mean': float(I_caused_mean) if I_caused_mean is not None else None,
        'I_caused_p95': float(I_caused_p95) if I_caused_p95 is not None else None,
        'rho_selected_mean': float(rho_selected_mean) if rho_selected_mean is not None else None,
        'rho_selected_p10': float(rho_selected_p10) if rho_selected_p10 is not None else None,
        'uncertainty_selected_mean': float(uncertainty_selected_mean) if uncertainty_selected_mean is not None else None,
        'uncertainty_selected_p90': float(uncertainty_selected_p90) if uncertainty_selected_p90 is not None else None,
        'risk_penalty_mean': float(risk_penalty_mean) if risk_penalty_mean is not None else None,
        'chain_tx_total': int(chain_tx_total) if chain_tx_total is not None else None,
        'chain_p95_mean': float(chain_p95_mean) if chain_p95_mean is not None else None,
        'chain_pfail_mean': float(chain_pfail_mean) if chain_pfail_mean is not None else None,
        'chain_risk_cost_total': float(chain_risk_cost_total) if chain_risk_cost_total is not None else None,
        'trust_attempts': int(trust_attempts) if trust_attempts is not None else None,
        'trust_failures': int(trust_failures) if trust_failures is not None else None,
        'trust_failure_rate': float(trust_failure_rate) if trust_failure_rate is not None else None,
        'trust_retry_count': int(trust_retry_count) if trust_retry_count is not None else None,
        'illegal_action_rate': float(illegal_action_rate) if illegal_action_rate is not None else None,
        'no_task_rate': float(no_task_rate) if no_task_rate is not None else None,
        'on_task_rate': float(on_task_rate) if on_task_rate is not None else None,
        'has_task_available_rate': float(has_task_available_rate) if has_task_available_rate is not None else None,
        'unified_illegal_trigger_rate': float(unified_illegal_trigger_rate) if unified_illegal_trigger_rate is not None else None,
        'episode_vehicle_count': episode_vehicle_count,
        'episode_task_count': episode_vehicle_count,  # 每辆车一个任务
        'v2v_gain_mean': collab_gain_mean if collab_gain_mean is not None else 0.0,
        'v2v_gain_pos_rate': collab_gain_pos_rate if collab_gain_pos_rate is not None else 0.0,
        'v2v_gain_pos_mean': collab_gain_pos_mean if collab_gain_pos_mean is not None else 0.0,
    }


def main():
    args = _parse_args()
    disable_baseline_eval = True
    
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
    env_reward_scheme = _env_str("REWARD_SCHEME")

    # Env/train overrides from environment variables (after profile/reward selection)
    apply_env_overrides()

    if Cfg.REWARD_SCHEME in ("PBRS_KP", "PBRS_KP_V2"):
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
    if env_reward_scheme:
        Cfg.REWARD_SCHEME = env_reward_scheme

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

    if env_disable_baseline is not None:
        disable_baseline_eval = env_disable_baseline.lower() in ("1", "true", "yes")
    if args.enable_baseline_eval:
        disable_baseline_eval = False
    if args.disable_baseline_eval:
        disable_baseline_eval = True

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
    log_step_logs = _bool_env("LOG_STEP_LOGS", False)
    if args.step_logs:
        log_step_logs = True
    if args.no_step_logs:
        log_step_logs = False

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
        # Keep existing default behavior unless caller explicitly requests exact run-dir.
        if (not args.exact_run_dir) and (not _has_ts(base)):
            run_dir = f"{run_dir_env}_{start_ts}"
            base = os.path.basename(run_dir)
        run_id = run_id_env or base
        if (not args.exact_run_dir) and (not _has_ts(run_id)):
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
    audit_results_dir = os.path.join(run_dir, "audit_results")
    _ensure_dir(run_dir)
    _ensure_dir(logs_dir)
    _ensure_dir(plots_dir)
    _ensure_dir(models_dir)
    _ensure_dir(audit_results_dir)
    os.environ["RUN_ID"] = run_id
    os.environ["RUN_DIR"] = run_dir
    os.environ["MAX_EPISODES"] = str(TC.MAX_EPISODES)
    os.environ["SEED"] = str(Cfg.SEED)
    os.environ["AUDIT_RESULTS_DIR"] = audit_results_dir
    os.environ["AUDIT_T_EST_REAL_PATH"] = os.path.join(audit_results_dir, "t_est_real_records.jsonl")

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

    # 固定算法定义：PS-MAPPO-CTDE-Hybrid（论文版）
    # 约束为严格CTDE（actor局部观测，critic读取集中摘要），不依赖可切换模式。
    TC.USE_SIMPLIFIED_CRITIC = False
    TC.COMMWAIT_DIRECT_TO_CRITIC = True
    TC.USE_LOGIT_BIAS = True
    TC.USE_FIXED_POWER = False
    if bool(getattr(TC, "USE_SIMPLIFIED_CRITIC", False)):
        raise ValueError(
            "PS-MAPPO-CTDE-Hybrid requires USE_SIMPLIFIED_CRITIC=False (centralized critic)."
        )
    if not bool(getattr(TC, "COMMWAIT_DIRECT_TO_CRITIC", False)):
        raise ValueError(
            "PS-MAPPO-CTDE-Hybrid requires COMMWAIT_DIRECT_TO_CRITIC=True."
        )
    if int(getattr(TC, "CTDE_GLOBAL_DIM", 0)) <= 0:
        raise ValueError(
            "PS-MAPPO-CTDE-Hybrid requires CTDE_GLOBAL_DIM > 0."
        )
    _print_unified_reward_scale_check()

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
        "GIT_COMMIT": _get_git_commit(),
    }
    snapshot = {
        "system_config": config_dict,
        "train_config": train_config_dict,
        "env": env_snapshot,
    }
    snapshot["config_hash"] = _stable_config_hash({
        "system_config": snapshot["system_config"],
        "train_config": snapshot["train_config"],
    })
    config_snapshot_path = os.path.join(logs_dir, "config_snapshot.json")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=True, indent=2, default=_json_default)
    run_meta = {
        "run_id": run_id,
        "run_dir": run_dir,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": env_snapshot["GIT_COMMIT"],
        "config_hash": snapshot["config_hash"],
    }
    run_meta_path = os.path.join(run_dir, "run_meta.json")
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=True, indent=2, default=_json_default)

    reward_scheme = Cfg.REWARD_SCHEME
    if reward_scheme == "PBRS_KP_V2":
        energy_lambda_effective = {
            "name": "ENERGY_LAMBDA",
            "value": float(getattr(Cfg, "ENERGY_LAMBDA", 0.0)),
        }
    elif reward_scheme == "PBRS_KP":
        energy_lambda_effective = {
            "name": "ENERGY_LAMBDA_PBRS",
            "value": float(getattr(Cfg, "ENERGY_LAMBDA_PBRS", 0.0)),
        }
    else:
        energy_lambda_effective = {
            "name": "DELTA_CFT_ENERGY_WEIGHT",
            "value": float(getattr(Cfg, "DELTA_CFT_ENERGY_WEIGHT", 0.0)),
        }

    config_dump = {
        "reward_scheme": reward_scheme,
        "reward_params": {
            "REWARD_ALPHA": Cfg.REWARD_ALPHA,
            "REWARD_BETA": Cfg.REWARD_BETA,
            "REWARD_GAMMA": Cfg.REWARD_GAMMA,
            "T_REF": Cfg.T_REF,
            "PHI_CLIP": Cfg.PHI_CLIP,
            "SHAPE_CLIP": Cfg.SHAPE_CLIP,
            "R_CLIP": Cfg.R_CLIP,
            "LAT_ALPHA": getattr(Cfg, "LAT_ALPHA", None),
        },
        "timeout_params": {
            "TIMEOUT_PENALTY_WEIGHT": getattr(Cfg, "TIMEOUT_PENALTY_WEIGHT", None),
            "TIMEOUT_STEEPNESS": getattr(Cfg, "TIMEOUT_STEEPNESS", None),
            "TIMEOUT_L1": getattr(Cfg, "TIMEOUT_L1", None),
            "TIMEOUT_L2": getattr(Cfg, "TIMEOUT_L2", None),
            "TIMEOUT_O0": getattr(Cfg, "TIMEOUT_O0", None),
            "TIMEOUT_K": getattr(Cfg, "TIMEOUT_K", None),
        },
        "energy_power_params": {
            "ENERGY_LAMBDA_PBRS": getattr(Cfg, "ENERGY_LAMBDA_PBRS", None),
            "ENERGY_LAMBDA": getattr(Cfg, "ENERGY_LAMBDA", None),
            "POWER_LAMBDA": getattr(Cfg, "POWER_LAMBDA", None),
            "E_REF": getattr(Cfg, "E_REF", None),
            "E_CLIP": getattr(Cfg, "E_CLIP", None),
            "DELTA_CFT_ENERGY_WEIGHT": getattr(Cfg, "DELTA_CFT_ENERGY_WEIGHT", None),
        },
        "energy_lambda_raw_fields": {
            "ENERGY_LAMBDA_PBRS": getattr(Cfg, "ENERGY_LAMBDA_PBRS", None),
            "ENERGY_LAMBDA": getattr(Cfg, "ENERGY_LAMBDA", None),
            "POWER_LAMBDA": getattr(Cfg, "POWER_LAMBDA", None),
            "E_REF": getattr(Cfg, "E_REF", None),
            "E_CLIP": getattr(Cfg, "E_CLIP", None),
            "DELTA_CFT_ENERGY_WEIGHT": getattr(Cfg, "DELTA_CFT_ENERGY_WEIGHT", None),
        },
        "energy_lambda_effective": energy_lambda_effective,
        "ppo_params": {
            "TARGET_KL": getattr(TC, "TARGET_KL", None),
            "MAX_GRAD_NORM": getattr(TC, "MAX_GRAD_NORM", None),
            "CLIP_PARAM": getattr(TC, "CLIP_PARAM", None),
            "LR_ACTOR": getattr(TC, "LR_ACTOR", None),
            "LR_CRITIC": getattr(TC, "LR_CRITIC", None),
        },
        "run": env_snapshot,
        "git_commit": env_snapshot["GIT_COMMIT"],
        "config_hash": snapshot["config_hash"],
    }
    config_dump_path = os.path.join(run_dir, "config_dump.json")
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(config_dump, f, ensure_ascii=True, indent=2, default=_json_default)
    print(f"[ConfigDump] saved: {config_dump_path}", flush=True)
    print(f"[ConfigSnapshot] saved: {config_snapshot_path}", flush=True)
    print(f"[RunMeta] saved: {run_meta_path}", flush=True)
    _print_startup_summary(
        run_id=run_id,
        run_dir=run_dir,
        logs_dir=logs_dir,
        plots_dir=plots_dir,
        models_dir=models_dir,
        device=device,
        baseline_eval_enabled=not disable_baseline_eval,
        log_step_metrics=log_step_metrics,
        log_step_logs=log_step_logs,
    )

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
    lagrange = _LagrangianController()

    best_reward = -float('inf')
    best_success_rate = 0.0  # 用于保存最佳模型
    recent_success_rates = deque(maxlen=50)  # 最近50轮的成功率
    best_success_episode = 0
    
    # Baseline策略列表
    baseline_policies = list(BASELINE_POLICIES)
    
    # 存储baseline的episode级指标（用于绘图）
    baseline_history = {policy: [] for policy in baseline_policies}
    
    # =========================================================================
    # training_stats.csv (用于plot_results.py)
    # =========================================================================
    training_stats_csv = os.path.join(logs_dir, "training_stats.csv")
    training_stats_header_written = os.path.exists(training_stats_csv) and os.path.getsize(training_stats_csv) > 0
    training_stats_fields = list(TRAINING_STATS_FIELDS)
    
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
        # UNIFIED component/audit (dominance check)
        "r_time",
        "r_interf",
        "r_risk",
        "r_illegal",
        "r_pbrs",
        "r_term",
        "r_step",
        "r_total",
        "abs_ratio_r_time",
        "abs_ratio_r_energy",
        "abs_ratio_r_interf",
        "abs_ratio_r_risk",
        "abs_ratio_r_illegal",
        "abs_ratio_r_pbrs",
        "abs_ratio_r_term",
        # Interference / trust oracle
        "I_total_mean",
        "I_total_p50",
        "I_total_p95",
        "I_caused_mean",
        "I_caused_p95",
        "rho_selected_mean",
        "rho_selected_p10",
        "uncertainty_selected_mean",
        "uncertainty_selected_p90",
        "rho_selected_p50",
        "rho_selected_p95",
        "rho_selected_lt_0p6_rate",
        "rho_selected_lt_0p7_rate",
        "risk_penalty_mean",
        # Chain / Trust
        "chain_tx_total",
        "chain_p95_mean",
        "chain_pfail_mean",
        "chain_risk_cost_total",
        "trust_attempts",
        "trust_failures",
        "trust_failure_rate",
        "trust_retry_count",
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
        "top_illegal_reason",
        "top_illegal_reason_count",
        "no_task_rate",
        "on_task_rate",
        "has_task_available_rate",
        "unified_illegal_trigger_rate",
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
        # PBRS_KP_V2 diagnostics
        "t_L",
        "t_R",
        "t_V",
        "t_a",
        "t_alt",
        "A_t",
        "r_lat",
        "cp_rem",
        "f_max",
        "d_cp_lb",
        "rate_best",
        "comm_lb",
        "queue_lb",
        "lb",
        "phi",
        "delta_phi",
        "r_shape",
        "overtime_ratio",
        "r_timeout",
        "e_tx",
        "r_energy",
        "r_power",
        "avg_power",
        "avg_rsu_queue",
        "rsu_queue_p95",
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
    log_header_every = max(20, int(TC.LOG_INTERVAL) if int(TC.LOG_INTERVAL) > 0 else 20)
    roll_reward = deque(maxlen=20)
    roll_task_sr = deque(maxlen=20)
    roll_miss = deque(maxlen=20)
    roll_tl = deque(maxlen=20)
    roll_v2v = deque(maxlen=20)

    training_state = {
        "current_episode": 0,
        "current_step": -1,
        "max_episodes": int(hyperparams["max_episodes"]),
        "run_id": run_id,
        "run_dir": run_dir,
        "completed": False,
    }
    prev_excepthook = sys.excepthook

    def _train_excepthook(exc_type, exc, tb):
        should_log = (
            not training_state["completed"]
            and training_state["current_episode"] < training_state["max_episodes"]
        )
        if should_log:
            error_log_path = os.path.join(run_dir, "train_error.log")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] Abnormal termination\n")
                    f.write(f"run_id={training_state['run_id']}\n")
                    f.write(f"run_dir={training_state['run_dir']}\n")
                    f.write(f"episode={training_state['current_episode']}\n")
                    f.write(f"step={training_state['current_step']}\n")
                    f.write(f"exception={exc_type.__name__}: {exc}\n")
                    f.write("traceback:\n")
                    f.write("".join(traceback.format_exception(exc_type, exc, tb)))
                    f.write("\n")
            except Exception:
                pass
        prev_excepthook(exc_type, exc, tb)

    sys.excepthook = _train_excepthook

    # V2V 探索 bias 全局步计数器
    _global_train_steps = 0

    for episode in range(1, hyperparams['max_episodes'] + 1):
        training_state["current_episode"] = episode
        training_state["current_step"] = -1

        # 学习率衰减
        if TC.USE_LR_DECAY and episode > 0 and episode % TC.LR_DECAY_STEPS == 0:
            agent.decay_lr()

        # 重置环境
        obs_list, _ = env.reset()
        last_step_info = {}
        _attach_global_state(obs_list, _build_ctde_global_state(env, obs_list, last_step_info))

        ep_reward = 0
        ep_start_time = time.time()
        step_logs_buffer = [] if log_step_logs else None
        ep_step_rewards = []
        step_metrics_rows = []
        rsu_queue_series = []

        # 统计容器
        stats = {
            "power_sum": 0.0,
            "power_values": [],
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
        ep_cost_sum = {"energy": 0.0, "interf": 0.0, "risk": 0.0}
        ep_cost_steps = 0

        # Rollout循环
        for step in range(hyperparams['max_steps_per_ep']):
            training_state["current_step"] = step
            _attach_global_state(obs_list, _build_ctde_global_state(env, obs_list, last_step_info))
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
            train_rewards, step_cost = lagrange.penalize_step(rewards, actions, obs_list, info)
            for _k in ep_cost_sum.keys():
                ep_cost_sum[_k] += float(step_cost.get(_k, 0.0))
            ep_cost_steps += 1

            # 统计
            stats["agent_rewards_sum"] += sum(train_rewards)
            stats["agent_rewards_count"] += len(train_rewards)
            active_mask = info.get("active_agent_mask")
            if not active_mask or len(active_mask) != len(train_rewards):
                active_mask = [1] * len(train_rewards)
            stats["active_sum"] += float(np.sum(active_mask))
            stats["active_total"] += float(len(active_mask))
            # 追踪每个Agent的累计奖励
            for agent_idx, r in enumerate(train_rewards):
                if agent_idx not in stats["agent_rewards_per_veh"]:
                    stats["agent_rewards_per_veh"][agent_idx] = 0.0
                stats["agent_rewards_per_veh"][agent_idx] += r

            # [修复] 存入Buffer时分离terminated和truncated
            buffer.add(
                obs_list,
                actions,
                train_rewards,
                values,
                log_probs,
                done,
                terminated=terminated,
                truncated=truncated,
                active_masks=active_mask,
            )

            # 过程统计
            num_agents = len(train_rewards) if len(train_rewards) > 0 else 1
            step_r = sum(train_rewards) / num_agents
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
                p = float(act.get("power", 0.0))
                stats['power_sum'] += p
                if int(tgt) != 0:
                    # Only count tx power for remote decisions; local action has no transmitter.
                    stats["power_values"].append(p)
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
                rsu_queue_series.append(rsu_queue_len)
            else:
                rsu_queue_series.append(0.0)

            # 记录详细日志
            if log_step_logs:
                for i, act in enumerate(actions):
                    step_logs_buffer.append({
                        "episode": episode, "step": step, "veh_id": i,
                        "target": act['target'],
                        "power": f"{act['power']:.3f}",
                        "reward": f"{train_rewards[i]:.3f}",
                        "q_len": env.vehicles[i].task_queue_len
                    })

            obs_list = next_obs_list
            last_step_info = info
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
        power_ratio_mean = float(np.mean(stats["power_values"])) if stats["power_values"] else 0.0
        power_ratio_p95 = float(np.percentile(stats["power_values"], 95)) if stats["power_values"] else 0.0
        avg_veh_queue = stats['queue_len_sum'] / total_decisions
        avg_rsu_queue = stats['rsu_queue_sum'] / total_steps
        rsu_queue_p95 = float(np.percentile(rsu_queue_series, 95)) if rsu_queue_series else 0.0

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
        top_illegal_reason = env_stats.get("top_illegal_reason") if env_stats else ""
        top_illegal_reason_count = env_stats.get("top_illegal_reason_count") if env_stats else 0
        no_task_rate = env_stats.get("no_task_rate") if env_stats else None
        on_task_rate = env_stats.get("on_task_rate") if env_stats else None
        has_task_available_rate = env_stats.get("has_task_available_rate") if env_stats else None
        unified_illegal_trigger_rate = env_stats.get("unified_illegal_trigger_rate") if env_stats else None
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
        # Prefer action-derived power stats (more reliable across reward schemes / env log settings).
        if power_ratio_mean is None:
            power_ratio_mean = env_metrics.get("power_ratio.mean")
        if power_ratio_p95 is None:
            power_ratio_p95 = env_metrics.get("power_ratio.p95")
        deadline_gamma = env_stats.get("deadline_gamma_mean") if env_stats else None
        deadline_seconds = env_stats.get("deadline_seconds_mean") if env_stats else None
        t_L = env_metrics.get("t_L.mean")
        t_R = env_metrics.get("t_R.mean")
        t_V = env_metrics.get("t_V.mean")
        t_a = env_metrics.get("t_a.mean")
        t_alt = env_metrics.get("t_alt.mean")
        A_t = env_metrics.get("A_t.mean")
        r_lat = env_metrics.get("r_lat.mean")
        cp_rem = env_metrics.get("cp_rem.mean")
        f_max = env_metrics.get("f_max.mean")
        d_cp_lb = env_metrics.get("d_cp_lb.mean")
        rate_best = env_metrics.get("rate_best.mean")
        comm_lb = env_metrics.get("comm_lb.mean")
        queue_lb = env_metrics.get("queue_lb.mean")
        lb = env_metrics.get("lb.mean")
        phi = env_metrics.get("phi.mean")
        delta_phi = env_metrics.get("delta_phi.mean")
        r_shape = env_metrics.get("r_shape.mean")
        overtime_ratio = env_metrics.get("overtime_ratio.mean")
        r_timeout_mean = env_metrics.get("r_timeout.mean")
        e_tx_mean = env_metrics.get("e_tx.mean")
        r_energy_mean = env_metrics.get("r_energy.mean")
        r_power_mean = env_metrics.get("r_power.mean")
        # UNIFIED component logging (means from env reward_stats)
        r_time_mean = env_metrics.get("r_time.mean")
        r_interf_mean = env_metrics.get("r_interf.mean")
        r_risk_mean = env_metrics.get("r_risk.mean")
        r_illegal_mean = env_metrics.get("r_illegal.mean")
        r_pbrs_mean = env_metrics.get("r_pbrs.mean")
        r_term_mean = env_metrics.get("r_term.mean")
        r_step_mean = env_metrics.get("r_step.mean")
        r_total_mean = env_metrics.get("reward.mean")
        # Abs means for dominance ratios
        r_time_abs_mean = env_metrics.get("r_time_abs.mean")
        r_energy_abs_mean = env_metrics.get("r_energy_abs.mean")
        r_interf_abs_mean = env_metrics.get("r_interf_abs.mean")
        r_risk_abs_mean = env_metrics.get("r_risk_abs.mean")
        r_illegal_abs_mean = env_metrics.get("r_illegal_abs.mean")
        r_pbrs_abs_mean = env_metrics.get("r_pbrs_abs.mean")
        r_term_abs_mean = env_metrics.get("r_term_abs.mean")

        # Interference / trust oracle (episode aggregates)
        I_total_mean = env_stats.get("I_total_mean") if env_stats else None
        I_total_p50 = env_stats.get("I_total_p50") if env_stats else None
        I_total_p95 = env_stats.get("I_total_p95") if env_stats else None
        I_caused_mean = env_stats.get("I_caused_mean") if env_stats else None
        I_caused_p95 = env_stats.get("I_caused_p95") if env_stats else None
        rho_selected_mean = env_stats.get("rho_selected_mean") if env_stats else None
        rho_selected_p10 = env_stats.get("rho_selected_p10") if env_stats else None
        uncertainty_selected_mean = env_stats.get("uncertainty_selected_mean") if env_stats else None
        uncertainty_selected_p90 = env_stats.get("uncertainty_selected_p90") if env_stats else None
        rho_selected_p50 = env_stats.get("rho_selected_p50") if env_stats else None
        rho_selected_p95 = env_stats.get("rho_selected_p95") if env_stats else None
        rho_selected_lt_0p6_rate = env_stats.get("rho_selected_lt_0p6_rate") if env_stats else None
        rho_selected_lt_0p7_rate = env_stats.get("rho_selected_lt_0p7_rate") if env_stats else None
        risk_penalty_mean = env_stats.get("risk_penalty_mean") if env_stats else None
        chain_tx_total = env_stats.get("chain_tx_total") if env_stats else None
        chain_p95_mean = env_stats.get("chain_p95_mean") if env_stats else None
        chain_pfail_mean = env_stats.get("chain_pfail_mean") if env_stats else None
        chain_risk_cost_total = env_stats.get("chain_risk_cost_total") if env_stats else None
        trust_attempts = env_stats.get("trust_attempts") if env_stats else None
        trust_failures = env_stats.get("trust_failures") if env_stats else None
        trust_failure_rate = env_stats.get("trust_failure_rate") if env_stats else None
        trust_retry_count = env_stats.get("trust_retry_count") if env_stats else None
        # mean_cft_rem: 优先使用env_stats，其次用env_metrics，最后fallback到deadline剩余时间
        mean_cft_rem = env_stats.get("mean_cft_rem") if env_stats else None
        if mean_cft_rem is None:
            mean_cft_rem = env_metrics.get("cft_curr_rem.mean")
        if mean_cft_rem is None and deadline_seconds is not None and episode_time_seconds is not None:
            mean_cft_rem = max(deadline_seconds - episode_time_seconds, 0.0)
        # mean_cft: 优先使用env_stats；若仅有剩余时间，则还原绝对CFT
        if mean_cft is None and mean_cft_rem is not None and episode_time_seconds is not None:
            mean_cft = mean_cft_rem + episode_time_seconds

        # Component dominance ratios (abs contribution shares)
        abs_parts = {
            "r_time": r_time_abs_mean,
            "r_energy": r_energy_abs_mean,
            "r_interf": r_interf_abs_mean,
            "r_risk": r_risk_abs_mean,
            "r_illegal": r_illegal_abs_mean,
            "r_pbrs": r_pbrs_abs_mean,
            "r_term": r_term_abs_mean,
        }
        abs_sum = 0.0
        for v in abs_parts.values():
            if v is not None and np.isfinite(v):
                abs_sum += float(v)
        if abs_sum <= 1e-12:
            abs_sum = 0.0

        def _abs_ratio(key: str) -> float:
            v = abs_parts.get(key)
            if abs_sum <= 0.0 or v is None or not np.isfinite(v):
                return 0.0
            return float(v) / abs_sum
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
        # 熵系数按全局步数线性退火：前期探索更强，后期收敛更稳。
        ent_start = float(getattr(TC, "ENTROPY_COEF_START", TC.ENTROPY_COEF))
        ent_end = float(getattr(TC, "ENTROPY_COEF_END", ent_start))
        ent_anneal_steps = int(max(getattr(TC, "ENTROPY_ANNEAL_STEPS", 0), 0))
        if ent_anneal_steps > 0:
            ent_frac = float(np.clip(_global_train_steps / max(ent_anneal_steps, 1), 0.0, 1.0))
            TC.ENTROPY_COEF = ent_start + (ent_end - ent_start) * ent_frac
        else:
            TC.ENTROPY_COEF = ent_end

        _attach_global_state(obs_list, _build_ctde_global_state(env, obs_list, last_step_info))
        last_value = agent.get_value(obs_list)
        buffer.compute_returns_and_advantages(last_value)
        update_loss = agent.update(buffer, batch_size=TC.MINI_BATCH_SIZE)
        buffer.clear()
        ep_cost_mean = {k: (ep_cost_sum[k] / max(ep_cost_steps, 1)) for k in ep_cost_sum.keys()}
        lagrange_state = lagrange.update_episode(ep_cost_mean, episode)
        update_stats = getattr(agent, "last_update_stats", {}) or {}
        policy_entropy_val = update_stats.get("policy_entropy", update_stats.get("entropy"))
        if policy_entropy_val is None:
            policy_entropy_val = 0.0
        entropy_loss_val = update_stats.get("entropy_loss")
        if entropy_loss_val is None and policy_entropy_val is not None:
            entropy_loss_val = -policy_entropy_val

        # 显存清理
        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # =====================================================================
        # Bias退火逻辑 (Bias Annealing)
        # [P18修复] 网络从TC直接读取bias值，无需额外同步
        # =====================================================================
        _global_train_steps += total_steps
        if TC.USE_LOGIT_BIAS:
            # 类别级轻先验：按线性日程从 INIT 退火到 END（通常END=0）。
            _total_plan_steps = max(int(getattr(TC, "MAX_EPISODES", 1) * getattr(TC, "MAX_STEPS", 1)), 1)
            _anneal_frac = float(np.clip(getattr(TC, "BIAS_ANNEAL_FRAC", 0.50), 0.05, 1.0))
            _anneal_steps = max(int(_total_plan_steps * _anneal_frac), 1)
            _prog = min(_global_train_steps / float(_anneal_steps), 1.0)
            _l0 = float(getattr(TC, "LOGIT_BIAS_LOCAL_INIT", 0.2))
            _l1 = float(getattr(TC, "LOGIT_BIAS_LOCAL_END", 0.0))
            _r0 = float(getattr(TC, "LOGIT_BIAS_RSU_INIT", 0.05))
            _r1 = float(getattr(TC, "LOGIT_BIAS_RSU_END", 0.0))
            _v0 = float(getattr(TC, "LOGIT_BIAS_V2V_INIT", 0.10))
            _v1 = float(getattr(TC, "LOGIT_BIAS_V2V_END", 0.0))
            # 线性插值: bias = init + prog * (end - init)
            TC.LOGIT_BIAS_LOCAL = _l0 + _prog * (_l1 - _l0)
            TC.LOGIT_BIAS_RSU = _r0 + _prog * (_r1 - _r0)
            TC._logit_bias_v2v_current = _v0 + _prog * (_v1 - _v0)
        else:
            TC.LOGIT_BIAS_LOCAL = 0.0
            TC.LOGIT_BIAS_RSU = 0.0
            TC._logit_bias_v2v_current = 0.0

        # =====================================================================
        # 控制台输出（每轮一行 + 周期诊断）
        # =====================================================================
        actor_loss = _safe_float(update_stats.get("policy_loss"))
        critic_loss = _safe_float(update_stats.get("value_loss"))
        entropy_val = _safe_float(update_stats.get("entropy"))
        approx_kl = _safe_float(update_stats.get("approx_kl"))
        clip_frac = _safe_float(update_stats.get("clip_fraction"))
        grad_norm_val = _safe_float(update_stats.get("grad_norm"))
        active_ratio_val = _safe_float(update_stats.get("active_ratio"))
        vehicle_sr = env_stats.get('vehicle_success_rate', veh_success_rate) if env_stats else veh_success_rate
        illegal_rate_display = 0.0 if illegal_action_rate is None else illegal_action_rate
        deadlock_count = env_stats.get('deadlock_vehicle_count', 0) if env_stats else 0
        deadline_misses = env_stats.get('audit_deadline_misses', 0) if env_stats else 0
        tx_created = env_stats.get('tx_tasks_created_count', 0) if env_stats else 0
        same_node_no_tx = env_stats.get('same_node_no_tx_count', 0) if env_stats else 0
        service_rate_active = env_stats.get('service_rate_when_active', 0.0) if env_stats else 0.0
        idle_fraction = env_stats.get('idle_fraction', 0.0) if env_stats else 0.0

        roll_reward.append(reward_mean)
        roll_task_sr.append(task_success_rate if task_success_rate is not None else 0.0)
        roll_miss.append(deadline_miss_rate if deadline_miss_rate is not None else 0.0)
        roll_tl.append(time_limit_rate if time_limit_rate is not None else 0.0)
        roll_v2v.append(frac_v2v if frac_v2v is not None else 0.0)

        if episode == 1 or episode % log_header_every == 1:
            print(
                "\n"
                f"{'Ep':>6} {'Wall':>7} {'R/step':>9} {'T_SR':>8} {'V_SR':>8} {'S_SR':>8} "
                f"{'L/R/V':>16} {'Lat(s)':>8} {'En':>8} {'Miss':>8} {'Ill':>8} {'Ent':>8} {'KL':>8} {'Clip':>8}",
                flush=True,
            )
            print("-" * 128, flush=True)

        deci_str = f"{_fmt_pct(frac_local, 0)}/{_fmt_pct(frac_rsu, 0)}/{_fmt_pct(frac_v2v, 0)}"
        print(
            f"{episode:6d} {duration:7.1f}s {reward_mean:9.4f} "
            f"{_fmt_pct(task_success_rate):>8} {_fmt_pct(vehicle_sr):>8} {_fmt_pct(subtask_success):>8} "
            f"{deci_str:>16} {_fmt_float(task_duration_mean, 3):>8} {_fmt_float(energy_norm_mean, 3):>8} "
            f"{_fmt_pct(deadline_miss_rate):>8} {_fmt_pct(illegal_rate_display):>8} "
            f"{_fmt_float(entropy_val, 4):>8} {_fmt_float(approx_kl, 4):>8} {_fmt_pct(clip_frac):>8}",
            flush=True,
        )

        if episode == 1 or episode % max(1, int(TC.LOG_INTERVAL)) == 0:
            ma_reward = float(np.mean(roll_reward)) if roll_reward else 0.0
            ma_sr = float(np.mean(roll_task_sr)) if roll_task_sr else 0.0
            ma_miss = float(np.mean(roll_miss)) if roll_miss else 0.0
            ma_tl = float(np.mean(roll_tl)) if roll_tl else 0.0
            ma_v2v = float(np.mean(roll_v2v)) if roll_v2v else 0.0
            print(
                f"  [MA20] reward={ma_reward:.4f} task_sr={ma_sr*100:.1f}% miss={ma_miss*100:.1f}% "
                f"time_limit={ma_tl*100:.1f}% v2v={ma_v2v*100:.1f}%",
                flush=True,
            )
            print(
                f"  [PPO ] p_loss={_fmt_float(actor_loss, 4)} v_loss={_fmt_float(critic_loss, 4)} "
                f"entropy={_fmt_float(entropy_val, 4)} grad={_fmt_float(grad_norm_val, 4)} "
                f"active={_fmt_pct(active_ratio_val)}",
                flush=True,
            )
            print(
                f"  [SIM ] avg_power(a)={_fmt_float(avg_power, 4)} rsu_q={_fmt_float(avg_rsu_queue, 3)} "
                f"tx={int(tx_created)} no_tx={int(same_node_no_tx)} deadlock={int(deadlock_count)} "
                f"svc={_fmt_float(service_rate_active / 1e9, 3)}GHz idle={_fmt_pct(idle_fraction)}",
                flush=True,
            )
            if lagrange.enabled:
                print(
                    f"  [CMDP] lambda(E/I/R)=({lagrange_state['energy']:.3f}/{lagrange_state['interf']:.3f}/{lagrange_state['risk']:.3f}) "
                    f"cost(E/I/R)=({ep_cost_mean['energy']:.3f}/{ep_cost_mean['interf']:.3f}/{ep_cost_mean['risk']:.3f})",
                    flush=True,
                )

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
            # UNIFIED component/audit (dominance check)
            "r_time": float(r_time_mean) if r_time_mean is not None and np.isfinite(r_time_mean) else 0.0,
            "r_interf": float(r_interf_mean) if r_interf_mean is not None and np.isfinite(r_interf_mean) else 0.0,
            "r_risk": float(r_risk_mean) if r_risk_mean is not None and np.isfinite(r_risk_mean) else 0.0,
            "r_illegal": float(r_illegal_mean) if r_illegal_mean is not None and np.isfinite(r_illegal_mean) else 0.0,
            "r_pbrs": float(r_pbrs_mean) if r_pbrs_mean is not None and np.isfinite(r_pbrs_mean) else 0.0,
            "r_term": float(r_term_mean) if r_term_mean is not None and np.isfinite(r_term_mean) else 0.0,
            "r_step": float(r_step_mean) if r_step_mean is not None and np.isfinite(r_step_mean) else 0.0,
            "r_total": float(r_total_mean) if r_total_mean is not None and np.isfinite(r_total_mean) else reward_mean,
            "abs_ratio_r_time": _abs_ratio("r_time"),
            "abs_ratio_r_energy": _abs_ratio("r_energy"),
            "abs_ratio_r_interf": _abs_ratio("r_interf"),
            "abs_ratio_r_risk": _abs_ratio("r_risk"),
            "abs_ratio_r_illegal": _abs_ratio("r_illegal"),
            "abs_ratio_r_pbrs": _abs_ratio("r_pbrs"),
            "abs_ratio_r_term": _abs_ratio("r_term"),
            # Interference / trust oracle (episode aggregates)
            "I_total_mean": float(I_total_mean) if I_total_mean is not None and np.isfinite(I_total_mean) else 0.0,
            "I_total_p50": float(I_total_p50) if I_total_p50 is not None and np.isfinite(I_total_p50) else 0.0,
            "I_total_p95": float(I_total_p95) if I_total_p95 is not None and np.isfinite(I_total_p95) else 0.0,
            "I_caused_mean": float(I_caused_mean) if I_caused_mean is not None and np.isfinite(I_caused_mean) else 0.0,
            "I_caused_p95": float(I_caused_p95) if I_caused_p95 is not None and np.isfinite(I_caused_p95) else 0.0,
            "rho_selected_mean": float(rho_selected_mean) if rho_selected_mean is not None and np.isfinite(rho_selected_mean) else 0.0,
            "rho_selected_p10": float(rho_selected_p10) if rho_selected_p10 is not None and np.isfinite(rho_selected_p10) else 0.0,
            "uncertainty_selected_mean": float(uncertainty_selected_mean) if uncertainty_selected_mean is not None and np.isfinite(uncertainty_selected_mean) else 0.0,
            "uncertainty_selected_p90": float(uncertainty_selected_p90) if uncertainty_selected_p90 is not None and np.isfinite(uncertainty_selected_p90) else 0.0,
            "rho_selected_p50": float(rho_selected_p50) if rho_selected_p50 is not None and np.isfinite(rho_selected_p50) else 0.0,
            "rho_selected_p95": float(rho_selected_p95) if rho_selected_p95 is not None and np.isfinite(rho_selected_p95) else 0.0,
            "rho_selected_lt_0p6_rate": float(rho_selected_lt_0p6_rate) if rho_selected_lt_0p6_rate is not None and np.isfinite(rho_selected_lt_0p6_rate) else 0.0,
            "rho_selected_lt_0p7_rate": float(rho_selected_lt_0p7_rate) if rho_selected_lt_0p7_rate is not None and np.isfinite(rho_selected_lt_0p7_rate) else 0.0,
            "risk_penalty_mean": float(risk_penalty_mean) if risk_penalty_mean is not None and np.isfinite(risk_penalty_mean) else 0.0,
            "chain_tx_total": int(chain_tx_total) if chain_tx_total is not None else 0,
            "chain_p95_mean": float(chain_p95_mean) if chain_p95_mean is not None and np.isfinite(chain_p95_mean) else 0.0,
            "chain_pfail_mean": float(chain_pfail_mean) if chain_pfail_mean is not None and np.isfinite(chain_pfail_mean) else 0.0,
            "chain_risk_cost_total": float(chain_risk_cost_total) if chain_risk_cost_total is not None and np.isfinite(chain_risk_cost_total) else 0.0,
            "trust_attempts": int(trust_attempts) if trust_attempts is not None else 0,
            "trust_failures": int(trust_failures) if trust_failures is not None else 0,
            "trust_failure_rate": float(trust_failure_rate) if trust_failure_rate is not None and np.isfinite(trust_failure_rate) else 0.0,
            "trust_retry_count": int(trust_retry_count) if trust_retry_count is not None else 0,
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
            "top_illegal_reason": str(top_illegal_reason or ""),
            "top_illegal_reason_count": int(top_illegal_reason_count or 0),
            "no_task_rate": no_task_rate if no_task_rate is not None else 0.0,
            "on_task_rate": on_task_rate if on_task_rate is not None else 0.0,
            "has_task_available_rate": has_task_available_rate if has_task_available_rate is not None else 0.0,
            "unified_illegal_trigger_rate": unified_illegal_trigger_rate if unified_illegal_trigger_rate is not None else 0.0,
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
            # PBRS_KP_V2 diagnostics
            "t_L": t_L if t_L is not None else 0.0,
            "t_R": t_R if t_R is not None else 0.0,
            "t_V": t_V if t_V is not None else 0.0,
            "t_a": t_a if t_a is not None else 0.0,
            "t_alt": t_alt if t_alt is not None else 0.0,
            "A_t": A_t if A_t is not None else 0.0,
            "r_lat": r_lat if r_lat is not None else 0.0,
            "cp_rem": cp_rem if cp_rem is not None else 0.0,
            "f_max": f_max if f_max is not None else 0.0,
            "d_cp_lb": d_cp_lb if d_cp_lb is not None else 0.0,
            "rate_best": rate_best if rate_best is not None else 0.0,
            "comm_lb": comm_lb if comm_lb is not None else 0.0,
            "queue_lb": queue_lb if queue_lb is not None else 0.0,
            "lb": lb if lb is not None else 0.0,
            "phi": phi if phi is not None else 0.0,
            "delta_phi": delta_phi if delta_phi is not None else 0.0,
            "r_shape": r_shape if r_shape is not None else 0.0,
            "overtime_ratio": overtime_ratio if overtime_ratio is not None else 0.0,
            "r_timeout": r_timeout_mean if r_timeout_mean is not None else 0.0,
            "e_tx": e_tx_mean if e_tx_mean is not None else 0.0,
            "r_energy": r_energy_mean if r_energy_mean is not None else 0.0,
            "r_power": r_power_mean if r_power_mean is not None else 0.0,
            "avg_power": avg_power,
            "avg_rsu_queue": avg_rsu_queue,
            "rsu_queue_p95": rsu_queue_p95,
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
        current_lr = None
        if hasattr(agent, "optimizer") and getattr(agent.optimizer, "param_groups", None):
            current_lr = agent.optimizer.param_groups[0].get("lr")

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
            "reward_abs_mean": reward_abs_mean if reward_abs_mean is not None else 0.0,
            # 成功率指标（0-1范围）
            "vehicle_sr": veh_success_rate,  # V_SR
            "task_sr": task_success_rate,    # T_SR
            "subtask_sr": subtask_success,   # S_SR
            # 物理性能指标
            "task_duration_mean": task_duration_mean if task_duration_mean is not None else 0.0,
            "task_duration_p95": task_duration_p95 if task_duration_p95 is not None else 0.0,
            "completed_tasks": completed_tasks_count if completed_tasks_count is not None else 0,
            "mean_cft_est": mean_cft_est if mean_cft_est is not None else 0.0,
            "episode_time_seconds": episode_time_seconds if episode_time_seconds is not None else 0.0,
            "energy_mean": energy_norm_mean if energy_norm_mean is not None else 0.0,
            "energy_p95": energy_norm_p95 if energy_norm_p95 is not None else 0.0,
            "t_tx_mean": t_tx_mean if t_tx_mean is not None else 0.0,
            "dT_eff_mean": dT_eff_mean if dT_eff_mean is not None else 0.0,
            "deadline_misses": deadline_misses,  # D_Miss
            "deadline_miss_rate": deadline_miss_rate if deadline_miss_rate is not None else 0.0,
            # 卸载决策分布（0-1范围）
            "ratio_local": frac_local,  # Local
            "ratio_rsu": frac_rsu,      # RSU
            "ratio_v2v": frac_v2v,      # V2V
            "decision_frac_local": frac_local,
            "decision_frac_rsu": frac_rsu,
            "decision_frac_v2v": frac_v2v,
            # 系统负载与资源
            "avg_power": avg_power if avg_power is not None else 0.0,
            "avg_rsu_queue": avg_rsu_queue if avg_rsu_queue is not None else 0.0,
            "rsu_queue_p95": rsu_queue_p95 if rsu_queue_p95 is not None else 0.0,
            "power_ratio_mean": power_ratio_mean if power_ratio_mean is not None else 0.0,
            "power_ratio_p95": power_ratio_p95 if power_ratio_p95 is not None else 0.0,
            "I_total_p50": I_total_p50 if I_total_p50 is not None else 0.0,
            "I_total_p95": I_total_p95 if I_total_p95 is not None else 0.0,
            "I_caused_mean": I_caused_mean if I_caused_mean is not None else 0.0,
            "I_caused_p95": I_caused_p95 if I_caused_p95 is not None else 0.0,
            "trust_failure_rate": trust_failure_rate if trust_failure_rate is not None else 0.0,
            "rho_selected_p10": rho_selected_p10 if rho_selected_p10 is not None else 0.0,
            "uncertainty_selected_p90": uncertainty_selected_p90 if uncertainty_selected_p90 is not None else 0.0,
            # 服务指标
            "tx_created": tx_created,             # TX
            "same_node_no_tx": same_node_no_tx,   # NoTX
            "service_rate_ghz": service_rate_active / 1e9,  # SvcRate (GHz)
            "idle_fraction": idle_fraction,       # Idle
            # 约束与安全
            "time_limit_rate": time_limit_rate if time_limit_rate is not None else 0.0,
            "illegal_action_rate": illegal_action_rate if illegal_action_rate is not None else 0.0,
            "no_task_rate": no_task_rate if no_task_rate is not None else 0.0,
            "on_task_rate": on_task_rate if on_task_rate is not None else 0.0,
            "has_task_available_rate": has_task_available_rate if has_task_available_rate is not None else 0.0,
            "unified_illegal_trigger_rate": unified_illegal_trigger_rate if unified_illegal_trigger_rate is not None else 0.0,
            "hard_trigger_rate": hard_trigger_rate if hard_trigger_rate is not None else 0.0,
            # 训练诊断指标
            "actor_loss": update_stats.get("policy_loss"),
            "critic_loss": update_stats.get("value_loss"),
            "entropy": update_stats.get("policy_entropy", update_stats.get("entropy")),
            "approx_kl": update_stats.get("approx_kl"),
            "clip_frac": update_stats.get("clip_fraction"),
            "grad_norm": update_stats.get("grad_norm"),
            "active_ratio": update_stats.get("active_ratio"),
            "value_clip_fraction": update_stats.get("value_clip_fraction"),
            "skipped_update_count": update_stats.get("skipped_update_count"),
            "early_stop": update_stats.get("early_stop"),
            "lr": current_lr,
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
        if log_step_logs and step_logs_buffer:
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
            baseline_stats_fields = list(BASELINE_STATS_FIELDS)
            baseline_header_written = os.path.exists(baseline_stats_csv) and os.path.getsize(baseline_stats_csv) > 0

            for policy_name in baseline_policies:
                baseline_metrics = evaluate_single_baseline_episode(
                    env,
                    policy_name,
                    episode_seed=int(getattr(Cfg, "SEED", 0)) + int(episode),
                )
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
                    "v2v_subtask_sr": baseline_metrics['v2v_subtask_success_rate'],
                    "ratio_local": baseline_metrics['decision_frac_local'],
                    "ratio_rsu": baseline_metrics['decision_frac_rsu'],
                    "ratio_v2v": baseline_metrics['decision_frac_v2v'],
                    "decision_frac_local": baseline_metrics['decision_frac_local'],
                    "decision_frac_rsu": baseline_metrics['decision_frac_rsu'],
                    "decision_frac_v2v": baseline_metrics['decision_frac_v2v'],
                    "avg_power": baseline_metrics['avg_power'],
                    "power_ratio_mean": baseline_metrics.get('power_ratio_mean'),
                    "power_ratio_p95": baseline_metrics.get('power_ratio_p95'),
                    "episode_time_seconds": baseline_metrics.get('episode_time_seconds'),
                    "mean_cft_est": baseline_metrics.get('mean_cft_est'),
                    "mean_cft_completed": baseline_metrics.get('mean_cft_completed'),
                    "task_duration_mean": baseline_metrics.get('task_duration_mean'),
                    "task_duration_p95": baseline_metrics.get('task_duration_p95'),
                    "deadline_miss_rate": baseline_metrics.get('deadline_miss_rate'),
                    "time_limit_rate": baseline_metrics.get('time_limit_rate'),
                    "illegal_action_rate": baseline_metrics.get('illegal_action_rate'),
                    "no_task_rate": baseline_metrics.get('no_task_rate'),
                    "on_task_rate": baseline_metrics.get('on_task_rate'),
                    "has_task_available_rate": baseline_metrics.get('has_task_available_rate'),
                    "unified_illegal_trigger_rate": baseline_metrics.get('unified_illegal_trigger_rate'),
                    "I_total_mean": baseline_metrics.get('I_total_mean'),
                    "I_total_p50": baseline_metrics.get('I_total_p50'),
                    "I_total_p95": baseline_metrics.get('I_total_p95'),
                    "I_caused_mean": baseline_metrics.get('I_caused_mean'),
                    "I_caused_p95": baseline_metrics.get('I_caused_p95'),
                    "rho_selected_mean": baseline_metrics.get('rho_selected_mean'),
                    "rho_selected_p10": baseline_metrics.get('rho_selected_p10'),
                    "uncertainty_selected_mean": baseline_metrics.get('uncertainty_selected_mean'),
                    "uncertainty_selected_p90": baseline_metrics.get('uncertainty_selected_p90'),
                    "risk_penalty_mean": baseline_metrics.get('risk_penalty_mean'),
                    "rho_selected_p50": baseline_metrics.get('rho_selected_p50'),
                    "rho_selected_p95": baseline_metrics.get('rho_selected_p95'),
                    "rho_selected_lt_0p6_rate": baseline_metrics.get('rho_selected_lt_0p6_rate'),
                    "rho_selected_lt_0p7_rate": baseline_metrics.get('rho_selected_lt_0p7_rate'),
                    "chain_tx_total": baseline_metrics.get('chain_tx_total'),
                    "chain_p95_mean": baseline_metrics.get('chain_p95_mean'),
                    "chain_pfail_mean": baseline_metrics.get('chain_pfail_mean'),
                    "chain_risk_cost_total": baseline_metrics.get('chain_risk_cost_total'),
                    "trust_attempts": baseline_metrics.get('trust_attempts'),
                    "trust_failures": baseline_metrics.get('trust_failures'),
                    "trust_failure_rate": baseline_metrics.get('trust_failure_rate'),
                    "trust_retry_count": baseline_metrics.get('trust_retry_count'),
                    "avg_queue_len": baseline_metrics['avg_queue_len'],
                    "avg_rsu_queue": baseline_metrics.get('avg_rsu_queue', 0.0),
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
        # 模型保存策略：仅保留 best_model / last_model 两个文件
        # =====================================================================
        # 维护 reward 指标用于日志摘要
        if ep_reward > best_reward:
            best_reward = ep_reward

        # 保存基于 task_success_rate 的最佳模型（50-ep滑动平均）
        avg_recent_sr = np.mean(recent_success_rates) if recent_success_rates else 0.0
        if episode == 1 or avg_recent_sr > best_success_rate:
            best_success_rate = avg_recent_sr
            agent.save(os.path.join(recorder.model_dir, "best_model.pth"))
            best_success_episode = int(episode)
            if episode == 1 or episode % TC.LOG_INTERVAL == 0:
                print(f"  → Best model saved: Success Rate = {best_success_rate:.3f} (50-ep avg)")

        # 周期性刷新 last_model（覆盖写，便于中途中断恢复）
        if episode % TC.SAVE_INTERVAL == 0:
            agent.save(os.path.join(recorder.model_dir, "last_model.pth"))

    training_state["completed"] = True
    sys.excepthook = prev_excepthook
    print(
        f"\n[Train] completed: episodes={TC.MAX_EPISODES} "
        f"best_reward={best_reward:.4f} best_success_rate(50ep)={best_success_rate:.4f}",
        flush=True,
    )
    # 训练结束时强制保存最后模型（覆盖 last_model）
    try:
        agent.save(os.path.join(recorder.model_dir, "last_model.pth"))
    except Exception as e:
        print(f"[Train] Warning: failed to save last_model.pth: {e}", flush=True)
    print(
        f"[Train] key outputs: {metrics_csv_path}, {training_stats_csv}, "
        f"{os.path.join(recorder.model_dir, 'best_model.pth')}, "
        f"{os.path.join(recorder.model_dir, 'last_model.pth')}",
        flush=True,
    )

    # =========================================================================
    # 训练结束：自动绘图 (Auto Plotting)
    # =========================================================================
    if not disable_auto_plot:
        print("\n[Auto Plot] Generating training plots...", flush=True)
        baseline_stats_csv = os.path.join(logs_dir, "baseline_stats.csv")
        plot_jobs: List[Dict[str, Any]] = []

        # 1) DataRecorder 基础训练图
        try:
            t0 = time.time()
            recorder.plot_training_stats(training_stats_csv, baseline_stats_csv)
            plot_jobs.append({"job": "DataRecorder.plot_training_stats", "ok": True, "seconds": time.time() - t0, "tail": []})
        except Exception as e:
            plot_jobs.append({"job": "DataRecorder.plot_training_stats", "ok": False, "seconds": 0.0, "tail": [str(e)]})
            print(f"[Auto Plot] DataRecorder.plot_training_stats failed: {e}", flush=True)

        # 2) plot_results.py（training_stats 主图）
        plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "plot_results.py")
        if os.path.exists(plot_script) and os.path.exists(training_stats_csv):
            plot_jobs.append(
                _run_plot_job(
                    "plot_results.py",
                    [sys.executable, plot_script, "--log-file", training_stats_csv, "--output-dir", plots_dir],
                )
            )
        else:
            plot_jobs.append({"job": "plot_results.py", "ok": False, "seconds": 0.0, "tail": ["missing script or training_stats.csv"]})

        # 3) generate_all_plots.py（episode_log 扩展图）
        generate_plots_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "generate_all_plots.py")
        if os.path.exists(generate_plots_script):
            plot_jobs.append(
                _run_plot_job(
                    "generate_all_plots.py",
                    [sys.executable, generate_plots_script, "--run-dir", run_dir],
                )
            )
        else:
            plot_jobs.append({"job": "generate_all_plots.py", "ok": False, "seconds": 0.0, "tail": ["script not found"]})

        # 4) 可选：MAPPO vs baseline 对比图
        compare_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "plot_mappo_vs_baselines.py")
        if os.path.exists(compare_script) and os.path.exists(baseline_stats_csv) and os.path.getsize(baseline_stats_csv) > 0:
            plot_jobs.append(
                _run_plot_job(
                    "plot_mappo_vs_baselines.py",
                    [sys.executable, compare_script, "--run-dir", run_dir, "--output-dir", plots_dir],
                )
            )
        else:
            plot_jobs.append(
                {
                    "job": "plot_mappo_vs_baselines.py",
                    "ok": True,
                    "skipped": True,
                    "seconds": 0.0,
                    "tail": ["baseline_stats.csv missing/empty or script not found"],
                }
            )

        # 5) DataRecorder 兼容图（episode_log）
        try:
            t0 = time.time()
            recorder.auto_plot(baseline_history=baseline_history)
            plot_jobs.append({"job": "DataRecorder.auto_plot", "ok": True, "seconds": time.time() - t0, "tail": []})
        except Exception as e:
            plot_jobs.append({"job": "DataRecorder.auto_plot", "ok": False, "seconds": 0.0, "tail": [str(e)]})
            print(f"[Auto Plot] DataRecorder.auto_plot failed: {e}", flush=True)

        manifest_path = _write_plot_manifest(plots_dir, plot_jobs)
        figures = _collect_plot_manifest(plots_dir)
        ok_jobs = sum(1 for job in plot_jobs if job.get("ok") and not job.get("skipped"))
        skipped_jobs = sum(1 for job in plot_jobs if job.get("skipped"))
        failed_jobs = len(plot_jobs) - ok_jobs - skipped_jobs
        print(f"[Auto Plot] jobs: ok={ok_jobs} skipped={skipped_jobs} failed={failed_jobs}", flush=True)
        print(f"[Auto Plot] generated figures: {len(figures)}", flush=True)
        if figures:
            preview = ", ".join([fig["file"] for fig in figures[:8]])
            print(f"[Auto Plot] sample: {preview}", flush=True)
        print(f"[Auto Plot] manifest: {manifest_path}", flush=True)
    else:
        print("[Auto Plot] disabled by DISABLE_AUTO_PLOT", flush=True)


if __name__ == "__main__":
    main()
