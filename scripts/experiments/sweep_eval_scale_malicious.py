"""
[RC1论文实验] sweep_eval_scale_malicious.py

目的：
  1) 在不写死参数数值的前提下，基于某个主 run 的 config_snapshot.json，
     对训练好的 MAPPO 模型与若干 baseline 在不同场景设置下进行评估；
  2) 支持两类 sweep：
     - 规模 sweep：NUM_VEHICLES = 10/20/40 ...
     - 恶意比例 sweep：MALICIOUS_RATIO = 0/0.1/0.2/0.3 ...

学术诚信说明：
  本脚本输出的 CSV/图表均由真实仿真评估得到，不生成“理想伪造数据”。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from train import apply_env_overrides, evaluate_single_baseline_episode, BASELINE_POLICIES


def _apply_config_snapshot(snapshot_path: Path) -> None:
    with snapshot_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    system_cfg: Dict[str, Any] = data.get("system_config", {}) or {}
    train_cfg: Dict[str, Any] = data.get("train_config", {}) or {}

    for key, val in system_cfg.items():
        if hasattr(Cfg, key):
            setattr(Cfg, key, val)
    for key, val in train_cfg.items():
        if hasattr(TC, key):
            setattr(TC, key, val)


@dataclass(frozen=True)
class SweepSetting:
    sweep: str  # "scale" or "malicious"
    value: float


def _set_env_var(name: str, value: Optional[str]) -> Optional[str]:
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)
    return old


class _EnvVarGuard:
    def __init__(self, mapping: Dict[str, Optional[str]]):
        self._mapping = dict(mapping)
        self._old: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self._mapping.items():
            self._old[k] = _set_env_var(k, v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, old in self._old.items():
            _set_env_var(k, old)
        return False


def _classify_action_type(action: Dict[str, Any], obs: Dict[str, Any]) -> str:
    """Return one of: local/rsu/v2v."""
    target = int(action.get("target", 0))
    candidate_types = obs.get("candidate_types")
    if candidate_types is not None and 0 <= target < len(candidate_types):
        ctype = int(candidate_types[target])
        if ctype == 1:
            return "local"
        if ctype == 2:
            return "rsu"
        return "v2v"
    if target == 0:
        return "local"
    if getattr(Cfg, "ENABLE_RSU_SELECTION", False):
        if 1 <= target <= int(getattr(Cfg, "NUM_RSU", 1)):
            return "rsu"
        return "v2v"
    return "rsu" if target == 1 else "v2v"


def evaluate_single_mappo_episode(
    env: VecOffloadingEnv,
    net: OffloadingPolicyNetwork,
    episode_seed: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Match train.evaluate_single_baseline_episode() metric style as much as possible."""
    obs_list, _ = env.reset(seed=int(episode_seed))
    ep_reward = 0.0
    total_steps = 0
    last_info: Optional[Dict[str, Any]] = None

    stats = {
        "power_sum": 0.0,
        "power_values": [],
        "local_cnt": 0,
        "rsu_cnt": 0,
        "neighbor_cnt": 0,
        "queue_len_sum": 0,
        "rsu_queue_sum": 0,
    }

    net.eval()
    for _ in range(int(getattr(TC, "MAX_STEPS", 0))):
        current_obs_list = obs_list
        with torch.no_grad():
            subtask_actions, target_actions, power_actions, _, _ = net.get_action_and_value(
                obs_list, deterministic=True, device=device
            )
        subtask_actions = subtask_actions.detach().cpu().numpy()
        target_actions = target_actions.detach().cpu().numpy()
        power_actions = power_actions.detach().cpu().numpy()

        actions: List[Dict[str, Any]] = []
        for i in range(len(obs_list)):
            act = {
                "subtask": int(subtask_actions[i]),
                "target": int(target_actions[i]),
                "power": float(power_actions[i]),
            }
            if isinstance(obs_list[i], dict) and "obs_stamp" in obs_list[i]:
                act["obs_stamp"] = int(obs_list[i]["obs_stamp"])
            actions.append(act)

        obs_list, rewards, done, truncated, info = env.step(actions)
        last_info = info

        ep_reward += float(np.mean(rewards)) if rewards is not None else 0.0
        total_steps += 1

        for i, act in enumerate(actions):
            obs_i = current_obs_list[i] if i < len(current_obs_list) else {}
            t = _classify_action_type(act, obs_i if isinstance(obs_i, dict) else {})
            if t == "local":
                stats["local_cnt"] += 1
            elif t == "rsu":
                stats["rsu_cnt"] += 1
            else:
                stats["neighbor_cnt"] += 1

            p = float(act.get("power", 0.0))
            stats["power_sum"] += p
            if int(act.get("target", 0)) != 0:
                stats["power_values"].append(p)
            stats["queue_len_sum"] += env.vehicles[i].task_queue_len if i < len(env.vehicles) else 0

        if env.rsus:
            rsu_queue_len = 0
            for rsu in env.rsus:
                proc_dict = env.rsu_cpu_q.get(rsu.id, {})
                rsu_queue_len += sum(len(q) for q in proc_dict.values())
            stats["rsu_queue_sum"] += rsu_queue_len

        if done or truncated:
            break

    avg_step_reward = ep_reward / max(total_steps, 1)
    total_decisions = stats["local_cnt"] + stats["rsu_cnt"] + stats["neighbor_cnt"]
    dec_den = total_decisions if total_decisions > 0 else 1

    # Success metrics (align with train.py baseline path)
    episode_vehicle_count = len(env.vehicles)
    success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished and not v.task_dag.is_failed])
    veh_success_rate = success_count / max(episode_vehicle_count, 1)
    task_success_rate = veh_success_rate

    total_subtasks = 0
    completed_subtasks = 0
    v2v_subtasks_attempted = 0
    v2v_subtasks_completed = 0
    for v in env.vehicles:
        total_subtasks += v.task_dag.num_subtasks
        completed_subtasks += int(np.sum(v.task_dag.status == 3))
        if hasattr(v.task_dag, "exec_locations"):
            for idx, loc in enumerate(v.task_dag.exec_locations):
                if isinstance(loc, int):
                    v2v_subtasks_attempted += 1
                    if v.task_dag.status[idx] == 3:
                        v2v_subtasks_completed += 1
    subtask_success_rate = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
    v2v_subtask_success_rate = (v2v_subtasks_completed / v2v_subtasks_attempted) if v2v_subtasks_attempted > 0 else 0.0

    frac_local = (stats["local_cnt"] / dec_den) if total_decisions > 0 else 0.0
    frac_rsu = (stats["rsu_cnt"] / dec_den) if total_decisions > 0 else 0.0
    frac_v2v = (stats["neighbor_cnt"] / dec_den) if total_decisions > 0 else 0.0
    avg_power = stats["power_sum"] / dec_den if total_decisions > 0 else 0.0
    power_ratio_mean = float(np.mean(stats["power_values"])) if stats["power_values"] else 0.0
    power_ratio_p95 = float(np.percentile(stats["power_values"], 95)) if stats["power_values"] else 0.0
    avg_queue_len = stats["queue_len_sum"] / dec_den if total_decisions > 0 else 0.0
    avg_rsu_queue = stats["rsu_queue_sum"] / max(total_steps, 1)

    epm = (last_info or {}).get("episode_metrics", {}) if isinstance(last_info, dict) else {}

    out: Dict[str, Any] = {
        "avg_step_reward": avg_step_reward,
        "total_reward": float(epm.get("r_total", avg_step_reward * max(total_steps, 1))),
        "veh_success_rate": veh_success_rate,
        "task_success_rate": task_success_rate,
        "subtask_success_rate": subtask_success_rate,
        "v2v_subtask_success_rate": v2v_subtask_success_rate,
        "decision_frac_local": frac_local,
        "decision_frac_rsu": frac_rsu,
        "decision_frac_v2v": frac_v2v,
        "avg_power": avg_power,
        "power_ratio_mean": power_ratio_mean,
        "power_ratio_p95": power_ratio_p95,
        "avg_queue_len": avg_queue_len,
        "avg_rsu_queue": float(epm.get("avg_rsu_queue", avg_rsu_queue)),
        # Episode-level metrics from env
        "episode_time_seconds": epm.get("episode_time_seconds"),
        "mean_cft_est": epm.get("mean_cft_est"),
        "mean_cft_completed": epm.get("mean_cft_completed"),
        "task_duration_mean": epm.get("task_duration_mean"),
        "task_duration_p95": epm.get("task_duration_p95"),
        "deadline_miss_rate": epm.get("deadline_miss_rate"),
        "deadline_meet_ratio": epm.get("deadline_meet_ratio"),
        "time_limit_rate": epm.get("time_limit_rate"),
        "energy_mean": epm.get("energy_norm_mean"),
        "energy_p95": epm.get("energy_norm_p95"),
        "illegal_action_rate": epm.get("illegal_action_rate"),
        "no_task_rate": epm.get("no_task_rate"),
        "on_task_rate": epm.get("on_task_rate"),
        "has_task_available_rate": epm.get("has_task_available_rate"),
        "unified_illegal_trigger_rate": epm.get("unified_illegal_trigger_rate"),
        "I_total_mean": epm.get("I_total_mean"),
        "I_total_p50": epm.get("I_total_p50"),
        "I_total_p95": epm.get("I_total_p95"),
        "I_caused_mean": epm.get("I_caused_mean"),
        "I_caused_p95": epm.get("I_caused_p95"),
        "rho_selected_mean": epm.get("rho_selected_mean"),
        "rho_selected_p10": epm.get("rho_selected_p10"),
        "uncertainty_selected_mean": epm.get("uncertainty_selected_mean"),
        "uncertainty_selected_p90": epm.get("uncertainty_selected_p90"),
        "risk_penalty_mean": epm.get("risk_penalty_mean"),
        "rho_selected_p50": epm.get("rho_selected_p50"),
        "rho_selected_p95": epm.get("rho_selected_p95"),
        "rho_selected_lt_0p6_rate": epm.get("rho_selected_lt_0p6_rate"),
        "rho_selected_lt_0p7_rate": epm.get("rho_selected_lt_0p7_rate"),
        "chain_tx_total": epm.get("chain_tx_total"),
        "chain_p95_mean": epm.get("chain_p95_mean"),
        "chain_pfail_mean": epm.get("chain_pfail_mean"),
        "chain_risk_cost_total": epm.get("chain_risk_cost_total"),
        "trust_attempts": epm.get("trust_attempts"),
        "trust_failures": epm.get("trust_failures"),
        "trust_failure_rate": epm.get("trust_failure_rate"),
        "trust_retry_count": epm.get("trust_retry_count"),
        "act_seconds": epm.get("act_seconds"),
        "makespan_seconds": epm.get("makespan_seconds"),
    }
    return out


def _aggregate_summary(rows: List[Dict[str, Any]], group_keys: List[str], metric_keys: List[str]) -> List[Dict[str, Any]]:
    summary: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in rows:
        g = tuple(r.get(k) for k in group_keys)
        d = summary.setdefault(g, {k: r.get(k) for k in group_keys})
        d.setdefault("_n", 0)
        d["_n"] += 1
        for mk in metric_keys:
            v = r.get(mk)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            d.setdefault(f"{mk}__vals", []).append(fv)

    out_rows: List[Dict[str, Any]] = []
    for g, d in summary.items():
        row = {k: d.get(k) for k in group_keys}
        row["episodes"] = int(d.get("_n", 0))
        for mk in metric_keys:
            vals = d.get(f"{mk}__vals", [])
            row[f"{mk}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{mk}_std"] = float(np.std(vals)) if vals else None
        out_rows.append(row)
    return out_rows


def _parse_csv_list(raw: str, kind: str) -> List[str]:
    items = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not items:
        return []
    if kind == "policy":
        allowed = set([p.lower() for p in BASELINE_POLICIES] + ["mappo"])
        bad = [x for x in items if x.lower() not in allowed]
        if bad:
            raise ValueError(f"Unknown policies: {bad}. Allowed: {sorted(allowed)}")
    return items


def _parse_num_list(raw: str, cast=float) -> List[Any]:
    items = [x.strip() for x in (raw or "").split(",") if x.strip()]
    return [cast(x) for x in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--policies", type=str, default="mappo,Local-Only,Greedy,EFT,LB-Greedy")
    ap.add_argument("--sweep", type=str, choices=["scale", "malicious", "main"], required=True)
    ap.add_argument("--scale-list", type=str, default="10,20,40")
    ap.add_argument("--malicious-list", type=str, default="0,0.1,0.2,0.3")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--summary-csv", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    snapshot_path = run_dir / "logs" / "config_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {snapshot_path}")

    # Load base snapshot once to get SEED/TC dims etc.
    _apply_config_snapshot(snapshot_path)
    if args.seed0 is None:
        args.seed0 = int(getattr(Cfg, "SEED", 0))

    model_path = Path(args.model_path) if args.model_path else (run_dir / "models" / "best_model.pth")
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    policies = _parse_csv_list(args.policies, kind="policy")
    if not policies:
        raise ValueError("Empty --policies.")

    # Instantiate model once (NUM_RSU etc are fixed across our sweeps).
    net = OffloadingPolicyNetwork()
    ckpt = torch.load(str(model_path), map_location="cpu")
    if isinstance(ckpt, dict) and "network_state_dict" in ckpt:
        net.load_state_dict(ckpt["network_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        net.load_state_dict(ckpt["policy_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        net.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        net.load_state_dict(ckpt, strict=False)

    settings: List[SweepSetting] = []
    if args.sweep == "main":
        settings = [SweepSetting("main", 0.0)]
    elif args.sweep == "scale":
        for v in _parse_num_list(args.scale_list, cast=int):
            settings.append(SweepSetting("scale", float(v)))
    else:
        for v in _parse_num_list(args.malicious_list, cast=float):
            settings.append(SweepSetting("malicious", float(v)))

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else out_csv.with_name(out_csv.stem + "_summary.csv")

    episode_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for setting in settings:
        # Reset to snapshot, then apply env overrides for this setting.
        _apply_config_snapshot(snapshot_path)
        override_env: Dict[str, Optional[str]] = {}
        if setting.sweep == "scale":
            override_env["NUM_VEHICLES"] = str(int(setting.value))
        elif setting.sweep == "malicious":
            override_env["MALICIOUS_RATIO"] = str(float(setting.value))
            # Explicitly keep the intended semantics (does not change if already default).
            override_env["REP_INIT_MODE"] = "beta"
            override_env["TRUST_FAIL_SCOPE"] = "v2v_only"

        with _EnvVarGuard(override_env):
            apply_env_overrides()
            env = VecOffloadingEnv()

            for ep in range(int(args.episodes)):
                episode_id = ep + 1
                seed = int(args.seed0) + int(episode_id)
                for policy in policies:
                    if policy.lower() == "mappo":
                        metrics = evaluate_single_mappo_episode(env, net, episode_seed=seed, device=args.device)
                        policy_name = "mappo"
                    else:
                        metrics = evaluate_single_baseline_episode(env, policy, episode_seed=seed)
                        policy_name = str(policy)

                    row = {
                        "sweep": setting.sweep,
                        "value": setting.value,
                        "policy": policy_name,
                        "episode": episode_id,
                        "seed": seed,
                    }
                    row.update(metrics)
                    episode_rows.append(row)

    # Write episode CSV
    fieldnames = sorted({k for r in episode_rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in episode_rows:
            w.writerow(r)

    # Summary CSV
    metric_keys = [
        "task_success_rate",
        "deadline_miss_rate",
        "mean_cft_est",
        "risk_penalty_mean",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
        "power_ratio_mean",
        "trust_failure_rate",
        "rho_selected_mean",
        "uncertainty_selected_mean",
        "illegal_action_rate",
        "time_limit_rate",
    ]
    summary_rows = _aggregate_summary(
        episode_rows,
        group_keys=["sweep", "value", "policy"],
        metric_keys=metric_keys,
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in summary_rows for k in r.keys()}))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    dt = time.time() - t0
    print(f"✓ Wrote: {out_csv}")
    print(f"✓ Wrote: {summary_csv}")
    print(f"✓ Done in {dt:.1f}s, rows={len(episode_rows)}")


if __name__ == "__main__":
    main()
