"""
[基线评估脚本] run_baselines.py
Baseline Evaluation Runner

作用 (Purpose):
    在独立进程中评估所有baseline策略，并将结果写入指定run目录的logs/baseline_stats.csv，
    便于与train.py生成的训练曲线进行统一绘图。

使用方法 (Usage):
    python scripts/run_baselines.py --run-dir runs/run_YYYYMMDD_HHMMSS --num-episodes 20
    python scripts/run_baselines.py --run-id run_YYYYMMDD_HHMMSS --num-episodes 20
"""

import argparse
import json
import os
import sys
import shutil
import time
import subprocess
from collections import defaultdict


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from train import (
    evaluate_single_baseline_episode,
    apply_env_overrides,
    BASELINE_POLICIES,
    BASELINE_STATS_FIELDS,
)


def _bind_baseline_gate_horizon():
    gate_steps = int(getattr(Cfg, "BASELINE_GATE_MAX_STEPS", getattr(Cfg, "MAX_STEPS", 0)))
    if gate_steps <= 0:
        raise ValueError(f"BASELINE_GATE_MAX_STEPS must be positive, got {gate_steps}")
    Cfg.MAX_STEPS = gate_steps
    TC.MAX_STEPS = gate_steps
    return gate_steps


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


def _find_latest_run(base_dir="runs"):
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        if name.startswith("run_"):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_run_dir(args):
    if args.run_dir:
        return os.path.abspath(args.run_dir)
    if args.run_id:
        return os.path.abspath(os.path.join("runs", args.run_id))
    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        return os.path.abspath(env_run_dir)
    latest = _find_latest_run()
    if latest:
        return os.path.abspath(latest)
    raise FileNotFoundError("未找到可用的run目录，请使用--run-dir或--run-id显式指定。")


def _apply_config_snapshot(snapshot_path):
    if not os.path.exists(snapshot_path):
        return False
    with open(snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    system_cfg = data.get("system_config", {})
    train_cfg = data.get("train_config", {})
    for key, val in system_cfg.items():
        if hasattr(Cfg, key):
            setattr(Cfg, key, val)
    for key, val in train_cfg.items():
        if hasattr(TC, key):
            setattr(TC, key, val)
    return True


def _apply_run_config_dump(run_dir):
    """
    Best-effort: apply a previous run's dumped config into the live Cfg/TC.
    Supports both run_dir/config.json and run_dir/config_dump.json.

    This is used when logs/config_snapshot.json is missing (e.g. older runs copied between machines).
    """
    candidates = [
        os.path.join(run_dir, "config.json"),
        os.path.join(run_dir, "config_dump.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    # Apply SystemConfig keys
    for key, val in data.items():
        if hasattr(Cfg, key):
            setattr(Cfg, key, val)

    # Apply common TrainConfig keys (config dump uses snake_case)
    tc_map = {
        "lr_actor": "LR_ACTOR",
        "lr_critic": "LR_CRITIC",
        "gamma": "GAMMA",
        "gae_lambda": "GAE_LAMBDA",
        "clip_param": "CLIP_PARAM",
        "batch_size": "MINI_BATCH_SIZE",
        "k_epochs": "PPO_EPOCH",
        "entropy_coef": "ENTROPY_COEF",
        "max_episodes": "MAX_EPISODES",
        "device": "DEVICE_NAME",
    }
    for src_key, dst_key in tc_map.items():
        if src_key in data and hasattr(TC, dst_key):
            setattr(TC, dst_key, data[src_key])

    return True


def _ensure_reward_jsonl(logs_dir):
    reward_jsonl_path = os.environ.get("REWARD_JSONL_PATH")
    if not reward_jsonl_path:
        reward_jsonl_path = os.path.join(logs_dir, "env_reward.jsonl")
        os.environ["REWARD_JSONL_PATH"] = reward_jsonl_path
    os.makedirs(os.path.dirname(reward_jsonl_path), exist_ok=True)
    if not os.path.exists(reward_jsonl_path):
        with open(reward_jsonl_path, "w", encoding="utf-8") as f:
            f.write("{}\n")
    return reward_jsonl_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Run baseline evaluation and write baseline_stats.csv.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--episode-start", type=int, default=1)
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Comma-separated subset of policies to run (e.g. 'Greedy,EFT'). Default runs all.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Write baseline stats to this CSV path instead of <run_dir>/logs/baseline_stats.csv.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = _resolve_run_dir(args)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 应用环境变量覆盖，再加载快照保证与训练一致，最后应用CLI覆盖
    apply_env_overrides()
    snapshot_path = os.path.join(logs_dir, "config_snapshot.json")
    loaded = _apply_config_snapshot(snapshot_path)
    snapshot_hash = None
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot_hash = json.load(f).get("config_hash")
        except Exception:
            snapshot_hash = None
    if not loaded:
        # Fallback for older runs: load run_dir/config.json or run_dir/config_dump.json
        _apply_run_config_dump(run_dir)

    if args.seed is not None:
        Cfg.SEED = int(args.seed)
    gate_steps = _bind_baseline_gate_horizon()

    _ensure_reward_jsonl(logs_dir)

    env = VecOffloadingEnv()

    baseline_meta = {
        "run_dir": run_dir,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _get_git_commit(),
        "config_hash": snapshot_hash,
        "seed": int(getattr(Cfg, "SEED", 0)),
        "max_steps": gate_steps,
        "num_episodes": int(args.num_episodes),
    }
    with open(os.path.join(run_dir, "baseline_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_meta, f, ensure_ascii=True, indent=2)

    # If this run uses the older layout (run_dir/metrics/*.csv), mirror them into logs/
    # so downstream plot scripts can find a consistent path.
    metrics_dir = os.path.join(run_dir, "metrics")
    if os.path.isdir(metrics_dir):
        try:
            src_metrics = os.path.join(metrics_dir, "metrics.csv")
            if os.path.exists(src_metrics):
                dst_metrics = os.path.join(logs_dir, "metrics.csv")
                if not os.path.exists(dst_metrics):
                    shutil.copyfile(src_metrics, dst_metrics)
        except Exception:
            pass
        try:
            src_train = os.path.join(metrics_dir, "train_metrics.csv")
            if os.path.exists(src_train):
                dst_train = os.path.join(logs_dir, "training_stats.csv")
                if not os.path.exists(dst_train):
                    shutil.copyfile(src_train, dst_train)
        except Exception:
            pass

    baseline_policies = list(BASELINE_POLICIES)
    if args.policies:
        want = [p.strip() for p in args.policies.split(",") if p.strip()]
        unknown = [p for p in want if p not in baseline_policies]
        if unknown:
            raise ValueError(f"Unknown policies: {unknown}. Supported: {baseline_policies}")
        baseline_policies = want
    baseline_stats_fields = list(BASELINE_STATS_FIELDS)
    baseline_stats_csv = os.path.abspath(args.output_csv) if args.output_csv else os.path.join(logs_dir, "baseline_stats.csv")
    os.makedirs(os.path.dirname(baseline_stats_csv), exist_ok=True)
    canonical_baseline_csv = os.path.abspath(os.path.join(logs_dir, "baseline_stats.csv"))

    if args.append:
        header_written = os.path.exists(baseline_stats_csv) and os.path.getsize(baseline_stats_csv) > 0
        file_mode = "a"
    else:
        header_written = False
        file_mode = "w"

    summary = {policy: [] for policy in baseline_policies}
    start_ep = int(args.episode_start)
    total_episodes = int(args.num_episodes)

    print(f"[Baselines] run_dir={run_dir}")
    print(f"[Baselines] num_episodes={total_episodes} seed={Cfg.SEED} max_steps={gate_steps}")
    print(f"[Baselines] policies={baseline_policies}")
    t0 = time.time()
    rows_written = 0

    try:
        with open(baseline_stats_csv, file_mode, newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=baseline_stats_fields, extrasaction="ignore")
            if not header_written:
                writer.writeheader()

            for ep_offset in range(total_episodes):
                episode = start_ep + ep_offset
                for policy_idx, policy_name in enumerate(baseline_policies, start=1):
                    t_policy0 = time.time()
                    print(f"[Baselines] ep={episode} ({ep_offset+1}/{total_episodes}) policy={policy_name} ({policy_idx}/{len(baseline_policies)}) ...")
                    metrics = evaluate_single_baseline_episode(
                        env,
                        policy_name,
                        episode_seed=int(getattr(Cfg, "SEED", 0)) + int(episode),
                    )
                    summary[policy_name].append(metrics)
                    row = {
                        "episode": episode,
                        "policy": policy_name,
                        "reward_mean": metrics["avg_step_reward"],
                        "reward_total": metrics["total_reward"],
                        "vehicle_sr": metrics["veh_success_rate"],
                        "task_sr": metrics.get("task_success_rate", metrics["veh_success_rate"]),
                        "subtask_sr": metrics["subtask_success_rate"],
                        "v2v_subtask_sr": metrics["v2v_subtask_success_rate"],
                        "ratio_local": metrics["decision_frac_local"],
                        "ratio_rsu": metrics["decision_frac_rsu"],
                        "ratio_v2v": metrics["decision_frac_v2v"],
                        "decision_frac_local": metrics["decision_frac_local"],
                        "decision_frac_rsu": metrics["decision_frac_rsu"],
                        "decision_frac_v2v": metrics["decision_frac_v2v"],
                        "avg_power": metrics["avg_power"],
                        "power_ratio_mean": metrics.get("power_ratio_mean"),
                        "power_ratio_p95": metrics.get("power_ratio_p95"),
                        "episode_time_seconds": metrics.get("episode_time_seconds"),
                        "mean_cft_est": metrics.get("mean_cft_est"),
                        "mean_cft_completed": metrics.get("mean_cft_completed"),
                        "act_seconds": metrics.get("act_seconds"),
                        "makespan_seconds": metrics.get("makespan_seconds"),
                        "task_duration_mean": metrics.get("task_duration_mean"),
                        "task_duration_p95": metrics.get("task_duration_p95"),
                        "deadline_miss_rate": metrics.get("deadline_miss_rate"),
                        "deadline_meet_ratio": metrics.get("deadline_meet_ratio"),
                        "time_limit_rate": metrics.get("time_limit_rate"),
                        "energy_mean": metrics.get("energy_mean"),
                        "energy_p95": metrics.get("energy_p95"),
                        "illegal_action_rate": metrics.get("illegal_action_rate"),
                        "no_task_rate": metrics.get("no_task_rate"),
                        "on_task_rate": metrics.get("on_task_rate"),
                        "has_task_available_rate": metrics.get("has_task_available_rate"),
                        "unified_illegal_trigger_rate": metrics.get("unified_illegal_trigger_rate"),
                        "I_total_mean": metrics.get("I_total_mean"),
                        "I_total_p50": metrics.get("I_total_p50"),
                        "I_total_p95": metrics.get("I_total_p95"),
                        "I_caused_mean": metrics.get("I_caused_mean"),
                        "I_caused_p95": metrics.get("I_caused_p95"),
                        "rho_selected_mean": metrics.get("rho_selected_mean"),
                        "rho_selected_p10": metrics.get("rho_selected_p10"),
                        "uncertainty_selected_mean": metrics.get("uncertainty_selected_mean"),
                        "uncertainty_selected_p90": metrics.get("uncertainty_selected_p90"),
                        "risk_penalty_mean": metrics.get("risk_penalty_mean"),
                        "rho_selected_p50": metrics.get("rho_selected_p50"),
                        "rho_selected_p95": metrics.get("rho_selected_p95"),
                        "rho_selected_lt_0p6_rate": metrics.get("rho_selected_lt_0p6_rate"),
                        "rho_selected_lt_0p7_rate": metrics.get("rho_selected_lt_0p7_rate"),
                        "chain_tx_total": metrics.get("chain_tx_total"),
                        "chain_p95_mean": metrics.get("chain_p95_mean"),
                        "chain_pfail_mean": metrics.get("chain_pfail_mean"),
                        "chain_risk_cost_total": metrics.get("chain_risk_cost_total"),
                        "trust_attempts": metrics.get("trust_attempts"),
                        "trust_failures": metrics.get("trust_failures"),
                        "trust_failure_rate": metrics.get("trust_failure_rate"),
                        "trust_retry_count": metrics.get("trust_retry_count"),
                        "avg_queue_len": metrics["avg_queue_len"],
                        "avg_rsu_queue": metrics.get("avg_rsu_queue", 0.0),
                    }
                    for field in baseline_stats_fields:
                        if field in row:
                            continue
                        if field in metrics:
                            row[field] = metrics.get(field)
                    writer.writerow(row)
                    rows_written += 1
                    f.flush()
                    dt = time.time() - t_policy0
                    print(
                        f"[Baselines]   done reward={row['reward_mean']:.4f} "
                        f"task_sr={row['task_sr']:.4f} subtask_sr={row['subtask_sr']:.4f} ({dt:.1f}s)"
                    )
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n[Baselines] interrupted after {elapsed:.1f}s, rows_written={rows_written}. Partial results kept.")
    finally:
        # Mirror only the canonical file to avoid races during parallel runs that write parts.
        if os.path.abspath(baseline_stats_csv) == canonical_baseline_csv:
            if os.path.exists(baseline_stats_csv) and os.path.getsize(baseline_stats_csv) > 0:
                if os.path.isdir(metrics_dir):
                    try:
                        dst = os.path.join(metrics_dir, "baseline_stats.csv")
                        shutil.copyfile(baseline_stats_csv, dst)
                        print(f"✓ Baseline stats mirrored: {dst}")
                    except Exception:
                        pass

    print(f"✓ Baseline stats saved: {baseline_stats_csv}")

    # 额外导出论文核心指标聚合摘要（按策略）
    core_summary = []
    for policy_name in baseline_policies:
        records = summary.get(policy_name, [])
        if not records:
            continue
        def _vals(key):
            out = []
            for r in records:
                v = r.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    out.append(fv)
            return out
        import numpy as np
        act_vals = _vals("act_seconds")
        makespan_vals = _vals("makespan_seconds")
        cft_vals = _vals("mean_cft_completed")
        duration_vals = _vals("task_duration_mean")
        duration_p95_vals = _vals("task_duration_p95")
        energy_vals = _vals("energy_mean")
        energy_p95_vals = _vals("energy_p95")
        tsr_vals = _vals("task_success_rate")
        dmr_vals = _vals("deadline_miss_rate")
        queue_vals = _vals("avg_rsu_queue")
        power_vals = _vals("avg_power")
        local_vals = _vals("decision_frac_local")
        rsu_vals = _vals("decision_frac_rsu")
        v2v_vals = _vals("decision_frac_v2v")
        core_summary.append({
            "policy": policy_name,
            "episodes": len(records),
            "act_mean": float(np.mean(act_vals)) if act_vals else None,
            "act_std": float(np.std(act_vals)) if act_vals else None,
            "makespan_mean": float(np.mean(makespan_vals)) if makespan_vals else None,
            "mean_cft_completed_mean": float(np.mean(cft_vals)) if cft_vals else None,
            "task_duration_mean": float(np.mean(duration_vals)) if duration_vals else None,
            "task_duration_p95_mean": float(np.mean(duration_p95_vals)) if duration_p95_vals else None,
            "energy_mean": float(np.mean(energy_vals)) if energy_vals else None,
            "energy_std": float(np.std(energy_vals)) if energy_vals else None,
            "energy_p95_mean": float(np.mean(energy_p95_vals)) if energy_p95_vals else None,
            "task_success_rate_mean": float(np.mean(tsr_vals)) if tsr_vals else None,
            "deadline_miss_rate_mean": float(np.mean(dmr_vals)) if dmr_vals else None,
            "deadline_meet_ratio_mean": (float(np.mean([max(0.0, min(1.0, 1.0 - x)) for x in dmr_vals])) if dmr_vals else None),
            "avg_rsu_queue_mean": float(np.mean(queue_vals)) if queue_vals else None,
            "avg_power_mean": float(np.mean(power_vals)) if power_vals else None,
            "decision_frac_local_mean": float(np.mean(local_vals)) if local_vals else None,
            "decision_frac_rsu_mean": float(np.mean(rsu_vals)) if rsu_vals else None,
            "decision_frac_v2v_mean": float(np.mean(v2v_vals)) if v2v_vals else None,
        })
    core_json = os.path.join(logs_dir, "baseline_eval_core_summary.json")
    with open(core_json, "w", encoding="utf-8") as f:
        json.dump(core_summary, f, ensure_ascii=True, indent=2)
    try:
        import csv
        core_csv = os.path.join(logs_dir, "baseline_eval_core_summary.csv")
        fieldnames = [
            "policy", "episodes",
            "act_mean", "act_std", "makespan_mean",
            "mean_cft_completed_mean",
            "task_duration_mean", "task_duration_p95_mean",
            "energy_mean", "energy_std", "energy_p95_mean",
            "task_success_rate_mean", "deadline_miss_rate_mean", "deadline_meet_ratio_mean",
            "avg_rsu_queue_mean", "avg_power_mean",
            "decision_frac_local_mean", "decision_frac_rsu_mean", "decision_frac_v2v_mean",
        ]
        with open(core_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in core_summary:
                writer.writerow(row)
        print(f"✓ Baseline core summary saved: {core_csv}")
    except Exception:
        pass
    print(f"✓ Baseline core summary saved: {core_json}")

    print("结果摘要 (均值):")
    for policy_name in baseline_policies:
        records = summary[policy_name]
        if not records:
            continue
        reward = sum(r["avg_step_reward"] for r in records) / len(records)
        task_sr = sum(r.get("task_success_rate", r["veh_success_rate"]) for r in records) / len(records)
        subtask_sr = sum(r["subtask_success_rate"] for r in records) / len(records)
        print(f"  {policy_name:<10} reward={reward:.4f} task_sr={task_sr:.4f} subtask_sr={subtask_sr:.4f}")


if __name__ == "__main__":
    main()
