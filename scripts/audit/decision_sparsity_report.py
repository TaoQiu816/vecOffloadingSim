#!/usr/bin/env python3
"""
Decision sparsity audit runner.

Runs N episodes with a random policy and writes per-episode + summary CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.config import SystemConfig as Cfg  # noqa: E402
from envs.vec_offloading_env import VecOffloadingEnv  # noqa: E402


def _parse_value(raw: str) -> Any:
    lower = raw.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        if "." in raw or "e" in lower:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _apply_overrides(overrides: List[str]) -> Dict[str, Any]:
    applied: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = _parse_value(value.strip())
        if not hasattr(Cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(Cfg, key, value)
        applied[key] = value
    return applied


def _calc_percentiles(values: List[float], ps: List[int]) -> Dict[str, float]:
    if not values:
        return {f"p{p}": float("nan") for p in ps}
    arr = np.asarray(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def _collect_queue_lengths(env: VecOffloadingEnv) -> Dict[str, List[int]]:
    veh_lengths = []
    for q in env.veh_cpu_q.values():
        veh_lengths.append(len(q))
    rsu_lengths = []
    for proc_dict in env.rsu_cpu_q.values():
        rsu_lengths.append(sum(len(q) for q in proc_dict.values()))
    return {"veh": veh_lengths, "rsu": rsu_lengths}


def _collect_inflight_counts(env: VecOffloadingEnv) -> List[int]:
    counts = []
    for v in env.vehicles:
        dag = v.task_dag
        inflight = 0
        for idx, loc in enumerate(dag.exec_locations):
            if loc is not None and dag.status[idx] < 3:
                inflight += 1
        counts.append(inflight)
    return counts


def run_episode(env: VecOffloadingEnv, rng: np.random.Generator, seed: int) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    num_vehicles = len(env.vehicles)

    active_sum = 0.0
    no_task_sum = 0.0
    per_vehicle_active = np.zeros(num_vehicles, dtype=float)

    ready_counts = []
    inflight_counts = []
    veh_queue_lengths = []
    rsu_queue_lengths = []
    service_steps = []

    comm_steps = 0
    compute_steps = 0
    wait_steps = 0
    finished_steps = 0

    assign_step: Dict[tuple, int] = {}
    prev_exec = {v.id: list(v.task_dag.exec_locations) for v in env.vehicles}
    prev_status = {v.id: v.task_dag.status.copy() for v in env.vehicles}

    step_count = 0
    last_info: Dict[str, Any] = {}

    while True:
        actions = []
        for _ in range(num_vehicles):
            target = int(rng.integers(0, env.config.MAX_TARGETS))
            power = float(rng.random())
            actions.append({"target": target, "power": power})

        obs, rewards, terminated, truncated, info = env.step(actions)
        last_info = info
        step_count += 1

        active_mask = info.get("active_agent_mask") or [1] * num_vehicles
        no_task_mask = info.get("no_task_step_mask") or [0] * num_vehicles
        active_sum += float(np.sum(active_mask))
        no_task_sum += float(np.sum(no_task_mask))
        per_vehicle_active += np.asarray(active_mask, dtype=float)

        ready_total = 0
        for v in env.vehicles:
            ready_total += int(np.sum(v.task_dag.status == 1))
        ready_counts.append(ready_total)

        inflight_counts.extend(_collect_inflight_counts(env))

        queues = _collect_queue_lengths(env)
        veh_queue_lengths.extend(queues["veh"])
        rsu_queue_lengths.extend(queues["rsu"])

        comm_active = any(len(q) > 0 for q in env.txq_v2i.values()) or any(len(q) > 0 for q in env.txq_v2v.values())
        compute_active = any(len(q) > 0 for q in env.veh_cpu_q.values()) or any(
            len(q) > 0 for proc_dict in env.rsu_cpu_q.values() for q in proc_dict.values()
        )
        all_finished = all(v.task_dag.is_finished for v in env.vehicles)

        if comm_active:
            comm_steps += 1
        if compute_active:
            compute_steps += 1
        if not comm_active and not compute_active and not all_finished:
            wait_steps += 1
        if all_finished:
            finished_steps += 1

        for v in env.vehicles:
            dag = v.task_dag
            for idx, loc in enumerate(dag.exec_locations):
                key = (v.id, idx)
                if prev_exec[v.id][idx] is None and loc is not None:
                    assign_step[key] = step_count
                if prev_status[v.id][idx] < 3 and dag.status[idx] == 3:
                    start = assign_step.get(key)
                    if start is not None:
                        service_steps.append(step_count - start + 1)

            prev_exec[v.id] = list(dag.exec_locations)
            prev_status[v.id] = dag.status.copy()

        if terminated or truncated:
            break

    steps_total = max(step_count * num_vehicles, 1)
    active_ratio = active_sum / steps_total
    no_task_ratio = no_task_sum / steps_total

    decision_counts = per_vehicle_active.tolist()
    decision_stats = {
        "decision_events_per_vehicle_mean": float(np.mean(decision_counts)) if decision_counts else 0.0,
        "decision_events_per_vehicle_p50": float(np.percentile(decision_counts, 50)) if decision_counts else 0.0,
        "decision_events_per_vehicle_p90": float(np.percentile(decision_counts, 90)) if decision_counts else 0.0,
    }

    ready_stats = _calc_percentiles(ready_counts, [50, 90, 95])
    inflight_stats = _calc_percentiles(inflight_counts, [95])
    service_stats = _calc_percentiles(service_steps, [50, 90])

    episode_metrics = last_info.get("episode_metrics", {})
    success_rate = episode_metrics.get("task_success_rate", last_info.get("task_success_rate", 0.0))
    deadline_miss_rate = episode_metrics.get("deadline_miss_rate", last_info.get("deadline_miss_rate", 0.0))
    illegal_count = episode_metrics.get("illegal_count", last_info.get("illegal_count", 0))
    illegal_reasons = episode_metrics.get("illegal_reasons", last_info.get("illegal_reasons", {})) or {}
    assign_failed_count = illegal_reasons.get("assign_failed", 0)

    return {
        "episode_seed": seed,
        "episode_steps": step_count,
        "active_ratio": float(active_ratio),
        "no_task_ratio": float(no_task_ratio),
        **decision_stats,
        "ready_subtask_count_mean": float(np.mean(ready_counts)) if ready_counts else 0.0,
        "ready_subtask_count_p50": ready_stats.get("p50", float("nan")),
        "ready_subtask_count_p90": ready_stats.get("p90", float("nan")),
        "ready_subtask_count_p95": ready_stats.get("p95", float("nan")),
        "inflight_subtask_count_mean": float(np.mean(inflight_counts)) if inflight_counts else 0.0,
        "inflight_subtask_count_p95": inflight_stats.get("p95", float("nan")),
        "avg_subtask_service_steps_mean": float(np.mean(service_steps)) if service_steps else float("nan"),
        "avg_subtask_service_steps_p50": service_stats.get("p50", float("nan")),
        "avg_subtask_service_steps_p90": service_stats.get("p90", float("nan")),
        "phase_comm_ratio": float(comm_steps / step_count) if step_count else 0.0,
        "phase_compute_ratio": float(compute_steps / step_count) if step_count else 0.0,
        "phase_wait_ratio": float(wait_steps / step_count) if step_count else 0.0,
        "phase_finished_ratio": float(finished_steps / step_count) if step_count else 0.0,
        "queue_len_vehicle_mean": float(np.mean(veh_queue_lengths)) if veh_queue_lengths else 0.0,
        "queue_len_vehicle_p95": float(np.percentile(veh_queue_lengths, 95)) if veh_queue_lengths else 0.0,
        "queue_len_rsu_mean": float(np.mean(rsu_queue_lengths)) if rsu_queue_lengths else 0.0,
        "queue_len_rsu_p95": float(np.percentile(rsu_queue_lengths, 95)) if rsu_queue_lengths else 0.0,
        "task_success_rate": float(success_rate),
        "deadline_miss_rate": float(deadline_miss_rate),
        "illegal_count": int(illegal_count),
        "assign_failed_count": int(assign_failed_count),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {"episodes": len(rows)}
    if not rows:
        return summary
    keys = [k for k in rows[0].keys() if k not in ("episode_seed",)]
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r.get(key))]
        if not values:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_p90"] = float("nan")
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_p90"] = float(np.percentile(values, 90))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision sparsity audit report")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--output_dir", type=str, default="audit_results/decision_sparsity", help="Output directory")
    parser.add_argument("--tag", type=str, default="baseline", help="Run tag for metadata")
    parser.add_argument("--set", action="append", default=[], help="Config override KEY=VALUE (repeatable)")
    args = parser.parse_args()

    applied = _apply_overrides(args.set)

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = VecOffloadingEnv(config=Cfg)
    rng = np.random.default_rng(args.seed)

    rows = []
    for ep in range(args.episodes):
        ep_seed = args.seed * 1000 + ep
        row = run_episode(env, rng, ep_seed)
        row["run_tag"] = args.tag
        row["config_overrides"] = ",".join(f"{k}={v}" for k, v in applied.items()) if applied else ""
        rows.append(row)

    episode_path = os.path.join(args.output_dir, "decision_sparsity_episode_stats.csv")
    write_csv(episode_path, rows)

    summary = summarize(rows)
    summary["run_tag"] = args.tag
    summary["config_overrides"] = ",".join(f"{k}={v}" for k, v in applied.items()) if applied else ""
    summary_path = os.path.join(args.output_dir, "decision_sparsity_summary.csv")
    write_csv(summary_path, [summary])

    print(f"Wrote: {episode_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
