#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _target_from_slot(slot_type: int, slot_id: int):
    if slot_type == 1:
        return "Local"
    if slot_type == 2 and slot_id >= 0:
        return ("RSU", int(slot_id))
    if slot_type == 3 and slot_id >= 0:
        return int(slot_id)
    return None


def _target_label(target) -> str:
    if target is None:
        return "None"
    if target == "Local":
        return "Local"
    if isinstance(target, tuple) and len(target) >= 2 and target[0] == "RSU":
        return f"RSU:{int(target[1])}"
    if isinstance(target, int):
        return f"V2V:{int(target)}"
    return str(target)


def _mode_label(slot_type: int) -> str:
    return {1: "local", 2: "rsu", 3: "v2v"}.get(int(slot_type), "none")


def _choose_ready_subtask(env: VecOffloadingEnv, vehicle) -> Optional[int]:
    dag = vehicle.task_dag
    ready = np.flatnonzero(np.asarray(dag.get_action_mask(), dtype=bool))
    if ready.size <= 0:
        return None
    priorities = dag.compute_all_priorities() if hasattr(dag, "compute_all_priorities") else np.zeros(dag.num_subtasks, dtype=np.float32)
    pri_ready = priorities[ready]
    return int(ready[int(np.argmax(pri_ready))])


def _random_action(obs: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    submask = np.asarray(obs.get("subtask_mask"), dtype=bool)
    actmask = np.asarray(obs.get("action_mask"), dtype=bool)
    ready = np.flatnonzero(submask)
    targets = np.flatnonzero(actmask)
    if ready.size <= 0:
        return {"subtask": 0, "target": 0, "power": 0.5}
    subtask = int(rng.choice(ready))
    target = int(rng.choice(targets)) if targets.size > 0 else 0
    return {"subtask": subtask, "target": target, "power": float(rng.random())}


def _estimate_prev_sojourn_target_cost(
    env: VecOffloadingEnv,
    vehicle,
    subtask_idx: int,
    target,
    task_comp=None,
    task_data=None,
    power_dbm=None,
    comm_wait_dict=None,
    active_v2i_count=None,
    active_v2i_vehicles=None,
    active_v2v_vehicles=None,
):
    est = env._estimate_snapshot_target_cost(
        vehicle,
        subtask_idx,
        target,
        task_comp=task_comp,
        task_data=task_data,
        power_dbm=power_dbm,
        comm_wait_dict=comm_wait_dict,
        active_v2i_count=active_v2i_count,
        active_v2i_vehicles=active_v2i_vehicles,
        active_v2v_vehicles=active_v2v_vehicles,
    )
    if not env._is_rsu_location(target) or not bool(est.get("available", False)):
        return est
    rsu_id = env._get_rsu_id_from_location(target)
    if rsu_id is None or not (0 <= int(rsu_id) < len(env.rsus)):
        return est
    service_rate = max(float(env._get_rsu_service_rate_snapshot(int(rsu_id))), 1e-9)
    queue_work = float(env._get_rsu_queue_load(int(rsu_id)))
    arrival_work = float(env._estimate_rsu_arrival_work_proxy(int(rsu_id), owner_vehicle_id=vehicle.id))
    task_cycles = float(task_comp if task_comp is not None else est.get("task_comp", 0.0))
    prev_est = dict(est)
    prev_est["cpu_wait"] = float((queue_work + arrival_work) / service_rate)
    prev_est["cpu_exec"] = float(max(task_cycles, 0.0) / service_rate)
    prev_est["J"] = float(prev_est["comm_wait"] + prev_est["tx_time"] + prev_est["cpu_wait"] + prev_est["cpu_exec"])
    return prev_est


def collect_snapshots(
    env: VecOffloadingEnv,
    episodes: int,
    max_steps: int,
    snapshots: int,
    seed: int,
    min_step: int = 0,
    compare_prev_sojourn: bool = True,
    max_snapshots_per_episode: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    snapshot_id = 0
    for ep in range(episodes):
        ep_snapshot_count = 0
        obs_list, _ = env.reset(seed=seed + ep)
        done = False
        step = 0
        while not done and step < max_steps and snapshot_id < snapshots:
            should_collect = step >= min_step
            for obs, veh in zip(obs_list, env.vehicles):
                if not should_collect:
                    continue
                if max_snapshots_per_episode is not None and ep_snapshot_count >= int(max_snapshots_per_episode):
                    continue
                subtask_idx = _choose_ready_subtask(env, veh)
                if subtask_idx is None:
                    continue
                dag = veh.task_dag
                criticality = dag.compute_all_priorities() if hasattr(dag, "compute_all_priorities") else np.zeros(dag.num_subtasks, dtype=np.float32)
                phi_weight = float(env._get_phi_subtask_weight(dag, subtask_idx)) if hasattr(env, "_get_phi_subtask_weight") else 1.0
                episode_domain_params = dict(getattr(env, "_episode_domain_params", {}) or {})
                cand_types = np.asarray(obs.get("candidate_types"), dtype=np.int64)
                cand_ids = np.asarray(obs.get("candidate_ids"), dtype=np.int64)
                action_mask = np.asarray(obs.get("action_mask"), dtype=bool)
                resource_raw = np.asarray(obs.get("resource_raw"), dtype=np.float32)
                task_comp = float(env._get_remaining_cycles(dag, subtask_idx))
                task_data = float(env._get_upload_bytes(dag, subtask_idx))
                comm_wait_dict = env._compute_comm_wait(veh.id)
                active_v2i_count, active_v2i_vehicles, active_v2v_vehicles = env._get_active_tx_context_from_queues()

                candidate_rows = []
                for idx, valid in enumerate(action_mask):
                    if not bool(valid):
                        continue
                    slot_type = int(cand_types[idx]) if idx < len(cand_types) else 0
                    slot_id = int(cand_ids[idx]) if idx < len(cand_ids) else -1
                    target = _target_from_slot(slot_type, slot_id)
                    est = env._estimate_snapshot_target_cost(
                        veh,
                        subtask_idx,
                        target,
                        task_comp=task_comp,
                        task_data=task_data,
                        comm_wait_dict=comm_wait_dict,
                        active_v2i_count=active_v2i_count,
                        active_v2i_vehicles=active_v2i_vehicles,
                        active_v2v_vehicles=active_v2v_vehicles,
                    )
                    prev_est = _estimate_prev_sojourn_target_cost(
                        env,
                        veh,
                        subtask_idx,
                        target,
                        task_comp=task_comp,
                        task_data=task_data,
                        comm_wait_dict=comm_wait_dict,
                        active_v2i_count=active_v2i_count,
                        active_v2i_vehicles=active_v2i_vehicles,
                        active_v2v_vehicles=active_v2v_vehicles,
                    ) if compare_prev_sojourn else est
                    rr = resource_raw[idx] if idx < len(resource_raw) else np.zeros(resource_raw.shape[1], dtype=np.float32)
                    candidate_rows.append({
                        "snapshot_id": snapshot_id,
                        "episode": ep,
                        "step": step,
                        "workload_level": str(episode_domain_params.get("workload_level", "unknown")),
                        "traffic_density": str(episode_domain_params.get("traffic_density", "unknown")),
                        "helper_availability": str(episode_domain_params.get("helper_availability", "unknown")),
                        "contact_stability": str(episode_domain_params.get("contact_stability", "unknown")),
                        "rsu_accessibility_or_load": str(episode_domain_params.get("rsu_accessibility_or_load", "unknown")),
                        "vehicle_id": int(veh.id),
                        "subtask_id": int(subtask_idx),
                        "mode": _mode_label(slot_type),
                        "target": _target_label(target),
                        "slot_idx": int(idx),
                        "criticality": float(criticality[subtask_idx]) if subtask_idx < len(criticality) else 0.0,
                        "is_sink": int(dag.out_degree[subtask_idx] == 0) if subtask_idx < len(dag.out_degree) else 0,
                        "l_bwd": int(dag.L_bwd[subtask_idx]) if getattr(dag, "L_bwd", None) is not None and subtask_idx < len(dag.L_bwd) else 0,
                        "deadline_rem": float(env._get_deadline_remaining_seconds(veh)),
                        "task_comp": task_comp,
                        "task_data": task_data,
                        "phi_weight": phi_weight,
                        "phi_total": float(env._estimate_decision_phi(veh)),
                        "cpu_norm": float(rr[0]) if rr.size > 0 else 0.0,
                        "comp_backlog_norm": float(rr[1]) if rr.size > 1 else 0.0,
                        "tx_backlog_norm": float(rr[2]) if rr.size > 2 else 0.0,
                        "dist_norm": float(rr[3]) if rr.size > 3 else 0.0,
                        "contact_norm": float(rr[7]) if rr.size > 7 else 0.0,
                        "contention_norm": float(rr[8]) if rr.size > 8 else 0.0,
                        "occupancy_norm": float(rr[9]) if rr.size > 9 else 0.0,
                        "wait_norm": float(rr[15]) if rr.size > 15 else 0.0,
                        "J": float(est.get("J", float("inf"))),
                        "comm_wait": float(est.get("comm_wait", 0.0)),
                        "tx_time": float(est.get("tx_time", 0.0)),
                        "cpu_wait": float(est.get("cpu_wait", 0.0)),
                        "cpu_exec": float(est.get("cpu_exec", 0.0)),
                        "contact_penalty": float(est.get("contact_penalty", 0.0)),
                        "delta_ext": float(est.get("delta_ext", 0.0)),
                        "queue_workload": float(est.get("queue_workload", 0.0)),
                        "arrival_workload": float(est.get("arrival_workload", 0.0)),
                        "service_rate": float(est.get("service_rate", 0.0)),
                        "prev_sojourn_J": float(prev_est.get("J", float("inf"))),
                        "prev_sojourn_cpu_wait": float(prev_est.get("cpu_wait", 0.0)),
                        "prev_sojourn_cpu_exec": float(prev_est.get("cpu_exec", 0.0)),
                        "weighted_J": float(phi_weight * est.get("J", float("inf"))),
                    })
                if candidate_rows:
                    best_row = min(candidate_rows, key=lambda r: r["J"])
                    for row in candidate_rows:
                        row["best_target"] = best_row["target"]
                        row["best_mode"] = best_row["mode"]
                        row["gap_to_best"] = float(row["J"] - best_row["J"])
                    rows.extend(candidate_rows)
                    snapshot_id += 1
                    ep_snapshot_count += 1
                    if snapshot_id >= snapshots:
                        break
            if snapshot_id >= snapshots:
                break
            actions = [_random_action(obs, rng) for obs in obs_list]
            obs_list, _, terminated, truncated, _ = env.step(actions)
            done = bool(terminated) or bool(truncated)
            step += 1
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "no snapshots collected"
    best_rows = {}
    for row in rows:
        sid = int(row["snapshot_id"])
        if sid not in best_rows or float(row["J"]) < float(best_rows[sid]["J"]):
            best_rows[sid] = row
    mode_counts = {"local": 0, "rsu": 0, "v2v": 0}
    dominant_comp = {"cpu_exec": 0, "cpu_wait": 0, "tx_time": 0, "comm_wait": 0, "contact_penalty": 0, "delta_ext": 0}
    rank_changed = 0
    lines = []
    for sid in sorted(best_rows.keys()):
        row = best_rows[sid]
        mode_counts[row["mode"]] = mode_counts.get(row["mode"], 0) + 1
        peers = [r for r in rows if int(r["snapshot_id"]) == sid]
        prev_best = None
        finite_prev = [r for r in peers if np.isfinite(float(r.get("prev_sojourn_J", float("inf"))))]
        if finite_prev:
            prev_best = min(finite_prev, key=lambda r: float(r.get("prev_sojourn_J", float("inf"))))
            if str(prev_best.get("target")) != str(row.get("target")) or str(prev_best.get("mode")) != str(row.get("mode")):
                rank_changed += 1
        if len(peers) > 1:
            alt = min([r for r in peers if r["target"] != row["target"]], key=lambda r: r["J"], default=None)
            if alt is not None:
                deltas = {
                    "cpu_exec": float(alt["cpu_exec"]) - float(row["cpu_exec"]),
                    "cpu_wait": float(alt["cpu_wait"]) - float(row["cpu_wait"]),
                    "tx_time": float(alt["tx_time"]) - float(row["tx_time"]),
                    "comm_wait": float(alt["comm_wait"]) - float(row["comm_wait"]),
                    "contact_penalty": float(alt["contact_penalty"]) - float(row["contact_penalty"]),
                    "delta_ext": float(alt["delta_ext"]) - float(row["delta_ext"]),
                }
                dominant = max(deltas.items(), key=lambda kv: abs(kv[1]))[0]
                dominant_comp[dominant] += 1
        lines.append(
            f"snapshot={sid} veh={row['vehicle_id']} subtask={row['subtask_id']} mode={row['mode']} target={row['target']} "
            f"J={row['J']:.3f} w={row['phi_weight']:.3f} crit={row['criticality']:.3f} sink={row['is_sink']} Lbwd={row['l_bwd']} "
            f"wait={row['wait_norm']:.3f} backlog={row['comp_backlog_norm']:.3f}/{row['tx_backlog_norm']:.3f} "
            f"prev_best={prev_best['mode']}:{prev_best['target']}" if prev_best is not None else
            f"snapshot={sid} veh={row['vehicle_id']} subtask={row['subtask_id']} mode={row['mode']} target={row['target']} "
            f"J={row['J']:.3f} w={row['phi_weight']:.3f} crit={row['criticality']:.3f} sink={row['is_sink']} Lbwd={row['l_bwd']} "
            f"wait={row['wait_norm']:.3f} backlog={row['comp_backlog_norm']:.3f}/{row['tx_backlog_norm']:.3f}"
        )
    lines.append(
        "best_mode_counts " + " ".join(f"{k}={v}" for k, v in mode_counts.items())
    )
    lines.append(
        "dominant_gap_component " + " ".join(f"{k}={v}" for k, v in dominant_comp.items())
    )
    lines.append(f"rank_changed_snapshots {rank_changed}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Diagnose phi vs actor-visible features on real env snapshots.")
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/phi_diag_20260311/phi_alignment.csv")
    args = parser.parse_args()

    env = VecOffloadingEnv(Cfg)
    rows = collect_snapshots(env, args.episodes, args.max_steps, args.snapshots, args.seed, min_step=args.min_step)
    out_path = Path(args.out)
    write_csv(rows, out_path)
    print(summarize(rows))
    print(f"csv={out_path}")


if __name__ == "__main__":
    main()
