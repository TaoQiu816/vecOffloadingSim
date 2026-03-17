#!/usr/bin/env python3
"""White-box consistency audit for snapshot latency semantics.

Purpose:
- No training.
- Roll a few short environment episodes.
- For the same state/action, align:
  1) candidate gate / ranking proxy
  2) phi candidate cost
  3) _estimate_t_actual() pre/post-commit estimate
- Report whether latency semantics are inconsistent across modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


@dataclass
class TargetRef:
    slot_idx: int
    target: Any
    mode: str
    target_id: int


def _target_from_slot(slot_type: int, slot_id: int) -> Any:
    if int(slot_type) == 1:
        return "Local"
    if int(slot_type) == 2 and int(slot_id) >= 0:
        return ("RSU", int(slot_id))
    if int(slot_type) == 3 and int(slot_id) >= 0:
        return int(slot_id)
    return None


def _target_label(target: Any) -> str:
    if target is None:
        return "None"
    if target == "Local":
        return "Local"
    if isinstance(target, tuple) and len(target) >= 2 and target[0] == "RSU":
        return f"RSU:{int(target[1])}"
    if isinstance(target, int):
        return f"V2V:{int(target)}"
    return str(target)


def _target_key(target: Any) -> Tuple[str, int]:
    if target == "Local":
        return ("local", -1)
    if isinstance(target, tuple) and len(target) >= 2 and target[0] == "RSU":
        return ("rsu", int(target[1]))
    if isinstance(target, int):
        return ("v2v", int(target))
    return ("none", -1)


def _mode_from_target(target: Any) -> str:
    return _target_key(target)[0]


def _target_id(target: Any) -> int:
    return _target_key(target)[1]


def _valid_ready_subtasks(obs: Dict[str, Any]) -> np.ndarray:
    return np.flatnonzero(np.asarray(obs.get("subtask_mask"), dtype=bool))


def _choose_subtask(obs: Dict[str, Any], rng: np.random.Generator, prefer_non_top: bool) -> Optional[int]:
    ready = _valid_ready_subtasks(obs)
    if ready.size <= 0:
        return None
    top = obs.get("subtask_index")
    try:
        top_idx = int(top)
    except Exception:
        top_idx = -1
    if prefer_non_top:
        non_top = ready[ready != top_idx]
        if non_top.size > 0:
            return int(rng.choice(non_top))
    return int(rng.choice(ready))


def _choose_target(obs: Dict[str, Any], rng: np.random.Generator) -> int:
    valid = np.flatnonzero(np.asarray(obs.get("action_mask"), dtype=bool))
    if valid.size <= 0:
        return 0
    return int(rng.choice(valid))


def _build_actions(
    obs_list: List[Dict[str, Any]],
    rng: np.random.Generator,
    prefer_non_top: bool,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for obs in obs_list:
        subtask = _choose_subtask(obs, rng, prefer_non_top=prefer_non_top)
        target = _choose_target(obs, rng)
        power = 0.0 if target == 0 else float(rng.uniform(0.2, 1.0))
        if subtask is None:
            actions.append({"subtask": 0, "target": 0, "power": 1.0})
        else:
            actions.append({"subtask": int(subtask), "target": int(target), "power": float(power)})
    return actions


def _candidate_targets_from_obs(obs: Dict[str, Any]) -> List[TargetRef]:
    refs: List[TargetRef] = []
    candidate_types = np.asarray(obs.get("candidate_types", []), dtype=np.int64).reshape(-1)
    candidate_ids = np.asarray(obs.get("candidate_ids", []), dtype=np.int64).reshape(-1)
    for idx in range(min(len(candidate_types), len(candidate_ids))):
        target = _target_from_slot(int(candidate_types[idx]), int(candidate_ids[idx]))
        if target is None:
            continue
        refs.append(
            TargetRef(
                slot_idx=int(idx),
                target=target,
                mode=_mode_from_target(target),
                target_id=_target_id(target),
            )
        )
    return refs


def _active_ctx(env: VecOffloadingEnv):
    return env._get_active_tx_context_from_queues()


def _estimate_snapshot_cost(
    env: VecOffloadingEnv,
    vehicle,
    subtask_idx: int,
    target: Any,
    task_comp: float,
    task_data: float,
    comm_wait_dict: Dict[str, float],
):
    active_v2i_count, active_v2i_vehicles, active_v2v_vehicles = _active_ctx(env)
    return env._estimate_snapshot_target_cost(
        vehicle,
        int(subtask_idx),
        target,
        task_comp=float(task_comp),
        task_data=float(task_data),
        comm_wait_dict=comm_wait_dict,
        active_v2i_count=active_v2i_count,
        active_v2i_vehicles=active_v2i_vehicles,
        active_v2v_vehicles=active_v2v_vehicles,
    )


def _build_gate_info(
    env: VecOffloadingEnv,
    vehicle,
    subtask_idx: int,
    target: Any,
    est: Dict[str, Any],
) -> Dict[str, Any]:
    rem_deadline = float(env._get_deadline_remaining_seconds(vehicle))
    info: Dict[str, Any] = {
        "remaining_deadline": rem_deadline,
        "t_tx_est": float(est.get("tx_time", 0.0)),
        "t_exec_est": float(est.get("cpu_exec", 0.0)),
        "predicted_finish_time": float(est.get("J", float("inf"))),
        "total_time": float(est.get("J", float("inf"))),
    }
    if isinstance(target, tuple) and len(target) >= 2 and target[0] == "RSU":
        info["t_queue_est"] = float(est.get("cpu_wait", 0.0))
        info["rsu_queue_wait_est"] = float(est.get("cpu_wait", 0.0))
    elif isinstance(target, int):
        contact_penalty, contact_time, _ = env._estimate_contact_penalty_snapshot(
            vehicle,
            target,
            float(est.get("comm_wait", 0.0)) + float(est.get("tx_time", 0.0)),
        )
        _ = contact_penalty
        info["helper_queue_wait_est"] = float(est.get("cpu_wait", 0.0))
        info["queue_wait"] = float(est.get("cpu_wait", 0.0))
        info["contact_window_est"] = float(contact_time)
        info["contact_time"] = float(contact_time)
    return info


def _gate_reason_rsu(cfg, info: Dict[str, Any]) -> List[str]:
    rem_deadline = max(float(info.get("remaining_deadline", 0.0)), 0.0)
    t_tx = max(float(info.get("t_tx_est", 0.0)), 0.0)
    t_exec = max(float(info.get("t_exec_est", 0.0)), 0.0)
    q_wait = max(float(info.get("rsu_queue_wait_est", info.get("queue_wait", 0.0))), 0.0)
    finish = max(float(info.get("predicted_finish_time", info.get("total_time", float("inf")))), 0.0)
    ratio = float(getattr(cfg, "RSU_CAND_DEADLINE_RATIO", 1.0))
    max_wait = float(getattr(cfg, "RSU_CAND_MAX_WAIT", float("inf")))
    soft = float(getattr(cfg, "RSU_CAND_ALLOW_SOFT_SLACK", 0.0))

    reasons: List[str] = []
    if rem_deadline <= 0.0:
        reasons.append("deadline_nonpositive")
    if (t_tx + q_wait + t_exec) >= ratio * rem_deadline:
        reasons.append("deadline_ratio")
    if q_wait >= max_wait:
        reasons.append("max_wait")
    if finish >= rem_deadline + soft:
        reasons.append("finish_slack")
    return reasons


def _gate_reason_v2v(cfg, info: Dict[str, Any]) -> List[str]:
    rem_deadline = max(float(info.get("remaining_deadline", 0.0)), 0.0)
    t_tx = max(float(info.get("t_tx_est", 0.0)), 0.0)
    t_exec = max(float(info.get("t_exec_est", 0.0)), 0.0)
    q_wait = max(float(info.get("helper_queue_wait_est", info.get("queue_wait", 0.0))), 0.0)
    contact = max(float(info.get("contact_window_est", info.get("contact_time", 0.0))), 0.0)
    finish = max(float(info.get("predicted_finish_time", info.get("total_time", float("inf")))), 0.0)
    ratio = float(getattr(cfg, "V2V_CAND_DEADLINE_RATIO", 1.0))
    margin = float(getattr(cfg, "V2V_CAND_CONTACT_MARGIN", 1.0))
    max_wait = float(getattr(cfg, "V2V_CAND_MAX_WAIT", float("inf")))
    soft = float(getattr(cfg, "V2V_CAND_ALLOW_SOFT_SLACK", 0.0))

    reasons: List[str] = []
    if rem_deadline <= 0.0:
        reasons.append("deadline_nonpositive")
    if (t_tx + t_exec) >= ratio * rem_deadline:
        reasons.append("deadline_ratio")
    if contact <= margin * t_tx:
        reasons.append("contact_margin")
    if q_wait >= max_wait:
        reasons.append("max_wait")
    if finish >= rem_deadline + soft:
        reasons.append("finish_slack")
    return reasons


def _decompose_t_actual(
    env: VecOffloadingEnv,
    vehicle,
    subtask_idx: int,
    target: Any,
    cycles: float,
) -> Dict[str, float]:
    freq_self = max(float(getattr(vehicle, "cpu_freq", env.config.MIN_VEHICLE_CPU_FREQ)), 1e-9)
    dag = vehicle.task_dag
    din = float(env._get_upload_bytes(dag, int(subtask_idx)))
    comm_wait_dict = env._compute_comm_wait(vehicle.id)
    eps_rate = float(getattr(env.config, "EPS_RATE", 1e-9))
    if target is None or target == "Local":
        cpu_wait = float(env._get_veh_queue_wait_time(vehicle.id, freq_self))
        exec_time = float(cycles / freq_self)
        return {
            "comm_wait": 0.0,
            "tx_time": 0.0,
            "cpu_wait": cpu_wait,
            "exec_time": exec_time,
            "total_cost": float(cpu_wait + exec_time),
        }
    if isinstance(target, tuple) and len(target) >= 2 and target[0] == "RSU":
        rsu_id = int(target[1])
        rate = float(env._get_rate_from_snapshot(("VEH", vehicle.id), ("RSU", rsu_id), "V2I"))
        rate = max(rate, eps_rate)
        tx_time = float(din / rate) if din > 0 else 0.0
        tx_timeout = float(getattr(env.config, "TX_TIMEOUT_SECONDS", 2.0))
        if tx_timeout > 0 and tx_time > tx_timeout:
            tx_time = tx_timeout
        cpu_wait = float(env._get_rsu_queue_wait_time(rsu_id))
        exec_freq = float(env.rsus[rsu_id].cpu_freq) if 0 <= rsu_id < len(env.rsus) else float(env.config.F_RSU)
        exec_time = float(cycles / max(exec_freq, 1e-9))
        comm_wait = float(comm_wait_dict.get("total_v2i", 0.0))
        return {
            "comm_wait": comm_wait,
            "tx_time": tx_time,
            "cpu_wait": cpu_wait,
            "exec_time": exec_time,
            "total_cost": float(comm_wait + tx_time + cpu_wait + exec_time),
        }
    if isinstance(target, int):
        rate = float(env._get_rate_from_snapshot(("VEH", vehicle.id), ("VEH", int(target)), "V2V"))
        rate = max(rate, eps_rate)
        tx_time = float(din / rate) if din > 0 else 0.0
        tx_timeout = float(getattr(env.config, "TX_TIMEOUT_SECONDS", 2.0))
        if tx_timeout > 0 and tx_time > tx_timeout:
            tx_time = tx_timeout
        helper = env._get_vehicle_by_id(int(target))
        exec_freq = float(getattr(helper, "cpu_freq", env.config.MIN_VEHICLE_CPU_FREQ)) if helper is not None else float(env.config.MIN_VEHICLE_CPU_FREQ)
        cpu_wait = float(env._get_veh_queue_wait_time(int(target), exec_freq)) if helper is not None else 0.0
        exec_time = float(cycles / max(exec_freq, 1e-9))
        comm_wait = float(comm_wait_dict.get("total_v2v", 0.0))
        return {
            "comm_wait": comm_wait,
            "tx_time": tx_time,
            "cpu_wait": cpu_wait,
            "exec_time": exec_time,
            "total_cost": float(comm_wait + tx_time + cpu_wait + exec_time),
        }
    return {
        "comm_wait": 0.0,
        "tx_time": 0.0,
        "cpu_wait": 0.0,
        "exec_time": 0.0,
        "total_cost": 0.0,
    }


def _rank_of_target(costs: Iterable[Tuple[str, float]], target_label: str) -> int:
    finite = [(label, float(cost)) for label, cost in costs if np.isfinite(cost)]
    finite.sort(key=lambda item: (item[1], item[0]))
    for idx, (label, _) in enumerate(finite, start=1):
        if label == target_label:
            return int(idx)
    return -1


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(decision_rows: List[Dict[str, Any]], soft_gate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "decision_count": int(len(decision_rows)),
        "soft_gate_mask_count": int(len(soft_gate_rows)),
    }
    if not decision_rows:
        return summary

    chosen_gate_mismatch = [int(bool(r["chosen_vs_gate_subtask_mismatch"])) for r in decision_rows]
    summary["chosen_vs_gate_subtask_mismatch_rate"] = float(np.mean(chosen_gate_mismatch))

    by_mode: Dict[str, Dict[str, float]] = {}
    for mode in ("local", "rsu", "v2v"):
        rows = [r for r in decision_rows if r["selected_mode"] == mode]
        if not rows:
            continue
        gate_delta = [float(r["gate_total_cost_for_selected"]) - float(r["phi_total_cost_for_selected"]) for r in rows if np.isfinite(r["gate_total_cost_for_selected"]) and np.isfinite(r["phi_total_cost_for_selected"])]
        pre_delta = [float(r["t_actual_pre_total"]) - float(r["phi_total_cost_for_selected"]) for r in rows if np.isfinite(r["t_actual_pre_total"]) and np.isfinite(r["phi_total_cost_for_selected"])]
        post_delta = [float(r["t_actual_post_total"]) - float(r["phi_total_cost_for_selected"]) for r in rows if np.isfinite(r["t_actual_post_total"]) and np.isfinite(r["phi_total_cost_for_selected"])]
        rows_rank = [r for r in rows if int(r["phi_rank_selected"]) > 0 and int(r["gate_rank_selected"]) > 0]
        by_mode[mode] = {
            "count": int(len(rows)),
            "median_gate_minus_phi": float(np.median(gate_delta)) if gate_delta else float("nan"),
            "median_tactual_pre_minus_phi": float(np.median(pre_delta)) if pre_delta else float("nan"),
            "median_tactual_post_minus_phi": float(np.median(post_delta)) if post_delta else float("nan"),
            "rank_mismatch_rate": float(np.mean([int(r["phi_rank_selected"] != r["gate_rank_selected"]) for r in rows_rank])) if rows_rank else float("nan"),
        }
    summary["by_mode"] = by_mode

    if soft_gate_rows:
        counts: Dict[str, int] = {}
        for row in soft_gate_rows:
            for reason in str(row["gate_fail_reasons"]).split("|"):
                if not reason:
                    continue
                counts[reason] = counts.get(reason, 0) + 1
        summary["soft_gate_reasons"] = counts
    return summary


def collect_diagnostics(
    env: VecOffloadingEnv,
    episodes: int,
    max_steps: int,
    seed: int,
    prefer_non_top: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    decision_rows: List[Dict[str, Any]] = []
    soft_gate_rows: List[Dict[str, Any]] = []

    for ep in range(episodes):
        obs_list, _ = env.reset(seed=seed + ep)
        done = False
        step = 0
        while (not done) and step < max_steps:
            actions = _build_actions(obs_list, rng, prefer_non_top=prefer_non_top)
            plans = env._plan_actions_snapshot(actions)
            plan_by_index = {int(plan.get("index", -1)): plan for plan in plans}
            commit_plans = [p for p in plans if p.get("subtask_idx") is not None]
            env._capture_rate_snapshot(commit_plans)
            pending_rows: List[Dict[str, Any]] = []

            for obs_idx, obs in enumerate(obs_list):
                plan = plan_by_index.get(int(obs_idx))
                if plan is None:
                    continue
                vehicle = plan["vehicle"]
                if plan.get("subtask_idx") is None:
                    continue

                chosen_subtask = int(plan["subtask_idx"])
                gate_subtask_raw = obs.get("subtask_index", -1)
                try:
                    gate_subtask = int(gate_subtask_raw)
                except Exception:
                    gate_subtask = -1
                target = plan.get("planned_target")
                target_label = _target_label(target)
                if target is None:
                    continue

                dag = vehicle.task_dag
                task_comp = float(env._get_remaining_cycles(dag, chosen_subtask))
                task_data = float(env._get_upload_bytes(dag, chosen_subtask))
                comm_wait_dict = env._compute_comm_wait(vehicle.id)
                phi_est = _estimate_snapshot_cost(
                    env,
                    vehicle,
                    chosen_subtask,
                    target,
                    task_comp,
                    task_data,
                    comm_wait_dict,
                )

                gate_total = float("nan")
                gate_comm = float("nan")
                gate_wait = float("nan")
                gate_exec = float("nan")
                gate_penalty = float("nan")
                if gate_subtask >= 0 and gate_subtask < dag.num_subtasks:
                    gate_comp = float(env._get_remaining_cycles(dag, gate_subtask))
                    gate_data = float(env._get_upload_bytes(dag, gate_subtask))
                    gate_est = _estimate_snapshot_cost(
                        env,
                        vehicle,
                        gate_subtask,
                        target,
                        gate_comp,
                        gate_data,
                        comm_wait_dict,
                    )
                    gate_total = float(gate_est.get("J", float("nan")))
                    gate_comm = float(gate_est.get("comm_wait", float("nan")))
                    gate_wait = float(gate_est.get("cpu_wait", float("nan")))
                    gate_exec = float(gate_est.get("cpu_exec", float("nan")))
                    gate_penalty = float(gate_est.get("contact_penalty", float("nan")))

                candidate_refs = _candidate_targets_from_obs(obs)
                valid_slots = set(np.flatnonzero(np.asarray(obs.get("action_mask"), dtype=bool)))
                chosen_costs: List[Tuple[str, float]] = []
                gate_costs: List[Tuple[str, float]] = []
                for ref in candidate_refs:
                    if ref.slot_idx not in valid_slots:
                        continue
                    est = _estimate_snapshot_cost(
                        env,
                        vehicle,
                        chosen_subtask,
                        ref.target,
                        float(env._get_remaining_cycles(dag, chosen_subtask)),
                        float(env._get_upload_bytes(dag, chosen_subtask)),
                        comm_wait_dict,
                    )
                    chosen_costs.append((_target_label(ref.target), float(est.get("J", float("inf")))))
                    if gate_subtask >= 0 and gate_subtask < dag.num_subtasks:
                        est_gate = _estimate_snapshot_cost(
                            env,
                            vehicle,
                            gate_subtask,
                            ref.target,
                            float(env._get_remaining_cycles(dag, gate_subtask)),
                            float(env._get_upload_bytes(dag, gate_subtask)),
                            comm_wait_dict,
                        )
                        gate_costs.append((_target_label(ref.target), float(est_gate.get("J", float("inf")))))

                chosen_vs_gate_mismatch = int(gate_subtask >= 0 and chosen_subtask != gate_subtask)
                selected_mode = _mode_from_target(target)
                queue_proxy = {
                    "local": float(env._get_veh_queue_load(vehicle.id)),
                    "rsu": float(env._get_rsu_queue_load(_target_id(target))) if selected_mode == "rsu" else float("nan"),
                    "helper": float(env._get_veh_queue_load(_target_id(target))) if selected_mode == "v2v" else float("nan"),
                }

                pending_rows.append(
                    {
                        "episode": int(ep),
                        "step": int(step),
                        "vehicle_id": int(vehicle.id),
                        "chosen_subtask": int(chosen_subtask),
                        "gate_subtask": int(gate_subtask),
                        "chosen_vs_gate_subtask_mismatch": int(chosen_vs_gate_mismatch),
                        "selected_slot_idx": int(plan.get("target_idx", -1) if plan.get("target_idx") is not None else -1),
                        "selected_target": target_label,
                        "selected_mode": selected_mode,
                        "selected_target_id": int(_target_id(target)),
                        "illegal_reason": str(plan.get("illegal_reason") or ""),
                        "phi_total_cost_for_selected": float(phi_est.get("J", float("nan"))),
                        "phi_comm_wait": float(phi_est.get("comm_wait", float("nan"))),
                        "phi_tx_time": float(phi_est.get("tx_time", float("nan"))),
                        "phi_cpu_wait": float(phi_est.get("cpu_wait", float("nan"))),
                        "phi_exec_time": float(phi_est.get("cpu_exec", float("nan"))),
                        "phi_contact_penalty": float(phi_est.get("contact_penalty", float("nan"))),
                        "phi_queue_workload": float(phi_est.get("queue_workload", float("nan"))),
                        "phi_arrival_workload": float(phi_est.get("arrival_workload", float("nan"))),
                        "phi_service_rate": float(phi_est.get("service_rate", float("nan"))),
                        "gate_total_cost_for_selected": gate_total,
                        "gate_comm_wait": gate_comm,
                        "gate_cpu_wait": gate_wait,
                        "gate_exec_time": gate_exec,
                        "gate_contact_penalty": gate_penalty,
                        "phi_rank_selected": int(_rank_of_target(chosen_costs, target_label)),
                        "gate_rank_selected": int(_rank_of_target(gate_costs, target_label)),
                        "queue_proxy_local": queue_proxy["local"],
                        "queue_proxy_rsu": queue_proxy["rsu"],
                        "queue_proxy_helper": queue_proxy["helper"],
                    }
                )

                if gate_subtask >= 0 and gate_subtask < dag.num_subtasks:
                    for rsu in env._get_all_rsus_in_range(vehicle.pos):
                        rsu_id = int(rsu.id)
                        if env._is_rsu_queue_full(rsu_id, 0.0):
                            continue
                        target_rsu = ("RSU", rsu_id)
                        est = _estimate_snapshot_cost(
                            env,
                            vehicle,
                            gate_subtask,
                            target_rsu,
                            float(env._get_remaining_cycles(dag, gate_subtask)),
                            float(env._get_upload_bytes(dag, gate_subtask)),
                            comm_wait_dict,
                        )
                        info = _build_gate_info(env, vehicle, gate_subtask, target_rsu, est)
                        reasons = _gate_reason_rsu(env.config, info)
                        if reasons and (not env.candidate_manager._passes_rsu_gate(info)):
                            soft_gate_rows.append(
                                {
                                    "episode": int(ep),
                                    "step": int(step),
                                    "vehicle_id": int(vehicle.id),
                                    "mode": "rsu",
                                    "target": _target_label(target_rsu),
                                    "gate_subtask": int(gate_subtask),
                                    "gate_fail_reasons": "|".join(reasons),
                                }
                            )
                    for other in env.vehicles:
                        if int(other.id) == int(vehicle.id):
                            continue
                        dist = float(np.linalg.norm(np.asarray(other.pos, dtype=float) - np.asarray(vehicle.pos, dtype=float)))
                        if dist > float(env.config.V2V_RANGE):
                            continue
                        if env._is_veh_queue_full(int(other.id), 0.0):
                            continue
                        target_v2v = int(other.id)
                        est = _estimate_snapshot_cost(
                            env,
                            vehicle,
                            gate_subtask,
                            target_v2v,
                            float(env._get_remaining_cycles(dag, gate_subtask)),
                            float(env._get_upload_bytes(dag, gate_subtask)),
                            comm_wait_dict,
                        )
                        info = _build_gate_info(env, vehicle, gate_subtask, target_v2v, est)
                        reasons = _gate_reason_v2v(env.config, info)
                        if reasons and (not env.candidate_manager._passes_v2v_gate(info)):
                            soft_gate_rows.append(
                                {
                                    "episode": int(ep),
                                    "step": int(step),
                                    "vehicle_id": int(vehicle.id),
                                    "mode": "v2v",
                                    "target": _target_label(target_v2v),
                                    "gate_subtask": int(gate_subtask),
                                    "gate_fail_reasons": "|".join(reasons),
                                }
                            )

            t_actual_records: Dict[Tuple[int, int, str, str], Dict[str, float]] = {}
            phase_state = {"phase": "pre_commit"}
            orig_estimate_t_actual = env._estimate_t_actual
            orig_commit = env._phase1_commit_offload_decisions

            def wrapped_estimate_t_actual(vehicle, subtask_idx, target, cycles, power_ratio=1.0):
                total, tx = orig_estimate_t_actual(vehicle, subtask_idx, target, cycles, power_ratio)
                parts = _decompose_t_actual(env, vehicle, int(subtask_idx), target, float(cycles))
                parts["total_cost"] = float(total)
                parts["tx_time"] = float(tx)
                t_actual_records[(int(vehicle.id), int(subtask_idx), _target_label(target), phase_state["phase"])] = parts
                return total, tx

            def wrapped_commit(plans_to_commit):
                result = orig_commit(plans_to_commit)
                phase_state["phase"] = "post_commit"
                return result

            env._estimate_t_actual = wrapped_estimate_t_actual
            env._phase1_commit_offload_decisions = wrapped_commit
            try:
                next_obs_list, _, terminated, truncated, _ = env.step(actions)
            finally:
                env._estimate_t_actual = orig_estimate_t_actual
                env._phase1_commit_offload_decisions = orig_commit

            for row in pending_rows:
                vid = int(row["vehicle_id"])
                chosen_subtask = int(row["chosen_subtask"])
                target_label = str(row["selected_target"])
                t_pre = t_actual_records.get((vid, chosen_subtask, target_label, "pre_commit"), {})
                t_post = t_actual_records.get((vid, chosen_subtask, target_label, "post_commit"), {})
                decision_rows.append(
                    {
                        **row,
                        "t_actual_pre_total": float(t_pre.get("total_cost", float("nan"))),
                        "t_actual_pre_comm_wait": float(t_pre.get("comm_wait", float("nan"))),
                        "t_actual_pre_tx_time": float(t_pre.get("tx_time", float("nan"))),
                        "t_actual_pre_cpu_wait": float(t_pre.get("cpu_wait", float("nan"))),
                        "t_actual_pre_exec_time": float(t_pre.get("exec_time", float("nan"))),
                        "t_actual_post_total": float(t_post.get("total_cost", float("nan"))),
                        "t_actual_post_comm_wait": float(t_post.get("comm_wait", float("nan"))),
                        "t_actual_post_tx_time": float(t_post.get("tx_time", float("nan"))),
                        "t_actual_post_cpu_wait": float(t_post.get("cpu_wait", float("nan"))),
                        "t_actual_post_exec_time": float(t_post.get("exec_time", float("nan"))),
                    }
                )

            env._clear_rate_snapshot()
            obs_list = next_obs_list
            done = bool(terminated) or bool(truncated)
            step += 1

    return decision_rows, soft_gate_rows, _summary(decision_rows, soft_gate_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-non-top", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("audit_results/latency_consistency"))
    args = parser.parse_args()

    env = VecOffloadingEnv(Cfg)
    decision_rows, soft_gate_rows, summary = collect_diagnostics(
        env,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        prefer_non_top=bool(args.prefer_non_top),
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(decision_rows, out_dir / "decision_alignment.csv")
    _write_csv(soft_gate_rows, out_dir / "soft_gate_masked.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"decision_rows={len(decision_rows)}")
    print(f"soft_gate_rows={len(soft_gate_rows)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
