import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from agents.mappo_agent import MAPPOAgent

try:
    from scripts.v2v_exec_audit import _load_mappo_network
except Exception:  # pragma: no cover - fallback if import fails
    _load_mappo_network = None


def _select_index_uniform(mask, rng):
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    return int(rng.choice(valid))


def _select_type_balanced(mask, rng):
    types = []
    if mask[0]:
        types.append("local")
    if len(mask) > 1 and mask[1]:
        types.append("rsu")
    if np.any(mask[2:]):
        types.append("v2v")
    if not types:
        return None
    choice = rng.choice(types)
    if choice == "local":
        return 0
    if choice == "rsu":
        return 1
    v2v_indices = np.where(mask[2:])[0]
    return int(rng.choice(v2v_indices) + 2) if len(v2v_indices) > 0 else None


def _calc_soft_and_hard(env, v, target, task_comp):
    soft_penalty = 0.0
    hard_triggered = False
    dist = None

    if env._is_rsu_location(target):
        in_range = False
        rsu_dist = float("inf")
        if isinstance(target, tuple) and len(target) == 2:
            rsu_id = target[1]
            if 0 <= rsu_id < len(env.rsus):
                in_range = env.rsus[rsu_id].is_in_coverage(v.pos)
                rsu_dist = np.linalg.norm(v.pos - env.rsus[rsu_id].position)
        else:
            if len(env.rsus) > 0:
                for rsu in env.rsus:
                    if rsu.is_in_coverage(v.pos):
                        in_range = True
                        rsu_dist = min(rsu_dist, np.linalg.norm(v.pos - rsu.position))
            else:
                rsu_dist = np.linalg.norm(v.pos - Cfg.RSU_POS)
                in_range = (rsu_dist <= Cfg.RSU_RANGE)
        dist = rsu_dist
        if not in_range:
            hard_triggered = True
        else:
            if Cfg.DIST_PENALTY_MODE != "off":
                safe_dist = Cfg.RSU_RANGE * Cfg.DIST_SAFE_FACTOR
                if rsu_dist > safe_dist:
                    dist_ratio = (rsu_dist - safe_dist) / (Cfg.RSU_RANGE - safe_dist + 1e-6)
                    dist_ratio = np.clip(dist_ratio, 0.0, 1.0)
                    soft_penalty += -Cfg.DIST_PENALTY_WEIGHT * (dist_ratio ** Cfg.DIST_SENSITIVITY)
    elif isinstance(target, int):
        t_veh = env._get_vehicle_by_id(target)
        if t_veh is None:
            hard_triggered = True
        else:
            dist = np.linalg.norm(v.pos - t_veh.pos)
            if dist > Cfg.V2V_RANGE:
                hard_triggered = True
            else:
                if Cfg.DIST_PENALTY_MODE != "off":
                    safe_dist = Cfg.V2V_RANGE * Cfg.DIST_SAFE_FACTOR
                    if dist > safe_dist:
                        dist_ratio = (dist - safe_dist) / (Cfg.V2V_RANGE - safe_dist + 1e-6)
                        dist_ratio = np.clip(dist_ratio, 0.0, 1.0)
                        soft_penalty += -Cfg.DIST_PENALTY_WEIGHT * (dist_ratio ** Cfg.DIST_SENSITIVITY)

    if not hard_triggered:
        q_after_load = 0.0
        q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
        if target == "Local":
            q_after_load = v.task_queue.get_total_load() + task_comp
            q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
        elif env._is_rsu_location(target):
            if isinstance(target, tuple) and len(target) == 2:
                rsu_id = target[1]
                if 0 <= rsu_id < len(env.rsus):
                    q_after_load = env.rsus[rsu_id].queue_manager.get_total_load() + task_comp
                else:
                    q_after_load = task_comp
                q_max_load = Cfg.RSU_QUEUE_CYCLES_LIMIT
            else:
                q_after_load = (sum([rsu.queue_manager.get_total_load() for rsu in env.rsus]) + task_comp) if len(env.rsus) > 0 else task_comp
                q_max_load = Cfg.RSU_QUEUE_CYCLES_LIMIT * len(env.rsus) if len(env.rsus) > 0 else Cfg.RSU_QUEUE_CYCLES_LIMIT
        elif isinstance(target, int):
            t_veh = env._get_vehicle_by_id(target)
            if t_veh is not None:
                q_after_load = t_veh.task_queue.get_total_load() + task_comp
                q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
        if q_after_load > q_max_load:
            hard_triggered = True

    return soft_penalty, hard_triggered, dist


def _compute_cost(env, v, task_idx, target, v2i_user_count):
    dag = v.task_dag
    task_comp = dag.total_comp[task_idx]
    task_data = dag.total_data[task_idx]

    queue_wait = 0.0
    cpu_freq = v.cpu_freq
    tx_time = 0.0
    rate = 0.0

    if target == "Local":
        queue_wait = v.task_queue.get_estimated_wait_time(v.cpu_freq)
        cpu_freq = v.cpu_freq
        tx_time = 0.0
        max_rate = env._get_norm_rate("V2I")
    elif env._is_rsu_location(target):
        rsu_id = env._get_rsu_id_from_location(target)
        if rsu_id is not None and 0 <= rsu_id < len(env.rsus):
            rsu = env.rsus[rsu_id]
            queue_wait = rsu.get_estimated_wait_time()
            cpu_freq = rsu.cpu_freq
            rate = env.channel.compute_one_rate(
                v, rsu.position, "V2I", env.time, v2i_user_count=v2i_user_count
            )
            rate = max(rate, 1e-6)
            tx_time = task_data / rate if task_data > 0 else 0.0
        max_rate = env._get_norm_rate("V2I")
    elif isinstance(target, int):
        t_veh = env._get_vehicle_by_id(target)
        if t_veh is not None:
            queue_wait = t_veh.task_queue.get_estimated_wait_time(t_veh.cpu_freq)
            cpu_freq = t_veh.cpu_freq
            rate = env.channel.compute_one_rate(v, t_veh.pos, "V2V", env.time)
            rate = max(rate, 1e-6)
            tx_time = task_data / rate if task_data > 0 else 0.0
        max_rate = env._get_norm_rate("V2V")
    else:
        max_rate = env._get_norm_rate("V2I")

    comp_time = task_comp / max(cpu_freq, 1e-6)
    max_tx_time = task_data / max(max_rate, 1e-6) if task_data > 0 else 1.0
    max_comp_time = task_comp / max(Cfg.MIN_VEHICLE_CPU_FREQ, 1e-6)
    delay_norm = (tx_time / max(max_tx_time, 1e-6) +
                  queue_wait / max(Cfg.NORM_MAX_WAIT_TIME, 1e-6) +
                  comp_time / max(max_comp_time, 1e-6))

    energy_norm = 0.0
    if tx_time > 0 and target != "Local":
        tx_power_w = Cfg.dbm2watt(v.tx_power_dbm)
        max_power_w = Cfg.dbm2watt(Cfg.TX_POWER_MAX_DBM)
        max_energy = max_power_w * max(max_tx_time, 1e-6)
        energy_norm = (tx_power_w * tx_time) / max(max_energy, 1e-6)

    r_timeout = 0.0
    if dag.deadline > 0:
        elapsed = env.time - dag.start_time
        if elapsed > dag.deadline and not dag.is_finished:
            overtime_ratio = (elapsed - dag.deadline) / dag.deadline
            r_timeout = -Cfg.TIMEOUT_PENALTY_WEIGHT * np.tanh(Cfg.TIMEOUT_STEEPNESS * overtime_ratio)

    soft_penalty, hard_triggered, dist = _calc_soft_and_hard(env, v, target, task_comp)
    reward = (-Cfg.DELAY_WEIGHT * delay_norm -
              Cfg.ENERGY_WEIGHT * energy_norm +
              soft_penalty +
              r_timeout)
    cost = -reward

    return {
        "tx_time": tx_time,
        "wait_time": queue_wait,
        "comp_time": comp_time,
        "delay_norm": delay_norm,
        "energy_norm": energy_norm,
        "dist_penalty": soft_penalty,
        "reward": reward,
        "cost": cost,
        "rate": rate,
        "hard_triggered": hard_triggered,
        "dist": dist,
    }


def _summary_stats(values):
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _write_md(path, sections):
    with open(path, "w", encoding="utf-8") as f:
        for title, content in sections:
            f.write(f"## {title}\n")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
            f.write("\n")


def run_audit(episodes=20, steps=50, seed=0, out_dir="results_dbg/decision_dominance",
              mappo_ckpt=None, policy_rollout="type_balanced_random", device="cpu"):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "decision_dominance_audit.csv")
    md_path = os.path.join(out_dir, "decision_dominance_audit.md")

    env = VecOffloadingEnv()
    rng = np.random.RandomState(seed)

    mappo_agent = None
    if device == "cuda" and not torch.cuda.is_available():
        print("[Audit] cuda requested but not available; falling back to cpu")
        device = "cpu"

    if mappo_ckpt:
        if _load_mappo_network is None:
            raise RuntimeError("MAPPO ckpt loader not available; cannot run mappo audit")
        net = _load_mappo_network(mappo_ckpt)
        mappo_agent = MAPPOAgent(net, device=device)

    argmin_counts = defaultdict(int)
    argmin_total = 0
    v2v_available = 0
    v2v_not_argmin = 0
    delta_delay = []
    delta_energy = []
    delta_v2v_vs_best = []
    worst_v2v = []

    conf_mat = defaultdict(lambda: defaultdict(int))
    chosen_counts = defaultdict(int)
    chosen_total = 0
    chosen_match = 0
    csv_rows = []
    mask_counts = {"local": 0, "rsu": 0, "v2v_slots": 0}
    samples_total = 0

    for ep in range(episodes):
        obs_list, _ = env.reset(seed=seed + ep)
        for step in range(steps):
            v2i_users = env._estimate_v2i_users()
            mappo_actions = None
            if mappo_agent is not None:
                out = mappo_agent.select_action(obs_list, deterministic=True)
                mappo_actions = out["actions"]
                assert len(mappo_actions) == len(obs_list) == len(env.vehicles), "actions/obs/vehicles length mismatch"
            argmin_by_index = [None] * len(obs_list)
            for i, obs in enumerate(obs_list):
                v = env.vehicles[i]
                task_idx = obs.get("subtask_index", -1)
                if task_idx is None or task_idx < 0:
                    continue

                mask = obs["action_mask"]
                mask_local = bool(mask[0])
                mask_rsu = bool(mask[1]) if len(mask) > 1 else False
                mask_v2v_slots = int(np.sum(mask[2:])) if len(mask) > 2 else 0
                mask_counts["local"] += int(mask_local)
                mask_counts["rsu"] += int(mask_rsu)
                mask_counts["v2v_slots"] += mask_v2v_slots
                samples_total += 1

                task_comp = v.task_dag.total_comp[task_idx]
                task_data = v.task_dag.total_data[task_idx]

                rsu_id = env._last_rsu_choice.get(v.id)
                rsu_target = ("RSU", rsu_id) if rsu_id is not None else None
                candidate_ids = env._last_candidates.get(v.id, [])
                v2v_candidates = [cid for cid in candidate_ids if cid is not None and cid >= 0]

                local_cost = _compute_cost(env, v, task_idx, "Local", v2i_users) if mask_local else None
                rsu_cost = None
                if rsu_target is not None:
                    rsu_cost = _compute_cost(env, v, task_idx, rsu_target, v2i_users) if mask_rsu else None
                v2v_best = None
                v2v_best_id = None
                v2v_best_idx = None
                for idx, cid in enumerate(candidate_ids):
                    if cid is None or cid < 0:
                        continue
                    if idx + 2 >= len(mask) or not mask[idx + 2]:
                        continue
                    cost = _compute_cost(env, v, task_idx, int(cid), v2i_users)
                    if v2v_best is None or cost["cost"] < v2v_best["cost"]:
                        v2v_best = cost
                        v2v_best_id = int(cid)
                        v2v_best_idx = idx

                v2v_is_available = v2v_best is not None
                if v2v_is_available:
                    v2v_available += 1

                candidates = []
                if local_cost is not None and not local_cost["hard_triggered"]:
                    candidates.append(("Local", local_cost))
                if rsu_cost is not None and not rsu_cost["hard_triggered"]:
                    candidates.append(("RSU", rsu_cost))
                if v2v_is_available and not v2v_best["hard_triggered"]:
                    candidates.append(("V2V", v2v_best))

                if not candidates:
                    continue

                argmin_target, argmin_cost = min(candidates, key=lambda x: x[1]["cost"])
                argmin_counts[argmin_target] += 1
                argmin_total += 1
                argmin_by_index[i] = argmin_target

                if v2v_is_available and argmin_target != "V2V":
                    v2v_not_argmin += 1

                if v2v_is_available and rsu_cost is not None and not rsu_cost["hard_triggered"]:
                    delta_delay.append(v2v_best["delay_norm"] - rsu_cost["delay_norm"])
                    delta_energy.append(v2v_best["energy_norm"] - rsu_cost["energy_norm"])

                if v2v_is_available:
                    non_v2v_costs = []
                    if local_cost is not None and not local_cost["hard_triggered"]:
                        non_v2v_costs.append(local_cost["cost"])
                    if rsu_cost is not None and not rsu_cost["hard_triggered"]:
                        non_v2v_costs.append(rsu_cost["cost"])
                    if non_v2v_costs:
                        gap = v2v_best["cost"] - min(non_v2v_costs)
                        delta_v2v_vs_best.append(gap)
                        worst_v2v.append({
                            "gap": gap,
                            "episode": ep,
                            "step": step,
                            "veh_id": v.id,
                            "subtask": task_idx,
                            "best_v2v_id": v2v_best_id,
                            "best_v2v_idx": v2v_best_idx,
                            "v2v_cost": v2v_best["cost"],
                            "best_non_v2v_cost": min(non_v2v_costs),
                            "dist_v2v": v2v_best["dist"],
                            "rate_v2v": v2v_best["rate"],
                            "queue_v2v": v2v_best["wait_time"],
                            "v2i_user_count": v2i_users,
                            "rsu_rate": rsu_cost["rate"] if rsu_cost else 0.0,
                            "rsu_wait": rsu_cost["wait_time"] if rsu_cost else 0.0,
                            "rsu_dist": rsu_cost["dist"] if rsu_cost else 0.0,
                            "task_comp": task_comp,
                            "task_data": task_data,
                        })

                chosen_type = None
                if mappo_actions is not None:
                    chosen_idx = int(mappo_actions[i]["target"])
                    if chosen_idx < 0 or chosen_idx >= len(mask) or not mask[chosen_idx]:
                        chosen_type = "illegal"
                    elif chosen_idx == 0:
                        chosen_type = "Local"
                    elif chosen_idx == 1:
                        chosen_type = "RSU"
                    else:
                        chosen_type = "V2V"
                    if argmin_by_index[i] is not None:
                        conf_mat[chosen_type][argmin_by_index[i]] += 1
                        chosen_total += 1
                        chosen_counts[chosen_type] += 1
                        if chosen_type == argmin_by_index[i]:
                            chosen_match += 1

                row = {
                    "episode": ep,
                    "step": step,
                    "veh_id": v.id,
                    "subtask": task_idx,
                    "v2i_user_count": v2i_users,
                    "mask_local": int(mask_local),
                    "mask_rsu": int(mask_rsu),
                    "mask_v2v": int(v2v_is_available),
                    "mask_v2v_slots": int(mask_v2v_slots),
                    "argmin_target": argmin_target,
                    "local_cost": local_cost["cost"] if local_cost else None,
                    "local_tx_time": local_cost["tx_time"] if local_cost else None,
                    "local_wait_time": local_cost["wait_time"] if local_cost else None,
                    "local_comp_time": local_cost["comp_time"] if local_cost else None,
                    "local_delay_norm": local_cost["delay_norm"] if local_cost else None,
                    "local_energy_norm": local_cost["energy_norm"] if local_cost else None,
                    "local_dist_penalty": local_cost["dist_penalty"] if local_cost else None,
                    "rsu_cost": rsu_cost["cost"] if rsu_cost else None,
                    "rsu_tx_time": rsu_cost["tx_time"] if rsu_cost else None,
                    "rsu_wait_time": rsu_cost["wait_time"] if rsu_cost else None,
                    "rsu_comp_time": rsu_cost["comp_time"] if rsu_cost else None,
                    "rsu_delay_norm": rsu_cost["delay_norm"] if rsu_cost else None,
                    "rsu_energy_norm": rsu_cost["energy_norm"] if rsu_cost else None,
                    "rsu_dist_penalty": rsu_cost["dist_penalty"] if rsu_cost else None,
                    "v2v_cost": v2v_best["cost"] if v2v_best else None,
                    "v2v_tx_time": v2v_best["tx_time"] if v2v_best else None,
                    "v2v_wait_time": v2v_best["wait_time"] if v2v_best else None,
                    "v2v_comp_time": v2v_best["comp_time"] if v2v_best else None,
                    "v2v_delay_norm": v2v_best["delay_norm"] if v2v_best else None,
                    "v2v_energy_norm": v2v_best["energy_norm"] if v2v_best else None,
                    "v2v_dist_penalty": v2v_best["dist_penalty"] if v2v_best else None,
                    "v2v_best_id": v2v_best_id,
                    "v2v_best_idx": v2v_best_idx,
                    "delta_v2v_vs_best": (v2v_best["cost"] - min([c[1]["cost"] for c in candidates if c[0] != "V2V"]))
                    if v2v_is_available and len([c for c in candidates if c[0] != "V2V"]) > 0 else None,
                    "chosen_target": chosen_type,
                }
                csv_rows.append(row)

            # choose actions to advance env
            if policy_rollout == "mappo":
                if mappo_agent is None:
                    raise RuntimeError("policy_rollout=mappo requires --mappo_ckpt")
                actions = mappo_actions
            else:
                actions = []
                for obs in obs_list:
                    mask = obs["action_mask"]
                    if policy_rollout == "index_uniform_random":
                        idx = _select_index_uniform(mask, rng)
                    else:
                        idx = _select_type_balanced(mask, rng)
                    if idx is None:
                        idx = 0
                    actions.append({"target": int(idx), "power": 1.0})

            obs_list, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

    # write CSV
    fieldnames = [
        "episode", "step", "veh_id", "subtask", "v2i_user_count", "argmin_target",
        "mask_local", "mask_rsu", "mask_v2v", "mask_v2v_slots",
        "local_cost", "local_tx_time", "local_wait_time", "local_comp_time", "local_delay_norm",
        "local_energy_norm", "local_dist_penalty",
        "rsu_cost", "rsu_tx_time", "rsu_wait_time", "rsu_comp_time", "rsu_delay_norm",
        "rsu_energy_norm", "rsu_dist_penalty",
        "v2v_cost", "v2v_tx_time", "v2v_wait_time", "v2v_comp_time", "v2v_delay_norm",
        "v2v_energy_norm", "v2v_dist_penalty",
        "v2v_best_id", "v2v_best_idx", "delta_v2v_vs_best", "chosen_target",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # prepare markdown
    argmin_lines = []
    if argmin_total > 0:
        for k in ("Local", "RSU", "V2V"):
            frac = argmin_counts.get(k, 0) / argmin_total
            argmin_lines.append(f"- {k}: {argmin_counts.get(k, 0)} ({frac:.4f})")
    v2v_not_ratio = (v2v_not_argmin / v2v_available) if v2v_available > 0 else 0.0
    argmin_text = "\n".join(argmin_lines) + f"\n- v2v_available: {v2v_available}\n- v2v_not_argmin_ratio: {v2v_not_ratio:.4f}\n"

    delta_delay_stats = _summary_stats(delta_delay)
    delta_energy_stats = _summary_stats(delta_energy)
    delta_v2v_stats = _summary_stats(delta_v2v_vs_best)
    delta_text = (
        f"- delta_delay_norm (v2v - rsu): mean={delta_delay_stats['mean']:.6f}, "
        f"p50={delta_delay_stats['p50']:.6f}, p90={delta_delay_stats['p90']:.6f}, p99={delta_delay_stats['p99']:.6f}\n"
        f"- delta_energy_norm (v2v - rsu): mean={delta_energy_stats['mean']:.6f}, "
        f"p50={delta_energy_stats['p50']:.6f}, p90={delta_energy_stats['p90']:.6f}, p99={delta_energy_stats['p99']:.6f}\n"
        f"- delta_v2v_vs_best (v2v - min(local,rsu)): mean={delta_v2v_stats['mean']:.6f}, "
        f"p50={delta_v2v_stats['p50']:.6f}, p90={delta_v2v_stats['p90']:.6f}, p99={delta_v2v_stats['p99']:.6f}\n"
    )

    worst_v2v.sort(key=lambda x: x["gap"], reverse=True)
    worst_rows = worst_v2v[:20]
    worst_table = "|gap|ep|step|veh|subtask|v2v_id|v2v_idx|v2v_cost|best_non_v2v_cost|dist_v2v|rate_v2v|queue_v2v|v2i_users|rsu_rate|rsu_wait|rsu_dist|\n"
    worst_table += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for row in worst_rows:
        worst_table += (
            f"|{row['gap']:.6f}|{row['episode']}|{row['step']}|{row['veh_id']}|{row['subtask']}|"
            f"{row['best_v2v_id']}|{row['best_v2v_idx']}|{row['v2v_cost']:.6f}|{row['best_non_v2v_cost']:.6f}|"
            f"{row['dist_v2v'] if row['dist_v2v'] is not None else 0.0:.3f}|{row['rate_v2v']:.3f}|"
            f"{row['queue_v2v']:.3f}|{row['v2i_user_count']}|{row['rsu_rate']:.3f}|{row['rsu_wait']:.3f}|"
            f"{row['rsu_dist']:.3f}|\n"
        )

    conf_text = ""
    chosen_text = ""
    if mappo_agent is not None:
        all_types = ["Local", "RSU", "V2V", "illegal"]
        conf_text = "|chosen \\ argmin|Local|RSU|V2V|\n|---|---|---|---|\n"
        for chosen in all_types:
            conf_text += f"|{chosen}|"
            for argmin in ("Local", "RSU", "V2V"):
                conf_text += f"{conf_mat[chosen].get(argmin, 0)}|"
            conf_text += "\n"
        acc = (chosen_match / max(chosen_total, 1))
        chosen_text = f"- chosen_total: {chosen_total}\n- accuracy: {acc:.4f}\n"
        for k in ("Local", "RSU", "V2V", "illegal"):
            chosen_text += f"- chosen_{k.lower()}_frac: {chosen_counts.get(k, 0) / max(chosen_total, 1):.4f}\n"
    else:
        conf_text = "MAPPO not provided.\n"
        chosen_text = "MAPPO not provided.\n"

    header = (
        f"Snapshot-consistent audit: costs computed on the same obs snapshot (no env.step side-effects).\n"
        f"seed={seed}, episodes={episodes}, steps={steps}, rollout_policy={policy_rollout}\n"
    )

    sections = [
        ("Overview", header),
        ("Argmin Distribution", argmin_text),
        ("V2V vs RSU Delta", delta_text),
        ("Top V2V Worst Samples", worst_table),
        ("Chosen Distribution (MAPPO)", chosen_text),
        ("Confusion Matrix", conf_text),
    ]
    _write_md(md_path, sections)

    # stdout summary
    print(f"[Audit] samples_total={samples_total}")
    print(f"[Audit] mask_local_true={mask_counts['local']} mask_rsu_true={mask_counts['rsu']} v2v_slots_true={mask_counts['v2v_slots']}")
    if argmin_total > 0:
        print("[Audit] argmin distribution:")
        for k in ("Local", "RSU", "V2V"):
            print(f"  - {k}: {argmin_counts.get(k, 0)} ({argmin_counts.get(k, 0) / argmin_total:.4f})")
    if mappo_agent is not None:
        acc = (chosen_match / max(chosen_total, 1))
        print(f"[Audit] chosen_total={chosen_total} accuracy={acc:.4f}")
        print("[Audit] confusion matrix (chosen -> argmin):")
        for chosen in ("Local", "RSU", "V2V", "illegal"):
            row = [conf_mat[chosen].get(argmin, 0) for argmin in ("Local", "RSU", "V2V")]
            print(f"  {chosen}: {row}")

    return {"csv": csv_path, "md": md_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results_dbg/decision_dominance")
    parser.add_argument("--mappo_ckpt", type=str, default=None)
    parser.add_argument("--policy_rollout", type=str, default="type_balanced_random",
                        choices=["index_uniform_random", "type_balanced_random", "mappo"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_audit(
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        out_dir=args.out_dir,
        mappo_ckpt=args.mappo_ckpt,
        policy_rollout=args.policy_rollout,
        device=args.device,
    )


if __name__ == "__main__":
    main()
