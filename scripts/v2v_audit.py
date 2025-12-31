import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from baselines.random_policy import RandomPolicy
from agents.mappo_agent import MAPPOAgent
from models.offloading_policy import OffloadingPolicyNetwork


def _contact_time(v, other, dist):
    rel_vel = other.vel - v.vel
    pos_diff = other.pos - v.pos
    pos_norm = np.linalg.norm(pos_diff)
    if pos_norm < 1e-6:
        return Cfg.V2V_RANGE / max(Cfg.VEL_MIN, 1e-6)
    rel_vel_proj = np.dot(rel_vel, pos_diff) / pos_norm
    if rel_vel_proj > 0.1:
        return max((Cfg.V2V_RANGE - dist) / rel_vel_proj, 0.0)
    return Cfg.V2V_RANGE / max(Cfg.VEL_MIN, 1e-6)


def _compute_times(env, v, target, task_idx):
    dag = v.task_dag
    task_comp = dag.total_comp[task_idx]
    task_data = dag.total_data[task_idx]
    queue_wait = 0.0
    cpu_freq = v.cpu_freq
    tx_time = 0.0
    rate = None
    if target == "Local":
        queue_wait = v.task_queue.get_estimated_wait_time(v.cpu_freq)
        cpu_freq = v.cpu_freq
        tx_time = 0.0
    elif isinstance(target, tuple) and target[0] == "RSU":
        rsu_id = target[1]
        if rsu_id is not None and 0 <= rsu_id < len(env.rsus):
            rsu = env.rsus[rsu_id]
            queue_wait = rsu.get_estimated_wait_time()
            cpu_freq = rsu.cpu_freq
            rate = env.channel.compute_one_rate(
                v, rsu.position, "V2I", env.time, v2i_user_count=env._estimate_v2i_users()
            )
            rate = max(rate, 1e-6)
            tx_time = task_data / rate if task_data > 0 else 0.0
    elif isinstance(target, int):
        t_veh = env._get_vehicle_by_id(target)
        if t_veh is not None:
            queue_wait = t_veh.task_queue.get_estimated_wait_time(t_veh.cpu_freq)
            cpu_freq = t_veh.cpu_freq
            rate = env.channel.compute_one_rate(v, t_veh.pos, "V2V", env.time)
            rate = max(rate, 1e-6)
            tx_time = task_data / rate if task_data > 0 else 0.0
    comp_time = task_comp / max(cpu_freq, 1e-6)
    return tx_time, queue_wait, comp_time, rate


def _cost_norm(env, v, target, task_idx, tx_time, queue_wait, comp_time):
    dag = v.task_dag
    task_comp = dag.total_comp[task_idx]
    task_data = dag.total_data[task_idx]
    if target == "Local":
        max_rate = env._get_norm_rate("V2I")
    elif isinstance(target, tuple) and target[0] == "RSU":
        max_rate = env._get_norm_rate("V2I")
    else:
        max_rate = env._get_norm_rate("V2V")
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
    return delay_norm, energy_norm


def _print_hist(values, bins, title):
    if not values:
        print(f"{title}: empty")
        return
    hist, edges = np.histogram(values, bins=bins)
    print(f"\n{title}")
    for i in range(len(hist)):
        print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}): {hist[i]}")


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
    v2v_indices = np.where(mask[2:])[0]
    if len(v2v_indices) > 0:
        types.append("v2v")
    if not types:
        return None
    choice = rng.choice(types)
    if choice == "local":
        return 0
    if choice == "rsu":
        return 1
    return int(rng.choice(v2v_indices) + 2)


def run_audit(episodes=50, steps=50, seed=0, mappo_ckpt=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = VecOffloadingEnv()
    rand_policy = RandomPolicy(seed=seed)
    mappo_agent = None
    if mappo_ckpt:
        net = OffloadingPolicyNetwork()
        state = torch.load(mappo_ckpt, map_location="cpu")
        net.load_state_dict(state)
        mappo_agent = MAPPOAgent(net, device="cpu")

    v2v_mask_true = 0
    v2v_mask_total = 0
    candidate_counts = []
    contact_times = []
    distances = []

    v2v_available = 0
    v2v_selected_random_idx = 0
    v2v_selected_random_type = 0
    v2v_selected_mappo_idx = 0
    v2v_selected_mappo_type = 0
    v2v_available_type_den = 0.0
    v2v_available_mappo = 0
    v2v_available_type_den_mappo = 0.0

    counterfactual_rows = []
    needed_triplets = 20
    triplets_collected = 0
    rng = np.random.RandomState(seed)

    for ep in range(episodes):
        obs_list, _ = env.reset(seed=seed + ep)
        for step in range(steps):
            # Collect availability stats
            for i, obs in enumerate(obs_list):
                mask = obs["action_mask"]
                v2v_mask = mask[2:]
                v2v_mask_true += int(np.sum(v2v_mask))
                v2v_mask_total += Cfg.MAX_NEIGHBORS
                candidate_counts.append(int(np.sum(v2v_mask)))

                v = env.vehicles[i]
                cand_ids = env._last_candidates.get(v.id, [])
                for n_id in cand_ids:
                    if n_id is None or n_id < 0:
                        continue
                    other = env._get_vehicle_by_id(n_id)
                    if other is None:
                        continue
                    dist = np.linalg.norm(other.pos - v.pos)
                    distances.append(dist)
                    contact_times.append(_contact_time(v, other, dist))

            # Counterfactual samples (same snapshot, same READY subtask)
            if triplets_collected < needed_triplets:
                v2i_users = env._estimate_v2i_users()
                for i, v in enumerate(env.vehicles):
                    if triplets_collected >= needed_triplets:
                        break
                    task_idx = v.task_dag.get_top_priority_task()
                    if task_idx is None:
                        continue
                    cand_ids = env._last_candidates.get(v.id, [])
                    if not cand_ids:
                        continue
                    rsu_id = env._last_rsu_choice.get(v.id)
                    if rsu_id is None:
                        continue
                    # Local
                    local = "Local"
                    tx_time, queue_wait, comp_time, rate = _compute_times(env, v, local, task_idx)
                    delay_norm, energy_norm = _cost_norm(env, v, local, task_idx, tx_time, queue_wait, comp_time)
                    dist_penalty, _ = env._calculate_constraint_penalty(i, local, task_idx, v.task_dag.total_comp[task_idx])
                    counterfactual_rows.append({
                        "ep": ep, "veh": v.id, "task": task_idx, "target": "Local",
                        "tx_time": tx_time, "queue_wait": queue_wait, "comp_time": comp_time,
                        "delay_norm": delay_norm, "energy_norm": energy_norm, "dist_penalty": dist_penalty,
                        "rate_v2i": 0.0, "rate_v2v": 0.0, "v2i_user_count": v2i_users
                    })
                    # RSU
                    rsu = ("RSU", rsu_id)
                    tx_time, queue_wait, comp_time, rate = _compute_times(env, v, rsu, task_idx)
                    delay_norm, energy_norm = _cost_norm(env, v, rsu, task_idx, tx_time, queue_wait, comp_time)
                    dist_penalty, _ = env._calculate_constraint_penalty(i, rsu, task_idx, v.task_dag.total_comp[task_idx])
                    counterfactual_rows.append({
                        "ep": ep, "veh": v.id, "task": task_idx, "target": f"RSU#{rsu_id}",
                        "tx_time": tx_time, "queue_wait": queue_wait, "comp_time": comp_time,
                        "delay_norm": delay_norm, "energy_norm": energy_norm, "dist_penalty": dist_penalty,
                        "rate_v2i": 0.0 if rate is None else rate, "rate_v2v": 0.0, "v2i_user_count": v2i_users
                    })
                    # V2V first candidate
                    for n_id in cand_ids:
                        if n_id is None or n_id < 0:
                            continue
                        tx_time, queue_wait, comp_time, rate = _compute_times(env, v, n_id, task_idx)
                        delay_norm, energy_norm = _cost_norm(env, v, n_id, task_idx, tx_time, queue_wait, comp_time)
                        dist_penalty, _ = env._calculate_constraint_penalty(i, n_id, task_idx, v.task_dag.total_comp[task_idx])
                        counterfactual_rows.append({
                            "ep": ep, "veh": v.id, "task": task_idx, "target": f"V2V#{n_id}",
                            "tx_time": tx_time, "queue_wait": queue_wait, "comp_time": comp_time,
                            "delay_norm": delay_norm, "energy_norm": energy_norm, "dist_penalty": dist_penalty,
                            "rate_v2i": 0.0, "rate_v2v": 0.0 if rate is None else rate, "v2i_user_count": v2i_users
                        })
                        break
                    triplets_collected += 1

            # Random policy selection stats (two variants)
            actions = []
            for obs in obs_list:
                mask = obs["action_mask"]
                target_idx = _select_index_uniform(mask, rng)
                if target_idx is None:
                    actions.append(None)
                    continue
                actions.append({"target": int(target_idx), "power": float(rng.uniform(0.2, 1.0))})
            actions_type = []
            for obs in obs_list:
                mask = obs["action_mask"]
                target_idx = _select_type_balanced(mask, rng)
                if target_idx is None:
                    actions_type.append(None)
                    continue
                actions_type.append({"target": int(target_idx), "power": float(rng.uniform(0.2, 1.0))})

            for obs, act, act_type in zip(obs_list, actions, actions_type):
                v2v_available_flag = bool(np.any(obs["action_mask"][2:]))
                if v2v_available_flag:
                    v2v_available += 1
                    if act is not None and act["target"] >= 2:
                        v2v_selected_random_idx += 1
                    if act_type is not None and act_type["target"] >= 2:
                        v2v_selected_random_type += 1
                    type_count = int(obs["action_mask"][0]) + int(obs["action_mask"][1]) + int(v2v_available_flag)
                    v2v_available_type_den += 1.0 / max(type_count, 1)

            obs_list, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        # keep collecting availability stats for all episodes/steps

    # MAPPO selection stats (optional)
    if mappo_agent is not None:
        obs_list, _ = env.reset(seed=seed + 999)
        for step in range(steps):
            action_dict = mappo_agent.select_action(obs_list, deterministic=True)
            actions = action_dict["actions"]
            for obs, act in zip(obs_list, actions):
                v2v_available_flag = bool(np.any(obs["action_mask"][2:]))
                if v2v_available_flag:
                    v2v_available_mappo += 1
                    v2v_selected_mappo_idx += 1 if act["target"] >= 2 else 0
                    type_count = int(obs["action_mask"][0]) + int(obs["action_mask"][1]) + int(v2v_available_flag)
                    v2v_available_type_den_mappo += 1.0 / max(type_count, 1)
                    v2v_selected_mappo_type += (1.0 / max(type_count, 1)) if act["target"] >= 2 else 0.0
            obs_list, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

    # Output summary table
    v2v_avail_ratio = v2v_mask_true / max(v2v_mask_total, 1)
    rand_select_ratio_idx = v2v_selected_random_idx / max(v2v_available, 1)
    rand_select_ratio_type = (v2v_selected_random_type / max(v2v_available, 1))
    mappo_select_ratio_idx = v2v_selected_mappo_idx / max(v2v_available_mappo, 1) if mappo_agent else None
    mappo_select_ratio_type = (v2v_selected_mappo_type / max(v2v_available_type_den_mappo, 1e-6)) if mappo_agent else None

    print("\n## V2V Availability & Selection Summary")
    print("| metric | value |")
    print("|---|---|")
    print(f"| v2v_mask_true/total_slots | {v2v_mask_true}/{v2v_mask_total} ({v2v_avail_ratio:.4f}) |")
    print(f"| candidate_count_mean | {np.mean(candidate_counts):.4f} |")
    print(f"| candidate_count_p50 | {np.percentile(candidate_counts, 50):.4f} |")
    print(f"| candidate_count_p90 | {np.percentile(candidate_counts, 90):.4f} |")
    print(f"| candidate_count_p99 | {np.percentile(candidate_counts, 99):.4f} |")
    print(f"| candidate_count_max | {np.max(candidate_counts):.4f} |")
    print(f"| random_v2v_select_ratio_idx | {rand_select_ratio_idx:.4f} |")
    print(f"| random_v2v_select_ratio_type | {rand_select_ratio_type:.4f} |")
    if mappo_agent:
        print(f"| mappo_v2v_select_ratio_idx | {mappo_select_ratio_idx:.4f} |")
        print(f"| mappo_v2v_select_ratio_type | {mappo_select_ratio_type:.4f} |")
    else:
        print("| mappo_v2v_select_ratio_idx | N/A |")
        print("| mappo_v2v_select_ratio_type | N/A |")

    _print_hist(candidate_counts, bins=[0, 1, 2, 3, 4, 5, 6, 10, 12], title="candidate_count histogram")
    _print_hist(distances, bins=10, title="distance histogram (meters)")
    _print_hist(contact_times, bins=10, title="contact_time histogram (seconds)")

    print("\n## Counterfactual Cost Samples (first 20 tasks)")
    print("| ep | veh | task | target | tx_time | wait_time | comp_time | delay_norm | energy_norm | dist_penalty | rate_v2i | rate_v2v | v2i_user_count |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in counterfactual_rows[:needed_triplets * 3]:
        print(f"| {row['ep']} | {row['veh']} | {row['task']} | {row['target']} | "
              f"{row['tx_time']:.4f} | {row['queue_wait']:.4f} | {row['comp_time']:.4f} | "
              f"{row['delay_norm']:.4f} | {row['energy_norm']:.4f} | {row['dist_penalty']:.4f} | "
              f"{row['rate_v2i']:.4f} | {row['rate_v2v']:.4f} | {row['v2i_user_count']} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mappo_ckpt", type=str, default=None)
    args = parser.parse_args()
    run_audit(args.episodes, args.steps, args.seed, args.mappo_ckpt)


if __name__ == "__main__":
    main()
