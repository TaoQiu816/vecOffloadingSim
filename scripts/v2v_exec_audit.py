import argparse
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
from models.offloading_policy import OffloadingPolicyNetwork


def _infer_num_layers(sd):
    max_idx = -1
    for k in sd.keys():
        if k.startswith("transformer.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                max_idx = max(max_idx, int(parts[2]))
    return max_idx + 1 if max_idx >= 0 else None


def _infer_d_model(sd):
    key = "transformer.layers.0.attention.W_q.weight"
    if key in sd and hasattr(sd[key], "shape"):
        return int(sd[key].shape[0])
    fallback_keys = [
        "actor_critic.layer_norm.weight",
        "layer_norm.weight",
        "transformer.layer_norm.weight",
    ]
    for k in fallback_keys:
        if k in sd and hasattr(sd[k], "shape"):
            return int(sd[k].shape[0])
    return None


def _infer_num_heads(d_model):
    if d_model is None:
        return 1
    if d_model % 8 == 0:
        return 8
    if d_model % 4 == 0:
        return 4
    return 1


def _load_mappo_network(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["network_state_dict"] if isinstance(ckpt, dict) and "network_state_dict" in ckpt else ckpt
    num_layers = _infer_num_layers(sd)
    d_model = _infer_d_model(sd)
    num_heads = _infer_num_heads(d_model)
    print(f"[MAPPO] inferred d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")

    net = OffloadingPolicyNetwork(d_model=d_model or 128, num_layers=num_layers or 4, num_heads=num_heads)
    try:
        net.load_state_dict(sd, strict=True)
    except Exception as exc:
        print("[MAPPO] strict=True load_state_dict failed:", str(exc))
        print("[MAPPO] retry with strict=False (audit-only)")
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print("[MAPPO] missing keys (first 20):", missing[:20])
        print("[MAPPO] unexpected keys (first 20):", unexpected[:20])
    net.eval()
    return net


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


def _compute_times(env, v, target, task_idx):
    dag = v.task_dag
    task_comp = dag.total_comp[task_idx]
    task_data = dag.total_data[task_idx]
    queue_wait = 0.0
    cpu_freq = v.cpu_freq
    tx_time = 0.0
    rate_v2i = 0.0
    rate_v2v = 0.0
    if target == "Local":
        queue_wait = v.task_queue.get_estimated_wait_time(v.cpu_freq)
        cpu_freq = v.cpu_freq
    elif isinstance(target, tuple) and target[0] == "RSU":
        rsu_id = target[1]
        if rsu_id is not None and 0 <= rsu_id < len(env.rsus):
            rsu = env.rsus[rsu_id]
            queue_wait = rsu.get_estimated_wait_time()
            cpu_freq = rsu.cpu_freq
            rate_v2i = env.channel.compute_one_rate(
                v, rsu.position, "V2I", env.time, v2i_user_count=env._estimate_v2i_users()
            )
            rate_v2i = max(rate_v2i, 1e-6)
            tx_time = task_data / rate_v2i if task_data > 0 else 0.0
    elif isinstance(target, int):
        t_veh = env._get_vehicle_by_id(target)
        if t_veh is not None:
            queue_wait = t_veh.task_queue.get_estimated_wait_time(t_veh.cpu_freq)
            cpu_freq = t_veh.cpu_freq
            rate_v2v = env.channel.compute_one_rate(v, t_veh.pos, "V2V", env.time)
            rate_v2v = max(rate_v2v, 1e-6)
            tx_time = task_data / rate_v2v if task_data > 0 else 0.0
    comp_time = task_comp / max(cpu_freq, 1e-6)
    return tx_time, queue_wait, comp_time, rate_v2i, rate_v2v


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


def _classify_illegal(reason):
    if reason == "queue_full":
        return "queue_full"
    if reason in ("rsu_out_of_coverage", "rsu_unavailable"):
        return "out_of_range"
    if reason in ("id_mapping_fail", "idx_out_of_range", "no_candidate_cache"):
        return "illegal_action"
    return "illegal_action"


def _distance(a, b):
    return float(np.linalg.norm(a - b))


def _run_policy(env, policy_name, policy_fn, episodes, steps, seed):
    rng = np.random.RandomState(seed)
    stats = {
        "decisions": defaultdict(int),
        "success": defaultdict(int),
        "timeout": defaultdict(int),
        "fail": defaultdict(int),
        "times": defaultdict(list),
        "delay_norm": defaultdict(list),
        "energy_norm": defaultdict(list),
        "fail_reasons": defaultdict(lambda: defaultdict(int)),
        "fail_examples": [],
    }
    v2v_decisions = 0

    for ep in range(episodes):
        obs_list, _ = env.reset(seed=seed + ep)
        pending = {}
        contact_break_flags = {}
        for step in range(steps):
            actions = []
            decision_records = []
            for i, obs in enumerate(obs_list):
                mask = obs["action_mask"]
                target_idx = policy_fn(mask, rng)
                if target_idx is None:
                    actions.append(None)
                    continue
                v = env.vehicles[i]
                subtask_idx = v.task_dag.get_top_priority_task()
                if subtask_idx is None:
                    actions.append(None)
                    continue

                chosen_type = "Local" if target_idx == 0 else ("RSU" if target_idx == 1 else "V2V")
                candidate_ids = env._last_candidates.get(v.id, [])
                neighbor_id = None
                if chosen_type == "V2V":
                    idx = target_idx - 2
                    if 0 <= idx < len(candidate_ids):
                        neighbor_id = candidate_ids[idx]
                rsu_id = env._last_rsu_choice.get(v.id)

                actual_target = "Local"
                if chosen_type == "RSU" and rsu_id is not None:
                    actual_target = ("RSU", rsu_id)
                elif chosen_type == "V2V" and neighbor_id is not None and neighbor_id >= 0:
                    actual_target = int(neighbor_id)

                tx_time, wait_time, comp_time, rate_v2i, rate_v2v = _compute_times(env, v, actual_target, subtask_idx)
                delay_norm, energy_norm = _cost_norm(env, v, actual_target, subtask_idx, tx_time, wait_time, comp_time)
                dist_penalty, _ = env._calculate_constraint_penalty(i, actual_target, subtask_idx, v.task_dag.total_comp[subtask_idx])

                decision_records.append({
                    "ep": ep,
                    "step": step,
                    "veh_id": v.id,
                    "subtask_id": subtask_idx,
                    "chosen_type": chosen_type,
                    "chosen_idx": target_idx,
                    "actual_target": actual_target,
                    "neighbor_id": neighbor_id,
                    "rsu_id": rsu_id,
                    "tx_time": tx_time,
                    "wait_time": wait_time,
                    "comp_time": comp_time,
                    "delay_norm": delay_norm,
                    "energy_norm": energy_norm,
                    "dist_penalty": dist_penalty,
                    "rate_v2i": rate_v2i,
                    "rate_v2v": rate_v2v,
                    "v2i_user_count": env._estimate_v2i_users(),
                })
                actions.append({"target": int(target_idx), "power": float(rng.uniform(0.2, 1.0))})

            obs_list, _, terminated, truncated, _ = env.step(actions)

            # evaluate decisions after step
            for rec in decision_records:
                v = env._get_vehicle_by_id(rec["veh_id"])
                if v is None:
                    continue
                stats["decisions"][rec["chosen_type"]] += 1
                if rec["chosen_type"] == "V2V":
                    v2v_decisions += 1
                stats["times"][rec["chosen_type"]].append((rec["tx_time"], rec["wait_time"], rec["comp_time"]))
                stats["delay_norm"][rec["chosen_type"]].append(rec["delay_norm"])
                stats["energy_norm"][rec["chosen_type"]].append(rec["energy_norm"])
                if getattr(v, "illegal_action", False):
                    reason = _classify_illegal(getattr(v, "illegal_reason", "unknown"))
                    stats["fail"][rec["chosen_type"]] += 1
                    stats["fail_reasons"][rec["chosen_type"]][reason] += 1
                    stats["fail_examples"].append((rec, reason))
                    continue

                if v.last_action_step == env.steps and v.last_scheduled_subtask == rec["subtask_id"]:
                    key = (rec["veh_id"], rec["subtask_id"])
                    pending[key] = rec
                    if rec["chosen_type"] == "V2V" and rec["neighbor_id"] is not None:
                        contact_break_flags[key] = False

            # update contact break flags
            for key, rec in list(pending.items()):
                if rec["chosen_type"] != "V2V":
                    continue
                v = env._get_vehicle_by_id(rec["veh_id"])
                t_veh = env._get_vehicle_by_id(rec["neighbor_id"]) if rec["neighbor_id"] is not None else None
                if v is None or t_veh is None:
                    contact_break_flags[key] = True
                    continue
                if _distance(v.pos, t_veh.pos) > Cfg.V2V_RANGE:
                    contact_break_flags[key] = True

            # check completion/timeout
            for key, rec in list(pending.items()):
                v = env._get_vehicle_by_id(rec["veh_id"])
                if v is None:
                    continue
                status = v.task_dag.status[rec["subtask_id"]]
                if status == 3:
                    stats["success"][rec["chosen_type"]] += 1
                    pending.pop(key, None)
                    contact_break_flags.pop(key, None)
                elif v.task_dag.is_failed:
                    stats["fail"][rec["chosen_type"]] += 1
                    stats["timeout"][rec["chosen_type"]] += 1
                    stats["fail_reasons"][rec["chosen_type"]]["deadline_timeout"] += 1
                    stats["fail_examples"].append((rec, "deadline_timeout"))
                    pending.pop(key, None)
                    contact_break_flags.pop(key, None)

            if terminated or truncated:
                break

        # end of episode, mark remaining pending as unfinished
        for key, rec in list(pending.items()):
            if contact_break_flags.get(key, False):
                reason = "contact_break"
            else:
                reason = "unfinished_end"
            stats["fail"][rec["chosen_type"]] += 1
            stats["fail_reasons"][rec["chosen_type"]][reason] += 1
            stats["fail_examples"].append((rec, reason))

    return stats, v2v_decisions


def _summarize_stats(stats):
    rows = []
    for target in ("Local", "RSU", "V2V"):
        decisions = stats["decisions"].get(target, 0)
        success = stats["success"].get(target, 0)
        fail = stats["fail"].get(target, 0)
        timeout = stats["timeout"].get(target, 0)
        times = stats["times"].get(target, [])
        delay_vals = stats["delay_norm"].get(target, [])
        energy_vals = stats["energy_norm"].get(target, [])
        mean_tx = float(np.mean([t[0] for t in times])) if times else 0.0
        mean_wait = float(np.mean([t[1] for t in times])) if times else 0.0
        mean_comp = float(np.mean([t[2] for t in times])) if times else 0.0
        mean_delay = float(np.mean(delay_vals)) if delay_vals else 0.0
        mean_energy = float(np.mean(energy_vals)) if energy_vals else 0.0
        fail_reason = stats["fail_reasons"].get(target, {})
        fail_reason_ratio = {}
        if fail > 0:
            for k, v in fail_reason.items():
                fail_reason_ratio[k] = round(v / fail, 4)
        rows.append((target, decisions, success, fail, timeout,
                     mean_tx, mean_wait, mean_comp, mean_delay, mean_energy, dict(fail_reason_ratio)))
    return rows


def _print_table(rows):
    total_decisions = sum(row[1] for row in rows)
    print("| target | decisions | decision_frac | success_count | fail_count | timeout_count | mean_tx | mean_wait | mean_comp | mean_delay_norm | mean_energy_norm | fail_reason_breakdown |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        target, decisions, success, fail, timeout, mean_tx, mean_wait, mean_comp, mean_delay, mean_energy, reasons = row
        frac = (decisions / total_decisions) if total_decisions > 0 else 0.0
        print(f"| {target} | {decisions} | {frac:.4f} | {success} | {fail} | {timeout} | "
              f"{mean_tx:.4f} | {mean_wait:.4f} | {mean_comp:.4f} | "
              f"{mean_delay:.4f} | {mean_energy:.4f} | {reasons} |")


def _print_fail_examples(fail_examples, limit=10):
    print("\nTop10 failure samples:")
    for rec, reason in fail_examples[:limit]:
        print(f"- ep={rec['ep']} step={rec['step']} veh={rec['veh_id']} subtask={rec['subtask_id']} "
              f"target={rec['chosen_type']} reason={reason} "
              f"tx={rec['tx_time']:.4f} wait={rec['wait_time']:.4f} comp={rec['comp_time']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", type=str, default="all",
                        choices=["all", "type_balanced_random", "index_uniform_random", "mappo"])
    parser.add_argument("--mappo_ckpt", type=str, default=None)
    args = parser.parse_args()

    policies = []
    if args.policy in ("all", "type_balanced_random"):
        policies.append(("type_balanced_random", _select_type_balanced))
    if args.policy in ("all", "index_uniform_random"):
        policies.append(("index_uniform_random", _select_index_uniform))

    if args.policy in ("all", "mappo") and args.mappo_ckpt:
        def mappo_select(mask, rng):
            return None
        policies.append(("mappo", mappo_select))

    for name, selector in policies:
        env = VecOffloadingEnv()
        if name == "mappo":
            if not args.mappo_ckpt:
                print("[MAPPO] ckpt not provided; skipping")
                continue
            net = _load_mappo_network(args.mappo_ckpt)
            agent = MAPPOAgent(net, device="cpu")
            def run_mappo():
                obs_list, _ = env.reset(seed=args.seed)
                stats = {
                    "decisions": defaultdict(int),
                    "success": defaultdict(int),
                    "timeout": defaultdict(int),
                    "fail": defaultdict(int),
                    "times": defaultdict(list),
                    "delay_norm": defaultdict(list),
                    "energy_norm": defaultdict(list),
                    "fail_reasons": defaultdict(lambda: defaultdict(int)),
                    "fail_examples": [],
                }
                v2v_decisions = 0
                for ep in range(args.episodes):
                    obs_list, _ = env.reset(seed=args.seed + ep)
                    pending = {}
                    contact_break_flags = {}
                    for step in range(args.steps):
                        action_dict = agent.select_action(obs_list, deterministic=True)
                        actions = action_dict["actions"]
                        assert len(actions) == len(obs_list) == len(env.vehicles), (
                            f"action/obs/vehicles mismatch: {len(actions)} {len(obs_list)} {len(env.vehicles)}"
                        )
                        decision_records = []
                        for i, obs in enumerate(obs_list):
                            if i >= len(actions):
                                continue
                            act = actions[i]
                            mask = obs["action_mask"]
                            if act is None:
                                continue
                            target_idx = int(act["target"])
                            if target_idx < 0 or target_idx >= len(mask):
                                continue
                            v = env.vehicles[i]
                            subtask_idx = v.task_dag.get_top_priority_task()
                            if subtask_idx is None:
                                continue
                            chosen_type = "Local" if target_idx == 0 else ("RSU" if target_idx == 1 else "V2V")
                            candidate_ids = env._last_candidates.get(v.id, [])
                            neighbor_id = None
                            if chosen_type == "V2V":
                                idx = target_idx - 2
                                if 0 <= idx < len(candidate_ids):
                                    neighbor_id = candidate_ids[idx]
                            rsu_id = env._last_rsu_choice.get(v.id)
                            actual_target = "Local"
                            if chosen_type == "RSU" and rsu_id is not None:
                                actual_target = ("RSU", rsu_id)
                            elif chosen_type == "V2V" and neighbor_id is not None and neighbor_id >= 0:
                                actual_target = int(neighbor_id)
                            tx_time, wait_time, comp_time, rate_v2i, rate_v2v = _compute_times(env, v, actual_target, subtask_idx)
                            delay_norm, energy_norm = _cost_norm(env, v, actual_target, subtask_idx, tx_time, wait_time, comp_time)
                            dist_penalty, _ = env._calculate_constraint_penalty(i, actual_target, subtask_idx, v.task_dag.total_comp[subtask_idx])
                            decision_records.append({
                                "ep": ep,
                                "step": step,
                                "veh_id": v.id,
                                "subtask_id": subtask_idx,
                                "chosen_type": chosen_type,
                                "chosen_idx": target_idx,
                                "actual_target": actual_target,
                                "neighbor_id": neighbor_id,
                                "rsu_id": rsu_id,
                                "tx_time": tx_time,
                                "wait_time": wait_time,
                                "comp_time": comp_time,
                                "delay_norm": delay_norm,
                                "energy_norm": energy_norm,
                                "dist_penalty": dist_penalty,
                                "rate_v2i": rate_v2i,
                                "rate_v2v": rate_v2v,
                                "v2i_user_count": env._estimate_v2i_users(),
                            })

                        obs_list, _, terminated, truncated, _ = env.step(actions)

                        for rec in decision_records:
                            v = env._get_vehicle_by_id(rec["veh_id"])
                            if v is None:
                                continue
                            stats["decisions"][rec["chosen_type"]] += 1
                            if rec["chosen_type"] == "V2V":
                                v2v_decisions += 1
                            stats["times"][rec["chosen_type"]].append((rec["tx_time"], rec["wait_time"], rec["comp_time"]))
                            stats["delay_norm"][rec["chosen_type"]].append(rec["delay_norm"])
                            stats["energy_norm"][rec["chosen_type"]].append(rec["energy_norm"])
                            if getattr(v, "illegal_action", False):
                                reason = _classify_illegal(getattr(v, "illegal_reason", "unknown"))
                                stats["fail"][rec["chosen_type"]] += 1
                                stats["fail_reasons"][rec["chosen_type"]][reason] += 1
                                stats["fail_examples"].append((rec, reason))
                                continue
                            if v.last_action_step == env.steps and v.last_scheduled_subtask == rec["subtask_id"]:
                                key = (rec["veh_id"], rec["subtask_id"])
                                pending[key] = rec
                                if rec["chosen_type"] == "V2V" and rec["neighbor_id"] is not None:
                                    contact_break_flags[key] = False

                        for key, rec in list(pending.items()):
                            if rec["chosen_type"] != "V2V":
                                continue
                            v = env._get_vehicle_by_id(rec["veh_id"])
                            t_veh = env._get_vehicle_by_id(rec["neighbor_id"]) if rec["neighbor_id"] is not None else None
                            if v is None or t_veh is None:
                                contact_break_flags[key] = True
                                continue
                            if _distance(v.pos, t_veh.pos) > Cfg.V2V_RANGE:
                                contact_break_flags[key] = True

                        for key, rec in list(pending.items()):
                            v = env._get_vehicle_by_id(rec["veh_id"])
                            if v is None:
                                continue
                            status = v.task_dag.status[rec["subtask_id"]]
                            if status == 3:
                                stats["success"][rec["chosen_type"]] += 1
                                pending.pop(key, None)
                                contact_break_flags.pop(key, None)
                            elif v.task_dag.is_failed:
                                stats["fail"][rec["chosen_type"]] += 1
                                stats["timeout"][rec["chosen_type"]] += 1
                                stats["fail_reasons"][rec["chosen_type"]]["deadline_timeout"] += 1
                                stats["fail_examples"].append((rec, "deadline_timeout"))
                                pending.pop(key, None)
                                contact_break_flags.pop(key, None)

                        if terminated or truncated:
                            break

                    for key, rec in list(pending.items()):
                        if contact_break_flags.get(key, False):
                            reason = "contact_break"
                        else:
                            reason = "unfinished_end"
                        stats["fail"][rec["chosen_type"]] += 1
                        stats["fail_reasons"][rec["chosen_type"]][reason] += 1
                        stats["fail_examples"].append((rec, reason))

                return stats, v2v_decisions

            stats, v2v_decisions = run_mappo()
        else:
            stats, v2v_decisions = _run_policy(env, name, selector, args.episodes, args.steps, args.seed)

        print(f"\n## Policy: {name}")
        rows = _summarize_stats(stats)
        _print_table(rows)
        _print_fail_examples(stats["fail_examples"])
        if v2v_decisions < 200:
            print(f"[Warn] V2V decision samples < 200 (count={v2v_decisions})")


if __name__ == "__main__":
    main()
