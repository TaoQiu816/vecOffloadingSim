#!/usr/bin/env python3
import contextlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from scripts.diagnose_phi_alignment import _choose_ready_subtask


@contextlib.contextmanager
def _patched(obj, **replacements):
    original = {}
    for key, value in replacements.items():
        original[key] = getattr(obj, key)
        setattr(obj, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(obj, key, value)


def _prepare_env(seed: int = 42):
    env = VecOffloadingEnv(Cfg)
    obs_list, _ = env.reset(seed=seed)
    for veh in env.vehicles:
        subtask_idx = _choose_ready_subtask(env, veh)
        if subtask_idx is not None:
            for helper in env.vehicles:
                if int(helper.id) != int(veh.id):
                    return env, veh, helper, int(subtask_idx), obs_list
    raise RuntimeError("No ready vehicle/helper pair found for static gate.")


def _estimate_three_modes(env, vehicle, helper, subtask_idx, task_comp, task_data):
    common = {
        "task_comp": float(task_comp),
        "task_data": float(task_data),
        "comm_wait_dict": env._compute_comm_wait(vehicle.id),
        "active_v2i_count": 0,
        "active_v2i_vehicles": [],
        "active_v2v_vehicles": [],
    }
    return {
        "local": env._estimate_snapshot_target_cost(vehicle, subtask_idx, "Local", **common),
        "rsu": env._estimate_snapshot_target_cost(vehicle, subtask_idx, ("RSU", 0), **common),
        "v2v": env._estimate_snapshot_target_cost(vehicle, subtask_idx, int(helper.id), **common),
    }


def _best_mode(costs):
    finite = [(mode, item) for mode, item in costs.items() if bool(item.get("available", False)) and np.isfinite(item.get("J", np.nan))]
    if not finite:
        return "none"
    return min(finite, key=lambda kv: kv[1]["J"])[0]


def _fmt_cost(mode, est):
    if not bool(est.get("available", False)) or not np.isfinite(est.get("J", np.nan)):
        return f"{mode}=NA"
    return (
        f"{mode}=J:{est['J']:.3f} "
        f"(comm_wait:{est.get('comm_wait', 0.0):.3f}, tx:{est.get('tx_time', 0.0):.3f}, "
        f"cpu_wait:{est.get('cpu_wait', 0.0):.3f}, cpu_exec:{est.get('cpu_exec', 0.0):.3f}, "
        f"contact:{est.get('contact_penalty', 0.0):.3f})"
    )


def _run_scenario(name, expected_mode, task_comp, task_data, rates, waits, contact_penalties, freqs):
    env, vehicle, helper, subtask_idx, _ = _prepare_env()
    rsu = env.rsus[0]
    vehicle.pos = np.asarray(rsu.position, dtype=float).copy()
    vehicle.vel = np.zeros(2, dtype=float)
    helper.pos = np.asarray(vehicle.pos, dtype=float) + np.array([1.0, 0.0], dtype=float)
    helper.vel = np.zeros(2, dtype=float)
    vehicle.cpu_freq = float(freqs["local"])
    helper.cpu_freq = float(freqs["helper"])
    rsu.cpu_freq = float(freqs["rsu"])

    def mock_comm_wait(_veh_id):
        return {"total_v2i": float(waits["comm_wait_rsu"]), "total_v2v": float(waits["comm_wait_v2v"])}

    def mock_rate(src_vehicle_id, dst_type, dst_id=None, power_dbm=None, active_v2i_count=None,
                  active_v2i_vehicles=None, active_v2v_vehicles=None):
        _ = (src_vehicle_id, dst_id, power_dbm, active_v2i_count, active_v2i_vehicles, active_v2v_vehicles)
        if dst_type in ("rsu", "RSU"):
            return float(rates["rsu"])
        if dst_type in ("v2v", "V2V"):
            return float(rates["v2v"])
        return float("inf")

    def mock_contact_penalty(_vehicle, target, _t_comm_phase):
        if isinstance(target, tuple) and len(target) >= 2 and target[0] == "RSU":
            penalty = float(contact_penalties["rsu"])
        elif isinstance(target, int):
            penalty = float(contact_penalties["v2v"])
        else:
            penalty = 0.0
        return penalty, float("inf"), float("inf")

    def mock_rsu_queue_load(rsu_id, processor_id=None):
        _ = processor_id
        return float(waits["rsu_queue_cycles"]) if int(rsu_id) == 0 else 0.0

    def mock_rsu_queue_wait_time(rsu_id):
        return float(waits["rsu_min_wait"]) if int(rsu_id) == 0 else 0.0

    def mock_arrival_work(rsu_id, owner_vehicle_id=None):
        _ = owner_vehicle_id
        return float(waits["rsu_arrival_cycles"]) if int(rsu_id) == 0 else 0.0

    def mock_veh_queue_load(veh_id):
        if int(veh_id) == int(vehicle.id):
            return float(waits["local_queue_cycles"])
        if int(veh_id) == int(helper.id):
            return float(waits["helper_queue_cycles"])
        return 0.0

    def mock_veh_queue_wait_time(veh_id, cpu_freq=None):
        _ = cpu_freq
        if int(veh_id) == int(vehicle.id):
            return float(waits["local_wait"])
        if int(veh_id) == int(helper.id):
            return float(waits["helper_wait"])
        return 0.0

    with _patched(
        env,
        _compute_comm_wait=mock_comm_wait,
        _estimate_link_rate_expected=mock_rate,
        _estimate_contact_penalty_snapshot=mock_contact_penalty,
        _get_rsu_queue_load=mock_rsu_queue_load,
        _get_rsu_queue_wait_time=mock_rsu_queue_wait_time,
        _estimate_rsu_arrival_work_proxy=mock_arrival_work,
        _get_veh_queue_load=mock_veh_queue_load,
        _get_veh_queue_wait_time=mock_veh_queue_wait_time,
        _is_rsu_queue_full=lambda rsu_id, cycles=0.0: False,
        _is_veh_queue_full=lambda veh_id, cycles=0.0: False,
    ):
        costs = _estimate_three_modes(env, vehicle, helper, subtask_idx, task_comp=task_comp, task_data=task_data)

    actual_mode = _best_mode(costs)
    passed = actual_mode == expected_mode
    lines = [
        f"scenario={name} expected={expected_mode} actual={actual_mode} pass={passed}",
        "  " + _fmt_cost("Local", costs["local"]),
        "  " + _fmt_cost("RSU", costs["rsu"]),
        "  " + _fmt_cost("V2V", costs["v2v"]),
    ]
    return passed, "\n".join(lines)


def main():
    scenarios = [
        {
            "name": "A_empty_queue_good_link",
            "expected": "rsu",
            "task_comp": 6.0e8,
            "task_data": 2.0e6,
            "rates": {"rsu": 4.0e8, "v2v": 2.0e8},
            "waits": {
                "comm_wait_rsu": 0.0,
                "comm_wait_v2v": 0.0,
                "rsu_queue_cycles": 0.0,
                "rsu_arrival_cycles": 0.0,
                "rsu_min_wait": 0.0,
                "local_queue_cycles": 0.0,
                "helper_queue_cycles": 0.0,
                "local_wait": 0.0,
                "helper_wait": 0.0,
            },
            "contact_penalties": {"rsu": 0.0, "v2v": 0.0},
            "freqs": {"local": 1.5e9, "helper": 1.8e9, "rsu": 4.0e9},
        },
        {
            "name": "B_rsu_heavy_queue_small_task_local_idle",
            "expected": "local",
            "task_comp": 1.5e8,
            "task_data": 1.0e6,
            "rates": {"rsu": 3.0e8, "v2v": 2.5e8},
            "waits": {
                "comm_wait_rsu": 0.0,
                "comm_wait_v2v": 0.0,
                "rsu_queue_cycles": 7.2e9,
                "rsu_arrival_cycles": 2.4e9,
                "rsu_min_wait": 0.22,
                "local_queue_cycles": 0.0,
                "helper_queue_cycles": 0.0,
                "local_wait": 0.0,
                "helper_wait": 0.0,
            },
            "contact_penalties": {"rsu": 0.0, "v2v": 0.0},
            "freqs": {"local": 1.5e9, "helper": 1.4e9, "rsu": 4.0e9},
        },
        {
            "name": "C_rsu_moderate_congestion_helper_idle_good_contact",
            "expected": "v2v",
            "task_comp": 5.0e8,
            "task_data": 2.0e6,
            "rates": {"rsu": 2.5e8, "v2v": 4.5e8},
            "waits": {
                "comm_wait_rsu": 0.0,
                "comm_wait_v2v": 0.0,
                "rsu_queue_cycles": 4.8e9,
                "rsu_arrival_cycles": 2.4e9,
                "rsu_min_wait": 0.12,
                "local_queue_cycles": 0.0,
                "helper_queue_cycles": 0.0,
                "local_wait": 0.0,
                "helper_wait": 0.0,
            },
            "contact_penalties": {"rsu": 0.0, "v2v": 0.0},
            "freqs": {"local": 1.4e9, "helper": 3.5e9, "rsu": 4.0e9},
        },
        {
            "name": "D_helper_poor_contact_or_high_tx",
            "expected": "rsu",
            "task_comp": 4.0e8,
            "task_data": 2.0e7,
            "rates": {"rsu": 3.5e8, "v2v": 3.0e7},
            "waits": {
                "comm_wait_rsu": 0.0,
                "comm_wait_v2v": 0.08,
                "rsu_queue_cycles": 1.2e9,
                "rsu_arrival_cycles": 0.0,
                "rsu_min_wait": 0.03,
                "local_queue_cycles": 0.0,
                "helper_queue_cycles": 0.0,
                "local_wait": 0.0,
                "helper_wait": 0.0,
            },
            "contact_penalties": {"rsu": 0.0, "v2v": 0.25},
            "freqs": {"local": 1.5e9, "helper": 3.5e9, "rsu": 4.0e9},
        },
    ]

    results = []
    for scenario in scenarios:
        passed, details = _run_scenario(
            scenario["name"],
            scenario["expected"],
            scenario["task_comp"],
            scenario["task_data"],
            scenario["rates"],
            scenario["waits"],
            scenario["contact_penalties"],
            scenario["freqs"],
        )
        results.append((passed, details))
        print(details)
    total_pass = sum(int(passed) for passed, _ in results)
    print(f"static_gate_passed {total_pass}/{len(results)}")


if __name__ == "__main__":
    main()
