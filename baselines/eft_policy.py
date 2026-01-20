"""
EFT Greedy Policy (Earliest Finish Time)

Selects the target with the smallest estimated completion time among:
- Local
- Serving RSU (index=1)
- Top-5 V2V neighbors (index>=2 from candidate set)

RSU choice is strictly serving RSU from candidate set.
"""

from typing import Dict, List

import numpy as np

from configs.config import SystemConfig as Cfg


class EFTPPolicy:
    def __init__(self, env, power=1.0):
        self.env = env
        self.power = float(np.clip(power, 0.0, 1.0))

    def _power_dbm(self):
        p_min = getattr(Cfg, "TX_POWER_MIN_DBM", Cfg.TX_POWER_MAX_DBM)
        p_max = getattr(Cfg, "TX_POWER_MAX_DBM", p_min)
        return float(np.clip(p_min + self.power * (p_max - p_min), p_min, p_max))

    def _estimate_local(self, vehicle, task_comp):
        queue_wait = self.env._get_veh_queue_wait_time(vehicle.id, vehicle.cpu_freq)
        comp_time = task_comp / max(vehicle.cpu_freq, 1e-6)
        return queue_wait + comp_time

    def _estimate_rsu(self, vehicle, serving_rsu_id, task_comp, task_data, comm_wait_v2i):
        rsu = self.env.rsus[serving_rsu_id]
        rate = self.env.channel.compute_one_rate(
            vehicle, rsu.position, "V2I", self.env.time,
            power_dbm_override=self._power_dbm(),
            v2i_user_count=self.env._estimate_v2i_users()
        )
        rate = max(rate, 1e-6)
        tx_time = task_data / rate if task_data > 0 else 0.0
        rsu_wait = self.env._get_rsu_queue_wait_time(serving_rsu_id)
        comp_time = task_comp / max(rsu.cpu_freq, 1e-6)
        return comm_wait_v2i + tx_time + rsu_wait + comp_time

    def _estimate_v2v(self, vehicle, target_veh, task_comp, task_data, comm_wait_v2v):
        rate = self.env.channel.compute_one_rate(
            vehicle, target_veh.pos, "V2V", self.env.time,
            power_dbm_override=self._power_dbm()
        )
        rate = max(rate, 1e-6)
        tx_time = task_data / rate if task_data > 0 else 0.0
        queue_wait = self.env._get_veh_queue_wait_time(target_veh.id, target_veh.cpu_freq)
        comp_time = task_comp / max(target_veh.cpu_freq, 1e-6)
        return comm_wait_v2v + tx_time + queue_wait + comp_time

    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        actions = []
        for i, obs in enumerate(obs_list):
            vehicle = self.env.vehicles[i]
            subtask_idx = vehicle.task_dag.get_top_priority_task()
            if subtask_idx is None:
                act = {"target": 0, "power": self.power}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            task_comp = (
                vehicle.task_dag.total_comp[subtask_idx]
                if subtask_idx < len(vehicle.task_dag.total_comp)
                else Cfg.MEAN_COMP_LOAD
            )
            task_data = (
                vehicle.task_dag.total_data[subtask_idx]
                if subtask_idx < len(vehicle.task_dag.total_data)
                else 0.0
            )

            comm_wait = self.env._compute_comm_wait(vehicle.id)
            comm_wait_v2i = comm_wait.get("total_v2i", 0.0)
            comm_wait_v2v = comm_wait.get("total_v2v", 0.0)

            candidate_ids = obs.get("candidate_ids")
            candidate_mask = obs.get("candidate_mask")
            if candidate_ids is None or candidate_mask is None:
                # Fallback to local if candidate set is missing
                act = {"target": 0, "power": self.power}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            best_idx = 0
            best_time = self._estimate_local(vehicle, task_comp)

            # Serving RSU (index 1)
            if candidate_mask[1] > 0.0:
                serving_rsu_id = int(candidate_ids[1])
                if 0 <= serving_rsu_id < len(self.env.rsus):
                    if self.env.rsus[serving_rsu_id].is_in_coverage(vehicle.pos):
                        t_rsu = self._estimate_rsu(
                            vehicle, serving_rsu_id, task_comp, task_data, comm_wait_v2i
                        )
                        if t_rsu < best_time:
                            best_time = t_rsu
                            best_idx = 1

            # V2V candidates (index >= 2)
            for idx in range(2, len(candidate_ids)):
                if candidate_mask[idx] <= 0.0:
                    continue
                neighbor_id = int(candidate_ids[idx])
                if neighbor_id < 0 or neighbor_id == vehicle.id:
                    continue
                target_veh = self.env._get_vehicle_by_id(neighbor_id)
                if target_veh is None:
                    continue
                t_v2v = self._estimate_v2v(
                    vehicle, target_veh, task_comp, task_data, comm_wait_v2v
                )
                if t_v2v < best_time:
                    best_time = t_v2v
                    best_idx = idx

            act = {"target": int(best_idx), "power": self.power}
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)

        return actions

    def reset(self):
        pass
