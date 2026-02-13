"""
Queue-aware EFT Policy (Earliest Finish Time).

接口约束:
- 仅输出 target/power，不参与子任务选择（与环境动作接口一致）。
"""

import numpy as np
from typing import Dict, List, Optional
from configs.config import SystemConfig as Cfg


class EFTPolicy:
    """Queue-aware EFT 启发式策略"""

    def __init__(self, env, target_snr_db: float = 10.0):
        self.env = env
        self.target_snr_db = float(target_snr_db)
        self.target_snr_linear = 10.0 ** (self.target_snr_db / 10.0)
        self._p_min_w = Cfg.dbm2watt(Cfg.TX_POWER_MIN_DBM)
        self._p_max_w = Cfg.dbm2watt(Cfg.TX_POWER_MAX_DBM)

    def _min_power_a(self, distance: float, link_type: str = "V2V") -> float:
        d = max(float(distance), 1.0)
        if link_type == "V2I":
            bw = Cfg.BW_V2I / max(self.env._estimate_v2i_users(), 1)
            alpha = Cfg.PL_BETA_V2I
        else:
            bw = Cfg.V2V_BW_PER_RB
            alpha = Cfg.PL_BETA_V2V

        beta0 = self.env.channel.beta0
        g_d = beta0 * (d ** (-alpha))
        noise_psd_dbm = Cfg.NOISE_POWER_DENSITY_DBM + Cfg.NOISE_FIGURE
        noise_psd_w = Cfg.dbm2watt(noise_psd_dbm)
        noise_w = noise_psd_w * bw
        p_req = self.target_snr_linear * noise_w / max(g_d, 1e-30)

        p_req = float(np.clip(p_req, self._p_min_w, self._p_max_w))
        log_ratio = np.log(self._p_max_w / max(self._p_min_w, 1e-30))
        if log_ratio < 1e-12:
            return 0.5
        a_power = np.log(p_req / max(self._p_min_w, 1e-30)) / log_ratio
        return float(np.clip(a_power, 0.0, 1.0))

    def _obs_comp_lb(self, obs: Dict, idx: int) -> Optional[float]:
        raw = obs.get("resource_raw")
        if raw is None:
            return None
        try:
            if idx >= len(raw):
                return None
            return float(max(raw[idx][11], 0.0) * 10.0)
        except Exception:
            return None

    def _obs_distance(self, obs: Dict, idx: int, ctype: int) -> Optional[float]:
        raw = obs.get("resource_raw")
        if raw is None:
            return None
        try:
            d_norm = float(raw[idx][2])
        except Exception:
            return None
        if ctype == 2:
            return float(max(d_norm, 0.0) * max(float(Cfg.RSU_RANGE), 1e-6))
        if ctype == 3:
            return float(max(d_norm, 0.0) * max(float(Cfg.V2V_RANGE), 1e-6))
        return 0.0

    def _rate_from_obs(self, obs: Dict, idx: int, ctype: int) -> Optional[float]:
        # First choice: previous-step realized rate (already aligned with env share/interference).
        rate_prev = obs.get("rate_prev")
        try:
            if rate_prev is not None and idx < len(rate_prev):
                rpn = float(rate_prev[idx])
                if np.isfinite(rpn) and rpn > 0.0:
                    norm = Cfg.NORM_MAX_RATE_V2I if ctype == 2 else Cfg.NORM_MAX_RATE_V2V
                    return float(max(rpn * norm, 1e-9))
        except Exception:
            pass

        # Fallback: resource_raw reference-rate channel.
        raw = obs.get("resource_raw")
        try:
            if raw is not None and idx < len(raw):
                rn = float(raw[idx][3])
                if ctype == 2:
                    return float(max(rn * Cfg.NORM_MAX_RATE_V2I, 1e-9))
                if ctype == 3:
                    log_max = float(getattr(self.env, "_v2v_ref_log_max", 0.0))
                    if log_max > 1e-12:
                        return float(max(np.expm1(np.clip(rn, 0.0, 1.0) * log_max), 1e-9))
                    return float(max(rn * Cfg.NORM_MAX_RATE_V2V, 1e-9))
        except Exception:
            pass
        return None

    def _estimate_rate(self, obs: Dict, idx: int, ctype: int, vehicle, target_pos, link_type: str, p_dbm: float) -> float:
        obs_rate = self._rate_from_obs(obs, idx, ctype)
        if obs_rate is not None:
            return float(max(obs_rate, 1e-6))
        if link_type == "V2I":
            rate = self.env.channel.compute_one_rate(
                vehicle, target_pos, "V2I", self.env.time,
                power_dbm_override=p_dbm,
                v2i_user_count=self.env._estimate_v2i_users(),
            )
        else:
            rate = self.env.channel.compute_one_rate(
                vehicle, target_pos, "V2V", self.env.time,
                power_dbm_override=p_dbm,
            )
        return float(max(rate, 1e-6))

    def _eft_local(self, vehicle, task_comp: float, obs: Dict) -> float:
        comp_lb = self._obs_comp_lb(obs, 0)
        if comp_lb is not None:
            return float(max(comp_lb, 0.0))
        backlog = self.env._get_veh_queue_load(vehicle.id)
        return float((task_comp + backlog) / max(vehicle.cpu_freq, 1e-6))

    def _eft_rsu(self, vehicle, rsu_id: int, task_comp: float, task_data: float, obs: Dict, idx: int):
        rsu = self.env.rsus[rsu_id]
        dist = self._obs_distance(obs, idx, ctype=2)
        if dist is None:
            dist = float(np.linalg.norm(np.array(rsu.position) - np.array(vehicle.pos)))
        a_pw = self._min_power_a(dist, "V2I")
        p_w = self._p_min_w * (self._p_max_w / max(self._p_min_w, 1e-12)) ** a_pw
        p_dbm = Cfg.watt2dbm(p_w)

        rate = self._estimate_rate(obs, idx, 2, vehicle, rsu.position, "V2I", p_dbm)
        t_tx = (task_data * 8.0) / rate if task_data > 0 else 0.0
        comm_wait = self.env._compute_comm_wait(vehicle.id).get("total_v2i", 0.0)

        comp_lb = self._obs_comp_lb(obs, idx)
        if comp_lb is None:
            rsu_backlog = self.env._get_rsu_queue_load(rsu_id)
            comp_lb = (task_comp + rsu_backlog) / max(rsu.cpu_freq, 1e-6)

        return float(comm_wait + t_tx + comp_lb), float(a_pw)

    def _eft_v2v(self, vehicle, target_veh, task_comp: float, task_data: float, obs: Dict, idx: int):
        dist = self._obs_distance(obs, idx, ctype=3)
        if dist is None:
            dist = float(np.linalg.norm(np.array(target_veh.pos) - np.array(vehicle.pos)))
        if dist > Cfg.V2V_RANGE:
            return float("inf"), 0.5
        a_pw = self._min_power_a(dist, "V2V")
        p_w = self._p_min_w * (self._p_max_w / max(self._p_min_w, 1e-12)) ** a_pw
        p_dbm = Cfg.watt2dbm(p_w)

        rate = self._estimate_rate(obs, idx, 3, vehicle, target_veh.pos, "V2V", p_dbm)
        t_tx = (task_data * 8.0) / rate if task_data > 0 else 0.0
        comm_wait = self.env._compute_comm_wait(vehicle.id).get("total_v2v", 0.0)

        comp_lb = self._obs_comp_lb(obs, idx)
        if comp_lb is None:
            nbr_backlog = self.env._get_veh_queue_load(target_veh.id)
            comp_lb = (task_comp + nbr_backlog) / max(target_veh.cpu_freq, 1e-6)

        return float(comm_wait + t_tx + comp_lb), float(a_pw)

    def _target_score(self, obs: Dict, idx: int, ctype: int, eft_time: float, task_comp: float, task_data: float, subtask_idx: int) -> float:
        _ = (obs, idx, ctype, task_comp, task_data, subtask_idx)
        return float(eft_time)

    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        actions = []
        for i, obs in enumerate(obs_list):
            vehicle = self.env.vehicles[i]
            self._current_vehicle = vehicle
            dag = vehicle.task_dag
            subtask_idx = dag.get_top_priority_task() if dag else None
            if subtask_idx is None or dag.is_finished or dag.is_failed:
                act = {"target": 0, "power": 0.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            task_comp = dag.total_comp[subtask_idx] if subtask_idx < len(dag.total_comp) else Cfg.MEAN_COMP_LOAD
            task_data = dag.total_data[subtask_idx] if subtask_idx < len(dag.total_data) else 0.0

            candidate_ids = obs.get("candidate_ids")
            candidate_types = obs.get("candidate_types")
            action_mask = obs.get("action_mask")
            if candidate_ids is None or action_mask is None:
                act = {"target": 0, "power": 0.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            best_idx = 0
            best_power = 0.0
            best_score = self._target_score(obs, 0, 1, self._eft_local(vehicle, task_comp, obs), task_comp, task_data, subtask_idx)

            for idx in range(1, len(candidate_ids)):
                if action_mask[idx] <= 0.0:
                    continue
                cid = int(candidate_ids[idx])
                if cid < 0:
                    continue
                ctype = int(candidate_types[idx]) if candidate_types is not None and idx < len(candidate_types) else 0
                if ctype == 2:
                    if 0 <= cid < len(self.env.rsus):
                        eft_time, a_pw = self._eft_rsu(vehicle, cid, task_comp, task_data, obs, idx)
                    else:
                        continue
                elif ctype == 3:
                    if cid == vehicle.id:
                        continue
                    target_veh = self.env._get_vehicle_by_id(cid)
                    if target_veh is None:
                        continue
                    eft_time, a_pw = self._eft_v2v(vehicle, target_veh, task_comp, task_data, obs, idx)
                else:
                    continue
                score = self._target_score(obs, idx, ctype, eft_time, task_comp, task_data, subtask_idx)
                if score < best_score:
                    best_score = score
                    best_idx = idx
                    best_power = a_pw

            act = {"target": int(best_idx), "power": float(best_power)}
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)

        self._current_vehicle = None
        return actions

    def reset(self):
        pass


EFTPPolicy = EFTPolicy
