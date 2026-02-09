"""
Queue-aware EFT Policy (Earliest Finish Time)

EFT(v,m) = t_wait_lb(m) + w_v/f_m + t_comm_lb(v,m)
选择 argmin EFT 的目标。

功率控制：给定目标 SNR0 计算最小功率并 clip 到 [P_min, P_max]。
"""

import numpy as np
from typing import Dict, List
from configs.config import SystemConfig as Cfg


class EFTPolicy:
    """Queue-aware EFT 启发式策略"""

    # 目标 SNR (线性)，默认 10 dB = 10.0
    TARGET_SNR_LINEAR = 10.0  # 10 dB

    def __init__(self, env, target_snr_db=10.0):
        self.env = env
        self.target_snr_db = target_snr_db
        self.target_snr_linear = 10.0 ** (target_snr_db / 10.0)
        # 功率上下限 (W)
        self._p_min_w = Cfg.dbm2watt(Cfg.TX_POWER_MIN_DBM)
        self._p_max_w = Cfg.dbm2watt(Cfg.TX_POWER_MAX_DBM)

    # ------------------------------------------------------------------
    # 功率控制：给定 SNR0 计算最小功率 a_power ∈ [0,1]
    # ------------------------------------------------------------------
    def _min_power_a(self, distance, link_type='V2V'):
        """
        计算达到目标 SNR 所需的最小 a_power ∈ [0,1]。
        P_req = SNR0 * N0 * B / G(d)
        a_power = log(P_req/P_min) / log(P_max/P_min)
        """
        d = max(distance, 1.0)
        if link_type == 'V2I':
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

        # 映射到 a_power ∈ [0,1]
        p_req = float(np.clip(p_req, self._p_min_w, self._p_max_w))
        log_ratio = np.log(self._p_max_w / max(self._p_min_w, 1e-30))
        if log_ratio < 1e-12:
            return 0.5
        a_power = np.log(p_req / max(self._p_min_w, 1e-30)) / log_ratio
        return float(np.clip(a_power, 0.0, 1.0))

    # ------------------------------------------------------------------
    # EFT 估计
    # ------------------------------------------------------------------
    def _eft_local(self, vehicle, task_comp):
        """Local EFT = queue_wait + comp_time"""
        backlog = self.env._get_veh_queue_load(vehicle.id)  # cycles
        comp_lb = (task_comp + backlog) / max(vehicle.cpu_freq, 1e-6)
        return comp_lb  # 无传输

    def _eft_rsu(self, vehicle, rsu_id, task_comp, task_data):
        """RSU EFT = t_tx_lb + tx_wait + comp_wait + comp_time"""
        rsu = self.env.rsus[rsu_id]
        dist = np.linalg.norm(np.array(rsu.position) - np.array(vehicle.pos))
        a_pw = self._min_power_a(dist, 'V2I')
        p_w = self._p_min_w * (self._p_max_w / max(self._p_min_w, 1e-12)) ** a_pw
        p_dbm = Cfg.watt2dbm(p_w)

        rate = self.env.channel.compute_one_rate(
            vehicle, rsu.position, 'V2I', self.env.time,
            power_dbm_override=p_dbm,
            v2i_user_count=self.env._estimate_v2i_users()
        )
        rate = max(rate, 1e-6)
        t_tx = (task_data * 8.0) / rate if task_data > 0 else 0.0

        # comm queue wait
        cw = self.env._compute_comm_wait(vehicle.id)
        t_tx_wait = cw.get('total_v2i', 0.0)

        # compute queue
        rsu_backlog = self.env._get_rsu_queue_load(rsu_id)
        t_comp = (task_comp + rsu_backlog) / max(rsu.cpu_freq, 1e-6)

        return t_tx_wait + t_tx + t_comp, a_pw

    def _eft_v2v(self, vehicle, target_veh, task_comp, task_data):
        """V2V EFT = t_tx_lb + tx_wait + comp_wait + comp_time"""
        dist = np.linalg.norm(np.array(target_veh.pos) - np.array(vehicle.pos))
        if dist > Cfg.V2V_RANGE:
            return float('inf'), 0.5
        a_pw = self._min_power_a(dist, 'V2V')
        p_w = self._p_min_w * (self._p_max_w / max(self._p_min_w, 1e-12)) ** a_pw
        p_dbm = Cfg.watt2dbm(p_w)

        rate = self.env.channel.compute_one_rate(
            vehicle, target_veh.pos, 'V2V', self.env.time,
            power_dbm_override=p_dbm
        )
        rate = max(rate, 1e-6)
        t_tx = (task_data * 8.0) / rate if task_data > 0 else 0.0

        cw = self.env._compute_comm_wait(vehicle.id)
        t_tx_wait = cw.get('total_v2v', 0.0)

        nbr_backlog = self.env._get_veh_queue_load(target_veh.id)
        t_comp = (task_comp + nbr_backlog) / max(target_veh.cpu_freq, 1e-6)

        return t_tx_wait + t_tx + t_comp, a_pw

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------
    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        actions = []
        for i, obs in enumerate(obs_list):
            vehicle = self.env.vehicles[i]
            dag = vehicle.task_dag
            subtask_idx = dag.get_top_priority_task() if dag else None
            if subtask_idx is None or dag.is_finished or dag.is_failed:
                act = {"target": 0, "power": 0.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            task_comp = (
                dag.total_comp[subtask_idx]
                if subtask_idx < len(dag.total_comp)
                else Cfg.MEAN_COMP_LOAD
            )
            task_data = (
                dag.total_data[subtask_idx]
                if subtask_idx < len(dag.total_data)
                else 0.0
            )

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
            best_time = self._eft_local(vehicle, task_comp)
            best_power = 0.0  # Local 无需功率

            for idx in range(1, len(candidate_ids)):
                if action_mask[idx] <= 0.0:
                    continue
                cid = int(candidate_ids[idx])
                if cid < 0:
                    continue
                ctype = int(candidate_types[idx]) if candidate_types is not None and idx < len(candidate_types) else 0

                if ctype == 2:  # RSU
                    if 0 <= cid < len(self.env.rsus):
                        t_eft, a_pw = self._eft_rsu(vehicle, cid, task_comp, task_data)
                        if t_eft < best_time:
                            best_time = t_eft
                            best_idx = idx
                            best_power = a_pw
                elif ctype == 3:  # V2V
                    if cid == vehicle.id:
                        continue
                    target_veh = self.env._get_vehicle_by_id(cid)
                    if target_veh is None:
                        continue
                    t_eft, a_pw = self._eft_v2v(vehicle, target_veh, task_comp, task_data)
                    if t_eft < best_time:
                        best_time = t_eft
                        best_idx = idx
                        best_power = a_pw

            act = {"target": int(best_idx), "power": float(best_power)}
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)

        return actions

    def reset(self):
        pass


# 兼容旧脚本拼写
EFTPPolicy = EFTPolicy
