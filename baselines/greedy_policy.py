"""
贪婪卸载策略 (Greedy Offloading Policy)

策略描述：
- 在合法目标中选择“估计完成时间”最小的目标（单步贪婪）
- 估计口径与环境保持一致：bits/bps、cycles/Hz、基础队列等待
- 功率控制：保持最大功率（与原策略一致）
"""

import numpy as np
from typing import List, Dict
from configs.config import SystemConfig as Cfg


class GreedyPolicy:
    """贪婪卸载策略"""
    
    def __init__(self, env):
        """
        Args:
            env: 环境实例，用于获取车辆和RSU的计算能力信息
        """
        self.env = env

    @staticmethod
    def _tx_time_seconds(task_data_bits: float, rate_bps: float) -> float:
        if task_data_bits <= 0:
            return 0.0
        return float(task_data_bits / max(rate_bps, 1e-6))

    def _rate_from_obs(self, obs: Dict, idx: int, ctype: int):
        rate_prev = obs.get("rate_prev")
        try:
            if rate_prev is not None and idx < len(rate_prev):
                rpn = float(rate_prev[idx])
                if np.isfinite(rpn) and rpn > 0.0:
                    norm = Cfg.NORM_MAX_RATE_V2I if ctype == 2 else Cfg.NORM_MAX_RATE_V2V
                    return float(max(rpn * norm, 1e-9))
        except Exception:
            pass

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

    def _estimate_rate(self, obs: Dict, idx: int, ctype: int, vehicle, target_pos, link_type: str) -> float:
        obs_rate = self._rate_from_obs(obs, idx, ctype)
        if obs_rate is not None:
            return float(max(obs_rate, 1e-6))
        p_dbm = float(getattr(vehicle, "tx_power_dbm", Cfg.TX_POWER_MAX_DBM))
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
    
    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        """
        根据观测选择动作
        
        Args:
            obs_list: 环境观测列表，每个元素包含一个车辆的观测
        
        Returns:
            actions: 动作列表，每个元素包含 {'target': int, 'power': float}
        """
        actions = []
        
        for i, obs in enumerate(obs_list):
            vehicle = self.env.vehicles[i]
            dag = vehicle.task_dag
            subtask_idx = dag.get_top_priority_task() if dag is not None else None
            if subtask_idx is None or dag.is_finished or dag.is_failed:
                act = {'target': 0, 'power': 1.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            task_comp = float(dag.total_comp[subtask_idx]) if subtask_idx < len(dag.total_comp) else float(Cfg.MEAN_COMP_LOAD)
            task_data = float(dag.total_data[subtask_idx]) if subtask_idx < len(dag.total_data) else 0.0

            # 使用统一动作掩码
            candidate_mask = obs['action_mask']
            candidate_ids = obs.get('candidate_ids')
            candidate_types = obs.get('candidate_types')
            valid_targets = np.where(candidate_mask > 0)[0]
            
            if len(valid_targets) == 0:
                # 如果没有合法目标，默认选择本地执行
                act = {'target': 0, 'power': 1.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue
            
            # 选择估计完成时间最小目标（贪婪一步）
            # 仅在低信誉/高不确定度场景下才触发远端回退，避免退化为纯本地。
            # 在先验信誉尚未收敛（rho≈0.5）时更保守地启用远端，
            # 仅当远端估计时延显著优于本地才切换，避免20ep校准阶段被高失败尾部拖垮。
            remote_switch_margin = 0.50
            scores = []
            local_score = float("inf")
            local_target = None
            for target_idx in valid_targets:
                cand_type = int(candidate_types[target_idx]) if candidate_types is not None and target_idx < len(candidate_types) else 0
                if cand_type == 1:  # Local
                    queue_cycles = float(self.env._get_veh_queue_load(vehicle.id))
                    score = (task_comp + queue_cycles) / max(float(vehicle.cpu_freq), 1e-6)
                    local_score = float(score)
                    local_target = int(target_idx)
                elif cand_type == 2:  # RSU
                    rsu_id = None
                    if candidate_ids is not None and target_idx < len(candidate_ids):
                        rsu_id = int(candidate_ids[target_idx])
                    if rsu_id is None or not (0 <= rsu_id < len(self.env.rsus)):
                        score = float("inf")
                    else:
                        rsu = self.env.rsus[rsu_id]
                        q_wait = float(self.env._compute_comm_wait(vehicle.id).get("total_v2i", 0.0))
                        rate = self._estimate_rate(obs, target_idx, 2, vehicle, rsu.position, "V2I")
                        t_tx = self._tx_time_seconds(task_data, rate)
                        rsu_q_cycles = float(self.env._get_rsu_queue_load(rsu_id))
                        t_comp = (task_comp + rsu_q_cycles) / max(float(rsu.cpu_freq), 1e-6)
                        score = q_wait + t_tx + t_comp
                        # 信誉风险保守修正：rho低时放大远端代价（与环境口径一致，不改动力学）
                        rho = 1.0
                        if obs.get("resource_raw") is not None and target_idx < obs["resource_raw"].shape[0] and obs["resource_raw"].shape[1] >= 13:
                            rho = float(np.clip(obs["resource_raw"][target_idx, 12], 0.05, 1.0))
                        score = score / rho
                        if rho < 0.70:
                            score *= 1.5
                elif cand_type == 3:  # V2V
                    neighbor_id = None
                    if candidate_ids is not None and target_idx < len(candidate_ids):
                        neighbor_id = int(candidate_ids[target_idx])
                    if neighbor_id is None or neighbor_id < 0 or neighbor_id == vehicle.id:
                        score = float("inf")
                    else:
                        neighbor_vehicle = self.env._get_vehicle_by_id(neighbor_id)
                        if neighbor_vehicle is None:
                            score = float("inf")
                        else:
                            q_wait = float(self.env._compute_comm_wait(vehicle.id).get("total_v2v", 0.0))
                            rate = self._estimate_rate(obs, target_idx, 3, vehicle, neighbor_vehicle.pos, "V2V")
                            t_tx = self._tx_time_seconds(task_data, rate)
                            nbr_q_cycles = float(self.env._get_veh_queue_load(neighbor_id))
                            t_comp = (task_comp + nbr_q_cycles) / max(float(neighbor_vehicle.cpu_freq), 1e-6)
                            score = q_wait + t_tx + t_comp
                            rho = 1.0
                            if obs.get("resource_raw") is not None and target_idx < obs["resource_raw"].shape[0] and obs["resource_raw"].shape[1] >= 13:
                                rho = float(np.clip(obs["resource_raw"][target_idx, 12], 0.05, 1.0))
                            score = score / rho
                            if rho < 0.70:
                                score *= 1.5
                else:
                    score = float("inf")
                scores.append(float(score))
            
            # 选择估计完成时间最小的目标
            best_idx = int(np.argmin(scores)) if len(scores) > 0 else 0
            best_target = valid_targets[best_idx]
            best_score = float(scores[best_idx]) if len(scores) > 0 else float("inf")
            best_type = int(candidate_types[best_target]) if candidate_types is not None and best_target < len(candidate_types) else 0
            best_rho = 1.0
            best_unc = 0.0
            if (
                obs.get("resource_raw") is not None
                and 0 <= int(best_target) < obs["resource_raw"].shape[0]
                and obs["resource_raw"].shape[1] >= 13
            ):
                best_rho = float(np.clip(obs["resource_raw"][int(best_target), 12], 0.05, 1.0))
            if (
                obs.get("resource_raw") is not None
                and 0 <= int(best_target) < obs["resource_raw"].shape[0]
                and obs["resource_raw"].shape[1] >= 14
            ):
                best_unc = float(np.clip(obs["resource_raw"][int(best_target), 13], 0.0, 1.0))

            # 条件回退：仅在低信誉/高不确定度时要求远端显著优于本地
            if local_target is not None and best_type in (2, 3):
                trust_bad = (best_rho < 0.70) or (best_unc > 0.75)
                if (
                    (not np.isfinite(best_score))
                    or (trust_bad and best_score >= (1.0 - remote_switch_margin) * local_score)
                ):
                    best_target = int(local_target)
            
            # 使用最大功率
            act = {
                'target': int(best_target),
                'power': 1.0
            }
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        
        return actions
    
    def reset(self):
        """重置策略状态（贪婪策略无状态）"""
        pass
