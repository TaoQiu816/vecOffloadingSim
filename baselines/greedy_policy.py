"""
贪婪卸载策略 (Greedy Offloading Policy)

主线口径：
- 在合法目标中选择当前一步估计完成时间最小的目标。
- 只使用主线 observation 的真实可观测量和环境当前队列状态。
- 功率控制保持最大值，作为当前主线的简单强基线。
"""

import numpy as np
from typing import List, Dict
from configs.config import SystemConfig as Cfg
from baselines.action_utils import attach_subtask


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
                act = attach_subtask(obs, act, preferred=subtask_idx)
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
                act = attach_subtask(obs, act, preferred=subtask_idx)
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue
            
            # 选择估计完成时间最小目标（贪婪一步）。
            scores = []
            for target_idx in valid_targets:
                cand_type = int(candidate_types[target_idx]) if candidate_types is not None and target_idx < len(candidate_types) else 0
                if cand_type == 1:  # Local
                    queue_cycles = float(self.env._get_veh_queue_load(vehicle.id))
                    score = (task_comp + queue_cycles) / max(float(vehicle.cpu_freq), 1e-6)
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
                        t_rsu_wait = float(self.env._get_rsu_queue_wait_time(rsu_id))
                        t_comp = task_comp / max(float(rsu.cpu_freq), 1e-6)
                        score = q_wait + t_tx + t_rsu_wait + t_comp
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
                else:
                    score = float("inf")
                scores.append(float(score))
            
            # 选择估计完成时间最小的目标
            best_idx = int(np.argmin(scores)) if len(scores) > 0 else 0
            best_target = valid_targets[best_idx]
            # 使用最大功率
            act = {
                'target': int(best_target),
                'power': 1.0
            }
            act = attach_subtask(obs, act, preferred=subtask_idx)
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        
        return actions
    
    def reset(self):
        """重置策略状态（贪婪策略无状态）"""
        pass
