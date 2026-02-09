"""
Critical-Path-First + EFT Policy

策略：
1. 任务选择：关键路径优先（L_bwd 最大的 READY 子任务）
2. 资源选择：Queue-aware EFT argmin
3. 功率控制：目标 SNR0 最小功率并 clip

与 EFTPolicy 区别：本策略显式优先调度关键路径任务，
而 EFTPolicy 使用环境默认的优先级选择。
"""

import numpy as np
from typing import Dict, List
from configs.config import SystemConfig as Cfg
from baselines.eft_policy import EFTPolicy


class CPFirstEFTPolicy(EFTPolicy):
    """Critical-Path-First + EFT 启发式策略"""

    def __init__(self, env, target_snr_db=10.0):
        super().__init__(env, target_snr_db=target_snr_db)

    def _get_critical_path_task(self, dag):
        """
        选择 L_bwd 最大的 READY 子任务（关键路径优先）。

        L_bwd[i] = 从节点 i 到 DAG 出口的最长路径长度（以 cycles 计）。
        """
        ready_mask = (dag.status == 1)
        ready_indices = np.where(ready_mask)[0]
        if len(ready_indices) == 0:
            return None

        # L_bwd: 关键路径后向长度
        l_bwd = getattr(dag, 'L_bwd', None)
        if l_bwd is None:
            # fallback: 使用计算量作为代理
            l_bwd = dag.total_comp

        # 选择 READY 中 L_bwd 最大的
        ready_lbwd = l_bwd[ready_indices]
        best_local = np.argmax(ready_lbwd)
        return int(ready_indices[best_local])

    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        actions = []
        for i, obs in enumerate(obs_list):
            vehicle = self.env.vehicles[i]
            dag = vehicle.task_dag

            if dag is None or dag.is_finished or dag.is_failed:
                act = {"target": 0, "power": 0.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue

            # 关键路径优先：选择 L_bwd 最大的 READY 任务
            subtask_idx = self._get_critical_path_task(dag)
            if subtask_idx is None:
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

            # EFT 资源选择（复用父类方法）
            best_idx = 0
            best_time = self._eft_local(vehicle, task_comp)
            best_power = 0.0

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
