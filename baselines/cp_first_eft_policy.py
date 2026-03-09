"""
CP-aware EFT Policy (interface-consistent).

说明:
- 与环境动作接口保持一致: 仅做 target/power 决策，不做子任务选择。
- CP-aware 只作为 target 评分附加项，不改变环境调度到的子任务。
"""

import numpy as np
from typing import Dict
from baselines.eft_policy import EFTPolicy


class CPFirstEFTPolicy(EFTPolicy):
    """Critical-path aware target scoring on top of EFT."""

    def __init__(self, env, target_snr_db: float = 10.0):
        super().__init__(env, target_snr_db=target_snr_db)

    def _criticality_weight(self, vehicle, subtask_idx: int) -> float:
        dag = vehicle.task_dag
        if dag is None or subtask_idx is None or subtask_idx < 0:
            return 0.0
        l_bwd = getattr(dag, "L_bwd", None)
        if l_bwd is None or len(l_bwd) == 0 or subtask_idx >= len(l_bwd):
            return 0.0
        val = float(l_bwd[subtask_idx])
        vmax = float(np.max(l_bwd)) if np.max(l_bwd) > 0 else 1.0
        return float(np.clip(val / max(vmax, 1e-9), 0.0, 1.0))

    def _target_score(self, obs: Dict, idx: int, ctype: int, eft_time: float, task_comp: float, task_data: float, subtask_idx: int) -> float:
        _ = (task_comp, task_data)
        vehicle = getattr(self, "_current_vehicle", None)
        if vehicle is None:
            return float(eft_time)
        crit = self._criticality_weight(vehicle, subtask_idx)
        if crit <= 0:
            return float(eft_time)

        # 主线 resource_raw(10维): col7 contact_norm, col8 contention, col9 occupancy
        raw = obs.get("resource_raw")
        if raw is None or idx >= len(raw):
            return float(eft_time)
        row = raw[idx]
        contact_norm = float(np.clip(row[7], 0.0, 1.0))
        contention = float(np.clip(row[8], 0.0, 1.0))
        occupancy = float(np.clip(row[9], 0.0, 1.0))

        # CP-aware correction: critical task prefers stable / less-congested targets.
        # Keep correction small to preserve EFT primary behavior.
        w_contact = 0.20
        w_contention = 0.15
        w_occupancy = 0.10
        correction = crit * (
            w_contact * (1.0 - contact_norm)
            + w_contention * contention
            + w_occupancy * occupancy
        )
        return float(eft_time + correction)
