"""
LB-Greedy baseline.

按动作口径一致的下界代价进行 target 选择:
LB(target) = t_comm_lb + t_comp_lb
"""

from typing import Dict
from baselines.eft_policy import EFTPolicy


class LBGreedyPolicy(EFTPolicy):
    """Lower-bound greedy target selection baseline."""

    def _target_score(self, obs: Dict, idx: int, ctype: int, eft_time: float, task_comp: float, task_data: float, subtask_idx: int) -> float:
        _ = (subtask_idx, task_comp)
        comp_lb = self._obs_comp_lb(obs, idx)
        if comp_lb is None:
            return float(eft_time)
        if idx == 0:
            return float(comp_lb)

        rate = self._rate_from_obs(obs, idx, ctype)
        if rate is None:
            return float(eft_time)
        # task_data 已是 bit，速率是 bit/s，因此无需额外 *8。
        tx_lb = float(task_data) / max(float(rate), 1e-6) if task_data > 0 else 0.0

        # 与环境推进口径对齐：加入基础通信排队等待下界（V2I / V2V）。
        comm_wait_lb = 0.0
        veh = getattr(self, "_current_vehicle", None)
        if veh is not None and ctype in (2, 3):
            cw = self.env._compute_comm_wait(veh.id)
            comm_wait_lb = float(cw.get("total_v2i" if ctype == 2 else "total_v2v", 0.0))

        return float(comp_lb + tx_lb + comm_wait_lb)
