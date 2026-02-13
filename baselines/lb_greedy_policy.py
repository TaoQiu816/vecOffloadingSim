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
        tx_lb = (float(task_data) * 8.0) / max(float(rate), 1e-6) if task_data > 0 else 0.0
        return float(comp_lb + tx_lb)

