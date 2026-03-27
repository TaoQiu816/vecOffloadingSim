"""
Nearest-RSU Offloading (NRO) baseline.

Policy:
- For the selected READY subtask, choose the nearest currently reachable RSU.
- If no legal RSU candidate exists, fall back to local execution.
- Use a fixed medium transmit power for remote offloading.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from baselines.action_utils import attach_subtask


class NearestRSUPolicy:
    """Nearest reachable RSU baseline with local fallback."""

    def __init__(self, env, fixed_power: float = 0.5):
        self.env = env
        self.fixed_power = float(np.clip(fixed_power, 0.0, 1.0))

    @staticmethod
    def _pick_nearest_rsu(obs: Dict) -> int | None:
        action_mask = obs.get("action_mask")
        candidate_types = obs.get("candidate_types")
        resource_raw = obs.get("resource_raw")
        if action_mask is None or candidate_types is None or resource_raw is None:
            return None

        valid = np.where(np.asarray(action_mask) > 0)[0]
        if len(valid) == 0:
            return None

        rsu_candidates = []
        for idx in valid:
            if int(candidate_types[idx]) != 2:
                continue
            try:
                dist_norm = float(resource_raw[idx][3])
            except Exception:
                dist_norm = float("inf")
            rsu_candidates.append((dist_norm, int(idx)))

        if not rsu_candidates:
            return None
        rsu_candidates.sort(key=lambda item: item[0])
        return int(rsu_candidates[0][1])

    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        actions = []
        for obs in obs_list:
            target = self._pick_nearest_rsu(obs)
            if target is None:
                act = {"target": 0, "power": 0.0}
            else:
                act = {"target": int(target), "power": self.fixed_power}
            act = attach_subtask(obs, act)
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        return actions

    def reset(self):
        pass
