"""
静态策略 (Static Policy)

主线口径：
- 在首个观测中为每个车辆选择一次目标
- 若提供 env，则按当前主线 EFT 估计一次性选定目标
- 后续步骤保持不变，不随动态环境调整
"""

import numpy as np
from typing import List, Dict
from baselines.action_utils import attach_subtask
from baselines.eft_policy import EFTPolicy


class StaticPolicy:
    """静态卸载策略（仅用初始观测做一次决策）"""

    def __init__(self, env=None):
        self.env = env
        self.fixed_targets = None
        self.fixed_power = None
        self._eft = EFTPolicy(env) if env is not None else None

    def reset(self):
        self.fixed_targets = None
        self.fixed_power = None

    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        if self.fixed_targets is None or len(self.fixed_targets) != len(obs_list):
            self.fixed_targets = []
            self.fixed_power = []
            for obs in obs_list:
                action_mask = obs['action_mask']
                valid_targets = np.where(action_mask > 0)[0]
                if len(valid_targets) == 0:
                    self.fixed_targets.append(0)
                    self.fixed_power.append(0.0)
                    continue

                if self._eft is None:
                    best_idx = int(valid_targets[0])
                    self.fixed_targets.append(best_idx)
                    self.fixed_power.append(1.0 if best_idx != 0 else 0.0)
                    continue

                act = self._eft.select_action([obs])[0]
                self.fixed_targets.append(int(act.get("target", 0)))
                self.fixed_power.append(float(act.get("power", 0.0)))

        actions = []
        for i, obs in enumerate(obs_list):
            target = self.fixed_targets[i] if i < len(self.fixed_targets) else 0
            action_mask = obs['action_mask']
            if target >= len(action_mask) or not action_mask[target]:
                target = 0
            power = self.fixed_power[i] if self.fixed_power is not None and i < len(self.fixed_power) else 0.0
            if target == 0:
                power = 0.0
            act = {'target': int(target), 'power': float(power)}
            act = attach_subtask(obs, act)
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        return actions
