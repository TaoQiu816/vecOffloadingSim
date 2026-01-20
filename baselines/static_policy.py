"""
静态策略 (Static Policy)

策略描述：
- 在首个观测中为每个车辆选择一次目标
- 后续步骤保持不变，不随动态环境调整
- 作为“非自适应”对比基线
"""

import numpy as np
from typing import List, Dict


class StaticPolicy:
    """静态卸载策略（仅用初始观测做一次决策）"""

    def __init__(self):
        self.fixed_targets = None

    def reset(self):
        self.fixed_targets = None

    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        if self.fixed_targets is None or len(self.fixed_targets) != len(obs_list):
            self.fixed_targets = []
            for obs in obs_list:
                action_mask = obs['action_mask']
                valid_targets = np.where(action_mask > 0)[0]
                if len(valid_targets) == 0:
                    self.fixed_targets.append(0)
                    continue

                resource_raw = obs.get('resource_raw')
                if resource_raw is None or resource_raw.shape[1] < 14:
                    self.fixed_targets.append(int(valid_targets[0]))
                    continue

                scores = []
                for tgt in valid_targets:
                    est_exec = resource_raw[tgt][11]
                    est_comm = resource_raw[tgt][12]
                    est_wait = resource_raw[tgt][13]
                    scores.append(est_exec + est_comm + est_wait)
                best_idx = int(valid_targets[int(np.argmin(scores))])
                self.fixed_targets.append(best_idx)

        actions = []
        for i, obs in enumerate(obs_list):
            target = self.fixed_targets[i] if i < len(self.fixed_targets) else 0
            action_mask = obs['action_mask']
            if target >= len(action_mask) or not action_mask[target]:
                target = 0
            power = 1.0 if target != 0 else 1.0
            act = {'target': int(target), 'power': float(power)}
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        return actions
