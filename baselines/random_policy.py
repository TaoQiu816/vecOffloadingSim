"""
随机卸载策略 (Random Offloading Policy)

策略描述：
- 在所有合法的卸载目标（Local/RSU/Neighbors）中随机选择
- 功率控制：随机选择[0, 1]范围内的功率比例
- 用于验证RL算法是否学到了有效策略
"""

import numpy as np
from typing import List, Dict


class RandomPolicy:
    """随机卸载策略"""
    
    def __init__(self, seed=None):
        """
        Args:
            seed: 随机种子，用于复现
        """
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, obs_list: List[Dict]) -> List[Dict]:
        """
        根据观测选择动作
        
        Args:
            obs_list: 环境观测列表，每个元素包含一个车辆的观测
        
        Returns:
            actions: 动作列表，每个元素包含 {'target': int, 'power': float}
        """
        actions = []
        
        for obs in obs_list:
            # 获取动作掩码（合法的卸载目标）
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask > 0)[0]
            
            if len(valid_targets) == 0:
                # 如果没有合法目标，默认选择本地执行
                target = 0
            else:
                # 从合法目标中随机选择
                target = self.rng.choice(valid_targets)
            
            # 随机选择功率比例 [0.2, 1.0]
            # 避免过低功率导致传输时间过长
            power = self.rng.uniform(0.2, 1.0)
            
            act = {
                'target': int(target),
                'power': float(power)
            }
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        
        return actions
    
    def reset(self):
        """重置策略状态（随机策略无状态）"""
        pass
