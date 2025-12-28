"""
全本地执行策略 (Local-Only Policy)

策略描述：
- 所有任务都在本地车辆上执行，不进行任何卸载
- 功率设置为最大值（虽然本地执行不需要传输）
- 用于评估卸载策略相比本地执行的性能提升
"""

import numpy as np
from typing import List, Dict


class LocalOnlyPolicy:
    """全本地执行策略"""
    
    def __init__(self):
        """初始化本地执行策略"""
        pass
    
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
            # 始终选择本地执行 (target=0)
            actions.append({
                'target': 0,
                'power': 1.0  # 本地执行不需要传输，功率设为最大值
            })
        
        return actions
    
    def reset(self):
        """重置策略状态（本地策略无状态）"""
        pass

