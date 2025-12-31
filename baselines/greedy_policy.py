"""
贪婪卸载策略 (Greedy Offloading Policy)

策略描述：
- 在通信范围内的所有合法目标（Local/RSU/Neighbors）中选择计算能力最强的节点
- 计算能力排序：RSU > 高频车辆 > 低频车辆
- 功率控制：使用最大功率以获得最高传输速率
- 不考虑队列拥塞情况（纯贪婪）

评估意义：
- 验证考虑队列状态和通信开销的重要性
- 作为启发式算法的上界参考
"""

import numpy as np
from typing import List, Dict
from configs.config import SystemConfig as Cfg


class GreedyPolicy:
    """贪婪卸载策略"""
    
    def __init__(self, env):
        """
        Args:
            env: 环境实例，用于获取车辆和RSU的计算能力信息
        """
        self.env = env
    
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
            
            # 获取动作掩码（合法的卸载目标）
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask > 0)[0]
            
            if len(valid_targets) == 0:
                # 如果没有合法目标，默认选择本地执行
                act = {'target': 0, 'power': 1.0}
                if "obs_stamp" in obs:
                    act["obs_stamp"] = int(obs["obs_stamp"])
                actions.append(act)
                continue
            
            # 计算每个合法目标的计算能力
            target_compute_power = []
            
            for target_idx in valid_targets:
                if target_idx == 0:
                    # Local: 本地车辆的CPU频率
                    compute_power = vehicle.cpu_freq
                elif target_idx == 1:
                    # RSU: RSU的CPU频率（通常最高）
                    compute_power = Cfg.F_RSU
                else:
                    # Neighbor: 邻居车辆的CPU频率
                    # target_idx = 2 + neighbor_index
                    resource_ids = obs['resource_ids']
                    neighbor_token = resource_ids[target_idx]
                    if neighbor_token >= 3:
                        neighbor_id = neighbor_token - 3
                        neighbor_vehicle = next(
                            (veh for veh in self.env.vehicles if veh.id == neighbor_id),
                            None
                        )
                        compute_power = neighbor_vehicle.cpu_freq if neighbor_vehicle else 0.0
                    else:
                        compute_power = 0.0
                
                target_compute_power.append(compute_power)
            
            # 选择计算能力最强的目标
            best_idx = np.argmax(target_compute_power)
            best_target = valid_targets[best_idx]
            
            # 使用最大功率
            act = {
                'target': int(best_target),
                'power': 1.0
            }
            if "obs_stamp" in obs:
                act["obs_stamp"] = int(obs["obs_stamp"])
            actions.append(act)
        
        return actions
    
    def reset(self):
        """重置策略状态（贪婪策略无状态）"""
        pass
