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

            # 以candidate_mask为准，保证与候选集一致
            candidate_mask = obs.get('candidate_mask', obs['action_mask'])
            candidate_ids = obs.get('candidate_ids')
            valid_targets = np.where(candidate_mask > 0)[0]
            
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
                    rsu_id = None
                    if candidate_ids is not None and len(candidate_ids) > 1:
                        rsu_id = int(candidate_ids[1])
                    if rsu_id is None or not (0 <= rsu_id < len(self.env.rsus)):
                        compute_power = 0.0
                    else:
                        compute_power = self.env.rsus[rsu_id].cpu_freq
                else:
                    # Neighbor: 邻居车辆的CPU频率
                    # target_idx = 2 + neighbor_index
                    neighbor_id = None
                    if candidate_ids is not None and target_idx < len(candidate_ids):
                        neighbor_id = int(candidate_ids[target_idx])
                    if neighbor_id is None or neighbor_id < 0 or neighbor_id == vehicle.id:
                        compute_power = 0.0
                    else:
                        neighbor_vehicle = self.env._get_vehicle_by_id(neighbor_id)
                        compute_power = neighbor_vehicle.cpu_freq if neighbor_vehicle else 0.0
                
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
