"""
奖励计算引擎：从状态变化计算奖励

职责：
- 计算CFT变化（dT_rem）
- 计算能耗惩罚（energy_norm）
- 组合奖励（时间收益 - 能耗惩罚）
- 处理硬约束触发和非法动作

设计原则：
- 纯函数风格（无副作用）
- 奖励公式、裁剪范围与原实现完全一致
- 统计收集可选（通过参数控制）
"""
from typing import TYPE_CHECKING, Dict, List, Tuple, Any
import numpy as np

if TYPE_CHECKING:
    from envs.vec_offloading_env import VecOffloadingEnv  # type: ignore
    from envs.entities.vehicle import Vehicle  # type: ignore
else:
    VecOffloadingEnv = object
    Vehicle = object


class RewardEngine:
    """
    奖励计算引擎：负责从状态变化计算奖励
    
    【重要】保持与原实现完全一致：
    - CFT计算方法（_compute_mean_cft_pi0）
    - 奖励公式（compute_absolute_reward）
    - 裁剪范围（REWARD_MIN, REWARD_MAX）
    - 硬约束触发逻辑
    """
    
    def __init__(self, config):
        """
        Args:
            config: 系统配置对象
        """
        self.config = config
    
    def compute(self, env: VecOffloadingEnv, prev_state: Dict[str, Any],
                curr_state: Dict[str, Any], step_events: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """
        [主入口] 计算所有车辆的奖励
        
        Args:
            env: 环境实例
            prev_state: 上一步状态快照（包含cft_prev等）
            curr_state: 当前步状态快照
            step_events: 本步事件记录（tx_time, power_ratio等）
            
        Returns:
            (rewards, reward_stats): 奖励列表和统计信息
        """
        # [委托到env的原方法] 当前阶段保持行为完全一致
        # 后续可逐步将逻辑迁移到此处
        raise NotImplementedError(
            "RewardEngine当前作为框架存在，实际逻辑仍在VecOffloadingEnv.step中。"
            "阶段6的目标是建立框架并集成，逐步迁移逻辑需要更细致的测试。"
        )

