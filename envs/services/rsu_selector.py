"""
[RSU选择服务] envs/services/rsu_selector.py
RSU Selection Service

作用 (Purpose):
    提供RSU选择和查询功能，包括：
    - 查找最近RSU
    - 查找覆盖范围内所有RSU
    - 选择最佳RSU（基于负载和距离）

设计原则 (Design Principles):
    - 无状态服务，依赖注入RSU列表
    - 支持多RSU场景
    - 可配置的选择策略
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.entities.rsu import RSU


class RSUSelector:
    """
    RSU选择器服务

    功能：
    - 查找最近RSU
    - 查找覆盖范围内所有RSU
    - 选择最佳RSU（考虑负载、距离、预计完成时间）
    """

    def __init__(self, rsus: List['RSU'], channel_model=None, config=None):
        """
        初始化RSU选择器

        Args:
            rsus: RSU实体列表
            channel_model: 信道模型（用于速率计算）
            config: 系统配置
        """
        self.rsus = rsus
        self.channel = channel_model
        self.config = config

    def get_nearest_rsu(self, position: np.ndarray) -> Optional['RSU']:
        """
        获取距离指定位置最近且在覆盖范围内的RSU

        Args:
            position: 位置坐标 [x, y]

        Returns:
            RSU or None: 最近的RSU，如果不在任何RSU覆盖范围内返回None
        """
        if len(self.rsus) == 0:
            return None

        nearest_rsu = None
        min_dist = float('inf')

        for rsu in self.rsus:
            if rsu.is_in_coverage(position):
                dist = rsu.get_distance(position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_rsu = rsu

        return nearest_rsu

    def get_all_rsus_in_range(self, position: np.ndarray) -> List['RSU']:
        """
        获取覆盖范围内所有RSU

        Args:
            position: 位置坐标 [x, y]

        Returns:
            list: 覆盖范围内的RSU列表
        """
        return [rsu for rsu in self.rsus if rsu.is_in_coverage(position)]

    def is_rsu_location(self, loc) -> bool:
        """
        判断位置是否为RSU位置

        Args:
            loc: 位置标识（'RSU'或('RSU', rsu_id)元组）

        Returns:
            bool: 是否为RSU位置
        """
        if loc == 'RSU':
            return True
        if isinstance(loc, tuple) and len(loc) == 2:
            if loc[0] == 'RSU':
                return True
        return False

    def get_rsu_id_from_location(self, loc) -> Optional[int]:
        """
        从位置标识提取RSU ID

        Args:
            loc: 位置标识

        Returns:
            int or None: RSU ID
        """
        if loc == 'RSU':
            return 0 if len(self.rsus) > 0 else None
        if isinstance(loc, tuple) and len(loc) == 2 and loc[0] == 'RSU':
            rsu_id = loc[1]
            if 0 <= rsu_id < len(self.rsus):
                return rsu_id
        return None

    def get_rsu_position(self, rsu_id: int) -> Optional[np.ndarray]:
        """
        获取指定RSU的位置

        Args:
            rsu_id: RSU ID

        Returns:
            np.ndarray or None: RSU位置坐标
        """
        if 0 <= rsu_id < len(self.rsus):
            return self.rsus[rsu_id].position
        return None

    def get_rsu_by_id(self, rsu_id: int) -> Optional['RSU']:
        """
        根据ID获取RSU实体

        Args:
            rsu_id: RSU ID

        Returns:
            RSU or None: RSU实体
        """
        if 0 <= rsu_id < len(self.rsus):
            return self.rsus[rsu_id]
        return None


# 导出列表
__all__ = ['RSUSelector']
