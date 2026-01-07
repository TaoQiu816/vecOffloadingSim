"""
[全局常量定义] constants.py
Global Constants Definition

作用 (Purpose):
    定义全局使用的常量值，确保一致性和可维护性。
    Defines global constant values to ensure consistency and maintainability.

主要常量 (Main Constants):
    - MASK_VALUE: 用于softmax前屏蔽无效位置的值
    - 任务状态常量

修复问题 (Fixes):
    - P33: 统一mask值，避免-1e9/-1e10/float('-inf')混用导致的数值差异
"""

import torch
from typing import Optional

# =============================================================================
# Mask值常量 (Mask Value Constants)
# =============================================================================

# 统一的mask值：用于softmax前屏蔽无效位置
# 选择-1e10而非float('-inf')的原因：
# 1. 避免softmax产生NaN（当所有值都是-inf时）
# 2. 数值稳定性更好
# 3. 与现有代码兼容（大部分使用-1e10）
MASK_VALUE: float = -1e10

def get_mask_value(dtype: Optional[torch.dtype] = None,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """
    获取适当类型和设备的mask值张量

    Args:
        dtype: 目标数据类型（如torch.float32）
        device: 目标设备（如'cuda'或'cpu'）

    Returns:
        torch.Tensor: mask值张量

    示例:
        >>> mask_val = get_mask_value(dtype=torch.float32, device='cuda')
        >>> masked_logits = torch.where(mask, logits, mask_val)
    """
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cpu')
    return torch.tensor(MASK_VALUE, dtype=dtype, device=device)


# =============================================================================
# 任务状态常量 (Task Status Constants)
# =============================================================================

class TaskStatus:
    """
    任务状态枚举

    状态转换: PENDING -> READY -> RUNNING -> COMPLETED
                                    └── FAILED
    """
    PENDING = 0     # 等待前驱完成
    READY = 1       # 就绪，可调度
    RUNNING = 2     # 执行中
    COMPLETED = 3   # 已完成
    FAILED = 4      # 执行失败

    @classmethod
    def to_string(cls, status: int) -> str:
        """将状态ID转换为字符串"""
        mapping = {
            cls.PENDING: "PENDING",
            cls.READY: "READY",
            cls.RUNNING: "RUNNING",
            cls.COMPLETED: "COMPLETED",
            cls.FAILED: "FAILED"
        }
        return mapping.get(status, f"UNKNOWN({status})")


# =============================================================================
# 资源角色常量 (Resource Role Constants)
# =============================================================================

class ResourceRole:
    """
    资源角色ID枚举

    用于resource_ids字段，表示执行目标的角色类型
    """
    PADDING = 0     # 填充位置（无效）
    LOCAL = 1       # 本地执行
    RSU = 2         # RSU执行
    NEIGHBOR = 3    # 邻居车辆（所有邻居统一为3）

    @classmethod
    def to_string(cls, role_id: int) -> str:
        """将角色ID转换为字符串"""
        if role_id == cls.PADDING:
            return "PADDING"
        elif role_id == cls.LOCAL:
            return "LOCAL"
        elif role_id == cls.RSU:
            return "RSU"
        elif role_id >= cls.NEIGHBOR:
            return f"NEIGHBOR_{role_id - cls.NEIGHBOR}"
        return f"UNKNOWN({role_id})"


# =============================================================================
# 动作索引常量 (Action Index Constants)
# =============================================================================

class ActionIndex:
    """
    动作索引常量

    用于action_mask和target动作
    """
    LOCAL = 0       # 本地执行
    RSU = 1         # RSU执行
    V2V_START = 2   # V2V动作起始索引（2, 3, 4, ...）

    @classmethod
    def get_v2v_index(cls, neighbor_idx: int) -> int:
        """获取第neighbor_idx个邻居的动作索引"""
        return cls.V2V_START + neighbor_idx

    @classmethod
    def is_v2v(cls, action_idx: int) -> bool:
        """判断是否为V2V动作"""
        return action_idx >= cls.V2V_START

    @classmethod
    def get_neighbor_idx(cls, action_idx: int) -> int:
        """从动作索引获取邻居索引"""
        return action_idx - cls.V2V_START


# =============================================================================
# 位置编码常量 (Location Encoding Constants)
# =============================================================================

class LocationCode:
    """
    位置编码常量

    用于exec_locations和task_locations字段
    """
    UNSCHEDULED = None  # 未调度
    LOCAL = 'Local'     # 本地执行

    @staticmethod
    def is_rsu(location) -> bool:
        """判断是否为RSU位置"""
        return isinstance(location, tuple) and location[0] == 'RSU'

    @staticmethod
    def is_v2v(location) -> bool:
        """判断是否为V2V位置（邻居车辆ID）"""
        return isinstance(location, int)

    @staticmethod
    def get_rsu_id(location) -> int:
        """从RSU位置获取RSU ID"""
        if isinstance(location, tuple) and location[0] == 'RSU':
            return location[1]
        return -1
