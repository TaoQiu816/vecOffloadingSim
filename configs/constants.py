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
    动作索引常量（静态默认值仅用于回退兼容）

    [通用化] 推荐使用 from_candidate_types() 动态获取边界，
    而非假设固定的 LOCAL=0/RSU=1/V2V_START=2 布局。
    """
    LOCAL = 0       # 本地执行（始终index 0）
    RSU = 1         # RSU执行（默认index 1，多RSU时为1..N_RSU）
    V2V_START = 2   # V2V动作起始索引（默认2，多RSU时为 1+NUM_RSU）

    @classmethod
    def from_config(cls):
        """从config动态计算V2V_START（不修改类属性，返回新值）"""
        from configs.config import SystemConfig as Cfg
        enable_rsu = getattr(Cfg, "ENABLE_RSU_SELECTION", False)
        num_rsu = getattr(Cfg, "NUM_RSU", 3)
        v2v_start = (1 + num_rsu) if enable_rsu else 2
        return cls.LOCAL, cls.RSU, v2v_start

    @classmethod
    def from_candidate_types(cls, types):
        """
        [通用化] 从 candidate_types 数组动态获取各类型索引。
        types: np.ndarray 或 list，1=Local, 2=RSU, 3=V2V
        返回: (local_indices, rsu_indices, v2v_indices) 三个列表
        """
        import numpy as np
        types = np.asarray(types)
        local_idx = np.where(types == 1)[0].tolist()
        rsu_idx = np.where(types == 2)[0].tolist()
        v2v_idx = np.where(types == 3)[0].tolist()
        return local_idx, rsu_idx, v2v_idx

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
