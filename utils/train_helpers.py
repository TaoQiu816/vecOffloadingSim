"""
[训练辅助函数] utils/train_helpers.py
Training Helper Functions

作用 (Purpose):
    提供训练脚本所需的通用工具函数，包括：
    - 文件/目录操作
    - 格式化输出
    - 环境变量解析
    - JSON序列化

设计原则 (Design Principles):
    - 纯函数，无副作用
    - 通用性，不依赖特定训练逻辑
"""

import os
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =========================================================================
# 文件/目录操作 (File/Directory Operations)
# =========================================================================

def ensure_dir(path):
    """
    确保目录存在，如不存在则创建

    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def read_last_jsonl(path):
    """
    读取JSONL文件的最后一行并解析为JSON

    Args:
        path: JSONL文件路径

    Returns:
        dict: 解析后的JSON对象，失败返回None
    """
    last = None
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


# =========================================================================
# 表格格式化 (Table Formatting)
# =========================================================================

def format_table_header(columns):
    """
    格式化表格头部

    Args:
        columns: [(label, width), ...] 列定义列表

    Returns:
        str: 格式化的表头字符串
    """
    parts = []
    for col in columns:
        label, width = col
        parts.append(str(label).center(width))
    return "| " + " | ".join(parts) + " |"


def format_table_divider(columns):
    """
    格式化表格分隔线

    Args:
        columns: [(label, width), ...] 列定义列表

    Returns:
        str: 分隔线字符串
    """
    parts = []
    for _, width in columns:
        parts.append("-" * width)
    return "+-" + "-+-".join(parts) + "-+"


def format_table_row(values, columns):
    """
    格式化表格数据行

    Args:
        values: {key: value, ...} 数据字典
        columns: [(key, width), ...] 列定义列表

    Returns:
        str: 格式化的数据行字符串
    """
    parts = []
    for col in columns:
        key, width = col
        val = values.get(key)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            cell = "-"
        elif isinstance(val, (int, np.integer)):
            cell = str(val)
        elif isinstance(val, float):
            cell = f"{val:.3f}"
        else:
            cell = str(val)
        parts.append(cell.rjust(width))
    return "| " + " | ".join(parts) + " |"


# =========================================================================
# 时间限制惩罚计算 (Time Limit Penalty)
# =========================================================================

def compute_time_limit_penalty(mode, remaining_time, deadline, base_penalty, k, ratio_clip):
    """
    计算时间限制惩罚

    Args:
        mode: "fixed" 或 "scaled"
        remaining_time: 剩余时间（秒）
        deadline: 截止时间（秒）
        base_penalty: 固定模式下的基础惩罚值
        k: scaled模式下的缩放系数
        ratio_clip: 比例裁剪上限

    Returns:
        tuple: (penalty, ratio) 惩罚值和时间比例
    """
    if mode == "scaled":
        if remaining_time is None or not np.isfinite(remaining_time):
            return 0.0, 0.0
        denom = max(deadline if deadline is not None and np.isfinite(deadline) else 1.0, 1e-6)
        ratio = np.clip(remaining_time / denom, 0.0, ratio_clip)
        penalty = -float(k) * float(ratio)
        return penalty, ratio
    else:
        return float(base_penalty), 0.0


# =========================================================================
# 环境变量解析 (Environment Variable Parsing)
# =========================================================================

def env_int(name):
    """
    从环境变量读取整数值

    Args:
        name: 环境变量名

    Returns:
        int or None: 整数值，解析失败返回None
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def env_float(name):
    """
    从环境变量读取浮点数值

    Args:
        name: 环境变量名

    Returns:
        float or None: 浮点数值，解析失败返回None
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def env_bool(name):
    """
    从环境变量读取布尔值

    Args:
        name: 环境变量名

    Returns:
        bool or None: 布尔值，未设置返回None
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    return str(raw).lower() in ("1", "true", "yes", "on")


def env_str(name):
    """
    从环境变量读取字符串值

    Args:
        name: 环境变量名

    Returns:
        str or None: 字符串值，空字符串返回None
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = str(raw).strip()
    return raw if raw else None


def bool_env(name, default=False):
    """
    从环境变量读取布尔值（带默认值）

    Args:
        name: 环境变量名
        default: 默认值

    Returns:
        bool: 布尔值
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# =========================================================================
# JSON序列化 (JSON Serialization)
# =========================================================================

def json_default(obj):
    """
    JSON序列化默认处理器，支持NumPy和PyTorch类型

    Args:
        obj: 待序列化对象

    Returns:
        序列化后的对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if TORCH_AVAILABLE and torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return str(obj)


# =========================================================================
# 导出列表
# =========================================================================

__all__ = [
    'ensure_dir',
    'read_last_jsonl',
    'format_table_header',
    'format_table_divider',
    'format_table_row',
    'compute_time_limit_penalty',
    'env_int',
    'env_float',
    'env_bool',
    'env_str',
    'bool_env',
    'json_default',
]
