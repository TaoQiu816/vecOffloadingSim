"""
基准策略模块

包含以下基准方法：
1. RandomPolicy: 随机卸载策略
2. LocalOnlyPolicy: 全本地执行策略
3. GreedyPolicy: 贪婪卸载策略（选择计算能力最强的节点）
4. StaticPolicy: 静态策略（仅使用初始观测）
"""

from .random_policy import RandomPolicy
from .local_only_policy import LocalOnlyPolicy
from .greedy_policy import GreedyPolicy
from .static_policy import StaticPolicy
from .eft_policy import EFTPPolicy

__all__ = ['RandomPolicy', 'LocalOnlyPolicy', 'GreedyPolicy', 'StaticPolicy', 'EFTPPolicy']
