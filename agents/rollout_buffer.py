"""
Rollout Buffer with GAE computation for PPO
支持动态车辆数量
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Generator


class RolloutBuffer:
    """
    经验回放缓冲区，用于存储单个episode的轨迹数据并计算GAE
    支持动态车辆数量（每步车辆数可能不同）
    """
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Args:
            gamma: 折扣因子
            gae_lambda: GAE平滑参数
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 存储容器 - 列表形式支持动态车辆数
        self.obs_list_buffer = []  # [T][N_t] 观测字典列表
        self.actions_buffer = []   # [T][N_t] 动作字典列表
        self.rewards_buffer = []   # [T] 每步是长度N_t的数组
        self.values_buffer = []    # [T] 每步是长度N_t的数组
        self.log_probs_buffer = [] # [T] 每步是长度N_t的数组
        self.dones_buffer = []     # [T] 每步是标量
        
        # GAE计算结果 - 列表形式
        self.advantages_buffer = []  # [T] 每步是长度N_t的数组
        self.returns_buffer = []     # [T] 每步是长度N_t的数组
        
    def add(self, obs_list: List[Dict], actions: List[Dict], 
            rewards: List[float], values: np.ndarray, 
            log_probs: np.ndarray, done: bool):
        """
        添加一步数据
        
        Args:
            obs_list: 观测列表（字典列表）
            actions: 动作列表（字典列表）
            rewards: 奖励列表
            values: 状态价值 (numpy数组或tensor)
            log_probs: 动作log概率 (numpy数组或tensor)
            done: 是否结束
        """
        self.obs_list_buffer.append(obs_list)
        self.actions_buffer.append(actions)
        
        # 确保rewards是numpy数组
        if isinstance(rewards, list):
            rewards = np.array(rewards, dtype=np.float32)
        elif not isinstance(rewards, np.ndarray):
            rewards = np.array([rewards], dtype=np.float32)
        self.rewards_buffer.append(rewards.astype(np.float32))
        
        # 确保values和log_probs是numpy数组
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
            
        self.values_buffer.append(values.astype(np.float32).flatten())
        self.log_probs_buffer.append(log_probs.astype(np.float32).flatten())
        self.dones_buffer.append(done)
    
    def compute_returns_and_advantages(self, last_value: np.ndarray):
        """
        计算GAE优势函数和returns
        支持动态车辆数量
        
        Args:
            last_value: 最后一步的状态价值（用于bootstrap）
        """
        # 确保last_value是numpy数组
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.cpu().numpy()
        last_value = last_value.flatten()
        
        T = len(self.rewards_buffer)
        
        # 清空之前的计算结果
        self.advantages_buffer = []
        self.returns_buffer = []
        
        # 为每个时间步预分配advantages数组
        for t in range(T):
            N_t = len(self.rewards_buffer[t])
            self.advantages_buffer.append(np.zeros(N_t, dtype=np.float32))
            self.returns_buffer.append(np.zeros(N_t, dtype=np.float32))
        
        # 获取最后一步的车辆数
        N_last = len(self.rewards_buffer[-1])
        
        # 从后向前计算GAE（每个车辆独立计算）
        # 注意：由于车辆数量可能变化，我们对每个车辆索引分别处理
        # 这里假设车辆ID在episode内是稳定的（索引0始终是同一辆车）
        
        # 获取所有时间步的最大车辆数
        max_N = max(len(r) for r in self.rewards_buffer)
        
        # 对每个车辆索引独立计算GAE
        for n in range(max_N):
            gae = 0.0
            for t in reversed(range(T)):
                N_t = len(self.rewards_buffer[t])
                N_v = len(self.values_buffer[t])
                
                # 如果当前时间步没有这个车辆（检查rewards和values两者），跳过
                if n >= N_t or n >= N_v:
                    gae = 0.0  # 重置GAE
                    continue
                
                # 获取当前时间步的数据
                reward = self.rewards_buffer[t][n]
                value = self.values_buffer[t][n]
                done = self.dones_buffer[t]
                
                # 获取下一步的value
                if t == T - 1:
                    # 最后一步使用bootstrap value
                    if n < len(last_value):
                        next_value = last_value[n]
                    else:
                        next_value = 0.0
                    next_non_terminal = 1.0 - float(done)
                else:
                    N_next = len(self.values_buffer[t + 1])
                    if n < N_next:
                        next_value = self.values_buffer[t + 1][n]
                    else:
                        next_value = 0.0
                    next_non_terminal = 1.0 - float(done)
                
                # TD error
                delta = reward + self.gamma * next_value * next_non_terminal - value
                
                # GAE
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                self.advantages_buffer[t][n] = gae
        
        # 计算returns = advantages + values
        for t in range(T):
            # 确保形状匹配
            N_adv = len(self.advantages_buffer[t])
            N_val = len(self.values_buffer[t])
            min_N = min(N_adv, N_val)
            self.returns_buffer[t] = self.advantages_buffer[t][:min_N] + self.values_buffer[t][:min_N]
            self.advantages_buffer[t] = self.advantages_buffer[t][:min_N]
    
    def get_batches(self, batch_size: int) -> Generator[Dict, None, None]:
        """
        生成mini-batch用于训练
        
        Args:
            batch_size: batch大小
            
        Yields:
            batch字典，包含obs_list, actions, old_log_probs, advantages, returns
        """
        T = len(self.obs_list_buffer)
        
        # 展平所有数据
        flat_obs_list = []
        flat_actions = []
        flat_log_probs = []
        flat_advantages = []
        flat_returns = []
        
        for t in range(T):
            N_t = len(self.obs_list_buffer[t])
            for n in range(N_t):
                flat_obs_list.append(self.obs_list_buffer[t][n])
                flat_actions.append(self.actions_buffer[t][n])
                flat_log_probs.append(self.log_probs_buffer[t][n])
                flat_advantages.append(self.advantages_buffer[t][n])
                flat_returns.append(self.returns_buffer[t][n])
        
        total_samples = len(flat_obs_list)
        
        if total_samples == 0:
            return
        
        # 转换为numpy数组
        flat_log_probs = np.array(flat_log_probs, dtype=np.float32)
        flat_advantages = np.array(flat_advantages, dtype=np.float32)
        flat_returns = np.array(flat_returns, dtype=np.float32)
        
        # 归一化advantages
        adv_std = flat_advantages.std()
        if adv_std > 1e-8:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (adv_std + 1e-8)
        
        # 随机打乱
        indices = np.random.permutation(total_samples)
        
        # 生成batches
        num_batches = max(1, total_samples // batch_size)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            if len(batch_indices) == 0:
                continue
            
            batch = {
                'obs_list': [flat_obs_list[idx] for idx in batch_indices],
                'actions': [flat_actions[idx] for idx in batch_indices],
                'old_log_probs': flat_log_probs[batch_indices],
                'advantages': flat_advantages[batch_indices],
                'returns': flat_returns[batch_indices]
            }
            
            yield batch
    
    def clear(self):
        """清空buffer"""
        self.obs_list_buffer.clear()
        self.actions_buffer.clear()
        self.rewards_buffer.clear()
        self.values_buffer.clear()
        self.log_probs_buffer.clear()
        self.dones_buffer.clear()
        self.advantages_buffer.clear()
        self.returns_buffer.clear()
    
    def __len__(self):
        return len(self.obs_list_buffer)
