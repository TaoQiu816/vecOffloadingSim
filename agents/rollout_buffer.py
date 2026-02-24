"""
[经验回放缓冲区] rollout_buffer.py
Rollout Buffer with GAE computation for PPO

作用 (Purpose):
    存储单个episode的轨迹数据，并计算Generalized Advantage Estimation (GAE)。
    支持动态车辆数量（每步车辆数可能不同）。
    Stores trajectory data for a single episode and computes GAE for PPO updates.
    Supports dynamic vehicle count (number of agents may vary per step).

核心功能 (Core Functions):
    1. add() - 添加一步的经验数据（obs, action, reward, value, log_prob）
    2. compute_returns_and_advantages() - 计算GAE优势和回报
    3. get_batches() - 生成mini-batch用于PPO更新
    4. clear() - 清空缓冲区（每个episode结束后调用）

GAE计算公式 (GAE Formula):
    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    A_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
    
    其中：
    - γ (gamma): 折扣因子，控制对未来奖励的重视程度
    - λ (lambda): GAE平滑因子，权衡偏差-方差
    - δ_t: TD残差
    - A_t: 优势函数

参考文献 (References):
    - GAE: Schulman et al., "High-Dimensional Continuous Control Using GAE" (2016)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Generator


class RolloutBuffer:
    """
    经验回放缓冲区 (Rollout Buffer)
    
    功能：
        - 存储单个episode的轨迹数据（obs, action, reward, value, log_prob）
        - 计算GAE优势和回报
        - 生成mini-batch用于PPO更新
        - 支持动态车辆数量（每步车辆数可能不同）
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
        self.active_masks_buffer = []  # [T] 每步是长度N_t的数组 (0/1)
        self.agent_ids_buffer = []  # [T] 每步是长度N_t的数组（尽量使用稳定车辆ID）
        
        # GAE计算结果 - 列表形式
        self.advantages_buffer = []  # [T] 每步是长度N_t的数组
        self.returns_buffer = []     # [T] 每步是长度N_t的数组
        
    def add(self, obs_list: List[Dict], actions: List[Dict],
            rewards: List[float], values: np.ndarray,
            log_probs: np.ndarray, done: bool,
            terminated: bool = None, truncated: bool = None,
            active_masks: List[int] = None):
        """
        添加一步数据
        
        Args:
            obs_list: 观测列表（字典列表）
            actions: 动作列表（字典列表）
            rewards: 奖励列表
            values: 状态价值 (numpy数组或tensor)
            log_probs: 动作log概率 (numpy数组或tensor)
            done: 是否结束（向后兼容）
            terminated: 是否真正终局（任务完成/失败）
            truncated: 是否时间截断
        """
        self.obs_list_buffer.append(obs_list)
        self.agent_ids_buffer.append(self._extract_agent_ids(obs_list))
        # 动作字典按原样透传，支持扩展键（如 subtask/target/power）
        if isinstance(actions, list):
            self.actions_buffer.append([
                dict(a) if isinstance(a, dict) else a for a in actions
            ])
        else:
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

        # active mask (1=有效决策，0=no_task/无决策)
        if active_masks is None:
            active_masks = np.ones_like(rewards, dtype=np.float32)
        else:
            if isinstance(active_masks, list):
                active_masks = np.array(active_masks, dtype=np.float32)
            elif not isinstance(active_masks, np.ndarray):
                active_masks = np.array([active_masks], dtype=np.float32)
            if len(active_masks) != len(rewards):
                active_masks = np.ones_like(rewards, dtype=np.float32)
        self.active_masks_buffer.append(active_masks.astype(np.float32))
        
        # [修复] 分离terminated和truncated
        # 如果提供了terminated/truncated，使用它们；否则向后兼容使用done
        if terminated is not None or truncated is not None:
            self.dones_buffer.append({
                'done': done,
                'terminated': terminated if terminated is not None else done,
                'truncated': truncated if truncated is not None else False
            })
        else:
            # 向后兼容：done作为terminated
            self.dones_buffer.append({
                'done': done,
                'terminated': done,
                'truncated': False
            })
    
    @staticmethod
    def _extract_agent_ids(obs_list: List[Dict]) -> np.ndarray:
        """从观测中提取稳定agent id；失败时回退为步内索引。"""
        ids = []
        for idx, obs in enumerate(obs_list or []):
            aid = None
            if isinstance(obs, dict):
                rid = obs.get("resource_ids")
                try:
                    if rid is not None and len(rid) > 0:
                        aid = int(np.asarray(rid).reshape(-1)[0])
                except Exception:
                    aid = None
            if aid is None:
                aid = int(idx)
            ids.append(aid)
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _build_value_map(agent_ids: np.ndarray, values: np.ndarray) -> Dict[int, float]:
        out = {}
        if agent_ids is None or values is None:
            return out
        n = min(len(agent_ids), len(values))
        for i in range(n):
            aid = int(agent_ids[i])
            # 同一步内若出现重复ID，保留首个以避免静默覆盖不同样本
            if aid not in out:
                out[aid] = float(values[i])
        return out

    def compute_returns_and_advantages(self, last_value: np.ndarray, last_obs_list: List[Dict] = None):
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
        
        # 基于稳定agent id计算GAE；若agent在下一步不存在，则该链条在此处截断（不bootstrap）。
        value_maps = []
        for t in range(T):
            ids_t = self.agent_ids_buffer[t] if t < len(self.agent_ids_buffer) else np.arange(len(self.values_buffer[t]))
            value_maps.append(self._build_value_map(ids_t, self.values_buffer[t]))

        if last_obs_list is not None:
            last_agent_ids = self._extract_agent_ids(last_obs_list)
        elif self.agent_ids_buffer:
            last_agent_ids = self.agent_ids_buffer[-1]
        else:
            last_agent_ids = np.arange(len(last_value), dtype=np.int64)
        last_value_map = self._build_value_map(last_agent_ids, last_value)

        gae_state: Dict[int, float] = {}
        for t in reversed(range(T)):
            rewards_t = self.rewards_buffer[t]
            values_t = self.values_buffer[t]
            ids_t = self.agent_ids_buffer[t] if t < len(self.agent_ids_buffer) else np.arange(len(rewards_t))
            done_info = self.dones_buffer[t]
            if isinstance(done_info, dict):
                terminated = bool(done_info.get('terminated', False))
            else:
                terminated = bool(done_info)
            next_map = last_value_map if t == T - 1 else value_maps[t + 1]

            n_cur = min(len(rewards_t), len(values_t), len(ids_t))
            for i in range(n_cur):
                aid = int(ids_t[i])
                reward = float(rewards_t[i])
                value = float(values_t[i])
                present_next = (aid in next_map)
                next_value = float(next_map.get(aid, 0.0))
                next_non_terminal = (1.0 - float(terminated)) * (1.0 if present_next else 0.0)
                prev_gae = gae_state.get(aid, 0.0) if present_next else 0.0
                delta = reward + self.gamma * next_value * next_non_terminal - value
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * prev_gae
                self.advantages_buffer[t][i] = gae
                gae_state[aid] = float(gae)
        
        # 计算returns = advantages + values
        for t in range(T):
            # 确保形状匹配
            N_adv = len(self.advantages_buffer[t])
            N_val = len(self.values_buffer[t])
            min_N = min(N_adv, N_val)
            self.returns_buffer[t] = self.advantages_buffer[t][:min_N] + self.values_buffer[t][:min_N]
            self.advantages_buffer[t] = self.advantages_buffer[t][:min_N]
    
    def get_batches(self, batch_size: int, active_only: bool = False) -> Generator[Dict, None, None]:
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
        flat_active_masks = []
        flat_values = []
        
        for t in range(T):
            N_t = len(self.obs_list_buffer[t])
            for n in range(N_t):
                flat_obs_list.append(self.obs_list_buffer[t][n])
                flat_actions.append(self.actions_buffer[t][n])
                flat_log_probs.append(self.log_probs_buffer[t][n])
                flat_advantages.append(self.advantages_buffer[t][n])
                flat_returns.append(self.returns_buffer[t][n])
                flat_active_masks.append(self.active_masks_buffer[t][n])
                flat_values.append(self.values_buffer[t][n])
        
        total_samples = len(flat_obs_list)
        
        if total_samples == 0:
            return
        
        # 转换为numpy数组
        flat_log_probs = np.array(flat_log_probs, dtype=np.float32)
        flat_advantages = np.array(flat_advantages, dtype=np.float32)
        flat_returns = np.array(flat_returns, dtype=np.float32)
        flat_active_masks = np.array(flat_active_masks, dtype=np.float32)
        flat_values = np.array(flat_values, dtype=np.float32)
        
        # 归一化advantages（优先使用active样本）
        active_idx = flat_active_masks > 0.0
        if np.any(active_idx):
            adv_mean = flat_advantages[active_idx].mean()
            adv_std = flat_advantages[active_idx].std()
        else:
            adv_mean = flat_advantages.mean()
            adv_std = flat_advantages.std()
        if adv_std > 1e-8:
            flat_advantages = (flat_advantages - adv_mean) / (adv_std + 1e-8)
        
        # 训练采样索引：可选仅采 active 样本，避免 idle/no-task 样本污染更新
        if active_only:
            candidate_idx = np.where(flat_active_masks > 0.0)[0]
            if candidate_idx.size == 0:
                return
            indices = np.random.permutation(candidate_idx)
        else:
            indices = np.random.permutation(total_samples)
        
        # 生成batches
        num_batches = max(1, len(indices) // batch_size)
        
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
                'returns': flat_returns[batch_indices],
                'active_masks': flat_active_masks[batch_indices],
                'old_values': flat_values[batch_indices],
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
        self.active_masks_buffer.clear()
        self.agent_ids_buffer.clear()

    def get_active_stats(self) -> Tuple[int, int]:
        """
        统计active样本数量与总样本数
        """
        total = 0
        active = 0
        for masks in self.active_masks_buffer:
            total += len(masks)
            active += int(np.sum(masks))
        return active, total

    def get_adv_stats(self) -> Tuple[float, float]:
        """
        统计active样本上的优势均值/方差
        """
        advantages = []
        masks = []
        for t in range(len(self.advantages_buffer)):
            if not self.advantages_buffer[t] is None:
                advantages.extend(list(self.advantages_buffer[t]))
                masks.extend(list(self.active_masks_buffer[t]))
        if not advantages:
            return 0.0, 0.0
        advantages = np.array(advantages, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        active_idx = masks > 0.0
        if np.any(active_idx):
            mean = float(np.mean(advantages[active_idx]))
            std = float(np.std(advantages[active_idx]))
        else:
            mean = float(np.mean(advantages))
            std = float(np.std(advantages))
        return mean, std
    
    def __len__(self):
        return len(self.obs_list_buffer)
