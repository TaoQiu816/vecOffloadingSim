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
        self.decision_steps_buffer = []  # [T] 每步是长度N_t的数组（对应decision发生的物理步）
        self.interval_steps_buffer = []  # [T] 每步是长度N_t的数组（decision间隔物理步数）
        self.bootstrap_cont_buffer = []  # [T] 每步是长度N_t的数组（0/1，是否允许bootstrap）
        self.bootstrap_values_buffer = []  # [T] 每步是长度N_t的数组
        self.bootstrap_cost_power_values_buffer = []  # [T] 每步是长度N_t的数组
        self.bootstrap_cost_trust_values_buffer = []  # [T] 每步是长度N_t的数组
        self.cost_power_buffer = []   # [T] 每步是长度N_t的数组
        self.cost_trust_buffer = []   # [T] 每步是长度N_t的数组
        self.cost_power_values_buffer = []  # [T] 每步是长度N_t的数组
        self.cost_trust_values_buffer = []  # [T] 每步是长度N_t的数组
        
        # GAE计算结果 - 列表形式
        self.advantages_buffer = []  # [T] 每步是长度N_t的数组
        self.returns_buffer = []     # [T] 每步是长度N_t的数组
        self.cost_power_advantages_buffer = []
        self.cost_power_returns_buffer = []
        self.cost_trust_advantages_buffer = []
        self.cost_trust_returns_buffer = []
        
    def add(self, obs_list: List[Dict], actions: List[Dict],
            rewards: List[float], values: np.ndarray,
            log_probs: np.ndarray, done: bool,
            terminated: bool = None, truncated: bool = None,
            active_masks: List[int] = None,
            agent_ids: np.ndarray = None,
            decision_steps: np.ndarray = None,
            interval_steps: np.ndarray = None,
            bootstrap_cont: np.ndarray = None,
            bootstrap_values: np.ndarray = None,
            cost_power: List[float] = None,
            cost_trust: List[float] = None,
            cost_power_values: np.ndarray = None,
            cost_trust_values: np.ndarray = None,
            bootstrap_cost_power_values: np.ndarray = None,
            bootstrap_cost_trust_values: np.ndarray = None):
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
            
        values_arr = values.astype(np.float32).flatten()
        log_probs_arr = log_probs.astype(np.float32).flatten()
        self.values_buffer.append(values_arr)
        self.log_probs_buffer.append(log_probs_arr)
        self.agent_ids_buffer.append(self._to_array(agent_ids, rewards, dtype=np.int64))
        self.decision_steps_buffer.append(self._to_array(decision_steps, rewards, dtype=np.int32))
        interval_arr = self._to_array(interval_steps, rewards, dtype=np.int32)
        if interval_arr.size > 0:
            interval_arr = np.maximum(interval_arr, 1)
        self.interval_steps_buffer.append(interval_arr)
        self.bootstrap_cont_buffer.append(self._to_array(bootstrap_cont, rewards))
        self.bootstrap_values_buffer.append(self._to_array(bootstrap_values, rewards))
        self.cost_power_buffer.append(self._to_array(cost_power, rewards))
        self.cost_trust_buffer.append(self._to_array(cost_trust, rewards))
        self.cost_power_values_buffer.append(self._to_array(cost_power_values, rewards))
        self.cost_trust_values_buffer.append(self._to_array(cost_trust_values, rewards))
        self.bootstrap_cost_power_values_buffer.append(self._to_array(bootstrap_cost_power_values, rewards))
        self.bootstrap_cost_trust_values_buffer.append(self._to_array(bootstrap_cost_trust_values, rewards))

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
    def _to_array(values, ref, dtype=np.float32) -> np.ndarray:
        if values is None:
            return np.zeros_like(ref, dtype=dtype)
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if isinstance(values, list):
            values = np.array(values, dtype=dtype)
        elif not isinstance(values, np.ndarray):
            values = np.array([values], dtype=dtype)
        out = values.astype(dtype).flatten()
        ref_arr = np.asarray(ref).flatten()
        if out.shape != ref_arr.shape:
            out = np.zeros_like(ref_arr, dtype=dtype)
        return out
    
    @staticmethod
    def _extract_agent_ids(obs_list: List[Dict]) -> np.ndarray:
        """从观测中提取稳定agent id；失败时回退为步内索引。"""
        ids = []
        for idx, obs in enumerate(obs_list or []):
            aid = None
            if isinstance(obs, dict):
                for key in ("agent_id", "vehicle_id"):
                    if obs.get(key) is None:
                        continue
                    try:
                        aid = int(obs.get(key))
                        break
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

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        last_obs_list: List[Dict] = None,
        last_cost_power_value: np.ndarray = None,
        last_cost_trust_value: np.ndarray = None,
    ):
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
        self.cost_power_advantages_buffer = []
        self.cost_power_returns_buffer = []
        self.cost_trust_advantages_buffer = []
        self.cost_trust_returns_buffer = []
        
        # 为每个时间步预分配advantages数组
        for t in range(T):
            N_t = len(self.rewards_buffer[t])
            self.advantages_buffer.append(np.zeros(N_t, dtype=np.float32))
            self.returns_buffer.append(np.zeros(N_t, dtype=np.float32))
            self.cost_power_advantages_buffer.append(np.zeros(N_t, dtype=np.float32))
            self.cost_power_returns_buffer.append(np.zeros(N_t, dtype=np.float32))
            self.cost_trust_advantages_buffer.append(np.zeros(N_t, dtype=np.float32))
            self.cost_trust_returns_buffer.append(np.zeros(N_t, dtype=np.float32))
        
        trajectories: Dict[int, List[Tuple[int, int, int]]] = {}
        for t in range(T):
            ids_t = self.agent_ids_buffer[t] if t < len(self.agent_ids_buffer) else np.arange(len(self.values_buffer[t]))
            decision_steps_t = self.decision_steps_buffer[t] if t < len(self.decision_steps_buffer) else np.arange(len(ids_t))
            n_cur = min(len(ids_t), len(decision_steps_t), len(self.values_buffer[t]))
            for i in range(n_cur):
                aid = int(ids_t[i])
                trajectories.setdefault(aid, []).append((int(decision_steps_t[i]), t, i))

        for samples in trajectories.values():
            samples.sort(key=lambda item: (item[0], item[1], item[2]))
            next_adv = 0.0
            next_cost_power_adv = 0.0
            next_cost_trust_adv = 0.0
            for pos in reversed(range(len(samples))):
                _, t, i = samples[pos]
                interval = int(max(self.interval_steps_buffer[t][i], 1)) if t < len(self.interval_steps_buffer) else 1
                gamma_eff = float(self.gamma ** interval)
                gae_eff = float((self.gamma * self.gae_lambda) ** interval)
                reward = float(self.rewards_buffer[t][i])
                value = float(self.values_buffer[t][i])
                cost_power = float(self.cost_power_buffer[t][i])
                cost_power_value = float(self.cost_power_values_buffer[t][i])
                cost_trust = float(self.cost_trust_buffer[t][i])
                cost_trust_value = float(self.cost_trust_values_buffer[t][i])

                if pos + 1 < len(samples):
                    _, next_t, next_i = samples[pos + 1]
                    cont = 1.0
                    next_value = float(self.values_buffer[next_t][next_i])
                    next_cost_power_value = float(self.cost_power_values_buffer[next_t][next_i])
                    next_cost_trust_value = float(self.cost_trust_values_buffer[next_t][next_i])
                else:
                    cont = float(self.bootstrap_cont_buffer[t][i]) if t < len(self.bootstrap_cont_buffer) else 0.0
                    next_value = float(self.bootstrap_values_buffer[t][i]) if t < len(self.bootstrap_values_buffer) else 0.0
                    next_cost_power_value = (
                        float(self.bootstrap_cost_power_values_buffer[t][i])
                        if t < len(self.bootstrap_cost_power_values_buffer) else 0.0
                    )
                    next_cost_trust_value = (
                        float(self.bootstrap_cost_trust_values_buffer[t][i])
                        if t < len(self.bootstrap_cost_trust_values_buffer) else 0.0
                    )
                    next_adv = 0.0
                    next_cost_power_adv = 0.0
                    next_cost_trust_adv = 0.0

                delta = reward + cont * gamma_eff * next_value - value
                gae = delta + cont * gae_eff * next_adv
                self.advantages_buffer[t][i] = gae
                next_adv = float(gae)

                delta_cost_power = cost_power + cont * gamma_eff * next_cost_power_value - cost_power_value
                gae_cost_power = delta_cost_power + cont * gae_eff * next_cost_power_adv
                self.cost_power_advantages_buffer[t][i] = gae_cost_power
                next_cost_power_adv = float(gae_cost_power)

                delta_cost_trust = cost_trust + cont * gamma_eff * next_cost_trust_value - cost_trust_value
                gae_cost_trust = delta_cost_trust + cont * gae_eff * next_cost_trust_adv
                self.cost_trust_advantages_buffer[t][i] = gae_cost_trust
                next_cost_trust_adv = float(gae_cost_trust)
        
        # 计算returns = advantages + values
        for t in range(T):
            # 确保形状匹配
            N_adv = len(self.advantages_buffer[t])
            N_val = len(self.values_buffer[t])
            min_N = min(N_adv, N_val)
            self.returns_buffer[t] = self.advantages_buffer[t][:min_N] + self.values_buffer[t][:min_N]
            self.advantages_buffer[t] = self.advantages_buffer[t][:min_N]
            self.cost_power_returns_buffer[t] = self.cost_power_advantages_buffer[t][:min_N] + self.cost_power_values_buffer[t][:min_N]
            self.cost_power_advantages_buffer[t] = self.cost_power_advantages_buffer[t][:min_N]
            self.cost_trust_returns_buffer[t] = self.cost_trust_advantages_buffer[t][:min_N] + self.cost_trust_values_buffer[t][:min_N]
            self.cost_trust_advantages_buffer[t] = self.cost_trust_advantages_buffer[t][:min_N]
    
    def flatten_samples(self, normalize_advantages: bool = True) -> Dict[str, object]:
        """
        展平buffer中的所有样本，供不同训练算法复用。
        """
        T = len(self.obs_list_buffer)
        flat_obs_list = []
        flat_actions = []
        flat_log_probs = []
        flat_advantages = []
        flat_returns = []
        flat_active_masks = []
        flat_values = []
        flat_cost_power_advantages = []
        flat_cost_power_returns = []
        flat_cost_power_values = []
        flat_cost_trust_advantages = []
        flat_cost_trust_returns = []
        flat_cost_trust_values = []
        flat_agent_ids = []

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
                flat_cost_power_advantages.append(self.cost_power_advantages_buffer[t][n])
                flat_cost_power_returns.append(self.cost_power_returns_buffer[t][n])
                flat_cost_power_values.append(self.cost_power_values_buffer[t][n])
                flat_cost_trust_advantages.append(self.cost_trust_advantages_buffer[t][n])
                flat_cost_trust_returns.append(self.cost_trust_returns_buffer[t][n])
                flat_cost_trust_values.append(self.cost_trust_values_buffer[t][n])
                flat_agent_ids.append(
                    self.agent_ids_buffer[t][n]
                    if t < len(self.agent_ids_buffer) and n < len(self.agent_ids_buffer[t])
                    else n
                )

        out = {
            "obs_list": flat_obs_list,
            "actions": flat_actions,
            "old_log_probs": np.array(flat_log_probs, dtype=np.float32),
            "advantages": np.array(flat_advantages, dtype=np.float32),
            "returns": np.array(flat_returns, dtype=np.float32),
            "active_masks": np.array(flat_active_masks, dtype=np.float32),
            "old_values": np.array(flat_values, dtype=np.float32),
            "cost_power_advantages": np.array(flat_cost_power_advantages, dtype=np.float32),
            "cost_power_returns": np.array(flat_cost_power_returns, dtype=np.float32),
            "old_cost_power_values": np.array(flat_cost_power_values, dtype=np.float32),
            "cost_trust_advantages": np.array(flat_cost_trust_advantages, dtype=np.float32),
            "cost_trust_returns": np.array(flat_cost_trust_returns, dtype=np.float32),
            "old_cost_trust_values": np.array(flat_cost_trust_values, dtype=np.float32),
            "agent_ids": np.array(flat_agent_ids, dtype=np.int64),
        }

        if not normalize_advantages or out["advantages"].size == 0:
            return out

        active_idx = out["active_masks"] > 0.0
        if np.any(active_idx):
            adv_mean = out["advantages"][active_idx].mean()
            adv_std = out["advantages"][active_idx].std()
        else:
            adv_mean = out["advantages"].mean()
            adv_std = out["advantages"].std()
        if adv_std > 1e-8:
            out["advantages"] = (out["advantages"] - adv_mean) / (adv_std + 1e-8)
        if np.any(active_idx):
            cp_mean = out["cost_power_advantages"][active_idx].mean()
            cp_std = out["cost_power_advantages"][active_idx].std()
            ct_mean = out["cost_trust_advantages"][active_idx].mean()
            ct_std = out["cost_trust_advantages"][active_idx].std()
            if cp_std > 1e-8:
                out["cost_power_advantages"] = (out["cost_power_advantages"] - cp_mean) / (cp_std + 1e-8)
            if ct_std > 1e-8:
                out["cost_trust_advantages"] = (out["cost_trust_advantages"] - ct_mean) / (ct_std + 1e-8)
        return out

    def get_agent_ids(self) -> np.ndarray:
        flat = self.flatten_samples(normalize_advantages=False)
        return np.unique(flat["agent_ids"]) if flat["agent_ids"].size > 0 else np.array([], dtype=np.int64)

    def get_batches(self, batch_size: int, active_only: bool = False, agent_id: int = None) -> Generator[Dict, None, None]:
        """
        生成mini-batch用于训练
        
        Args:
            batch_size: batch大小
            
        Yields:
            batch字典，包含obs_list, actions, old_log_probs, advantages, returns
        """
        flat = self.flatten_samples(normalize_advantages=True)
        flat_obs_list = flat["obs_list"]
        flat_actions = flat["actions"]
        flat_log_probs = flat["old_log_probs"]
        flat_advantages = flat["advantages"]
        flat_returns = flat["returns"]
        flat_active_masks = flat["active_masks"]
        flat_values = flat["old_values"]
        flat_cost_power_advantages = flat["cost_power_advantages"]
        flat_cost_power_returns = flat["cost_power_returns"]
        flat_cost_power_values = flat["old_cost_power_values"]
        flat_cost_trust_advantages = flat["cost_trust_advantages"]
        flat_cost_trust_returns = flat["cost_trust_returns"]
        flat_cost_trust_values = flat["old_cost_trust_values"]
        flat_agent_ids = flat["agent_ids"]

        total_samples = len(flat_obs_list)
        
        if total_samples == 0:
            return
        
        # 训练采样索引：可选仅采 active 样本，避免 idle/no-task 样本污染更新
        candidate_idx = np.arange(total_samples)
        if agent_id is not None:
            candidate_idx = candidate_idx[flat_agent_ids == int(agent_id)]
        if active_only:
            candidate_idx = candidate_idx[flat_active_masks[candidate_idx] > 0.0]
        if candidate_idx.size == 0:
            return
        indices = np.random.permutation(candidate_idx)
        
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
                'cost_power_advantages': flat_cost_power_advantages[batch_indices],
                'cost_power_returns': flat_cost_power_returns[batch_indices],
                'old_cost_power_values': flat_cost_power_values[batch_indices],
                'cost_trust_advantages': flat_cost_trust_advantages[batch_indices],
                'cost_trust_returns': flat_cost_trust_returns[batch_indices],
                'old_cost_trust_values': flat_cost_trust_values[batch_indices],
                'agent_ids': flat_agent_ids[batch_indices],
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
        self.decision_steps_buffer.clear()
        self.interval_steps_buffer.clear()
        self.bootstrap_cont_buffer.clear()
        self.bootstrap_values_buffer.clear()
        self.bootstrap_cost_power_values_buffer.clear()
        self.bootstrap_cost_trust_values_buffer.clear()
        self.cost_power_buffer.clear()
        self.cost_trust_buffer.clear()
        self.cost_power_values_buffer.clear()
        self.cost_trust_values_buffer.clear()
        self.cost_power_advantages_buffer.clear()
        self.cost_power_returns_buffer.clear()
        self.cost_trust_advantages_buffer.clear()
        self.cost_trust_returns_buffer.clear()

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

    def get_oracle_group_adv_stats(self) -> Dict[str, float]:
        """
        基于raw GAE/returns（未归一化）统计若干关键 oracle×chosen 组别的均值。
        仅统计 active 且 oracle/chosen 标签齐全的样本。
        """
        groups = (
            ("v2v", "rsu"),
            ("v2v", "v2v"),
            ("rsu", "rsu"),
            ("rsu", "v2v"),
        )
        acc = {
            g: {"count": 0, "adv_sum": 0.0, "ret_sum": 0.0, "rew_sum": 0.0}
            for g in groups
        }
        T = len(self.obs_list_buffer)
        for t in range(T):
            obs_t = self.obs_list_buffer[t]
            rew_t = self.rewards_buffer[t] if t < len(self.rewards_buffer) else None
            adv_t = self.advantages_buffer[t] if t < len(self.advantages_buffer) else None
            ret_t = self.returns_buffer[t] if t < len(self.returns_buffer) else None
            mask_t = self.active_masks_buffer[t] if t < len(self.active_masks_buffer) else None
            if obs_t is None or rew_t is None or adv_t is None or ret_t is None or mask_t is None:
                continue
            n = min(len(obs_t), len(rew_t), len(adv_t), len(ret_t), len(mask_t))
            for i in range(n):
                if float(mask_t[i]) <= 0.0:
                    continue
                obs = obs_t[i] if i < len(obs_t) else None
                if not isinstance(obs, dict):
                    continue
                oracle_mode = str(obs.get("oracle_mode", "") or "").lower()
                chosen_mode = str(obs.get("chosen_mode", "") or "").lower()
                key = (oracle_mode, chosen_mode)
                if key not in acc:
                    continue
                rec = acc[key]
                rec["count"] += 1
                rec["adv_sum"] += float(adv_t[i])
                rec["ret_sum"] += float(ret_t[i])
                rec["rew_sum"] += float(rew_t[i])

        out: Dict[str, float] = {}
        for (oracle_mode, chosen_mode), rec in acc.items():
            prefix = f"oracle_{oracle_mode}_chosen_{chosen_mode}"
            cnt = int(rec["count"])
            out[f"{prefix}_count"] = cnt
            if cnt > 0:
                out[f"{prefix}_adv_mean"] = float(rec["adv_sum"] / cnt)
                out[f"{prefix}_ret_mean"] = float(rec["ret_sum"] / cnt)
                out[f"{prefix}_rew_mean"] = float(rec["rew_sum"] / cnt)
            else:
                out[f"{prefix}_adv_mean"] = 0.0
                out[f"{prefix}_ret_mean"] = 0.0
                out[f"{prefix}_rew_mean"] = 0.0
        return out
    
    def __len__(self):
        return len(self.obs_list_buffer)
