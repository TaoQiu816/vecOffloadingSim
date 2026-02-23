"""
[MAPPO智能体] mappo_agent.py
MAPPO Agent - Lightweight wrapper for OffloadingPolicyNetwork

作用 (Purpose):
    封装策略网络的训练和推理接口，实现PPO算法的核心逻辑。
    Wraps policy network for training and inference, implements core PPO algorithm logic.

核心功能 (Core Functions):
    1. select_action() - 根据观测选择动作（支持确定性/随机策略）
    2. evaluate_actions() - 重新评估动作的log_prob和value（用于PPO更新）
    3. update() - 执行PPO更新（Clip Loss + Value Loss + Entropy Regularization）
    4. get_value() - 获取状态价值（用于GAE计算）

PPO更新流程 (PPO Update Pipeline):
    1. 从RolloutBuffer采样mini-batch
    2. 重新评估动作得到新的log_prob和value
    3. 计算ratio = exp(new_log_prob - old_log_prob)
    4. 应用Clip约束防止策略突变
    5. 反向传播并更新网络参数
    6. 返回训练诊断指标（loss, entropy, kl, clip_frac等）

参考文献 (References):
    - PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from models.offloading_policy import OffloadingPolicyNetwork
from agents.rollout_buffer import RolloutBuffer
from configs.train_config import TrainConfig as TC


class MAPPOAgent:
    """
    MAPPO智能体 (Multi-Agent PPO Agent)
    
    功能：
        - 封装策略网络的训练和推理接口
        - 实现PPO算法的核心更新逻辑
        - 管理优化器和学习率调度
    """
    
    def __init__(self, network: OffloadingPolicyNetwork, device: str = 'cpu'):
        """
        Args:
            network: 策略网络
            device: 计算设备
        """
        self.network = network.to(device)
        self.device = device

        # 单优化器 + 参数组：
        # 共享trunk同时接收Actor/Critic梯度，Critic head使用更高学习率，避免表示漂移
        named_params = [(n, p) for n, p in self.network.named_parameters() if p.requires_grad]
        actor_prefixes = (
            "actor_critic.cross_attention.",
            "actor_critic.subtask_head.",
            "actor_critic.actor_head.",
            "actor_critic.layer_norm.",
            "power_cond_mlp.",
        )
        critic_prefix = "actor_critic.critic_head."

        self.shared_trunk_params = []
        self.actor_head_params = []
        self.critic_head_params = []
        for name, param in named_params:
            if name.startswith(critic_prefix):
                self.critic_head_params.append(param)
            elif any(name.startswith(prefix) for prefix in actor_prefixes):
                self.actor_head_params.append(param)
            else:
                self.shared_trunk_params.append(param)

        if len(self.shared_trunk_params) == 0 or len(self.actor_head_params) == 0 or len(self.critic_head_params) == 0:
            raise RuntimeError(
                "Failed to build parameter groups for MAPPOAgent "
                f"(shared={len(self.shared_trunk_params)}, actor={len(self.actor_head_params)}, critic={len(self.critic_head_params)})"
            )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.shared_trunk_params, "lr": getattr(TC, "LR_ACTOR", 3e-4)},
                {"params": self.actor_head_params, "lr": getattr(TC, "LR_ACTOR", 3e-4)},
                {"params": self.critic_head_params, "lr": getattr(TC, "LR_CRITIC", 1e-3)},
            ]
        )
        
        # 学习率调度器
        if TC.USE_LR_DECAY:
            # train.py 仅在每个 decay interval 调用一次 decay_lr()；
            # 这里 step_size 需为 1，才能按 interval 实际衰减。
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,
                gamma=TC.LR_DECAY_RATE
            )
        else:
            self.scheduler = None

    @staticmethod
    def _has_invalid_grad(params) -> bool:
        for param in params:
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                return True
        return False
    
    def select_action(self, obs_list: List[Dict], deterministic: bool = False) -> Dict:
        """
        选择动作

        Args:
            obs_list: 观测列表
            deterministic: 是否使用确定性策略

        Returns:
            动作字典，包含actions, log_probs, values
        """
        with torch.no_grad():
            subtask_actions, target_actions, power_actions, log_probs, values = self.network.get_action_and_value(
                obs_list, deterministic=deterministic, device=self.device
            )
        
        subtask_actions_np = torch.atleast_1d(subtask_actions).cpu().numpy().astype(int).flatten()
        target_actions_np = torch.atleast_1d(target_actions).cpu().numpy().astype(int).flatten()
        power_actions_np = torch.atleast_1d(power_actions).cpu().numpy().flatten()

        # 转换为环境可用的动作格式
        actions = []
        for i in range(len(obs_list)):
            obs_stamp = obs_list[i].get("obs_stamp")
            actions.append({
                'subtask': int(subtask_actions_np[i]),
                'target': int(target_actions_np[i]),
                'power': float(power_actions_np[i]),
                **({'obs_stamp': int(obs_stamp)} if obs_stamp is not None else {})
            })

        return {
            'actions': actions,
            'log_probs': log_probs.cpu().numpy(),
            'values': values.cpu().squeeze(-1).numpy()
        }
    
    def get_value(self, obs_list: List[Dict]) -> np.ndarray:
        """
        获取状态价值
        
        Args:
            obs_list: 观测列表
            
        Returns:
            状态价值数组
        """
        with torch.no_grad():
            inputs = self.network.prepare_inputs(obs_list, self.device)
            
            _, _, _, _, values = self.network.forward(
                node_x=inputs['node_x'],
                adj=inputs['adj'],
                status=inputs['status'],
                location=inputs['location'],
                L_fwd=inputs['L_fwd'],
                L_bwd=inputs['L_bwd'],
                data_matrix=inputs['data_matrix'],
                delta=inputs['delta'],
                resource_ids=inputs['resource_ids'],
                resource_raw=inputs['resource_raw'],
                subtask_index=inputs['subtask_index'],
                action_mask=inputs['action_mask'],
                subtask_mask=inputs.get('subtask_mask'),
                node_valid_mask=inputs.get('node_valid_mask'),
                task_mask=inputs['task_mask'],
                rate_prev=inputs.get('rate_prev'),
                serving_rsu_onehot=inputs.get('serving_rsu_onehot'),
                global_state=inputs.get('global_state'),
            )
        
        return values.cpu().squeeze(-1).numpy()
    
    def evaluate_actions(self, obs_list: List[Dict], actions: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重新评估动作（用于PPO更新）
        
        Args:
            obs_list: 观测列表
            actions: 动作列表
            
        Returns:
            log_probs: 动作log概率
            values: 状态价值
            entropy: 熵
        """
        subtask_values = []
        for obs, act in zip(obs_list, actions):
            if isinstance(act, dict) and ('subtask' in act):
                sval = int(act.get('subtask', 0))
            else:
                sval = int(obs.get('subtask_index', 0))
            if sval < 0:
                sval = 0
            subtask_values.append(sval)
        subtask_actions = torch.tensor(subtask_values, dtype=torch.long, device=self.device)
        target_actions = torch.tensor([a['target'] for a in actions], dtype=torch.long, device=self.device)
        power_actions = torch.tensor([a['power'] for a in actions], dtype=torch.float32, device=self.device)
        log_probs, entropy, values = self.network.evaluate_actions(
            obs_list=obs_list,
            subtask_actions=subtask_actions,
            target_actions=target_actions,
            power_actions=power_actions,
            device=self.device,
        )
        return log_probs, values.squeeze(-1), entropy
    
    def update(self, buffer: RolloutBuffer, batch_size: int = 64) -> float:
        """
        PPO更新
        
        Args:
            buffer: 经验缓冲区
            batch_size: mini-batch大小
            
        Returns:
            平均损失
        """
        total_loss = 0.0
        total_entropy = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_kl = 0.0
        total_clip = 0.0
        total_grad_norm = 0.0
        total_value_clip = 0.0
        total_value_target_mean = 0.0
        total_value_target_std = 0.0
        total_value_pred_mean = 0.0
        total_value_pred_std = 0.0
        total_critic_loss_active = 0.0
        total_critic_loss_inactive = 0.0
        total_value_norm = 0.0
        total_mask_active = 0.0
        total_mask_count = 0.0
        num_updates = 0
        ppo_epochs_executed = 0
        num_minibatches_executed = 0
        mb_kl_values = []
        mb_kl_max = 0.0
        early_stop_epoch_idx = -1
        early_stop_batch_idx = -1
        active_samples, total_samples = buffer.get_active_stats()
        adv_mean, adv_std = buffer.get_adv_stats()

        if active_samples < max(1, TC.MIN_ACTIVE_SAMPLES):
            self.last_update_stats = {
                "loss": 0.0,
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "entropy_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "value_loss_raw_mean": 0.0,
                "normalized_value_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
                "active_samples": int(active_samples),
                "total_samples": int(total_samples),
                "active_ratio": (float(active_samples) / float(total_samples)) if total_samples > 0 else 0.0,
                "actor_update_active_frac": 0.0,
                "adv_mean": float(adv_mean),
                "adv_std": float(adv_std),
                "value_target_mean": 0.0,
                "value_target_std": 0.0,
                "value_pred_mean": 0.0,
                "value_pred_std": 0.0,
                "value_clip_fraction": 0.0,
                "critic_loss_active": 0.0,
                "critic_loss_inactive": 0.0,
                "ppo_epochs_executed": 0,
                "num_minibatches_executed": 0,
                "mb_kl_max": 0.0,
                "mb_kl_p95": 0.0,
                "early_stop_epoch_idx": -1,
                "early_stop_batch_idx": -1,
                "skipped_update_count": 1,
            }
            return 0.0

        early_stop = False
        target_kl = getattr(TC, "TARGET_KL", None)
        kl_stop_mult = float(max(getattr(TC, "TARGET_KL_STOP_MULT", 1.5), 1.0))
        critic_soft_mask_enabled = bool(getattr(TC, "CRITIC_SOFT_ACTIVE_MASK", True))
        critic_inactive_weight = float(getattr(TC, "CRITIC_INACTIVE_SAMPLE_WEIGHT", 0.2))
        critic_inactive_weight = max(0.0, min(1.0, critic_inactive_weight))
        for epoch_idx in range(TC.PPO_EPOCH):
            if early_stop:
                break
            ppo_epochs_executed += 1
            # 固化: mini-batch 包含 all-steps；actor/critic 再用各自mask计算
            for batch_idx, batch in enumerate(buffer.get_batches(batch_size, active_only=False)):
                # 提取batch数据
                obs_list = batch['obs_list']
                actions = batch['actions']
                old_log_probs = torch.tensor(batch['old_log_probs'], dtype=torch.float32, device=self.device)
                advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
                returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
                active_masks = torch.tensor(batch['active_masks'], dtype=torch.float32, device=self.device)
                old_values = torch.tensor(batch['old_values'], dtype=torch.float32, device=self.device)
                actor_masks = active_masks
                if critic_soft_mask_enabled:
                    inactive_valid_masks = (active_masks <= 0.0).float()
                    critic_masks = actor_masks + critic_inactive_weight * inactive_valid_masks
                else:
                    critic_masks = torch.ones_like(active_masks)
                
                # 重新评估动作
                log_probs, values, entropy = self.evaluate_actions(obs_list, actions)
                
                # PPO Clip Loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - TC.CLIP_PARAM, 1.0 + TC.CLIP_PARAM) * advantages
                actor_mask_sum = actor_masks.sum()
                critic_mask_sum = critic_masks.sum()
                total_mask_active += float(actor_mask_sum.item())
                total_mask_count += float(actor_masks.numel())
                if critic_mask_sum.item() < 1e-6:
                    continue
                if actor_mask_sum.item() < 1.0:
                    policy_loss = torch.tensor(0.0, device=self.device)
                else:
                    policy_loss = -(torch.min(surr1, surr2) * actor_masks).sum() / actor_mask_sum
                
                # Value Loss（归一化处理，解决梯度主导问题）
                critic_idx = critic_masks > 0.0
                masked_returns = returns[critic_idx]
                masked_values = values[critic_idx]
                masked_old_values = old_values[critic_idx]

                value_target_mean = masked_returns.mean() if masked_returns.numel() > 0 else torch.tensor(0.0, device=self.device)
                value_target_std = masked_returns.std(unbiased=False) if masked_returns.numel() > 0 else torch.tensor(0.0, device=self.device)
                value_pred_mean = masked_values.mean() if masked_values.numel() > 0 else torch.tensor(0.0, device=self.device)
                value_pred_std = masked_values.std(unbiased=False) if masked_values.numel() > 0 else torch.tensor(0.0, device=self.device)

                if TC.USE_VALUE_TARGET_NORM and masked_returns.numel() > 0 and value_target_std.item() > 1e-6:
                    norm_std = value_target_std + 1e-8
                    returns_used = (returns - value_target_mean) / norm_std
                    values_used = (values - value_target_mean) / norm_std
                    old_values_used = (old_values - value_target_mean) / norm_std
                else:
                    returns_used = returns
                    values_used = values
                    old_values_used = old_values

                value_diff = values_used - old_values_used
                if TC.USE_VALUE_CLIP:
                    value_pred_clipped = old_values_used + torch.clamp(
                        value_diff, -TC.VALUE_CLIP_RANGE, TC.VALUE_CLIP_RANGE
                    )
                else:
                    value_pred_clipped = values_used

                value_loss_raw = (values_used - returns_used) ** 2
                value_loss_clip = (value_pred_clipped - returns_used) ** 2
                value_loss_max = torch.max(value_loss_raw, value_loss_clip) if TC.USE_VALUE_CLIP else value_loss_raw
                value_loss_raw_mean = (value_loss_max * critic_masks).sum() / critic_mask_sum
                active_idx = actor_masks > 0.0
                inactive_idx = (active_masks <= 0.0)
                critic_loss_active = value_loss_max[active_idx].mean() if torch.any(active_idx) else torch.tensor(0.0, device=self.device)
                critic_loss_inactive = value_loss_max[inactive_idx].mean() if torch.any(inactive_idx) else torch.tensor(0.0, device=self.device)
                # 将Value Loss归一化到与Policy Loss相近的量级
                # 使用动态归一化：除以returns的方差
                masked_returns_used = returns_used[critic_idx]
                returns_var = masked_returns_used.var(unbiased=False) + 1e-8 if masked_returns_used.numel() > 0 else torch.tensor(1.0, device=self.device)
                value_loss = value_loss_raw_mean / returns_var
                
                # Entropy Loss
                if actor_mask_sum.item() < 1.0:
                    entropy_mean = torch.tensor(0.0, device=self.device)
                    entropy_loss = torch.tensor(0.0, device=self.device)
                    approx_kl = torch.tensor(0.0, device=self.device)
                    clip_frac = torch.tensor(0.0, device=self.device)
                else:
                    entropy_mean = (entropy * actor_masks).sum() / actor_mask_sum
                    entropy_loss = -entropy_mean
                    approx_kl = ((old_log_probs - log_probs) * actor_masks).sum() / actor_mask_sum
                    clip_frac = ((torch.abs(ratio - 1.0) > TC.CLIP_PARAM).float() * actor_masks).sum() / actor_mask_sum
                mb_kl = float(approx_kl.item())
                num_minibatches_executed += 1
                mb_kl_values.append(mb_kl)
                mb_kl_max = max(mb_kl_max, mb_kl)
                value_clip_frac = (
                    (torch.abs(values_used - old_values_used) > TC.VALUE_CLIP_RANGE).float() * critic_masks
                ).sum() / critic_mask_sum if TC.USE_VALUE_CLIP else torch.tensor(0.0, device=self.device)
                
                # 单优化器 + 参数组：共享trunk同时接收Actor/Critic梯度
                actor_loss_total = policy_loss + TC.ENTROPY_COEF * entropy_loss
                critic_loss_total = TC.VF_COEF * value_loss
                loss = actor_loss_total + critic_loss_total
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), TC.MAX_GRAD_NORM)
                has_invalid_grad = self._has_invalid_grad(self.network.parameters())

                if not has_invalid_grad:
                    self.optimizer.step()
                    total_loss += loss.item()
                    total_entropy += entropy_mean.item()
                    total_policy += policy_loss.item()
                    total_value += value_loss_raw_mean.item()  # 记录mask后的Value Loss用于诊断
                    total_value_norm += value_loss.item()      # 记录实际用于优化的归一化Value Loss
                    total_kl += approx_kl.item()
                    total_clip += clip_frac.item()
                    total_grad_norm += float(grad_norm) if grad_norm is not None else 0.0
                    total_value_clip += float(value_clip_frac.item())
                    total_value_target_mean += float(value_target_mean.item())
                    total_value_target_std += float(value_target_std.item())
                    total_value_pred_mean += float(value_pred_mean.item())
                    total_value_pred_std += float(value_pred_std.item())
                    total_critic_loss_active += float(critic_loss_active.item())
                    total_critic_loss_inactive += float(critic_loss_inactive.item())
                    num_updates += 1
                if (
                    target_kl is not None
                    and target_kl > 0.0
                    and mb_kl > (target_kl * kl_stop_mult)
                ):
                    early_stop = True
                    early_stop_epoch_idx = int(epoch_idx)
                    early_stop_batch_idx = int(batch_idx)
                    break

        mb_kl_p95 = float(np.percentile(mb_kl_values, 95)) if mb_kl_values else 0.0
        if num_updates > 0:
            avg_entropy = total_entropy / num_updates
            # 确保entropy是有效的正数（策略分布的熵应该 > 0）
            if avg_entropy < 1e-6:
                # 如果熵过小，可能是数值问题或策略过于确定
                avg_entropy = max(avg_entropy, 0.0)
            
            self.last_update_stats = {
                "loss": total_loss / num_updates,
                "entropy": avg_entropy,  # 真实的策略熵
                "policy_entropy": avg_entropy,
                "entropy_loss": -avg_entropy,  # 熵损失（负号因为我们要最大化熵）
                "policy_loss": total_policy / num_updates,
                "value_loss": total_value / num_updates,  # raw mse-like diagnostic (legacy key)
                "value_loss_raw_mean": total_value / num_updates,
                "normalized_value_loss": total_value_norm / num_updates,
                "approx_kl": total_kl / num_updates,
                "clip_fraction": total_clip / num_updates,
                "grad_norm": total_grad_norm / num_updates,
                "active_samples": int(active_samples),
                "total_samples": int(total_samples),
                "active_ratio": (float(active_samples) / float(total_samples)) if total_samples > 0 else 0.0,
                "actor_update_active_frac": (total_mask_active / total_mask_count) if total_mask_count > 0 else 0.0,
                "adv_mean": float(adv_mean),
                "adv_std": float(adv_std),
                "value_target_mean": total_value_target_mean / num_updates,
                "value_target_std": total_value_target_std / num_updates,
                "value_pred_mean": total_value_pred_mean / num_updates,
                "value_pred_std": total_value_pred_std / num_updates,
                "value_clip_fraction": total_value_clip / num_updates,
                "critic_loss_active": total_critic_loss_active / num_updates,
                "critic_loss_inactive": total_critic_loss_inactive / num_updates,
                "ppo_epochs_executed": int(ppo_epochs_executed),
                "num_minibatches_executed": int(num_minibatches_executed),
                "mb_kl_max": float(mb_kl_max),
                "mb_kl_p95": float(mb_kl_p95),
                "early_stop_epoch_idx": int(early_stop_epoch_idx),
                "early_stop_batch_idx": int(early_stop_batch_idx),
                "skipped_update_count": 0,
                "early_stop": early_stop,
            }
        else:
            # 如果没有有效更新，保留上一次的stats或使用默认值
            self.last_update_stats = {
                "loss": 0.0,
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "entropy_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "value_loss_raw_mean": 0.0,
                "normalized_value_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
                "active_samples": int(active_samples),
                "total_samples": int(total_samples),
                "active_ratio": (float(active_samples) / float(total_samples)) if total_samples > 0 else 0.0,
                "actor_update_active_frac": (total_mask_active / total_mask_count) if total_mask_count > 0 else 0.0,
                "adv_mean": float(adv_mean),
                "adv_std": float(adv_std),
                "value_target_mean": 0.0,
                "value_target_std": 0.0,
                "value_pred_mean": 0.0,
                "value_pred_std": 0.0,
                "value_clip_fraction": 0.0,
                "critic_loss_active": 0.0,
                "critic_loss_inactive": 0.0,
                "ppo_epochs_executed": int(ppo_epochs_executed),
                "num_minibatches_executed": int(num_minibatches_executed),
                "mb_kl_max": float(mb_kl_max),
                "mb_kl_p95": float(mb_kl_p95),
                "early_stop_epoch_idx": int(early_stop_epoch_idx),
                "early_stop_batch_idx": int(early_stop_batch_idx),
                "skipped_update_count": 0,
                "early_stop": early_stop,
            }

        return total_loss / num_updates if num_updates > 0 else 0.0
    
    def decay_lr(self):
        """学习率衰减"""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def save(self, path: str):
        """保存模型"""
        payload = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            payload['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(payload, path)
    
    def load(self, path: str, restore_optimizer: bool = True, restore_scheduler: bool = True):
        """加载模型。
        restore_optimizer=False 时只恢复网络参数，保留当前 optimizer/scheduler 状态，
        用于 LateGuard rollback（只回滚网络权重，LR 轨迹不被重置）。
        resume 断点续训时保持默认 True 以完整恢复训练状态。
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        if restore_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception:
                    # 兼容旧双优化器checkpoint（param_groups结构不一致时跳过）
                    pass
        if restore_scheduler:
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
