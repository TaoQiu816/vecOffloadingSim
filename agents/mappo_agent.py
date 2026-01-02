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

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=TC.LR_ACTOR
        )
        
        # 学习率调度器
        if TC.USE_LR_DECAY:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=TC.LR_DECAY_STEPS,
                gamma=TC.LR_DECAY_RATE
            )
        else:
            self.scheduler = None
    
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
            target_actions, power_actions, log_probs, values = self.network.get_action_and_value(
                obs_list, deterministic=deterministic, device=self.device
            )
        
        # 转换为环境可用的动作格式
        actions = []
        for i in range(len(obs_list)):
            obs_stamp = obs_list[i].get("obs_stamp")
            actions.append({
                'target': int(target_actions[i].cpu().item()),
                'power': float(power_actions[i].cpu().item()),
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
            
            _, _, _, values = self.network.forward(
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
                task_mask=inputs['task_mask']
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
        # 提取target和power
        target_actions = torch.tensor([a['target'] for a in actions], dtype=torch.long, device=self.device)
        power_actions = torch.tensor([a['power'] for a in actions], dtype=torch.float32, device=self.device)
        
        # 准备输入
        inputs = self.network.prepare_inputs(obs_list, self.device)
        
        # 前向传播
        target_logits, alpha, beta, values = self.network.forward(
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
            task_mask=inputs['task_mask']
        )
        
        # 计算target的log_prob和entropy
        # [Logit Bias] 解决动作空间不平衡问题：给Local和RSU添加偏置
        # 索引映射：Index 0=Local, Index 1=RSU, Index 2+=Neighbors
        from configs.train_config import TrainConfig as TC
        if TC.USE_LOGIT_BIAS:
            logit_bias = torch.zeros_like(target_logits)
            logit_bias[:, 0] = TC.LOGIT_BIAS_LOCAL  # Local (Index 0)
            logit_bias[:, 1] = TC.LOGIT_BIAS_RSU    # RSU (Index 1)
            target_logits = target_logits + logit_bias
        
        # 应用action_mask
        action_mask_tensor = inputs['action_mask']
        masked_logits = torch.where(
            action_mask_tensor > 0,
            target_logits,
            torch.tensor(-1e10, dtype=target_logits.dtype, device=target_logits.device)
        )
        
        target_probs = torch.softmax(masked_logits, dim=-1)
        target_dist = torch.distributions.Categorical(target_probs)
        log_prob_target = target_dist.log_prob(target_actions)
        entropy_target = target_dist.entropy()
        
        # 计算power的log_prob和entropy
        power_dist = torch.distributions.Beta(alpha.squeeze(-1), beta.squeeze(-1))
        log_prob_power = power_dist.log_prob(power_actions)
        entropy_power = power_dist.entropy()
        
        # 联合log_prob和entropy
        log_probs = log_prob_target + log_prob_power
        entropy = entropy_target + entropy_power
        
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
        num_updates = 0
        
        for _ in range(TC.PPO_EPOCH):
            for batch in buffer.get_batches(batch_size):
                # 提取batch数据
                obs_list = batch['obs_list']
                actions = batch['actions']
                old_log_probs = torch.tensor(batch['old_log_probs'], dtype=torch.float32, device=self.device)
                advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
                returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
                
                # 重新评估动作
                log_probs, values, entropy = self.evaluate_actions(obs_list, actions)
                
                # PPO Clip Loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - TC.CLIP_PARAM, 1.0 + TC.CLIP_PARAM) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy Loss
                entropy_mean = entropy.mean()
                entropy_loss = -entropy_mean
                approx_kl = (old_log_probs - log_probs).mean()
                clip_frac = (torch.abs(ratio - 1.0) > TC.CLIP_PARAM).float().mean()
                
                # Total Loss
                loss = policy_loss + TC.VF_COEF * value_loss + TC.ENTROPY_COEF * entropy_loss
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), TC.MAX_GRAD_NORM)
                
                # 检查梯度是否有效
                has_invalid_grad = False
                for param in self.network.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_invalid_grad = True
                        break
                
                if not has_invalid_grad:
                    self.optimizer.step()
                    total_loss += loss.item()
                    total_entropy += entropy_mean.item()
                    total_policy += policy_loss.item()
                    total_value += value_loss.item()
                    total_kl += approx_kl.item()
                    total_clip += clip_frac.item()
                    total_grad_norm += float(grad_norm) if grad_norm is not None else 0.0
                    num_updates += 1

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
                "value_loss": total_value / num_updates,
                "approx_kl": total_kl / num_updates,
                "clip_fraction": total_clip / num_updates,
                "grad_norm": total_grad_norm / num_updates,
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
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
            }

        return total_loss / num_updates if num_updates > 0 else 0.0
    
    def decay_lr(self):
        """学习率衰减"""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
