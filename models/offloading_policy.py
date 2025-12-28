"""
完整的卸载策略网络

整合所有模块：
1. DAG特征嵌入
2. 边增强Transformer
3. 资源特征编码
4. Actor-Critic输出（Beta分布版本）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Dict, List, Tuple, Optional
import numpy as np

from models.dag_embedding import (
    DAGNodeEmbedding,
    EdgeFeatureEncoder,
    SpatialDistanceEncoder
)
from models.edge_enhanced_transformer import EdgeEnhancedTransformer
from models.resource_features import ResourceFeatureEncoder
from models.actor_critic import ActorCriticNetwork
from configs.config import SystemConfig as Cfg


class OffloadingPolicyNetwork(nn.Module):
    """
    完整的卸载决策网络
    """
    
    def __init__(self,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 continuous_dim: int = 7):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            d_ff: 前馈层维度
            dropout: Dropout率
            continuous_dim: 连续特征维度
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 1. DAG节点嵌入
        self.dag_embedding = DAGNodeEmbedding(d_model, continuous_dim)
        
        # 2. 边特征和空间距离编码器
        self.edge_encoder = EdgeFeatureEncoder(num_heads)
        self.spatial_encoder = SpatialDistanceEncoder(num_heads)
        
        # 3. 边增强Transformer
        self.transformer = EdgeEnhancedTransformer(
            num_layers, d_model, num_heads, d_ff, dropout
        )
        
        # 4. 资源特征编码器
        self.resource_encoder = ResourceFeatureEncoder(
            max_vehicle_id=Cfg.MAX_VEHICLE_ID,
            d_model=d_model
        )
        
        # 5. Actor-Critic网络
        self.actor_critic = ActorCriticNetwork(
            d_model, num_heads, num_layers, d_ff, dropout
        )
    
    def prepare_inputs(self, obs_list: List[Dict], device='cpu') -> Dict[str, torch.Tensor]:
        """
        从环境观测中准备网络输入
        
        Args:
            obs_list: 环境返回的观测列表
            device: 目标设备
        
        Returns:
            输入字典，包含所有必要的Tensor
        """
        batch_size = len(obs_list)
        
        # 提取特征
        node_x_list = []
        adj_list = []
        task_mask_list = []
        action_mask_list = []
        subtask_index_list = []
        resource_ids_list = []
        
        # DAG拓扑特征（环境已提供）
        status_list = []
        location_list = []
        L_fwd_list = []
        L_bwd_list = []
        data_matrix_list = []
        delta_list = []
        
        for obs in obs_list:
            node_x_list.append(obs['node_x'])
            adj_list.append(obs['adj'])
            task_mask_list.append(obs['task_mask'])
            action_mask_list.append(obs['action_mask'])
            subtask_index_list.append(obs['subtask_index'])
            resource_ids_list.append(obs['resource_ids'])
            
            # 从环境提供的字段中获取
            status_list.append(obs['status'])
            location_list.append(obs['location'])
            L_fwd_list.append(obs['L_fwd'])
            L_bwd_list.append(obs['L_bwd'])
            data_matrix_list.append(obs['data_matrix'])
            delta_list.append(obs['Delta'])
        
        # 转换为Tensor并移到目标设备
        inputs = {
            'node_x': torch.from_numpy(np.stack(node_x_list)).float().to(device),
            'adj': torch.from_numpy(np.stack(adj_list)).float().to(device),
            'task_mask': torch.from_numpy(np.stack(task_mask_list)).bool().to(device),
            'action_mask': torch.from_numpy(np.stack(action_mask_list)).bool().to(device),
            'subtask_index': torch.from_numpy(np.array(subtask_index_list, dtype=np.int64)).long().to(device),
            'resource_ids': torch.from_numpy(np.stack(resource_ids_list)).long().to(device),
            'status': torch.from_numpy(np.stack(status_list)).long().to(device),
            'location': torch.from_numpy(np.stack(location_list)).long().to(device),
            'L_fwd': torch.from_numpy(np.stack(L_fwd_list)).long().to(device),
            'L_bwd': torch.from_numpy(np.stack(L_bwd_list)).long().to(device),
            'data_matrix': torch.from_numpy(np.stack(data_matrix_list)).float().to(device),
            'delta': torch.from_numpy(np.stack(delta_list)).long().to(device)
        }
        
        return inputs
    
    def forward(self,
                node_x: torch.Tensor,
                adj: torch.Tensor,
                status: torch.Tensor,
                location: torch.Tensor,
                L_fwd: torch.Tensor,
                L_bwd: torch.Tensor,
                data_matrix: torch.Tensor,
                delta: torch.Tensor,
                resource_ids: torch.Tensor,
                subtask_index: torch.Tensor,
                action_mask: torch.Tensor,
                task_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播（Beta分布版本）
        
        Args:
            node_x: [Batch, MAX_NODES, continuous_dim], 连续特征
            adj: [Batch, MAX_NODES, MAX_NODES], 邻接矩阵
            status: [Batch, MAX_NODES], 状态ID
            location: [Batch, MAX_NODES], 位置ID
            L_fwd: [Batch, MAX_NODES], 前向层级
            L_bwd: [Batch, MAX_NODES], 后向层级
            data_matrix: [Batch, MAX_NODES, MAX_NODES], 边数据量
            delta: [Batch, MAX_NODES, MAX_NODES], 最短路径距离
            resource_ids: [Batch, N_res], 资源ID列表
            subtask_index: [Batch], 当前选中任务索引
            action_mask: [Batch, N_res], 动作掩码（True=可选）
            task_mask: [Batch, MAX_NODES], 有效节点mask
        
        Returns:
            target_logits: [Batch, N_res]
            alpha: [Batch, 1], Beta分布参数
            beta: [Batch, 1], Beta分布参数
            value: [Batch, 1]
        """
        # 1. DAG节点嵌入
        node_emb = self.dag_embedding(node_x, status, location, L_fwd, L_bwd)
        
        # 2. 计算边偏置和空间偏置
        edge_bias = self.edge_encoder(data_matrix)
        spatial_bias = self.spatial_encoder(delta)
        
        # 3. Transformer编码
        # 构造padding mask（从task_mask）
        if task_mask is not None:
            key_padding_mask = ~task_mask  # True表示需要mask
        else:
            key_padding_mask = None
        
        dag_features = self.transformer(
            node_emb,
            edge_bias=edge_bias,
            spatial_bias=spatial_bias,
            key_padding_mask=key_padding_mask
        )
        
        # 4. 构建资源raw特征（用于物理偏置计算）
        # 这里需要从obs中构建，暂时用零占位
        batch_size = node_x.shape[0]
        n_res = resource_ids.shape[1]
        resource_raw = torch.zeros(batch_size, n_res, 9, device=node_x.device)
        
        # 5. 资源特征编码（物理特征 + ID嵌入）
        resource_encoded = self.resource_encoder(resource_raw, resource_ids)
        
        # 6. 生成资源padding mask（所有资源ID=0的位置）
        resource_padding_mask = (resource_ids == 0)
        
        # 7. Actor-Critic输出
        target_logits, alpha, beta, value = self.actor_critic(
            dag_features=dag_features,
            resource_encoded=resource_encoded,
            resource_raw=resource_raw,
            subtask_index=subtask_index,
            action_mask=action_mask,
            task_mask=task_mask,
            resource_padding_mask=resource_padding_mask
        )
        
        return target_logits, alpha, beta, value
    
    def get_action_and_value(self,
                            obs_list: List[Dict],
                            deterministic: bool = False,
                            device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从观测获取动作和价值（用于训练和推理）
        
        Args:
            obs_list: 环境观测列表
            deterministic: 是否使用确定性策略
            device: 计算设备
        
        Returns:
            target_actions: [Batch], 目标选择动作
            power_actions: [Batch], 功率比例动作
            log_probs: [Batch], 联合动作的log概率
            values: [Batch, 1], 状态价值估计
        """
        # 1. 准备输入
        inputs = self.prepare_inputs(obs_list, device)
        
        # 2. 前向传播
        target_logits, alpha, beta, values = self.forward(
            node_x=inputs['node_x'],
            adj=inputs['adj'],
            status=inputs['status'],
            location=inputs['location'],
            L_fwd=inputs['L_fwd'],
            L_bwd=inputs['L_bwd'],
            data_matrix=inputs['data_matrix'],
            delta=inputs['delta'],
            resource_ids=inputs['resource_ids'],
            subtask_index=inputs['subtask_index'],
            action_mask=inputs['action_mask'],
            task_mask=inputs['task_mask']
        )
        
        # 3. Target采样（Categorical分布）
        # [Logit Bias] 解决动作空间不平衡问题：给Local和RSU添加偏置
        # 索引映射：Index 0=Local, Index 1=RSU, Index 2+=Neighbors
        from configs.train_config import TrainConfig as TC
        if TC.USE_LOGIT_BIAS:
            logit_bias = torch.zeros_like(target_logits)
            logit_bias[:, 0] = TC.LOGIT_BIAS_LOCAL  # Local (Index 0)
            logit_bias[:, 1] = TC.LOGIT_BIAS_RSU    # RSU (Index 1)
            target_logits = target_logits + logit_bias
        
        # 应用action_mask，将无效动作的logits设为极小值
        action_mask_tensor = inputs['action_mask']
        masked_logits = torch.where(
            action_mask_tensor > 0,
            target_logits,
            torch.tensor(-1e10, dtype=target_logits.dtype, device=target_logits.device)
        )
        
        target_probs = F.softmax(masked_logits, dim=-1)
        target_dist = Categorical(target_probs)
        
        if deterministic:
            target_actions = torch.argmax(target_probs, dim=-1)
        else:
            target_actions = target_dist.sample()
        
        log_prob_target = target_dist.log_prob(target_actions)
        
        # 4. Power采样（Beta分布）
        power_dist = Beta(alpha.squeeze(-1), beta.squeeze(-1))
        
        if deterministic:
            # 使用期望值作为确定性动作
            power_actions = alpha / (alpha + beta)
        else:
            power_actions = power_dist.sample()
        
        log_prob_power = power_dist.log_prob(power_actions.squeeze(-1))
        
        # 5. 联合log概率
        log_probs = log_prob_target + log_prob_power
        
        return target_actions, power_actions.squeeze(-1), log_probs, values
    
    def evaluate_actions(self,
                        obs_list: List[Dict],
                        target_actions: torch.Tensor,
                        power_actions: torch.Tensor,
                        device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的log概率和熵（用于PPO训练）
        
        Args:
            obs_list: 环境观测列表
            target_actions: [Batch], 目标选择动作
            power_actions: [Batch], 功率比例动作
            device: 计算设备
        
        Returns:
            log_probs: [Batch], 联合动作的log概率
            entropy: [Batch], 联合动作的熵
            values: [Batch, 1], 状态价值估计
        """
        # 1. 准备输入
        inputs = self.prepare_inputs(obs_list, device)
        
        # 2. 前向传播
        target_logits, alpha, beta, values = self.forward(
            node_x=inputs['node_x'],
            adj=inputs['adj'],
            status=inputs['status'],
            location=inputs['location'],
            L_fwd=inputs['L_fwd'],
            L_bwd=inputs['L_bwd'],
            data_matrix=inputs['data_matrix'],
            delta=inputs['delta'],
            resource_ids=inputs['resource_ids'],
            subtask_index=inputs['subtask_index'],
            action_mask=inputs['action_mask'],
            task_mask=inputs['task_mask']
        )
        
        # 3. Target分布评估
        target_probs = F.softmax(target_logits, dim=-1)
        target_dist = Categorical(target_probs)
        log_prob_target = target_dist.log_prob(target_actions)
        entropy_target = target_dist.entropy()
        
        # 4. Power分布评估
        power_dist = Beta(alpha.squeeze(-1), beta.squeeze(-1))
        log_prob_power = power_dist.log_prob(power_actions)
        entropy_power = power_dist.entropy()
        
        # 5. 联合log概率和熵
        log_probs = log_prob_target + log_prob_power
        entropy = entropy_target + entropy_power
        
        return log_probs, entropy, values

