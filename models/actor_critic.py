"""
Actor-Critic网络模块

功能：
- Cross-Attention融合DAG特征与资源特征
- Actor: 双头输出（Target + Power）
- Critic: 全局池化估值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from configs.config import SystemConfig as Cfg


class CrossAttentionWithPhysicsBias(nn.Module):
    """
    带物理偏置的交叉注意力模块
    
    双流输入架构：
    - 语义流：编码后的特征 → K/V（学习的表示）
    - 物理流：原始9维特征 → 物理偏置（显式先验）
    
    物理偏置公式：Bias = -λ_dist * Dist - λ_load * Load
    """
    
    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
        
        # [关键] 可学习的物理偏置权重（初始化为建议值）
        self.lambda_dist = nn.Parameter(torch.tensor(1.0))
        self.lambda_load = nn.Parameter(torch.tensor(0.5))
    
    def forward(self,
                query: torch.Tensor,
                resource_encoded: torch.Tensor,
                resource_raw: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [Batch, 1, d_model], 查询（当前任务特征）
            resource_encoded: [Batch, N_res, d_model], 编码后的资源特征（用于K/V）
            resource_raw: [Batch, N_res, 9], 原始资源特征（用于物理偏置）
            key_padding_mask: [Batch, N_res], Padding mask
        
        Returns:
            [Batch, 1, d_model], 融合后的特征
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = resource_encoded.shape[1]
        
        # 1. 语义流：计算Q, K, V
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = self.W_k(resource_encoded).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        V = self.W_v(resource_encoded).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        
        # 转置 [B, H, N, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算语义注意力分数 [B, H, 1, N_res]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 2. 物理流：计算物理偏置
        # resource_raw: [B, N_res, 9]
        # [CPU, Queue, Dist, Rate, Rel_X, Rel_Y, Vel_X, Vel_Y, Node_Type]
        dist_norm = resource_raw[:, :, 2]  # Dist_Norm
        load_norm = resource_raw[:, :, 1]  # Queue_Norm
        
        # Bias = -λ_dist * dist - λ_load * load
        # [B, N_res] -> [B, 1, 1, N_res] (广播到多头)
        bias_phy = -self.lambda_dist * dist_norm - self.lambda_load * load_norm
        bias_phy = bias_phy.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N_res]
        
        # 3. 融合：语义分数 + 物理偏置
        attn_scores = attn_scores + bias_phy
        
        # 4. 应用Padding Mask
        if key_padding_mask is not None:
            # [B, N_res] -> [B, 1, 1, N_res]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # 5. Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. 加权求和 [B, H, 1, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # 7. 合并多头 [B, 1, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.d_model)
        
        # 8. 输出投影
        output = self.W_o(attn_output)
        
        return output


class ActorHead(nn.Module):
    """
    Actor输出头（Beta分布版本）
    
    双头输出：
    1. Target (Discrete): 成对拼接 + 批量计算
    2. Power (Continuous): Beta分布参数 (alpha, beta)
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 输入特征维度
        """
        super().__init__()
        self.d_model = d_model
        self.max_targets = 2 + Cfg.NUM_VEHICLES
        
        # Target头（成对特征MLP）
        # 输入：[h_fused, h_res] 拼接后为 2*d_model
        self.target_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)  # 输出单个logit
        )
        
        # Power头（Beta分布参数）
        # 输出alpha和beta两个参数
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 初始化bias，使初期输出Beta(2,2)
        # softplus(0.54) + 1.0 ≈ 2.0
        self._init_beta_params()
    
    def _init_beta_params(self):
        """初始化Beta分布参数，使初期集中在0.5附近"""
        # 设置bias使初期alpha≈2, beta≈2
        init_bias = 0.54
        self.alpha_head[-1].bias.data.fill_(init_bias)
        self.beta_head[-1].bias.data.fill_(init_bias)
        
        # 权重设小，让bias主导
        self.alpha_head[-1].weight.data.mul_(0.01)
        self.beta_head[-1].weight.data.mul_(0.01)
    
    def forward(self, 
                h_fused: torch.Tensor,
                h_res: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_fused: [Batch, d_model], 融合后的全局特征
            h_res: [Batch, N_res, d_model], 资源节点特征
            action_mask: [Batch, N_res], 动作掩码（True=可选）
        
        Returns:
            target_logits: [Batch, N_res], Target的logits
            alpha: [Batch, 1], Beta分布的alpha参数
            beta: [Batch, 1], Beta分布的beta参数
        """
        batch_size = h_fused.shape[0]
        n_res = h_res.shape[1]
        
        # 1. Target Head：成对拼接 + 批量计算
        # Broadcast h_fused: [B, d] -> [B, N_res, d]
        h_fused_expanded = h_fused.unsqueeze(1).expand(-1, n_res, -1)
        
        # 拼接：[B, N_res, 2*d_model]
        concat_all = torch.cat([h_fused_expanded, h_res], dim=-1)
        
        # 共享MLP计算所有logits：[B, N_res, 1] -> [B, N_res]
        target_logits = self.target_mlp(concat_all).squeeze(-1)
        
        # 应用action_mask（将不可选动作设为极小值）
        if action_mask is not None:
            target_logits = target_logits.masked_fill(~action_mask, -1e9)
        
        # 2. Power Head：Beta分布参数
        raw_alpha = self.alpha_head(h_fused)  # [B, 1]
        raw_beta = self.beta_head(h_fused)    # [B, 1]
        
        # 使用Softplus + 1确保 > 1（避免U型分布）
        alpha = F.softplus(raw_alpha) + 1.0
        beta = F.softplus(raw_beta) + 1.0
        
        return target_logits, alpha, beta


class CriticHead(nn.Module):
    """
    Critic输出头（原版 - 基于DAG全局池化）
    
    全局池化 + Value估计
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 输入特征维度
        """
        super().__init__()
        self.d_model = d_model
        
        # 注意力权重层（用于加权池化）
        self.attention_weight = nn.Linear(d_model, 1)
        
        # Value头
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self,
                dag_features: torch.Tensor,
                task_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            dag_features: [Batch, MAX_NODES, d_model], DAG节点特征
            task_mask: [Batch, MAX_NODES], 有效节点mask
        
        Returns:
            value: [Batch, 1], 状态价值估计
        """
        # 1. 计算注意力权重
        attn_scores = self.attention_weight(dag_features)  # [B, N, 1]
        
        # 应用task_mask
        if task_mask is not None:
            mask = task_mask.unsqueeze(-1)  # [B, N, 1]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]
        # 处理全-inf行导致的NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # 2. 加权池化
        global_feature = torch.sum(dag_features * attn_weights, dim=1)  # [B, d_model]
        
        # 3. Value估计
        value = self.value_head(global_feature)  # [B, 1]
        
        return value


class SimplifiedCriticHead(nn.Module):
    """
    简化版Critic输出头（Phase 1）
    
    输入：局部观测特征聚合
    - DAG全局特征（Attention Pooling）
    - 邻居平均特征（Mean Pooling）
    - RSU特征
    
    类似IPPO，用于快速验证Actor逻辑
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 输入特征维度
        """
        super().__init__()
        self.d_model = d_model
        
        # DAG特征池化权重
        self.dag_attention_weight = nn.Linear(d_model, 1)
        
        # Value头（输入：DAG + Neighbors + RSU = 3*d_model）
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self,
                dag_features: torch.Tensor,
                resource_features: torch.Tensor,
                task_mask: Optional[torch.Tensor] = None,
                resource_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            dag_features: [Batch, MAX_NODES, d_model], DAG节点特征
            resource_features: [Batch, N_res, d_model], 资源节点特征
            task_mask: [Batch, MAX_NODES], DAG有效节点mask
            resource_mask: [Batch, N_res], 资源有效节点mask
        
        Returns:
            value: [Batch, 1], 状态价值估计
        """
        batch_size = dag_features.shape[0]
        
        # 1. DAG全局特征（Attention Pooling）
        dag_attn_scores = self.dag_attention_weight(dag_features)  # [B, N_dag, 1]
        if task_mask is not None:
            mask = task_mask.unsqueeze(-1)
            dag_attn_scores = dag_attn_scores.masked_fill(~mask, float('-inf'))
        dag_attn_weights = F.softmax(dag_attn_scores, dim=1)
        # 处理全-inf行导致的NaN（当整行都被mask时）
        dag_attn_weights = torch.nan_to_num(dag_attn_weights, nan=0.0)
        dag_global = torch.sum(dag_features * dag_attn_weights, dim=1)  # [B, d]
        
        # 2. 邻居平均特征（Mean Pooling，排除Local和RSU）
        # resource_features: [B, N_res, d], 前2个是Local/RSU，后面是Neighbors
        if resource_features.shape[1] > 2:
            neighbor_feats = resource_features[:, 2:, :]  # [B, N_neighbors, d]
            if resource_mask is not None:
                neighbor_mask = resource_mask[:, 2:]  # [B, N_neighbors]
                # 计算有效邻居的平均
                neighbor_mask_expanded = neighbor_mask.unsqueeze(-1)  # [B, N_neighbors, 1]
                valid_count = torch.sum(neighbor_mask_expanded, dim=1, keepdim=True).clamp(min=1)
                neighbor_avg = torch.sum(neighbor_feats * neighbor_mask_expanded, dim=1) / valid_count.squeeze(-1)
            else:
                neighbor_avg = torch.mean(neighbor_feats, dim=1)  # [B, d]
        else:
            # 无邻居，用零向量
            neighbor_avg = torch.zeros(batch_size, self.d_model, device=dag_features.device)
        
        # 3. RSU特征（Index 1）
        rsu_feat = resource_features[:, 1, :]  # [B, d]
        
        # 4. 拼接所有特征
        critic_input = torch.cat([dag_global, neighbor_avg, rsu_feat], dim=-1)  # [B, 3*d]
        
        # 5. Value估计
        value = self.value_head(critic_input)  # [B, 1]
        
        return value


class ActorCriticNetwork(nn.Module):
    """
    完整的Actor-Critic网络（更新版）
    
    流程：
    1. DAG特征嵌入 + Transformer
    2. 资源特征编码（物理特征 + ID嵌入）
    3. Actor: Gather选中任务 -> Cross-Attention（物理偏置） -> 双头输出
    4. Critic: 简化版（局部观测聚合）或完整版（DAG池化）
    """
    
    def __init__(self,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 use_simplified_critic: bool = True):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            d_ff: 前馈层维度
            dropout: Dropout率
            use_simplified_critic: 是否使用简化版Critic（Phase 1）
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_simplified_critic = use_simplified_critic
        
        # Cross-Attention（带物理偏置）
        self.cross_attention = CrossAttentionWithPhysicsBias(d_model, num_heads, dropout)
        
        # Actor头（成对拼接版）
        self.actor_head = ActorHead(d_model)
        
        # Critic头（根据配置选择）
        if use_simplified_critic:
            self.critic_head = SimplifiedCriticHead(d_model)
        else:
            self.critic_head = CriticHead(d_model)
        
        # Layer Norm（用于Cross-Attention后）
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward_actor(self,
                     dag_features: torch.Tensor,
                     resource_encoded: torch.Tensor,
                     resource_raw: torch.Tensor,
                     subtask_index: torch.Tensor,
                     action_mask: torch.Tensor,
                     resource_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actor前向传播（Beta分布版）
        
        Args:
            dag_features: [Batch, MAX_NODES, d_model], Transformer输出
            resource_encoded: [Batch, N_res, d_model], 编码后的资源特征
            resource_raw: [Batch, N_res, 9], 原始资源特征（用于物理偏置）
            subtask_index: [Batch], 当前选中的任务索引
            action_mask: [Batch, N_res], 动作掩码（True=可选）
            resource_padding_mask: [Batch, N_res], 资源Padding mask
        
        Returns:
            target_logits: [Batch, N_res]
            alpha: [Batch, 1], Beta分布参数
            beta: [Batch, 1], Beta分布参数
        """
        batch_size = dag_features.shape[0]
        
        # 1. Gather当前选中任务的特征
        # subtask_index: [B] -> [B, 1, 1] -> [B, 1, d_model]
        # 处理-1情况（无ready任务）：使用第0个节点作为默认
        safe_subtask_index = torch.clamp(subtask_index, min=0)
        indices = safe_subtask_index.unsqueeze(1).unsqueeze(2).expand(-1, -1, dag_features.shape[-1])  # [B, 1, d_model]
        query = torch.gather(dag_features, 1, indices)  # [B, 1, d_model]
        
        # 2. Cross-Attention（带物理偏置）融合资源特征
        h_fused = self.cross_attention(
            query=query,
            resource_encoded=resource_encoded,
            resource_raw=resource_raw,
            key_padding_mask=resource_padding_mask
        )  # [B, 1, d_model]
        
        # 3. Layer Norm + Residual
        residual = query + h_fused
        # 防止NaN：检查residual
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
        h_fused = self.layer_norm(residual)
        
        # 4. Squeeze到[B, d_model]
        h_fused = h_fused.squeeze(1)
        
        # 5. Actor双头输出（Beta分布）
        target_logits, alpha, beta = self.actor_head(
            h_fused=h_fused,
            h_res=resource_encoded,
            action_mask=action_mask
        )
        
        return target_logits, alpha, beta
    
    def forward_critic(self,
                      dag_features: torch.Tensor,
                      resource_features: Optional[torch.Tensor] = None,
                      task_mask: Optional[torch.Tensor] = None,
                      resource_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Critic前向传播（更新版）
        
        Args:
            dag_features: [Batch, MAX_NODES, d_model], Transformer输出
            resource_features: [Batch, N_res, d_model], 资源特征（简化版需要）
            task_mask: [Batch, MAX_NODES], DAG有效节点mask
            resource_mask: [Batch, N_res], 资源有效节点mask（简化版需要）
        
        Returns:
            value: [Batch, 1], 状态价值估计
        """
        if self.use_simplified_critic:
            # 简化版Critic需要资源特征
            assert resource_features is not None, "Simplified critic requires resource_features"
            value = self.critic_head(dag_features, resource_features, task_mask, resource_mask)
        else:
            # 原版Critic只需要DAG特征
            value = self.critic_head(dag_features, task_mask)
        
        return value
    
    def forward(self,
                dag_features: torch.Tensor,
                resource_encoded: torch.Tensor,
                resource_raw: torch.Tensor,
                subtask_index: torch.Tensor,
                action_mask: torch.Tensor,
                task_mask: Optional[torch.Tensor] = None,
                resource_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播（同时计算Actor和Critic）（Beta分布版）
        
        Args:
            dag_features: [Batch, MAX_NODES, d_model], DAG Transformer输出
            resource_encoded: [Batch, N_res, d_model], 编码后的资源特征
            resource_raw: [Batch, N_res, 9], 原始资源特征
            subtask_index: [Batch], 选中的任务索引
            action_mask: [Batch, N_res], 动作掩码
            task_mask: [Batch, MAX_NODES], DAG节点掩码
            resource_padding_mask: [Batch, N_res], 资源Padding掩码
        
        Returns:
            target_logits: [Batch, N_res]
            alpha: [Batch, 1], Beta分布参数
            beta: [Batch, 1], Beta分布参数
            value: [Batch, 1]
        """
        # Actor
        target_logits, alpha, beta = self.forward_actor(
            dag_features, resource_encoded, resource_raw,
            subtask_index, action_mask, resource_padding_mask
        )
        
        # Critic
        value = self.forward_critic(
            dag_features, resource_encoded, task_mask, resource_padding_mask
        )
        
        return target_logits, alpha, beta, value

