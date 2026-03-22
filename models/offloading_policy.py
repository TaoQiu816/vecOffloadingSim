"""
[卸载策略网络] offloading_policy.py
Complete Offloading Policy Network

作用 (Purpose):
    整合所有模块构建完整的卸载决策网络，从原始观测到动作输出的端到端流程。
    Integrates all modules to build complete offloading decision network, 
    end-to-end pipeline from raw observations to action outputs.

网络架构 (Network Architecture):
    1. DAG特征嵌入 (DAG Feature Embedding)
       - 节点嵌入：comp, data, status, in_degree, location
       - 边特征编码：data_matrix → edge_bias
       - 空间距离编码：shortest_path_matrix → spatial_bias
    
    2. 边增强Transformer (Edge-Enhanced Transformer)
       - 多头自注意力 + 边偏置 + 空间偏置
       - 捕获DAG依赖关系和拓扑结构
    
    3. 资源特征编码 (Resource Feature Encoding)
       - 10维资源特征：CPU, backlog, tx backlog, distance, contact, competition等
       - 角色嵌入：Local/RSU/Neighbor（无ID泄漏）
    
    4. Actor-Critic输出 (Actor-Critic Heads)
       - Actor: 双头输出（Target选择 + Power控制）
         * Target: Categorical分布（Local/RSU/V2V）
         * Power: Beta分布（连续功率比例）
       - Critic: 全局池化估值（状态价值函数）

输入输出 (Input/Output):
    输入 (Input):
        - node_x: [B, N, 7] - DAG节点特征
        - adj: [B, N, N] - 邻接矩阵
        - data_matrix: [B, N, N] - 边数据量
        - delta: [B, N, N] - 最短路径矩阵
        - resource_raw: [B, M, 10] - 资源原始特征
        - resource_ids: [B, M] - 资源角色ID
        - subtask_index: [B] - 兼容fallback索引，不代表环境预选任务
        - action_mask: [B, M] - 动作掩码
        - task_mask: [B, N] - 任务掩码
    
    输出 (Output):
        - target_logits: [B, M] - 目标选择logits
        - alpha, beta: [B, 1] - Beta分布参数（功率）
        - values: [B, 1] - 状态价值

参考文献 (References):
    - Transformer: Vaswani et al., "Attention Is All You Need" (2017)
    - Actor-Critic: Mnih et al., "Asynchronous Methods for Deep RL" (2016)
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
    SpatialDistanceEncoder,
    RankBiasEncoder
)
from models.edge_enhanced_transformer import EdgeEnhancedTransformer
from models.resource_features import ResourceFeatureEncoder
from models.actor_critic import ActorCriticNetwork
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


class OffloadingPolicyNetwork(nn.Module):
    """
    完整的卸载决策网络 (Complete Offloading Policy Network)
    
    功能：
        - 整合DAG编码、Transformer、资源编码和Actor-Critic模块
        - 端到端学习从观测到动作的映射
        - 支持离散动作（Target）和连续动作（Power）的混合动作空间
    """
    
    def __init__(self,
                 d_model: int = None,
                 num_heads: int = None,
                 num_layers: int = None,
                 d_ff: int = None,
                 dropout: float = None,
                 continuous_dim: int = 7):
        """
        Args:
            d_model: 模型维度（默认从 TrainConfig.EMBED_DIM 读取）
            num_heads: 注意力头数（默认从 TrainConfig.NUM_HEADS 读取）
            num_layers: Transformer层数（默认从 TrainConfig.NUM_LAYERS 读取）
            d_ff: 前馈层维度（默认从 TrainConfig.D_FF 读取）
            dropout: Dropout率（默认从 TrainConfig.DROPOUT 读取）
            continuous_dim: 连续特征维度
        """
        super().__init__()

        # 从配置文件读取默认值
        d_model = d_model if d_model is not None else TC.EMBED_DIM
        num_heads = num_heads if num_heads is not None else TC.NUM_HEADS
        num_layers = num_layers if num_layers is not None else TC.NUM_LAYERS
        d_ff = d_ff if d_ff is not None else TC.D_FF
        dropout = dropout if dropout is not None else TC.DROPOUT
        
        self.d_model = d_model
        
        # 1. DAG节点嵌入
        self.dag_embedding = DAGNodeEmbedding(d_model, continuous_dim)
        
        # 2. 边特征和空间距离编码器
        self.edge_encoder = EdgeFeatureEncoder(num_heads)
        self.spatial_encoder = SpatialDistanceEncoder(num_heads)

        # 2.5 [暂时停用] Rank偏置编码器（priority/rank 相关路径）
        # 为避免影响当前RC1仿真与训练口径，先显式关闭rank-bias前向路径。
        # 保留priority相关函数签名与调用参数以兼容旧脚本/模型代码。
        # if TC.USE_RANK_BIAS:
        #     self.rank_bias_encoder = RankBiasEncoder(num_heads)
        # else:
        #     self.rank_bias_encoder = None
        self.rank_bias_encoder = None

        # 3. 边增强Transformer
        self.transformer = EdgeEnhancedTransformer(
            num_layers, d_model, num_heads, d_ff, dropout
        )
        
        # 4. 资源特征编码器
        # [修复] 移除max_vehicle_id参数（已在审计中删除，ResourceFeatureEncoder已标记为废弃参数）
        self.resource_encoder = ResourceFeatureEncoder(
            d_model=d_model
        )
        
        # 4.5 rate_prev 轻量投影（上步链路速率 → 资源编码空间）
        self.rate_prev_proj = nn.Linear(1, d_model)
        nn.init.zeros_(self.rate_prev_proj.weight)  # 初始化为0，默认不影响已有训练
        nn.init.zeros_(self.rate_prev_proj.bias)

        # 4.6 serving_rsu_onehot 投影（当前服务RSU → 全局条件，广播到所有资源候选）
        num_rsu = getattr(Cfg, 'NUM_RSU', 3)
        self.serving_rsu_proj = nn.Linear(num_rsu, d_model)
        nn.init.zeros_(self.serving_rsu_proj.weight)  # 零初始化，默认不影响已有训练
        nn.init.zeros_(self.serving_rsu_proj.bias)

        # 5. Actor-Critic网络
        self.actor_critic = ActorCriticNetwork(
            d_model, num_heads, num_layers, d_ff, dropout,
            # 论文固定版：严格CTDE（不依赖可切换模式）
            use_simplified_critic=False,
            use_subtask_cond_critic=True,
            use_no_ready_embed=True,
            use_commwait_direct=True,
            commwait_dim=int(getattr(TC, "CTDE_GLOBAL_DIM", 30)),
        )

        # 条件功率头：让 power 分布显式依赖选中的 target（混合动作一致性）
        cond_in_dim = int(Cfg.RESOURCE_RAW_DIM) + 3  # resource_raw + one-hot(candidate_type)
        cond_hidden = max(d_model // 2, 16)
        self.power_cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, cond_hidden),
            nn.ReLU(),
            nn.Linear(cond_hidden, 2),
        )
        # 零初始化最后一层，保证启用时不破坏已有训练稳定性
        nn.init.zeros_(self.power_cond_mlp[-1].weight)
        nn.init.zeros_(self.power_cond_mlp[-1].bias)
    
    @staticmethod
    def _ablation_flags() -> Tuple[bool, bool, str]:
        mode = str(getattr(TC, "ABLATION_MODE", "full")).strip().lower()
        disable_dag = mode in {"no_dag", "no_dag_resource"}
        disable_resource = mode in {"no_resource", "no_dag_resource"}
        return disable_dag, disable_resource, mode

    def _sanitize_policy_side_channels(
        self,
        resource_raw: torch.Tensor,
        candidate_types: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disable_dag, disable_resource, _ = self._ablation_flags()
        if not disable_resource:
            return resource_raw, candidate_types, action_mask
        # Keep action_mask only as legality constraint; strip all resource semantics.
        return (
            torch.zeros_like(resource_raw),
            torch.zeros_like(candidate_types),
            action_mask,
        )

    @staticmethod
    def _apply_logit_bias(target_logits: torch.Tensor,
                          candidate_types: torch.Tensor,
                          action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        [通用化] 根据candidate_types动态赋logit_bias，取代硬编码index 0/1。
        candidate_types: [B, M]，1=Local, 2=RSU, 3=V2V

        类别平衡校正（仅概率质量分配，不是硬约束）:
        - Local: +b_L
        - RSU  : +b_R
        - V2V每个候选: +(b_V - log(max(n_V, 1)))
          其中 n_V 为当前步可行动作中 V2V 候选数量（按 action_mask 计数）。
        """
        if not TC.USE_LOGIT_BIAS:
            return target_logits
        logit_bias = torch.zeros_like(target_logits)
        logit_bias[candidate_types == 1] = TC.LOGIT_BIAS_LOCAL   # Local槽位
        logit_bias[candidate_types == 2] = TC.LOGIT_BIAS_RSU     # RSU槽位（所有RSU统一）
        # V2V 探索 bias（线性退火，由 train.py 更新 _logit_bias_v2v_current）
        # 叠加类别规模校正: -log(n_V)，抵消 ALL 候选下 V2V 槽位数过多的先天优势。
        v2v_bias = float(getattr(TC, '_logit_bias_v2v_current', 0.0))
        v2v_mask = (candidate_types == 3)
        if action_mask is not None:
            feasible_mask = action_mask > 0
            feasible_v2v = v2v_mask & feasible_mask
        else:
            feasible_v2v = v2v_mask
        n_v = feasible_v2v.sum(dim=-1, keepdim=True).to(dtype=target_logits.dtype)  # [B,1]
        # In ALL-candidate mode, only a small subset of neighbors is usually competitive.
        # Keep the original full correction by default, but allow a capped / scaled correction
        # for controlled ablations when full -log(n_V) over-suppresses the whole V2V mode.
        cap = int(getattr(TC, "LOGIT_BIAS_V2V_SIZE_CORR_CAP", 0) or 0)
        if cap > 0:
            n_v_eff = torch.clamp(n_v, min=1.0, max=float(cap))
        else:
            n_v_eff = torch.clamp(n_v, min=1.0)
        corr_coef = float(getattr(TC, "LOGIT_BIAS_V2V_SIZE_CORR_COEF", 1.0))
        v2v_size_correction = corr_coef * torch.log(n_v_eff)
        v2v_slot_bias = (v2v_bias - v2v_size_correction)  # [B,1], broadcast to V2V slots
        logit_bias = logit_bias + v2v_mask.to(dtype=target_logits.dtype) * v2v_slot_bias
        return target_logits + logit_bias

    @staticmethod
    def _gather_selected(x: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Gather per-batch selected target features from x:[B,M,D] or x:[B,M]."""
        ta = target_actions.long().view(-1)
        if x.dim() == 3:
            idx = ta.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            return torch.gather(x, 1, idx).squeeze(1)
        idx = ta.view(-1, 1)
        return torch.gather(x, 1, idx).squeeze(1)

    def _build_target_conditioned_power_dist(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        resource_raw: torch.Tensor,
        candidate_types: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Beta:
        """
        Build Beta(power | o, target).
        Keep base alpha/beta from actor head, then modulate by selected target features.
        """
        base_alpha = alpha.squeeze(-1)
        base_beta = beta.squeeze(-1)

        sel_raw = self._gather_selected(resource_raw, target_actions)  # [B, R]
        sel_type = self._gather_selected(candidate_types, target_actions).long()  # [B]
        sel_type = torch.clamp(sel_type - 1, min=0, max=2)
        type_oh = F.one_hot(sel_type, num_classes=3).to(dtype=sel_raw.dtype, device=sel_raw.device)
        cond_in = torch.cat([sel_raw, type_oh], dim=-1)

        delta = torch.tanh(self.power_cond_mlp(cond_in))
        cond_scale = float(getattr(TC, "POWER_COND_SCALE", 0.20))
        delta = delta * cond_scale

        alpha_cond = torch.clamp(base_alpha * torch.exp(delta[:, 0]), min=1.001, max=100.0)
        beta_cond = torch.clamp(base_beta * torch.exp(delta[:, 1]), min=1.001, max=100.0)
        return Beta(alpha_cond, beta_cond)

    @staticmethod
    def _masked_subtask_logits(
        subtask_logits: torch.Tensor,
        subtask_mask: torch.Tensor,
        node_valid_mask: Optional[torch.Tensor] = None,
        fallback_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        为 subtask logits 应用掩码，并处理全0 mask 的保底逻辑。
        """
        mask = (subtask_mask > 0).bool()
        if mask.ndim != 2:
            raise ValueError(f"subtask_mask dim must be 2, got {mask.shape}")
        if node_valid_mask is not None:
            mask = mask & node_valid_mask.bool()

        no_valid = ~mask.any(dim=-1)
        if torch.any(no_valid):
            mask = mask.clone()
            if fallback_index is not None:
                safe_fb = torch.clamp(fallback_index.long(), min=0, max=mask.shape[-1] - 1)
                row_idx = torch.nonzero(no_valid, as_tuple=False).squeeze(-1)
                mask[row_idx, safe_fb[row_idx]] = True
            else:
                mask[no_valid, 0] = True
            # 若fallback位置无效，再兜底开第一个有效节点
            if node_valid_mask is not None:
                still_invalid = ~mask.any(dim=-1)
                if torch.any(still_invalid):
                    row_idx = torch.nonzero(still_invalid, as_tuple=False).squeeze(-1)
                    first_valid = torch.argmax(node_valid_mask[row_idx].long(), dim=-1)
                    mask[row_idx, first_valid] = True
        return subtask_logits.masked_fill(~mask, -1e9)

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
        subtask_mask_list = []
        node_valid_mask_list = []
        action_mask_list = []
        subtask_index_list = []
        resource_ids_list = []
        resource_raw_list = []
        candidate_types_list = []  # [通用化] 候选类型数组
        rate_prev_list = []        # 上步链路速率
        serving_rsu_onehot_list = []  # 当前服务RSU的one-hot编码
        global_state_list = []        # centralized critic全局摘要（CTDE）

        # DAG拓扑特征（环境已提供）
        status_list = []
        location_list = []
        L_fwd_list = []
        L_bwd_list = []
        data_matrix_list = []
        delta_list = []
        priority_list = []  # [方案A] 用于Rank Bias（可选）
        
        for obs in obs_list:
            node_x_list.append(obs['node_x'])
            adj_list.append(obs['adj'])
            # 语义解绑：task_mask(旧)可能是可调度掩码；新字段优先
            subtask_mask_list.append(obs.get('subtask_mask', obs.get('task_mask')))
            node_valid_mask_list.append(obs.get('node_valid_mask', obs.get('task_mask')))
            task_mask_list.append(obs.get('node_valid_mask', obs.get('task_mask')))
            action_mask_list.append(obs['action_mask'])
            subtask_index_list.append(obs['subtask_index'])
            resource_ids_list.append(obs['resource_ids'])
            resource_raw_list.append(obs['resource_raw'])
            # [通用化] 提取candidate_types (1=Local, 2=RSU, 3=V2V)
            candidate_types_list.append(
                obs.get('candidate_types', np.zeros(obs['action_mask'].shape[0], dtype=np.int8))
            )
            # 上步链路速率 [MAX_TARGETS]
            rate_prev_list.append(
                obs.get('rate_prev', np.zeros(obs['action_mask'].shape[0], dtype=np.float32))
            )
            # 当前服务RSU [NUM_RSU]
            num_rsu = getattr(Cfg, 'NUM_RSU', 3)
            serving_rsu_onehot_list.append(
                obs.get('serving_rsu_onehot', np.zeros(num_rsu, dtype=np.float32))
            )
            gdim = int(getattr(TC, "CTDE_GLOBAL_DIM", 0))
            g = obs.get("global_state", None)
            if g is None:
                g = np.zeros(gdim, dtype=np.float32)
            else:
                g = np.asarray(g, dtype=np.float32).reshape(-1)
                if gdim > 0:
                    if g.shape[0] < gdim:
                        g = np.pad(g, (0, gdim - g.shape[0]))
                    elif g.shape[0] > gdim:
                        g = g[:gdim]
            global_state_list.append(g)
            
            # 从环境提供的字段中获取
            if 'status' in obs:
                status_list.append(obs['status'])
            else:
                node_x = obs['node_x']
                status_col = node_x[:, 2] if node_x.shape[1] > 2 else np.zeros(node_x.shape[0])
                status_arr = np.rint(np.clip(status_col * 3.0, 0.0, 3.0)).astype(np.int64)
                status_list.append(status_arr)
            location_list.append(obs['location'])
            L_fwd_list.append(obs['L_fwd'])
            L_bwd_list.append(obs['L_bwd'])
            data_matrix_list.append(obs['data_matrix'])
            delta_list.append(obs['Delta'])
            # [暂时停用] priority/rank 路径
            # 保留兼容变量priority_list，但不再从obs提取priority，避免误启用rank-bias。
            # if 'priority' in obs and obs.get('priority') is not None:
            #     priority_list.append(np.asarray(obs['priority'], dtype=np.float32))
        
        # 转换为Tensor并移到目标设备
        inputs = {
            'node_x': torch.from_numpy(np.stack(node_x_list)).float().to(device),
            'adj': torch.from_numpy(np.stack(adj_list)).float().to(device),
            'task_mask': torch.from_numpy(np.stack(task_mask_list)).bool().to(device),  # alias: DAG有效节点mask
            'subtask_mask': torch.from_numpy(np.stack(subtask_mask_list)).bool().to(device),
            'node_valid_mask': torch.from_numpy(np.stack(node_valid_mask_list)).bool().to(device),
            'action_mask': torch.from_numpy(np.stack(action_mask_list)).bool().to(device),
            'subtask_index': torch.from_numpy(np.array(subtask_index_list, dtype=np.int64)).long().to(device),
            'resource_ids': torch.from_numpy(np.stack(resource_ids_list)).long().to(device),
            'resource_raw': torch.from_numpy(np.stack(resource_raw_list)).float().to(device),
            'status': torch.from_numpy(np.stack(status_list)).long().to(device),
            'location': torch.from_numpy(np.stack(location_list)).long().to(device),
            'L_fwd': torch.from_numpy(np.stack(L_fwd_list)).long().to(device),
            'L_bwd': torch.from_numpy(np.stack(L_bwd_list)).long().to(device),
            'data_matrix': torch.from_numpy(np.stack(data_matrix_list)).float().to(device),
            'delta': torch.from_numpy(np.stack(delta_list)).long().to(device),
            'candidate_types': torch.from_numpy(np.stack(candidate_types_list)).long().to(device),  # [通用化]
            'rate_prev': torch.from_numpy(np.stack(rate_prev_list)).float().to(device),  # [B, M]
            'serving_rsu_onehot': torch.from_numpy(np.stack(serving_rsu_onehot_list)).float().to(device),  # [B, NUM_RSU]
            'global_state': torch.from_numpy(np.stack(global_state_list)).float().to(device),  # [B, G]
        }
        # [暂时停用] priority/rank 路径：不向前向传播提供inputs['priority']
        # if len(priority_list) == len(obs_list) and len(priority_list) > 0:
        #     inputs['priority'] = torch.from_numpy(np.stack(priority_list)).float().to(device)
        
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
                resource_raw: torch.Tensor,
                subtask_index: torch.Tensor,
                action_mask: torch.Tensor,
                subtask_mask: Optional[torch.Tensor] = None,
                node_valid_mask: Optional[torch.Tensor] = None,
                task_mask: Optional[torch.Tensor] = None,
                priority: Optional[torch.Tensor] = None,
                rate_prev: Optional[torch.Tensor] = None,
                serving_rsu_onehot: Optional[torch.Tensor] = None,
                global_state: Optional[torch.Tensor] = None,
                candidate_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播（Beta分布版本）

        【一致性保证】
        forward() 和 evaluate_actions() 都调用此方法，确保rank_bias使用一致。

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
            resource_raw: [Batch, N_res, RESOURCE_RAW_DIM], 资源原始特征
            subtask_index: [Batch], 兼容fallback索引
            action_mask: [Batch, N_res], 动作掩码（True=可选）
            subtask_mask: [Batch, MAX_NODES], 可调度子任务mask（READY且未分配）
            node_valid_mask: [Batch, MAX_NODES], DAG有效节点mask（非padding）
            task_mask: [Batch, MAX_NODES], 兼容别名（若node_valid_mask为空则使用）
            priority: [Batch, MAX_NODES], 节点优先级分数（0-1，越大越重要）[方案A新增]
            rate_prev: [Batch, N_res], 上步链路速率（归一化）[时序信号补齐]
            serving_rsu_onehot: [Batch, NUM_RSU], 当前服务RSU的one-hot编码 [多RSU感知]

        Returns:
            subtask_logits: [Batch, MAX_NODES]
            target_logits: [Batch, N_res]
            alpha: [Batch, 1], Beta分布参数
            beta: [Batch, 1], Beta分布参数
            value: [Batch, 1]
        """
        if node_valid_mask is None:
            node_valid_mask = task_mask
        disable_dag, disable_resource, _ = self._ablation_flags()

        node_raw_for_head = node_x
        if disable_dag:
            node_raw_for_head = node_x.clone()
            for col in (2, 3, 4, 6):
                if col < node_raw_for_head.shape[-1]:
                    node_raw_for_head[..., col] = 0.0
            adj = torch.zeros_like(adj)
            data_matrix = torch.zeros_like(data_matrix)
            delta = torch.zeros_like(delta)

        # 1. DAG节点嵌入
        node_emb = self.dag_embedding(node_x, status, location, L_fwd, L_bwd)

        # 2. 计算边偏置和空间偏置（支持消融开关）
        edge_bias = None
        spatial_bias = None
        if not disable_dag and getattr(TC, "USE_EDGE_BIAS", True):
            edge_bias = self.edge_encoder(data_matrix)
        if not disable_dag and getattr(TC, "USE_SPATIAL_BIAS", True):
            spatial_bias = self.spatial_encoder(delta)

        # 2.5 [暂时停用] Rank偏置（priority/rank 相关路径）
        rank_bias = None
        # if self.rank_bias_encoder is not None and priority is not None:
        #     rank_bias = self.rank_bias_encoder(
        #         priority=priority,
        #         adj=adj,
        #         tau=TC.RANK_BIAS_TAU,
        #         kappa=TC.RANK_BIAS_KAPPA,
        #         cover_mode=TC.RANK_BIAS_COVER,
        #         task_mask=node_valid_mask
        #     )

        # 3. Transformer编码
        # 构造padding mask（从task_mask）
        if node_valid_mask is not None:
            key_padding_mask = ~node_valid_mask  # True表示需要mask
        else:
            key_padding_mask = None

        dag_features = self.transformer(
            node_emb,
            edge_bias=edge_bias,
            spatial_bias=spatial_bias,
            rank_bias=rank_bias,  # [方案A] 传递rank_bias
            key_padding_mask=key_padding_mask
        )
        
        # 4. 资源特征编码（物理特征 + ID嵌入）
        assert resource_raw.shape[-1] == Cfg.RESOURCE_RAW_DIM, (
            f"resource_raw dim mismatch: got {resource_raw.shape[-1]}, expected {Cfg.RESOURCE_RAW_DIM}"
        )
        resource_encoded = self.resource_encoder(resource_raw, resource_ids)

        # 4.5 将上步链路速率投影并加到资源编码上（时序信号补齐）
        if rate_prev is not None:
            # rate_prev: [B, M] → [B, M, 1] → proj → [B, M, d_model]
            rate_prev_emb = self.rate_prev_proj(rate_prev.unsqueeze(-1))
            resource_encoded = resource_encoded + rate_prev_emb

        # 4.6 将当前服务RSU信息投影并广播加到所有资源候选上（多RSU感知）
        if serving_rsu_onehot is not None:
            # serving_rsu_onehot: [B, NUM_RSU] → proj → [B, d_model] → unsqueeze → [B, 1, d_model]
            rsu_ctx = self.serving_rsu_proj(serving_rsu_onehot).unsqueeze(1)
            resource_encoded = resource_encoded + rsu_ctx  # broadcast到 [B, M, d_model]

        resource_raw_for_head = resource_raw
        if disable_resource:
            resource_encoded = torch.zeros_like(resource_encoded)
            resource_raw_for_head = torch.zeros_like(resource_raw)
            if global_state is not None:
                global_state = torch.zeros_like(global_state)

        # 6. 生成资源padding mask（padding或不可选动作）
        resource_padding_mask = (resource_ids == 0)
        if action_mask is not None:
            resource_padding_mask = resource_padding_mask | (~action_mask)

        # 论文固定版：critic始终读取集中摘要global_state
        # 测试/独立前向场景下若未提供，回退为零向量以保持输入维度一致。
        if global_state is None:
            gdim = int(getattr(TC, "CTDE_GLOBAL_DIM", 0))
            if gdim > 0:
                global_state = torch.zeros(
                    (node_x.shape[0], gdim),
                    dtype=node_x.dtype,
                    device=node_x.device,
                )
        commwait_extra = global_state
        subtask_mask_for_model = subtask_mask if subtask_mask is not None else node_valid_mask
        if disable_dag:
            # Keep readiness only in the external legality mask, not in critic conditioning.
            subtask_mask_for_model = node_valid_mask

        # 7. Actor-Critic输出
        subtask_logits, target_logits, alpha, beta, value, cost_power_value, cost_trust_value = self.actor_critic(
            dag_features=dag_features,
            node_raw=node_raw_for_head,
            resource_encoded=resource_encoded,
            resource_raw=resource_raw_for_head,
            subtask_index=subtask_index,
            action_mask=action_mask,
            candidate_types=candidate_types,
            subtask_mask=subtask_mask_for_model,
            task_mask=node_valid_mask,
            resource_padding_mask=resource_padding_mask,
            commwait_extra=commwait_extra,
        )
        
        return subtask_logits, target_logits, alpha, beta, value, cost_power_value, cost_trust_value
    
    def get_action_and_value(self,
                            obs_list: List[Dict],
                            deterministic: bool = False,
                            device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从观测获取动作和价值（用于训练和推理）
        
        Args:
            obs_list: 环境观测列表
            deterministic: 是否使用确定性策略
            device: 计算设备
        
        Returns:
            subtask_actions: [Batch], 子任务动作
            target_actions: [Batch], 目标选择动作
            power_actions: [Batch], 功率比例动作
            log_probs: [Batch], 联合动作的log概率
            values: [Batch, 1], 状态价值估计
        """
        # 1. 准备输入
        inputs = self.prepare_inputs(obs_list, device)
        policy_resource_raw, policy_candidate_types, policy_action_mask = self._sanitize_policy_side_channels(
            inputs["resource_raw"],
            inputs["candidate_types"],
            inputs["action_mask"],
        )
        
        # 2. 前向传播
        subtask_logits, target_logits, alpha, beta, values_env, cost_power_values, cost_trust_values = self.forward(
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
            priority=inputs.get('priority'),  # [方案A] 传递priority（可选）
            rate_prev=inputs['rate_prev'],  # 上步链路速率
            serving_rsu_onehot=inputs['serving_rsu_onehot'],  # 当前服务RSU
            global_state=inputs.get('global_state'),
        )

        # 2.5 Subtask采样（标准建模：策略自己选择待执行子任务）
        masked_subtask_logits = self._masked_subtask_logits(
            subtask_logits=subtask_logits,
            subtask_mask=inputs.get('subtask_mask', inputs['task_mask']),
            node_valid_mask=inputs.get('node_valid_mask'),
            fallback_index=inputs.get('subtask_index'),
        )
        subtask_dist = Categorical(logits=masked_subtask_logits)
        if deterministic:
            subtask_actions = torch.argmax(subtask_dist.probs, dim=-1)
        else:
            subtask_actions = subtask_dist.sample()
        log_prob_subtask = subtask_dist.log_prob(subtask_actions)

        # 2.6 使用 sampled subtask 重算 target/power/value（value 允许使用 sampled subtask 作为条件）
        _, target_logits, alpha, beta, values_env, cost_power_values, cost_trust_values = self.forward(
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
            subtask_index=subtask_actions,
            action_mask=inputs['action_mask'],
            subtask_mask=inputs.get('subtask_mask'),
            node_valid_mask=inputs.get('node_valid_mask'),
            task_mask=inputs['task_mask'],
            priority=inputs.get('priority'),
            rate_prev=inputs['rate_prev'],
            serving_rsu_onehot=inputs['serving_rsu_onehot'],
            global_state=inputs.get('global_state'),
        )

        # 3. Target采样（Categorical分布）
        # [通用化] 根据candidate_types动态赋logit_bias
        target_logits = self._apply_logit_bias(
            target_logits,
            policy_candidate_types,
            action_mask=policy_action_mask,
        )
        
        # 应用action_mask（在Categorical/softmax前执行），非法动作logit强制压到极小值
        action_mask_tensor = policy_action_mask > 0
        if action_mask_tensor.ndim == 2:
            no_valid = ~action_mask_tensor.any(dim=-1)
            if torch.any(no_valid):
                # 保底：防止全0 mask 导致分布退化；回退开放 Local(槽位0)
                action_mask_tensor = action_mask_tensor.clone()
                action_mask_tensor[no_valid, 0] = True
        masked_logits = target_logits.masked_fill(~action_mask_tensor, -1e9)

        target_dist = Categorical(logits=masked_logits)
        target_probs = target_dist.probs

        if deterministic:
            target_actions = torch.argmax(target_probs, dim=-1)
        else:
            target_actions = target_dist.sample()
        
        log_prob_target = target_dist.log_prob(target_actions)

        # target类型：1=Local, 2=RSU, 3=V2V
        _, disable_resource, _ = self._ablation_flags()
        if disable_resource:
            remote_mask = (target_actions != 0).to(dtype=log_prob_target.dtype)
        else:
            sel_type = self._gather_selected(policy_candidate_types, target_actions).long()
            remote_mask = (sel_type != 1).to(dtype=log_prob_target.dtype)

        # 4. Power采样（条件Beta分布，显式依赖target）
        power_dist = self._build_target_conditioned_power_dist(
            alpha=alpha,
            beta=beta,
            resource_raw=policy_resource_raw,
            candidate_types=policy_candidate_types,
            target_actions=target_actions,
        )
        
        if deterministic:
            power_actions = power_dist.mean
        else:
            power_actions = power_dist.sample()

        if bool(getattr(TC, "USE_FIXED_POWER", False)):
            # 固定功率：不学习连续功率分支，避免无意义梯度干扰
            power_actions = torch.full_like(power_actions, 0.5)
            log_prob_power = torch.zeros_like(log_prob_target)
        else:
            # Local动作无发送功率语义：固定到0.5并mask掉power分量损失/熵
            power_actions = torch.where(remote_mask > 0.0, power_actions, torch.full_like(power_actions, 0.5))
            power_actions = torch.clamp(power_actions, 1e-6, 1.0 - 1e-6)
            log_prob_power = power_dist.log_prob(power_actions) * remote_mask
        
        # 5. 联合log概率
        log_probs = log_prob_subtask + log_prob_target + log_prob_power
        
        return subtask_actions, target_actions, power_actions, log_probs, values_env, cost_power_values, cost_trust_values
    
    def evaluate_actions(self,
                        obs_list: List[Dict],
                        subtask_actions: torch.Tensor,
                        target_actions: torch.Tensor,
                        power_actions: torch.Tensor,
                        device='cpu',
                        return_aux: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的log概率和熵（用于PPO训练）
        
        Args:
            obs_list: 环境观测列表
            subtask_actions: [Batch], 子任务选择动作
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
        policy_resource_raw, policy_candidate_types, policy_action_mask = self._sanitize_policy_side_channels(
            inputs["resource_raw"],
            inputs["candidate_types"],
            inputs["action_mask"],
        )
        
        # 2. 前向传播
        subtask_logits, _, _, _, values, cost_power_values, cost_trust_values = self.forward(
            node_x=inputs['node_x'],
            adj=inputs['adj'],
            status=inputs['status'],
            location=inputs['location'],
            L_fwd=inputs['L_fwd'],
            L_bwd=inputs['L_bwd'],
            data_matrix=inputs['data_matrix'],
            delta=inputs['delta'],
            resource_ids=inputs['resource_ids'],
            resource_raw=inputs['resource_raw'],  # P38修复: 添加缺失的resource_raw参数
            subtask_index=inputs['subtask_index'],
            action_mask=inputs['action_mask'],
            subtask_mask=inputs.get('subtask_mask'),
            node_valid_mask=inputs.get('node_valid_mask'),
            task_mask=inputs['task_mask'],
            priority=inputs.get('priority'),  # [方案A] 传递priority（可选）
            rate_prev=inputs['rate_prev'],  # 上步链路速率
            serving_rsu_onehot=inputs['serving_rsu_onehot'],  # 当前服务RSU
            global_state=inputs.get('global_state'),
        )

        # 先评估 subtask 分布（与采样路径一致）
        masked_subtask_logits = self._masked_subtask_logits(
            subtask_logits=subtask_logits,
            subtask_mask=inputs.get('subtask_mask', inputs['task_mask']),
            node_valid_mask=inputs.get('node_valid_mask'),
            fallback_index=inputs.get('subtask_index'),
        )
        subtask_dist = Categorical(logits=masked_subtask_logits)
        log_prob_subtask = subtask_dist.log_prob(subtask_actions)
        entropy_subtask = subtask_dist.entropy()

        # 再以给定 subtask 评估 target/power/value（自回归一致性）
        _, target_logits, alpha, beta, values, cost_power_values, cost_trust_values = self.forward(
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
            subtask_index=subtask_actions,
            action_mask=inputs['action_mask'],
            subtask_mask=inputs.get('subtask_mask'),
            node_valid_mask=inputs.get('node_valid_mask'),
            task_mask=inputs['task_mask'],
            priority=inputs.get('priority'),
            rate_prev=inputs['rate_prev'],
            serving_rsu_onehot=inputs['serving_rsu_onehot'],
            global_state=inputs.get('global_state'),
        )

        # [通用化] 根据candidate_types动态赋logit_bias（与采样时一致）
        target_logits = self._apply_logit_bias(
            target_logits,
            policy_candidate_types,
            action_mask=policy_action_mask,
        )

        # 应用action_mask（在Categorical/softmax前执行），非法动作logit强制压到极小值
        action_mask_tensor = policy_action_mask > 0
        if action_mask_tensor.ndim == 2:
            no_valid = ~action_mask_tensor.any(dim=-1)
            if torch.any(no_valid):
                action_mask_tensor = action_mask_tensor.clone()
                action_mask_tensor[no_valid, 0] = True
        masked_logits = target_logits.masked_fill(~action_mask_tensor, -1e9)

        # 3. Target分布评估
        target_dist = Categorical(logits=masked_logits)
        log_prob_target = target_dist.log_prob(target_actions)
        entropy_target = target_dist.entropy()
        
        # 4. Power分布评估（条件于target，必须与采样路径一致）
        power_dist = self._build_target_conditioned_power_dist(
            alpha=alpha,
            beta=beta,
            resource_raw=policy_resource_raw,
            candidate_types=policy_candidate_types,
            target_actions=target_actions,
        )
        _, disable_resource, _ = self._ablation_flags()
        if disable_resource:
            remote_mask = (target_actions != 0).to(dtype=log_prob_target.dtype)
        else:
            sel_type = self._gather_selected(policy_candidate_types, target_actions).long()
            remote_mask = (sel_type != 1).to(dtype=log_prob_target.dtype)
        if bool(getattr(TC, "USE_FIXED_POWER", False)):
            log_prob_power = torch.zeros_like(log_prob_target)
            entropy_power = torch.zeros_like(entropy_target)
        else:
            power_actions = torch.clamp(power_actions, 1e-6, 1.0 - 1e-6)
            log_prob_power = power_dist.log_prob(power_actions) * remote_mask
            entropy_power = power_dist.entropy() * remote_mask
        
        # 5. 联合log概率和熵
        log_probs = log_prob_subtask + log_prob_target + log_prob_power
        entropy = entropy_subtask + entropy_target + entropy_power
        
        if not return_aux:
            return log_probs, entropy, values, cost_power_values, cost_trust_values

        aux = {
            'masked_target_logits': masked_logits,
            'candidate_types': policy_candidate_types,
            'action_mask': policy_action_mask,
            'cost_power_values': cost_power_values,
            'cost_trust_values': cost_trust_values,
        }
        return log_probs, entropy, values, cost_power_values, cost_trust_values, aux
