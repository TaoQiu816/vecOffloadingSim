"""
DAG特征嵌入模块

功能：
- 连续特征投影
- 离散特征嵌入（Status, Location, Topological Position）
- 双向拓扑位置编码（BTPE）
"""

import torch
import torch.nn as nn
from configs.config import SystemConfig as Cfg
from configs.constants import MASK_VALUE


class LocationEncoder(nn.Module):
    """
    位置编码器
    
    将子任务的执行位置编码为嵌入向量
    位置ID（角色级别）：
    - 0: Unscheduled (未调度)
    - 1: Local (本车)
    - 2: RSU
    - 3: Neighbor (任意邻居车辆，不区分具体ID)
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 嵌入维度
        """
        super().__init__()
        self.d_model = d_model
        
        # [修复] 位置嵌入层：仅4个角色，移除Vehicle ID依赖
        num_locations = 4  # {Unscheduled, Local, RSU, Neighbor}
        self.location_embedding = nn.Embedding(num_locations, d_model)
    
    def forward(self, location_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            location_ids: [Batch, MAX_NODES], dtype=long
                         原始值：0=Unscheduled, 1=Local, 2=RSU, 3+=具体邻居ID
        
        Returns:
            [Batch, MAX_NODES, d_model]
        """
        # [修复] 将所有邻居ID统一映射为角色3（Neighbor）
        # 这样网络只看到"是邻居"，不会过拟合具体ID
        role_ids = torch.clamp(location_ids, max=3)
        
        return self.location_embedding(role_ids)


class StatusEncoder(nn.Module):
    """
    状态编码器
    
    将子任务的执行状态编码为嵌入向量
    状态：
    - 0: PENDING (等待前驱)
    - 1: READY (可调度)
    - 2: RUNNING (执行中)
    - 3: COMPLETED (已完成)
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 嵌入维度
        """
        super().__init__()
        self.d_model = d_model
        
        # 状态嵌入层
        self.status_embedding = nn.Embedding(4, d_model)  # 4种状态
    
    def forward(self, status_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            status_ids: [Batch, MAX_NODES], dtype=long
        
        Returns:
            [Batch, MAX_NODES, d_model]
        """
        return self.status_embedding(status_ids)


class BidirectionalTopologicalPositionEncoder(nn.Module):
    """
    双向拓扑位置编码器（BTPE）
    
    使用L_fwd和L_bwd作为位置编码
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 嵌入维度
        """
        super().__init__()
        self.d_model = d_model
        
        # 前向层级嵌入
        # MAX_NODES是最大可能的层级
        self.L_fwd_embedding = nn.Embedding(Cfg.MAX_NODES, d_model)
        
        # 后向层级嵌入
        self.L_bwd_embedding = nn.Embedding(Cfg.MAX_NODES, d_model)
    
    def forward(self, L_fwd: torch.Tensor, L_bwd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L_fwd: [Batch, MAX_NODES], dtype=long, 前向层级
            L_bwd: [Batch, MAX_NODES], dtype=long, 后向层级
        
        Returns:
            [Batch, MAX_NODES, d_model], 双向位置编码（相加）
        """
        fwd_emb = self.L_fwd_embedding(L_fwd)  # [B, N, d]
        bwd_emb = self.L_bwd_embedding(L_bwd)  # [B, N, d]
        
        # 相加融合
        return fwd_emb + bwd_emb


class DAGNodeEmbedding(nn.Module):
    """
    DAG节点特征嵌入模块
    
    输入：
    - 连续特征: [comp, input_data, total_comp, CT, in_degree, out_degree, priority]
    - 离散特征: status, location, L_fwd, L_bwd
    
    输出：
    - 节点嵌入向量 [Batch, MAX_NODES, d_model]
    """
    
    def __init__(self, d_model: int = 128, continuous_dim: int = 7):
        """
        Args:
            d_model: 嵌入维度
            continuous_dim: 连续特征维度（node_x的最后一维）
        """
        super().__init__()
        self.d_model = d_model
        self.continuous_dim = continuous_dim
        
        # 连续特征投影
        self.continuous_proj = nn.Linear(continuous_dim, d_model)
        
        # 离散特征编码器
        self.status_encoder = StatusEncoder(d_model)
        self.location_encoder = LocationEncoder(d_model)
        self.topo_position_encoder = BidirectionalTopologicalPositionEncoder(d_model)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                node_x: torch.Tensor,
                status: torch.Tensor,
                location: torch.Tensor,
                L_fwd: torch.Tensor,
                L_bwd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_x: [Batch, MAX_NODES, continuous_dim], 连续特征
            status: [Batch, MAX_NODES], dtype=long, 状态ID
            location: [Batch, MAX_NODES], dtype=long, 位置ID
            L_fwd: [Batch, MAX_NODES], dtype=long, 前向层级
            L_bwd: [Batch, MAX_NODES], dtype=long, 后向层级
        
        Returns:
            [Batch, MAX_NODES, d_model], 节点嵌入
        """
        # 1. 连续特征投影
        continuous_emb = self.continuous_proj(node_x)  # [B, N, d_model]
        
        # 2. 状态嵌入
        status_emb = self.status_encoder(status)  # [B, N, d_model]
        
        # 3. 位置嵌入
        location_emb = self.location_encoder(location)  # [B, N, d_model]
        
        # 4. 拓扑位置嵌入
        topo_emb = self.topo_position_encoder(L_fwd, L_bwd)  # [B, N, d_model]
        
        # 5. 求和融合（Transformer标准做法）
        node_emb = continuous_emb + status_emb + location_emb + topo_emb
        
        # 6. Layer Normalization
        node_emb = self.layer_norm(node_emb)
        
        return node_emb


class SpatialDistanceEncoder(nn.Module):
    """
    空间距离编码器
    
    将最短路径距离矩阵编码为注意力偏置
    """
    
    def __init__(self, num_heads: int = 8):
        """
        Args:
            num_heads: 注意力头数
        """
        super().__init__()
        self.num_heads = num_heads
        
        # 距离嵌入层
        # MAX_NODES: 最大距离，MAX_NODES+1: 不连通
        num_distances = Cfg.MAX_NODES + 1
        self.distance_embedding = nn.Embedding(num_distances, num_heads)
    
    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distance_matrix: [Batch, MAX_NODES, MAX_NODES], dtype=long
                            最短路径距离矩阵
        
        Returns:
            [Batch, num_heads, MAX_NODES, MAX_NODES], 空间偏置
        """
        # 嵌入距离
        spatial_bias = self.distance_embedding(distance_matrix)  # [B, N, N, H]
        
        # 转置以匹配注意力维度 [B, H, N, N]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)
        
        return spatial_bias


class EdgeFeatureEncoder(nn.Module):
    """
    边特征编码器
    
    将边上的数据依赖量编码为注意力偏置
    """
    
    def __init__(self, num_heads: int = 8):
        """
        Args:
            num_heads: 注意力头数
        """
        super().__init__()
        self.num_heads = num_heads
        
        # 边特征投影（单个标量 -> num_heads维）
        self.edge_proj = nn.Linear(1, num_heads)
    
    def forward(self, edge_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_data: [Batch, MAX_NODES, MAX_NODES], 边数据量（归一化）
        
        Returns:
            [Batch, num_heads, MAX_NODES, MAX_NODES], 边特征偏置
        """
        # [修复] 识别无连接边（零值）
        no_edge_mask = edge_data.abs() < 1e-6  # [B, N, N]
        
        # 增加最后一维
        edge_data = edge_data.unsqueeze(-1)  # [B, N, N, 1]
        
        # 投影到num_heads维
        edge_bias = self.edge_proj(edge_data)  # [B, N, N, H]
        
        # 转置以匹配注意力维度 [B, H, N, N]
        edge_bias = edge_bias.permute(0, 3, 1, 2)
        
        # [修复] 将无连接边的bias设为极小值，物理上切断注意力
        # 广播mask: [B, N, N] -> [B, 1, N, N]
        # [P33修复] 使用统一的MASK_VALUE常量
        no_edge_mask = no_edge_mask.unsqueeze(1)
        edge_bias = edge_bias.masked_fill(no_edge_mask, MASK_VALUE)

        return edge_bias


class RankBiasEncoder(nn.Module):
    """
    Rank偏置编码器（GA-DRL启发的邻居重要性先验 - 方案A）

    【设计说明】
    作用：在attention logits中新增rank_prior_bias，作为可控的注意力先验偏置，
         提升DAG表征质量与跨拓扑泛化能力。
    不做：不参与调度决策、不改变action/reward/队列逻辑。

    【数学原理】
    1. 得到节点先验权重: w_j = softmax(priority_j / tau)
    2. 转换为可加bias: rank_bias_{i,j} = kappa * log(w_j + eps)
    3. 物理意义: attention ∝ exp(logits) * (w_j)^kappa
       - kappa控制先验强度
       - 方向一致性: priority大 => w大 => bias大 => attention高

    【覆盖模式】
    - M1 (cover='all'): rank_bias对所有(i,j)生效，仅依赖key节点j（最稳，推荐）
    - M2 (cover='adj'): rank_bias只对adj[i,j]==1的位置生效（更贴DAG结构）

    Reference: GA-DRL Rank-Guided Neighbor Sampling (adapted for attention bias)
    """

    def __init__(self, num_heads: int = 8):
        """
        Args:
            num_heads: 注意力头数，与EdgeEnhancedAttention保持一致
        """
        super().__init__()
        self.num_heads = num_heads

        # Rank偏置投影：将标量偏置扩展到num_heads维
        # 使用无偏置的线性层，确保零输入产生零输出（M2模式需要）
        self.rank_proj = nn.Linear(1, num_heads, bias=False)

        # 初始化权重为正值（都是1），保持方向一致性
        # 这样所有head的行为一致，方向与bias_1d相同
        nn.init.ones_(self.rank_proj.weight)

    def forward(self,
                priority: torch.Tensor,
                adj: torch.Tensor = None,
                tau: float = 1.0,
                kappa: float = 0.5,
                cover_mode: str = 'all',
                task_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将Priority/Rank值转换为注意力偏置

        【重要】方向约定：
        - 输入priority是"越大越重要"（与compute_task_priority一致）
        - 如果输入是rank（越小越重要），调用前需要取负或转换

        Args:
            priority: [Batch, N], 优先级分数（越大越重要，已归一化到[0,1]）
            adj: [Batch, N, N], 邻接矩阵（cover_mode='adj'时必需）
            tau: 温度参数，控制softmax尖锐程度
            kappa: 强度系数，rank_bias = kappa * log(w + eps)
            cover_mode: 'all'=M1模式(所有位置), 'adj'=M2模式(仅邻接边)
            task_mask: [Batch, N], 有效节点掩码（True=有效）

        Returns:
            rank_bias: [Batch, num_heads, N, N], 注意力偏置
        """
        batch_size, N = priority.shape
        device = priority.device
        eps = 1e-8

        # 1. 计算节点重要性分数（priority越大 => 分数越大）
        importance_score = priority / max(tau, eps)  # [B, N]

        # 2. 应用task_mask（padding节点不参与softmax）
        if task_mask is not None:
            # padding节点设为极小值，softmax后概率趋近0
            importance_score = importance_score.masked_fill(~task_mask, MASK_VALUE)

        # 3. 对每行应用softmax得到节点先验权重 w_j
        # w_j = softmax(priority_j / tau)
        w = torch.softmax(importance_score, dim=-1)  # [B, N]

        # 4. 转换为可加到logits的bias
        # rank_bias = kappa * log(w + eps)
        # 物理意义: attention ∝ exp(logits + rank_bias) = exp(logits) * (w)^kappa
        log_w = torch.log(w + eps)  # [B, N]
        bias_1d = kappa * log_w  # [B, N]

        # 5. 扩展到[B, N, N]（对所有query i，bias仅依赖key j）
        # bias_1d: [B, N] -> [B, 1, N] 广播到 [B, N, N]
        rank_bias_2d = bias_1d.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]

        # 6. 根据cover_mode决定覆盖范围
        if cover_mode == 'adj' and adj is not None:
            # M2模式：只对adj[i,j]==1的位置加bias，其他位置精确为0
            # 使用torch.where确保非邻接位置为0（乘法可能产生微小误差）
            adj_mask = (adj > 0.5)  # [B, N, N], 二值化
            rank_bias_2d = torch.where(adj_mask, rank_bias_2d, torch.zeros_like(rank_bias_2d))
        # M1模式(cover_mode='all')：不改变，bias对所有位置生效

        # 7. 投影到num_heads维
        # [B, N, N] -> [B, N, N, 1] -> [B, N, N, H] -> [B, H, N, N]
        rank_bias_2d = rank_bias_2d.unsqueeze(-1)  # [B, N, N, 1]
        rank_bias = self.rank_proj(rank_bias_2d)  # [B, N, N, H]
        rank_bias = rank_bias.permute(0, 3, 1, 2)  # [B, H, N, N]

        return rank_bias
