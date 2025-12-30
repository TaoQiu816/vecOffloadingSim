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


class LocationEncoder(nn.Module):
    """
    位置编码器
    
    将子任务的执行位置编码为嵌入向量
    位置ID：
    - 0: Unscheduled (未调度)
    - 1: Local (本车)
    - 2: RSU
    - 3 ~ 2+MAX_NEIGHBORS: Neighbor Vehicles
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: 嵌入维度
        """
        super().__init__()
        self.d_model = d_model
        
        # 位置嵌入层
        # num_embeddings = 3 + MAX_VEHICLE_ID (覆盖所有可能的邻居ID)
        num_locations = 3 + Cfg.MAX_VEHICLE_ID
        self.location_embedding = nn.Embedding(num_locations, d_model)
    
    def forward(self, location_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            location_ids: [Batch, MAX_NODES], dtype=long
        
        Returns:
            [Batch, MAX_NODES, d_model]
        """
        return self.location_embedding(location_ids)


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
        # 增加最后一维
        edge_data = edge_data.unsqueeze(-1)  # [B, N, N, 1]
        
        # 投影到num_heads维
        edge_bias = self.edge_proj(edge_data)  # [B, N, N, H]
        
        # 转置以匹配注意力维度 [B, H, N, N]
        edge_bias = edge_bias.permute(0, 3, 1, 2)
        
        return edge_bias
