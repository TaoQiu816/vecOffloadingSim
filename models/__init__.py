"""
Models模块

新架构（基于边增强Transformer）：
- dag_features: 拓扑特征计算
- dag_embedding: DAG节点嵌入
- edge_enhanced_transformer: 边增强Transformer
- resource_features: 资源特征构建
- actor_critic: Actor-Critic网络
- offloading_policy: 完整策略网络

旧架构（已废弃，备份为.bak文件）：
- policy_old.py.bak: 旧的策略网络
- encoders_old.py.bak: 旧的编码器
"""

# 导出新模块
from .dag_features import (
    compute_forward_levels,
    compute_backward_levels,
    compute_shortest_path_matrix,
    normalize_distance_matrix
)

from .dag_embedding import (
    DAGNodeEmbedding,
    LocationEncoder,
    StatusEncoder,
    BidirectionalTopologicalPositionEncoder,
    EdgeFeatureEncoder,
    SpatialDistanceEncoder
)

from .edge_enhanced_transformer import (
    EdgeEnhancedAttention,
    EdgeEnhancedTransformerLayer,
    EdgeEnhancedTransformer
)

from .resource_features import (
    ResourceFeatureBuilder,
    ResourceIDEncoder,
    ResourceFeatureEncoder
)

from .actor_critic import (
    CrossAttentionWithPhysicsBias,
    ActorHead,
    CriticHead,
    SimplifiedCriticHead,
    ActorCriticNetwork
)

from .offloading_policy import (
    OffloadingPolicyNetwork
)

__all__ = [
    # DAG特征
    'compute_forward_levels',
    'compute_backward_levels',
    'compute_shortest_path_matrix',
    'normalize_distance_matrix',
    
    # DAG嵌入
    'DAGNodeEmbedding',
    'LocationEncoder',
    'StatusEncoder',
    'BidirectionalTopologicalPositionEncoder',
    'EdgeFeatureEncoder',
    'SpatialDistanceEncoder',
    
    # Transformer
    'EdgeEnhancedAttention',
    'EdgeEnhancedTransformerLayer',
    'EdgeEnhancedTransformer',
    
    # 资源特征
    'ResourceFeatureBuilder',
    'ResourceIDEncoder',
    'ResourceFeatureEncoder',
    
    # Actor-Critic
    'CrossAttentionWithPhysicsBias',
    'ActorHead',
    'CriticHead',
    'SimplifiedCriticHead',
    'ActorCriticNetwork',
    
    # 完整策略
    'OffloadingPolicyNetwork',
]

