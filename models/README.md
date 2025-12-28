# Models 模块说明

## 新架构（边增强Transformer）

本目录包含基于边增强Transformer的DAG任务卸载策略网络。

### 核心模块

#### 1. `dag_features.py`
拓扑特征计算工具：
- `compute_forward_levels()` - 计算前向层级（入口到节点）
- `compute_backward_levels()` - 计算后向层级（节点到出口，关键路径指标）
- `compute_shortest_path_matrix()` - 计算最短路径距离矩阵
- `normalize_distance_matrix()` - 归一化距离矩阵

#### 2. `dag_embedding.py`
DAG节点嵌入模块：
- `DAGNodeEmbedding` - 完整节点嵌入（连续+离散特征）
- `LocationEncoder` - 位置编码（Unscheduled/Local/RSU/Neighbor）
- `StatusEncoder` - 状态编码（PENDING/READY/RUNNING/COMPLETED）
- `BidirectionalTopologicalPositionEncoder` - 双向拓扑位置编码（BTPE）
- `EdgeFeatureEncoder` - 边数据依赖编码（注意力偏置）
- `SpatialDistanceEncoder` - 空间距离编码（注意力偏置）

#### 3. `edge_enhanced_transformer.py`
边增强Transformer：
- `EdgeEnhancedAttention` - 边增强注意力机制
- `EdgeEnhancedTransformerLayer` - Transformer层
- `EdgeEnhancedTransformer` - 多层Transformer

特点：
- 支持边特征偏置（数据依赖）
- 支持空间偏置（拓扑距离）
- Padding mask支持

#### 4. `resource_features.py`
资源节点特征构建：
- `ResourceFeatureBuilder` - 9维统一特征构建器
- `ResourceFeatureEncoder` - 特征编码器

9维特征：`[CPU, Queue, Dist, Rate, Rel_X, Rel_Y, Vel_X, Vel_Y, Node_Type]`

#### 5. `actor_critic.py`
Actor-Critic网络：
- `CrossAttention` - 跨注意力（DAG特征×资源特征）
- `ActorHead` - 双头输出（Target离散 + Power连续）
- `CriticHead` - 价值估计（全局池化）
- `ActorCriticNetwork` - 完整网络

#### 6. `offloading_policy.py`
完整策略网络（待完善）

### 使用示例

```python
from models import (
    DAGNodeEmbedding,
    EdgeEnhancedTransformer,
    ResourceFeatureEncoder,
    ActorCriticNetwork
)

# 1. 创建DAG嵌入
dag_embedding = DAGNodeEmbedding(d_model=128, continuous_dim=7)

# 2. 创建Transformer
transformer = EdgeEnhancedTransformer(
    num_layers=4,
    d_model=128,
    num_heads=8
)

# 3. 创建资源编码器
resource_encoder = ResourceFeatureEncoder(d_model=128)

# 4. 创建Actor-Critic
actor_critic = ActorCriticNetwork(d_model=128, num_heads=8)

# 5. 前向传播
# dag_features = transformer(node_emb, edge_bias, spatial_bias, mask)
# target_logits, power_ratio, value = actor_critic(
#     dag_features, resource_features, subtask_index, target_mask, task_mask
# )
```

### 测试

运行测试验证所有模块：

```bash
# Phase 1 测试（环境+观测+动作）
python tests/test_phase1_comprehensive.py

# Phase 2 测试（神经网络模块）
python tests/test_phase2_modules.py
```

### 旧架构（已废弃）

以下文件已备份为 `.bak` 格式：
- `policy_old.py.bak` - 基于PyG的旧策略网络
- `encoders_old.py.bak` - 基于GATv2的旧编码器

如需参考旧代码，可查看备份文件。

## 架构对比

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 基础框架 | PyTorch Geometric | Pure PyTorch |
| 图编码 | GATv2Conv | Edge-Enhanced Transformer |
| 边特征 | 不支持 | 数据依赖+拓扑距离 |
| 位置编码 | 无 | BTPE（双向拓扑） |
| 资源特征 | 分散 | 9维统一 |
| 任务选择 | Agent决策 | 环境自动（优先级） |
| Batch支持 | 需要PyG Batch | 原生Tensor |

## 下一步

1. 完善 `offloading_policy.py` 的 `get_action()` 方法
2. 创建新的 `mappo_agent.py` 适配新架构
3. 更新训练脚本使用新网络

