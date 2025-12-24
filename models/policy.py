import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

# =========================================================================
# 1. 导入依赖与配置
# =========================================================================
# [关键修改] 从 models.encoders 导入独立的编码器类
# 这确保了我们使用的是包含 Fallback 机制和 Edge Feature 支持的最新版本
from models.encoders import DAGTaskEncoder, HeteroTopologyEncoder

# [关键修改] 引入配置类，用于获取 EDGE_INPUT_DIM
from configs.train_config import TrainConfig as TC


# =========================================================================
# 2. 策略网络 (Actor & Critic)
# =========================================================================

class TransformerHybridActor(nn.Module):
    """
    [Actor 网络]
    融合 DAG 任务特征与网络拓扑特征，输出卸载决策(离散)和功率控制(连续)。

    架构流程:
    1. Task Encoder -> 提取任务节点特征和全局特征
    2. Topo Encoder -> 提取车辆和 RSU 的拓扑特征 (含边特征)
    3. Attention -> 计算任务与目标节点的匹配分数 (Logits)
    4. Power Head -> 基于任务和自身状态输出传输功率
    """

    def __init__(self, task_feat_dim, veh_feat_dim, rsu_feat_dim,
                 embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerHybridActor, self).__init__()

        self.embed_dim = embed_dim

        # --- 1. DAG Encoder ---
        # 使用导入的 DAGTaskEncoder
        self.task_encoder = DAGTaskEncoder(
            input_dim=task_feat_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            heads=num_heads
        )

        # --- 2. Topology Encoder ---
        # [关键修复] 显式传入 edge_dim=TC.EDGE_INPUT_DIM
        # 对应 models/encoders.py 中 __init__(..., edge_dim, ...) 的签名
        self.topo_encoder = HeteroTopologyEncoder(
            vehicle_dim=veh_feat_dim,
            rsu_dim=rsu_feat_dim,
            edge_dim=TC.EDGE_INPUT_DIM,  # <--- 必须传入此参数，否则会报错缺少参数
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            heads=num_heads
        )

        # --- 3. Power Head (功率控制) ---
        # 输入: Task Global (Embed) + Self State (Embed) -> 2 * Embed
        self.power_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出归一化功率 [0, 1]
        )

    def forward(self, dag_batch, topo_batch, candidates_mask=None):
        """
        前向传播
        Args:
            dag_batch: PyG Batch (任务图)
            topo_batch: PyG Hetero Batch (拓扑图)
            candidates_mask: 可选的动作掩码
        """
        batch_size = dag_batch.num_graphs

        # 1. Encoding
        # Task: [Total_Subtasks, Dim]
        task_node_emb, task_global_emb = self.task_encoder(dag_batch.x, dag_batch.edge_index, dag_batch.batch)

        # Topology: {'vehicle': [Total_Veh, Dim], 'rsu': ...}
        # [关键] 传入 edge_attr_dict 以利用边特征 (带宽/算力比)
        topo_node_dict = self.topo_encoder(
            topo_batch.x_dict,
            topo_batch.edge_index_dict,
            topo_batch.edge_attr_dict
        )
        veh_emb = topo_node_dict['vehicle']
        rsu_emb = topo_node_dict['rsu']

        # 2. Batch Loop (处理变长图 Masking)
        logits_list = []
        power_list = []

        dag_ptr = dag_batch.ptr
        veh_ptr = topo_batch['vehicle'].ptr
        rsu_ptr = topo_batch['rsu'].ptr

        for i in range(batch_size):
            # --- A. 任务侧数据切片 ---
            t_start, t_end = dag_ptr[i], dag_ptr[i + 1]
            curr_task_emb = task_node_emb[t_start:t_end]  # [Num_Sub, Dim]

            # 获取 Mask (逻辑依赖 Mask)
            if hasattr(dag_batch, 'mask'):
                curr_task_mask = dag_batch.mask[t_start:t_end]
            else:
                curr_task_mask = torch.ones(curr_task_emb.size(0), dtype=torch.bool, device=curr_task_emb.device)

            # 获取 Mask (物理可达 Target Mask)
            if hasattr(dag_batch, 'target_mask'):
                tm = dag_batch.target_mask[t_start:t_end]
                # 确保维度是 [Num_Sub, Num_Targets]
                curr_target_mask = tm if tm.dim() > 1 else tm.unsqueeze(0).expand(curr_task_emb.size(0), -1)
            else:
                curr_target_mask = None

            # --- B. 拓扑侧数据切片 ---
            v_start, v_end = veh_ptr[i], veh_ptr[i + 1]
            r_start, r_end = rsu_ptr[i], rsu_ptr[i + 1]
            curr_veh_emb = veh_emb[v_start:v_end]
            curr_rsu_emb = rsu_emb[r_start:r_end]

            # 拼接目标集合: [RSU(1) + Vehicles(N)]
            # 注意: 这里 vehicle 包含 Self 和 Neighbors
            targets_emb = torch.cat([curr_rsu_emb, curr_veh_emb], dim=0)

            # --- C. Attention 匹配 (Pointer Network) ---
            # Q: Task, K: Targets
            # Scores: [Num_Sub, Num_Targets]
            scores = torch.matmul(curr_task_emb, targets_emb.t()) / (self.embed_dim ** 0.5)

            # Mask 1: 屏蔽不可调度的任务 (Row Mask)
            scores = scores.masked_fill(~curr_task_mask.unsqueeze(1), -1e9)

            # Mask 2: 屏蔽物理不可达的目标 (Col Mask)
            if curr_target_mask is not None:
                # 安全切片，防止 target_mask 维度与 targets_emb 不一致
                min_tgt = min(scores.size(1), curr_target_mask.size(1))
                scores[:, :min_tgt] = scores[:, :min_tgt].masked_fill(~curr_target_mask[:, :min_tgt], -1e9)

            logits_list.append(scores.flatten())

            # --- D. 功率控制 ---
            curr_task_global = task_global_emb[i:i + 1]

            # [优化] 获取"自身"的 Embedding
            # 在 data_utils 构建的 local_topo 中，Vehicles 的第 0 个通常是 Self
            # 使用 curr_veh_emb[0:1] 比 mean 更精准地代表 Agent 自己的状态
            self_veh_feat = curr_veh_emb[0:1]

            p_feat = torch.cat([curr_task_global, self_veh_feat], dim=1)
            power_list.append(self.power_head(p_feat))

        # 3. Output Assembly
        # Pad logits to same length for batch processing
        padded_logits = nn.utils.rnn.pad_sequence(logits_list, batch_first=True, padding_value=-1e16)
        power_tensor = torch.cat(power_list, dim=0)

        return padded_logits, None, power_tensor


class TransformerHybridCritic(nn.Module):
    """
    [Critic 网络]
    评估状态价值 V(s)。
    """

    def __init__(self, task_feat_dim, veh_feat_dim, rsu_feat_dim,
                 embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerHybridCritic, self).__init__()

        # 1. DAG Encoder
        self.task_encoder = DAGTaskEncoder(
            input_dim=task_feat_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            heads=num_heads
        )

        # 2. Topology Encoder
        # [关键修复] 显式传入 edge_dim=TC.EDGE_INPUT_DIM
        self.topo_encoder = HeteroTopologyEncoder(
            vehicle_dim=veh_feat_dim,
            rsu_dim=rsu_feat_dim,
            edge_dim=TC.EDGE_INPUT_DIM,  # <--- 必须传入此参数
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            heads=num_heads
        )

        # 3. Value Head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, dag_batch, topo_batch):
        # 1. DAG Global Feature
        _, task_global_emb = self.task_encoder(dag_batch.x, dag_batch.edge_index, dag_batch.batch)

        # 2. Topology Global Feature
        # [关键] 传入 edge_attr_dict
        topo_node_dict = self.topo_encoder(
            topo_batch.x_dict,
            topo_batch.edge_index_dict,
            topo_batch.edge_attr_dict
        )

        veh_emb = topo_node_dict['vehicle']
        rsu_emb = topo_node_dict['rsu']

        # 聚合所有车辆信息 (Global Mean Pool)
        if hasattr(topo_batch['vehicle'], 'batch'):
            veh_global = global_mean_pool(veh_emb, topo_batch['vehicle'].batch)
        else:
            veh_global = veh_emb.mean(dim=0, keepdim=True)

        # 聚合 RSU 信息
        if hasattr(topo_batch['rsu'], 'batch') and topo_batch['rsu'].batch is not None:
            rsu_global = global_mean_pool(rsu_emb, topo_batch['rsu'].batch)
        else:
            # 如果 RSU batch 属性丢失 (常发生于单图 Batch)，取平均
            rsu_mean = torch.mean(rsu_emb, dim=0, keepdim=True)
            rsu_global = rsu_mean.expand(veh_global.size(0), -1)

        # 融合网络上下文
        net_context = veh_global + rsu_global

        # 3. Value Prediction
        cat_feat = torch.cat([task_global_emb, net_context], dim=1)
        value = self.value_head(cat_feat)

        return value