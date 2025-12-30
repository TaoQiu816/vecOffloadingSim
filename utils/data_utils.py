import torch
import numpy as np
from torch_geometric.data import Data, HeteroData, Batch
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


def process_env_obs(obs_list, device):
    """
    [CTDE 数据处理 - 最终修正版]
    将环境观测转换为 PyG 图数据。

    关键修正:
    1. 构建 "Self-First, Fixed-Size" 的局部拓扑图，确保 Actor 动作空间固定。
    2. 对齐 TrainConfig 中的 7 维特征输入。
    3. 正确生成 Policy Mask 以支持 decode_actions。

    维度对应:
    - DAG Node: [N, 7]
    - Vehicle:  [N, 7] (补齐1维)
    - RSU:      [1, 7] (补齐到与Vehicle一致)
    """
    num_vehicles = len(obs_list)
    edge_dim = TC.EDGE_INPUT_DIM

    # 容器
    dag_data_list = []
    local_topo_list = []

    # =========================================================================
    # 0. 准备全局特征 (用于 Critic 和 Actor 的 Others 部分)
    # =========================================================================
    # obs['self_info'] 已经是 7 维: [vx, vy, wait, cpu_freq, v2i_rate, pos_x, pos_y]
    # 与 TrainConfig 中的 VEH_INPUT_DIM = 7 一致
    all_veh_feats_np = np.stack([obs['self_info'] for obs in obs_list])

    # 验证维度一致性
    if all_veh_feats_np.shape[1] != 7:
        raise ValueError(f"Expected self_info to be 7-dimensional, got {all_veh_feats_np.shape[1]}")

    all_veh_tensor = torch.FloatTensor(all_veh_feats_np)

    # RSU Feature (与TrainConfig.RSU_INPUT_DIM=1一致)
    # [load] - 仅保留归一化负载
    rsu_load = obs_list[0]['rsu_info'][0]
    rsu_feat_vec = torch.tensor([[rsu_load]], dtype=torch.float32)

    # =========================================================================
    # Part 1: 遍历每个 Agent，构建 DAG 和 Actor 局部数据
    # =========================================================================
    for v_id, obs in enumerate(obs_list):
        # --- A. DAG 数据 ---
        # 这里的 node_x 已经在 Env 中修正为 7 维
        x = torch.FloatTensor(obs['node_x'])
        edge_index = torch.LongTensor(obs['adj']).nonzero().t().contiguous()
        dag = Data(x=x, edge_index=edge_index)

        # --- B. 构建 Masks (逻辑核心) ---
        # 1. Task Mask: 哪些子任务是 Ready 的
        dag.mask = torch.BoolTensor(obs['task_mask'])

        # 2. Target Mask: 使用环境的标准顺序
        # Canonical Order: [Local, RSU, V2V_0, V2V_1, ...]
        policy_target_mask = torch.as_tensor(obs['target_mask'], dtype=torch.bool)
        if policy_target_mask.numel() != Cfg.MAX_TARGETS:
            raise ValueError(f"target_mask length mismatch: {policy_target_mask.numel()} != {Cfg.MAX_TARGETS}")

        # 解析 Env Neighbors 列表，构建邻居字典（用于拓扑边）
        nbr_dict = {}
        for i, nbr in enumerate(obs['neighbors']):
            nid = int(nbr[0])
            nbr_dict[nid] = bool(obs['target_mask'][i + 2])

        # 构建一个不包含自己的 ID 列表
        other_ids = [i for i in range(num_vehicles) if i != v_id]

        # 扩展 Mask 维度: [Num_Subtasks, Num_Targets]
        num_sub = x.size(0)
        dag.target_mask = policy_target_mask.unsqueeze(0).expand(num_sub, -1)

        dag_data_list.append(dag)

        # --- C. Local Topology (固定结构: Self + Others) ---
        local_topo = HeteroData()

        # 特征矩阵构建:
        # Row 0: Self
        # Row 1..N-1: Others (按 ID 排序)
        self_row = all_veh_tensor[v_id:v_id + 1]
        others_rows = torch.cat([all_veh_tensor[:v_id], all_veh_tensor[v_id + 1:]], dim=0)

        local_veh_x = torch.cat([self_row, others_rows], dim=0)

        local_topo['vehicle'].x = local_veh_x
        local_topo['rsu'].x = rsu_feat_vec

        # 构建边 (Star Topology: Self <-> Neighbors)
        # 只有在通信范围内的邻居才有边连接，虽然图中包含了所有节点
        # src: Self (0)
        # dst: Indices in 'local_veh_x' that correspond to neighbors

        valid_dsts = []
        for i, other_id in enumerate(other_ids):
            # i+1 是 other_id 在 local_veh_x 中的索引 (因为 0 是 self)
            if other_id in nbr_dict:  # 只要在邻居列表里，就有边 (不管是否拥堵)
                valid_dsts.append(i + 1)

        if len(valid_dsts) > 0:
            src = torch.zeros(len(valid_dsts), dtype=torch.long)
            dst = torch.tensor(valid_dsts, dtype=torch.long)

            # 双向边
            u = torch.cat([src, dst])
            v = torch.cat([dst, src])
            local_topo['vehicle', 'v2v', 'vehicle'].edge_index = torch.stack([u, v])
            # 填充V2V边属性（防止维度错误）
            num_edges = u.size(0)
            local_topo['vehicle', 'v2v', 'vehicle'].edge_attr = torch.ones((num_edges, edge_dim), dtype=torch.float32)
        else:
            local_topo['vehicle', 'v2v', 'vehicle'].edge_index = torch.empty((2, 0), dtype=torch.long)
            local_topo['vehicle', 'v2v', 'vehicle'].edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)

        # V2I (Self <-> RSU)
        # V2I Edges (Self <-> RSU)
        # 正向
        local_topo['vehicle', 'v2i', 'rsu'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        local_topo['vehicle', 'v2i', 'rsu'].edge_attr = torch.ones((1, edge_dim), dtype=torch.float32)

        # 反向 (I2V)
        local_topo['rsu', 'i2v', 'vehicle'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        local_topo['rsu', 'i2v', 'vehicle'].edge_attr = torch.ones((1, edge_dim), dtype=torch.float32)

        local_topo_list.append(local_topo)

    # =========================================================================
    # Part 2: Critic 全局数据 (Global Graph)
    # =========================================================================
    global_topo = HeteroData()
    global_topo['vehicle'].x = all_veh_tensor
    global_topo['rsu'].x = rsu_feat_vec

    # 全局 V2V 边 (基于真实物理连接)
    g_v2v_src, g_v2v_dst = [], []
    g_v2i_src, g_v2i_dst = [], []

    for v_id, obs in enumerate(obs_list):
        # V2I
        if obs['target_mask'][1]:
            g_v2i_src.append(v_id)
            g_v2i_dst.append(0)

        # V2V
        for nbr_info in obs['neighbors']:
            nbr_id = int(nbr_info[0])
            g_v2v_src.append(v_id)
            g_v2v_dst.append(nbr_id)

    # 转 Tensor
    # V2V 填充
    if g_v2v_src:
        edge_index = torch.tensor([g_v2v_src, g_v2v_dst], dtype=torch.long)
        global_topo['vehicle', 'v2v', 'vehicle'].edge_index = edge_index
        # 填充全局拓扑V2V边属性
        num_e = len(g_v2v_src)
        global_topo['vehicle', 'v2v', 'vehicle'].edge_attr = torch.ones((num_e, edge_dim), dtype=torch.float32)
    else:
        global_topo['vehicle', 'v2v', 'vehicle'].edge_index = torch.empty((2, 0), dtype=torch.long)
        global_topo['vehicle', 'v2v', 'vehicle'].edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)

    # V2I 填充
    if g_v2i_src:
        # V2I
        edge_index = torch.tensor([g_v2i_src, g_v2i_dst], dtype=torch.long)
        global_topo['vehicle', 'v2i', 'rsu'].edge_index = edge_index
        global_topo['vehicle', 'v2i', 'rsu'].edge_attr = torch.ones((len(g_v2i_src), edge_dim), dtype=torch.float32)

        # I2V (反向)
        edge_index_rev = torch.tensor([g_v2i_dst, g_v2i_src], dtype=torch.long)
        global_topo['rsu', 'i2v', 'vehicle'].edge_index = edge_index_rev
        global_topo['rsu', 'i2v', 'vehicle'].edge_attr = torch.ones((len(g_v2i_dst), edge_dim), dtype=torch.float32)
    else:
        empty_idx = torch.empty((2, 0), dtype=torch.long)
        empty_attr = torch.empty((0, edge_dim), dtype=torch.float32)

        global_topo['vehicle', 'v2i', 'rsu'].edge_index = empty_idx
        global_topo['vehicle', 'v2i', 'rsu'].edge_attr = empty_attr

        global_topo['rsu', 'i2v', 'vehicle'].edge_index = empty_idx
        global_topo['rsu', 'i2v', 'vehicle'].edge_attr = empty_attr

    # =========================================================================
    # Part 3: 批处理与设备转移
    # =========================================================================
    dag_batch = Batch.from_data_list(dag_data_list).to(device)
    local_topo_batch = Batch.from_data_list(local_topo_list).to(device)
    global_topo = global_topo.to(device)

    # 返回3个值，去除多余的 None
    return dag_batch, local_topo_batch, global_topo
