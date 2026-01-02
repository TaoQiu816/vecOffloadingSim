"""
资源节点特征构建模块

功能：
- 构建统一的资源节点特征向量
- 支持Local、RSU、Neighbor三种资源类型
- 用于Cross-Attention的Key/Value输入
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from configs.config import SystemConfig as Cfg


class ResourceFeatureBuilder:
    """
    资源节点特征构建器
    
    输出统一的特征向量（14维）：
    [CPU_Norm, Queue_Norm, Dist_Norm, Rate_Norm, Rel_X, Rel_Y, Vel_X, Vel_Y, 
     Node_Type, Slack_Norm, Contact_Norm, Est_Exec_Time, Est_Comm_Time, Est_Wait_Time]
    """
    
    def __init__(self):
        """初始化归一化常数"""
        self._inv_max_cpu = 1.0 / Cfg.NORM_MAX_CPU
        self._inv_max_wait = 1.0 / Cfg.NORM_MAX_WAIT_TIME
        self._inv_map_size = 1.0 / Cfg.MAP_SIZE
        self._inv_max_rate_v2i = 1.0 / Cfg.NORM_MAX_RATE_V2I
        self._inv_max_rate_v2v = 1.0 / Cfg.NORM_MAX_RATE_V2V
        self._inv_max_vel = 1.0 / Cfg.MAX_VELOCITY
        
        # 时间预估归一化常数（假设最大预估时间为10秒）
        self._inv_max_time_est = 1.0 / 10.0
    
    def build_local_feature(self, 
                           cpu_freq: float,
                           queue_wait: float,
                           vel_x: float,
                           vel_y: float,
                           task_comp: float = 0.0,
                           task_data: float = 0.0) -> np.ndarray:
        """
        构建Local资源节点特征
        
        Args:
            cpu_freq: CPU频率 (Hz)
            queue_wait: 队列等待时间 (s)
            vel_x, vel_y: 车辆速度 (m/s)
            task_comp: 任务计算量 (cycles)
            task_data: 任务数据量 (bits)
        
        Returns:
            [CPU, Queue, 0, 0, 0, 0, Vel_x, Vel_y, 1, Slack, Contact, Est_Exec, Est_Comm, Est_Wait]
        """
        # 时间预估（物理公式）
        est_exec_time = task_comp / max(cpu_freq, 1e-6) if task_comp > 0 else 0.0
        est_comm_time = 0.0  # Local无传输
        est_wait_time = queue_wait
        
        return np.array([
            cpu_freq * self._inv_max_cpu,
            np.clip(queue_wait * self._inv_max_wait, 0, 1),
            0.0,  # 距离为0（本地）
            0.0,  # [修复] Local无传输，Rate=0而非1
            0.0,  # 相对位置X为0
            0.0,  # 相对位置Y为0
            vel_x * self._inv_max_vel,
            vel_y * self._inv_max_vel,
            1.0,  # Node_Type = 1 (Local)
            0.0,  # Slack_Norm (外部填充)
            1.0,  # Contact_Norm = 1 (永久连接)
            np.clip(est_exec_time * self._inv_max_time_est, 0, 1),
            np.clip(est_comm_time * self._inv_max_time_est, 0, 1),
            np.clip(est_wait_time * self._inv_max_time_est, 0, 1)
        ], dtype=np.float32)
    
    def build_rsu_feature(self,
                         cpu_freq: float,
                         queue_wait: float,
                         distance: float,
                         v2i_rate: float,
                         rel_x: float,
                         rel_y: float,
                         task_comp: float = 0.0,
                         task_data: float = 0.0) -> np.ndarray:
        """
        构建RSU资源节点特征
        
        Args:
            cpu_freq: RSU CPU频率 (Hz)
            queue_wait: RSU队列等待时间 (s)
            distance: 到RSU的距离 (m)
            v2i_rate: V2I通信速率 (bps)
            rel_x, rel_y: 相对位置 (归一化)
            task_comp: 任务计算量 (cycles)
            task_data: 任务数据量 (bits)
        
        Returns:
            [CPU, Queue, Dist, V2I_Rate, Rel_X, Rel_Y, 0, 0, 2, Slack, Contact, Est_Exec, Est_Comm, Est_Wait]
        """
        # 时间预估（物理公式）
        est_exec_time = task_comp / max(cpu_freq, 1e-6) if task_comp > 0 else 0.0
        est_comm_time = task_data / max(v2i_rate, 1e-6) if task_data > 0 else 0.0
        est_wait_time = queue_wait
        
        return np.array([
            cpu_freq * self._inv_max_cpu,
            np.clip(queue_wait * self._inv_max_wait, 0, 1),
            distance * self._inv_map_size,
            np.clip(v2i_rate * self._inv_max_rate_v2i, 0, 1),
            rel_x,  # 已归一化
            rel_y,  # 已归一化
            0.0,    # RSU速度为0
            0.0,    # RSU速度为0
            2.0,    # Node_Type = 2 (RSU)
            0.0,    # Slack_Norm (外部填充)
            0.0,    # Contact_Norm (外部填充)
            np.clip(est_exec_time * self._inv_max_time_est, 0, 1),
            np.clip(est_comm_time * self._inv_max_time_est, 0, 1),
            np.clip(est_wait_time * self._inv_max_time_est, 0, 1)
        ], dtype=np.float32)
    
    def build_neighbor_feature(self,
                              cpu_freq: float,
                              queue_wait: float,
                              distance: float,
                              v2v_rate: float,
                              rel_x: float,
                              rel_y: float,
                              vel_x: float,
                              vel_y: float,
                              task_comp: float = 0.0,
                              task_data: float = 0.0) -> np.ndarray:
        """
        构建Neighbor资源节点特征
        
        Args:
            cpu_freq: 邻居CPU频率 (Hz)
            queue_wait: 邻居队列等待时间 (s)
            distance: 到邻居的距离 (m)
            v2v_rate: V2V通信速率 (bps)
            rel_x, rel_y: 相对位置 (归一化)
            vel_x, vel_y: 邻居速度 (m/s)
            task_comp: 任务计算量 (cycles)
            task_data: 任务数据量 (bits)
        
        Returns:
            [CPU, Queue, Dist, V2V_Rate, Rel_X, Rel_Y, Vel_x, Vel_y, 3, Slack, Contact, Est_Exec, Est_Comm, Est_Wait]
        """
        # 时间预估（物理公式）
        est_exec_time = task_comp / max(cpu_freq, 1e-6) if task_comp > 0 else 0.0
        est_comm_time = task_data / max(v2v_rate, 1e-6) if task_data > 0 else 0.0
        est_wait_time = queue_wait
        
        return np.array([
            cpu_freq * self._inv_max_cpu,
            np.clip(queue_wait * self._inv_max_wait, 0, 1),
            distance * self._inv_map_size,
            np.clip(v2v_rate * self._inv_max_rate_v2v, 0, 1),
            rel_x,  # 已归一化
            rel_y,  # 已归一化
            vel_x * self._inv_max_vel,
            vel_y * self._inv_max_vel,
            3.0,    # Node_Type = 3 (Neighbor)
            0.0,    # Slack_Norm (外部填充)
            0.0,    # Contact_Norm (外部填充)
            np.clip(est_exec_time * self._inv_max_time_est, 0, 1),
            np.clip(est_comm_time * self._inv_max_time_est, 0, 1),
            np.clip(est_wait_time * self._inv_max_time_est, 0, 1)
        ], dtype=np.float32)
    
    def build_batch_resource_features(self,
                                     obs_list: List[Dict]) -> torch.Tensor:
        """
        批量构建资源节点特征（从环境观测中提取）
        
        Args:
            obs_list: 环境返回的观测列表
        
        Returns:
            Tensor [Batch, 2+MAX_NEIGHBORS, RESOURCE_RAW_DIM]
        """
        batch_size = len(obs_list)
        max_neighbors = Cfg.MAX_NEIGHBORS
        max_targets = Cfg.MAX_TARGETS
        
        resource_features = np.zeros((batch_size, max_targets, Cfg.RESOURCE_RAW_DIM), dtype=np.float32)
        
        for b, obs in enumerate(obs_list):
            if 'resource_raw' in obs:
                resource_features[b] = obs['resource_raw']
                continue
            # 从观测中提取特征
            self_info = obs['self_info']
            rsu_info = obs['rsu_info']
            neighbors = obs['neighbors']
            
            # Local节点特征 (Index 0)
            # self_info: [x, y, vx, vy, cpu, queue_wait, rsu_dist]
            resource_features[b, 0] = self.build_local_feature(
                cpu_freq=self_info[4],
                queue_wait=self_info[5],
                vel_x=self_info[2],
                vel_y=self_info[3]
            )
            
            # RSU节点特征 (Index 1)
            # rsu_info: [queue_wait_norm]
            # 需要从self_info中获取距离和位置
            rsu_dist = self_info[6]  # 已归一化
            # 相对位置需要计算（这里简化，使用距离的投影估计）
            rel_x = rsu_dist * 0.5  # 简化处理
            rel_y = 0.0
            
            # 估算V2I速率（基于距离）
            # 这里简化，实际应从环境获取
            v2i_rate = Cfg.NORM_MAX_RATE_V2I * (1.0 - rsu_dist) if rsu_dist < 1.0 else 1.0
            
            resource_features[b, 1] = self.build_rsu_feature(
                cpu_freq=Cfg.F_RSU,
                queue_wait=rsu_info[0] * Cfg.NORM_MAX_WAIT_TIME,  # 反归一化
                distance=rsu_dist * Cfg.MAP_SIZE,  # 反归一化
                v2i_rate=v2i_rate,
                rel_x=rel_x,
                rel_y=rel_y
            )
            
            # Neighbor节点特征 (Index 2 ~ 1+MAX_NEIGHBORS)
            for n_idx in range(max_neighbors):
                if n_idx < len(neighbors):
                    neighbor_feat = neighbors[n_idx]
                    # neighbor: [rel_x, rel_y, dist, vx, vy, queue_wait, cpu, v2v_rate]
                    if np.any(neighbor_feat):  # 非零表示有效邻居
                        resource_features[b, 2 + n_idx] = self.build_neighbor_feature(
                            cpu_freq=neighbor_feat[6] * Cfg.NORM_MAX_CPU,  # 反归一化
                            queue_wait=neighbor_feat[5] * Cfg.NORM_MAX_WAIT_TIME,
                            distance=neighbor_feat[2] * Cfg.MAP_SIZE,
                            v2v_rate=neighbor_feat[7] * Cfg.NORM_MAX_RATE_V2V,
                            rel_x=neighbor_feat[0],
                            rel_y=neighbor_feat[1],
                            vel_x=neighbor_feat[3] * Cfg.MAX_VELOCITY,
                            vel_y=neighbor_feat[4] * Cfg.MAX_VELOCITY
                        )
                    # else: 保持全0（Padding）
        
        return torch.from_numpy(resource_features)


class ResourceIDEncoder(nn.Module):
    """
    资源节点ID编码器（Type-Based）
    
    将资源类型（不含具体车辆ID）映射为嵌入向量
    ID映射：
    - 0: Padding（不使用）
    - 1: Local
    - 2: RSU
    - 3: Neighbor（所有邻居统一）
    """
    
    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        """
        Args:
            d_model: 嵌入维度
            dropout: Dropout率（训练时防止过拟合）
        """
        super().__init__()
        self.d_model = d_model
        
        # [修复] ID嵌入层：仅4个角色类型
        num_embeddings = 4  # {Padding, Local, RSU, Neighbor}
        self.id_embedding = nn.Embedding(num_embeddings, d_model, padding_idx=0)
        
        # Dropout层（防止网络只依赖类型判断节点好坏）
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, resource_ids: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            resource_ids: [Batch, N_res], dtype=long, Resource ID列表
                         原始值：0=Padding, 1=Local, 2=RSU, 3+=具体邻居ID
        
        Returns:
            [Batch, N_res, d_model], 类型嵌入向量
        """
        # [修复] 将所有邻居ID统一映射为类型3（Neighbor）
        # 这样网络只看到"是邻居"，不会过拟合具体ID
        type_ids = torch.clamp(resource_ids, max=3)
        
        # 嵌入
        id_emb = self.id_embedding(type_ids)  # [B, N_res, d_model]
        
        # Dropout（仅训练时）
        if training and self.training:
            id_emb = self.dropout(id_emb)
        
        return id_emb


class ResourceFeatureEncoder(nn.Module):
    """
    资源节点特征编码器
    
    融合物理特征和类型特征
    """
    
    def __init__(self, max_vehicle_id: int = None, d_model: int = 128, id_dropout: float = 0.2):
        """
        Args:
            max_vehicle_id: (已废弃，保留为兼容性参数)
            d_model: 嵌入维度
            id_dropout: ID嵌入的Dropout率
        """
        super().__init__()
        self.d_model = d_model
        
        # 物理特征投影
        self.feature_proj = nn.Linear(Cfg.RESOURCE_RAW_DIM, d_model)
        
        # [修复] ID编码器：不再接收max_vehicle_id
        self.id_encoder = ResourceIDEncoder(d_model, id_dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                resource_features: torch.Tensor,
                resource_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            resource_features: [Batch, N_res, RESOURCE_RAW_DIM], 原始物理特征
            resource_ids: [Batch, N_res], Global ID列表
        
        Returns:
            [Batch, N_res, d_model], 融合后的特征
        """
        # 1. 物理特征投影
        h_raw = self.feature_proj(resource_features)  # [B, N_res, d_model]
        
        # 2. ID嵌入（带Dropout）
        h_id = self.id_encoder(resource_ids, training=self.training)  # [B, N_res, d_model]
        
        # 3. 求和融合
        h_res = h_raw + h_id
        
        # 4. Layer Norm
        h_res = self.layer_norm(h_res)
        
        return h_res
