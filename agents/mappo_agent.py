import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch_geometric.data import Batch
from models.policy import TransformerHybridActor, TransformerHybridCritic
# 引入配置类
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


class MAPPOAgent:
    """
    [MAPPO Agent - Final Version]
    多智能体 PPO 算法 (Centralized Training Decentralized Execution) 的实现。

    核心特性:
    1. 混合动作空间: 同时输出离散动作(卸载目标+子任务)和连续动作(传输功率)。
    2. 参数共享: 所有车辆智能体共享同一个 Actor-Critic 网络参数。
    3. 集中式训练: Critic 网络利用 Transformer/GNN 提取的全局特征进行价值评估。

    适配确认:
    - 适配 Buffer 的 Flatten 存储结构。
    - 适配 Policy 的 logits 输出 (数值稳定)。
    - 适配 TrainConfig 的超参数引用。
    """

    def __init__(self, task_dim, veh_dim, rsu_dim, device='cpu'):
        """
        初始化 MAPPO Agent。

        Args:
            task_dim (int): 任务特征维度 (来自 Encoder, usually 7)。
            veh_dim (int): 车辆特征维度 (usually 7)。
            rsu_dim (int): RSU 特征维度 (usually 7)。
            device (str): 运行设备 ('cpu' or 'cuda').
        """
        self.device = device

        # --- 1. 超参数加载 (从 TrainConfig) ---
        lr_actor = TC.LR_ACTOR
        lr_critic = TC.LR_CRITIC
        self.gamma = TC.GAMMA
        self.gae_lambda = TC.GAE_LAMBDA
        self.eps_clip = TC.CLIP_PARAM
        self.K_epochs = TC.PPO_EPOCH
        self.entropy_coef = TC.ENTROPY_COEF
        self.max_grad_norm = TC.MAX_GRAD_NORM

        # --- 2. 网络初始化 ---
        # Actor: 输出动作策略 (logits 和 raw_power)
        self.actor = TransformerHybridActor(
            task_feat_dim=task_dim, veh_feat_dim=veh_dim, rsu_feat_dim=rsu_dim,
            embed_dim=TC.EMBED_DIM, num_layers=TC.NUM_LAYERS, num_heads=TC.NUM_HEADS
        ).to(device)

        # Critic: 输出状态价值 (Value)
        self.critic = TransformerHybridCritic(
            task_feat_dim=task_dim, veh_feat_dim=veh_dim, rsu_feat_dim=rsu_dim,
            embed_dim=TC.EMBED_DIM, num_heads=TC.NUM_HEADS, num_layers=TC.NUM_LAYERS
        ).to(device)

        # --- 3. 优化器配置 ---
        # 分离 Actor 和 Critic 的学习率
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-5)

        # --- 4. 连续动作方差 (Learnable Log Std) ---
        # 功率控制是一个连续动作，我们学习一个与状态无关的对数标准差 (Log Std)
        # 初始化为 -0.5 (对应 std ≈ 0.6)，允许网络在训练中调整探索噪声
        self.power_log_std = nn.Parameter(torch.zeros(1).to(device) - 0.5)
        # 将其归类到 Actor 参数组，跟随 Actor 更新
        self.optimizer.add_param_group({'params': [self.power_log_std], 'lr': lr_actor})

        self.mse_loss = nn.MSELoss()

    def decay_lr(self, decay_rate=0.9):
        """
        [辅助] 学习率衰减
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay_rate

    def select_action(self, dag_list, topo_list, candidates_mask=None):
        """
        [推理/交互阶段] 选择动作。

        Args:
            dag_list: DAG 任务图列表 (PyG Data) 或 Batch 对象。
            topo_list: 拓扑图列表 (PyG HeteroData) 或 Batch 对象。
            candidates_mask: 离散动作的掩码 (Tensor or None)。

        Returns:
            dict: 包含离散动作、连续动作(raw & clamped)、对数概率等。
        """
        # 1. 在线 Batching: 如果是列表则转换为 Batch 对象，否则直接使用
        if isinstance(dag_list, list):
            dag_batch = Batch.from_data_list(dag_list).to(self.device)
        else:
            # 已经是Batch对象，确保在正确的device上
            if hasattr(dag_list, 'x') and dag_list.x.device != self.device:
                dag_batch = dag_list.to(self.device)
            else:
                dag_batch = dag_list
            
        if isinstance(topo_list, list):
            topo_batch = Batch.from_data_list(topo_list).to(self.device)
        else:
            # 对于HeteroData Batch，检查device属性
            if hasattr(topo_list, 'x_dict'):
                first_tensor = next(iter(topo_list.x_dict.values()))
                if first_tensor.device != self.device:
                    topo_batch = topo_list.to(self.device)
                else:
                    topo_batch = topo_list
            else:
                topo_batch = topo_list.to(self.device) if hasattr(topo_list, 'to') else topo_list

        # [修复] 强制转换为 Tensor 并移动到 Device (无论输入是 list 还是 numpy)
        if candidates_mask is not None:
            candidates_mask = torch.as_tensor(candidates_mask, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            # Actor Forward
            # 返回 logits (未归一化概率，含 -1e9 mask) 和 raw_power (未截断功率)
            logits, _, raw_power = self.actor(dag_batch, topo_batch, candidates_mask)

            # --- A. 离散动作采样 (Target Selection) ---
            # 使用 logits 初始化 Categorical，它会自动处理 Softmax 和数值稳定性
            dist_discrete = Categorical(logits=logits)
            action_discrete = dist_discrete.sample()  # [Batch]
            action_discrete_logprob = dist_discrete.log_prob(action_discrete)

            # --- B. 连续动作采样 (Power Control) ---
            # 训练时加噪声，实现探索
            power_std = torch.exp(self.power_log_std)
            dist_continuous = Normal(raw_power, power_std)

            action_power_sample = dist_continuous.sample()
            # 对于多维连续动作，log_prob 需要求和 (这里 dim=1，sum(-1))
            action_power_logprob = dist_continuous.log_prob(action_power_sample).sum(dim=-1)

            # --- C. 动作后处理 ---
            action_power_clamped = torch.clamp(action_power_sample, 0.01, 1.0)

            # [新增关键修正] 计算联合 LogProb
            # 原因: PPO 更新时计算 ratios 需要 (log_d + log_p) - old_log_total
            # 如果不在此处合并，Buffer 可能会存错，导致 Loss 爆炸
            logprob_total = action_discrete_logprob + action_power_logprob

        return {
            'action_d': action_discrete,
            'action_p': action_power_sample,
            'power_val': action_power_clamped,
            'logprob_d': action_discrete_logprob,
            'logprob_p': action_power_logprob,
            'logprob_total': logprob_total  # [新增] 存入 Buffer 供 update 使用
        }

    def get_value(self, dag_list, topo_list):
        """[辅助] 获取状态价值"""
        # 支持列表和Batch对象
        if isinstance(dag_list, list):
            dag_batch = Batch.from_data_list(dag_list).to(self.device)
        else:
            if hasattr(dag_list, 'x') and dag_list.x.device != self.device:
                dag_batch = dag_list.to(self.device)
            else:
                dag_batch = dag_list
            
        if isinstance(topo_list, list):
            topo_batch = Batch.from_data_list(topo_list).to(self.device)
        else:
            if hasattr(topo_list, 'x_dict'):
                first_tensor = next(iter(topo_list.x_dict.values()))
                if first_tensor.device != self.device:
                    topo_batch = topo_list.to(self.device)
                else:
                    topo_batch = topo_list
            else:
                topo_batch = topo_list.to(self.device) if hasattr(topo_list, 'to') else topo_list
                
        with torch.no_grad():
            values = self.critic(dag_batch, topo_batch).squeeze(-1)
        return values

    def decode_actions(self, action_d_tensor, power_tensor, num_targets):
        """
        [推理阶段专用] 解码动作
        将神经网络输出的 Flat Index 解码为 {subtask, target, power} 字典。

        注意: 
        - Policy View: [0: RSU, 1: Self, 2+: Others(按全局ID排序)]
        - Env View: [0: Local, 1: RSU, 2+: Vehicle ID + 2]
        - data_utils中Others顺序: [0, 1, ..., i-1, i+1, ..., N-1] (按全局ID)
        """
        actions_list = []
        act_d = action_d_tensor.cpu().numpy()
        act_p = power_tensor.cpu().numpy()
        
        # num_targets应该是 1(RSU) + 1(Self) + (N-1)(Others) = N+1
        num_vehicles = num_targets - 1

        for i in range(len(act_d)):
            idx = int(act_d[i])
            subtask_idx = idx // num_targets
            policy_target_idx = idx % num_targets

            # --- Target 映射逻辑 ---
            if policy_target_idx == 0:
                # Policy Index 0 = RSU -> Env Index 1
                target_env = 1
            elif policy_target_idx == 1:
                # Policy Index 1 = Self -> Env Index 0 (Local)
                target_env = 0
            else:
                # Policy Index 2+ = Others
                # data_utils中Others顺序: [0, 1, ..., i-1, i+1, ..., N-1]
                # list_index = policy_target_idx - 2 (在Others列表中的索引)
                list_index = policy_target_idx - 2
                
                # 计算真实的车辆ID
                # 如果list_index < i，则对应车辆ID = list_index
                # 如果list_index >= i，则对应车辆ID = list_index + 1
                if list_index < i:
                    real_veh_id = list_index
                else:
                    real_veh_id = list_index + 1
                
                # Env期望: Vehicle ID + 2
                target_env = real_veh_id + 2

            actions_list.append({
                'subtask': subtask_idx,
                'target': target_env,
                'power': float(act_p[i])
            })

        return actions_list

    def update(self, buffer, batch_size=64):
        """
        [PPO 更新] 核心训练循环。
        """
        avg_loss = 0

        for _ in range(self.K_epochs):
            # 获取 Batch 数据 (Shuffle inside)
            data_generator = buffer.get_generator(batch_size)

            for sample in data_generator:
                (mb_dag, mb_topo, mb_masks,
                 mb_act_d, mb_act_c,
                 mb_old_logprobs, mb_returns, mb_advs) = sample

                # --- 1. 数据转移到 GPU ---
                mb_dag = mb_dag.to(self.device)
                mb_topo = mb_topo.to(self.device)

                # mb_masks 在 Buffer 中通常为 None (因为 mask 内嵌在 dag_batch 中)
                # 这里做兼容处理，防止出错
                if mb_masks is not None:
                    mb_masks = mb_masks.to(self.device)

                # 处理 Action 维度 [Batch, 1] -> [Batch]
                mb_act_d = mb_act_d.to(self.device)
                if mb_act_d.dim() > 1:
                    mb_act_d = mb_act_d.squeeze(-1)

                # 处理 Logprobs 维度 [Batch, 1] -> [Batch]
                # [关键修正] 确保 old_logprobs 是一维的 [Batch]
                mb_old_logprobs = mb_old_logprobs.to(self.device)
                if mb_old_logprobs.dim() > 1:
                    mb_old_logprobs = mb_old_logprobs.squeeze(-1)

                mb_act_c = mb_act_c.to(self.device)
                mb_returns = mb_returns.to(self.device)
                mb_advs = mb_advs.to(self.device)

                # --- 2. 重新评估 (Forward New Policy) ---
                # 获取 logits (含 mask) 和 raw_power
                # 注意: Actor 会自动优先使用 mb_dag.target_mask
                logits, _, raw_power = self.actor(mb_dag, mb_topo, mb_masks)

                # 获取新价值
                new_values = self.critic(mb_dag, mb_topo).squeeze(-1)

                # --- 3. 计算损失 (Calculate Loss) ---

                # A. 离散动作 Loss
                dist_discrete = Categorical(logits=logits)
                new_logprobs_d = dist_discrete.log_prob(mb_act_d)
                entropy_d = dist_discrete.entropy().mean()

                # B. 连续动作 Loss
                power_std = torch.exp(self.power_log_std)
                dist_continuous = Normal(raw_power, power_std)
                new_logprobs_p = dist_continuous.log_prob(mb_act_c).sum(dim=-1)
                entropy_p = dist_continuous.entropy().mean()

                # C. 联合概率 (Total LogProb)
                # 假设离散和连续动作相互独立
                total_new_logprob = new_logprobs_d + new_logprobs_p

                # D. 概率比率 (Importance Sampling Ratio)
                # [关键修正] 确保维度对齐，防止 broadcasting 错误导致 ratio 异常
                # mb_old_logprobs 必须是在 select_action 里计算好的 'logprob_total'
                ratios = torch.exp(total_new_logprob - mb_old_logprobs)

                # E. PPO Clip Loss
                surr1 = ratios * mb_advs
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advs
                loss_actor = -torch.min(surr1, surr2).mean()

                # F. Value Loss (MSE)
                loss_critic = self.mse_loss(new_values, mb_returns.squeeze(-1))

                # G. Total Loss
                loss_entropy = entropy_d + entropy_p
                loss = loss_actor + TC.VF_COEF * loss_critic - self.entropy_coef * loss_entropy

                # --- 4. 反向传播 (Backward) ---
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪 (防止梯度爆炸)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                avg_loss += loss.item()
                # [新增关键修正] 显式释放 PyG Batch 对象，解决 27G 内存泄漏
                # PyG 的 Batch 对象比较重，且容易在 Loop 中残留引用
                del mb_dag, mb_topo, logits, raw_power

        # [新增] 每一轮 update 结束后清理缓存
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return avg_loss / self.K_epochs