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
    [MAPPO Agent]
    多智能体 PPO 算法 (Centralized Training Decentralized Execution) 的实现。

    核心特性:
    1. 混合动作空间: 同时输出离散动作(卸载目标+子任务)和连续动作(传输功率)。
    2. 参数共享: 所有车辆智能体共享同一个 Actor-Critic 网络参数。
    3. 集中式训练: Critic 网络利用 Transformer 提取的全局特征进行价值评估。

    修改记录:
    - [修复] 将 Actor 输出明确为 logits，并使用 Categorical(logits=...) 初始化分布，
      解决因 Mask 导致的概率为 0 时的数值不稳定问题。
    - [新增] decode_actions 方法，处理动作解码和功率门控 (Gating)。
    - [参数引用] 修改为引用 TrainConfig 中的参数。
    """

    def __init__(self, task_dim, veh_dim, rsu_dim,
                 device='cpu'):  # 减少直接传入参数，改为使用 TC 配置
        """
        初始化 MAPPO Agent。

        Args:
            task_dim (int): 任务特征维度 (来自 Encoder)。
            veh_dim (int): 车辆特征维度。
            rsu_dim (int): RSU 特征维度。
            device (str): 运行设备 ('cpu' or 'cuda').
        """
        self.device = device

        # 使用 TrainConfig 中的超参数
        lr_actor = TC.LR_ACTOR
        lr_critic = TC.LR_CRITIC
        self.gamma = TC.GAMMA
        self.gae_lambda = TC.GAE_LAMBDA
        self.eps_clip = TC.CLIP_PARAM
        self.K_epochs = TC.PPO_EPOCH
        self.entropy_coef = TC.ENTROPY_COEF
        self.max_grad_norm = TC.MAX_GRAD_NORM

        # --- 初始化 Actor 和 Critic 网络 ---
        # Actor: 输出动作策略 (logits 和 raw_power)
        # 使用 TC 配置的网络参数
        self.actor = TransformerHybridActor(task_feat_dim=task_dim, veh_feat_dim=veh_dim, rsu_feat_dim=rsu_dim,
                                            embed_dim=TC.EMBED_DIM).to(device)

        # Critic: 输出状态价值 (Value)
        self.critic = TransformerHybridCritic(
            task_feat_dim=task_dim,  # 修改为 task_feat_dim
            veh_feat_dim=veh_dim,  # 修改为 veh_feat_dim
            rsu_feat_dim=rsu_dim,  # 修改为 rsu_feat_dim
            embed_dim=TC.EMBED_DIM,
            num_heads=TC.NUM_HEADS,
            num_layers=TC.NUM_LAYERS
        ).to(device)

        # --- 优化器 ---
        # 分离 Actor 和 Critic 的学习率
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-5)

        # --- 连续动作方差参数 ---
        # 功率控制是一个连续动作，我们学习一个与状态无关的对数标准差 (Log Std)
        # 初始化为 -0.5，对应 std ≈ 0.6
        self.power_log_std = nn.Parameter(torch.zeros(1).to(device) - 0.5)
        # 将其归类到 Actor 参数组，使用 actor 的学习率
        self.optimizer.add_param_group({'params': [self.power_log_std], 'lr': lr_actor})

        self.mse_loss = nn.MSELoss()

    def decay_lr(self, decay_rate=0.9):
        """
        [辅助] 学习率衰减函数
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay_rate

    def select_action(self, dag_list, topo_list, candidates_mask=None):
        """
        [推理/交互阶段] 选择动作。

        Args:
            dag_list: DAG 任务图列表。
            topo_list: 拓扑图列表 (包含车辆和 RSU 节点)。
            candidates_mask: 离散动作的掩码 (可选)。

        Returns:
            dict: 包含离散动作、连续动作(raw & clamped)、对数概率等。
        """
        # 1. 在线 Batching: 将列表转换为 PyG Batch 对象，以便并行处理
        dag_batch = Batch.from_data_list(dag_list).to(self.device)
        topo_batch = Batch.from_data_list(topo_list).to(self.device)

        #GPU： [修改位置]: 在 with torch.no_grad() 之前添加
        if candidates_mask is not None and not isinstance(candidates_mask, torch.Tensor):
            candidates_mask = torch.as_tensor(candidates_mask, dtype=torch.bool).to(self.device)

        # 如果有 mask，也需要 batch (假设传入的已经是 tensor batch，或者需要在这里 stack)
        # 这里假设调用处已经处理好 mask 的 tensor 形式，或者暂时不处理 mask 里的 batch 逻辑
        # 如果 candidates_mask 是 list，需 torch.stack(candidates_mask).to(device)

        with torch.no_grad():
            # Actor Forward
            # [关键修改] Actor 返回的是 logits (包含用于 Mask 的负无穷大值)，而不是归一化的 probs
            # 这对于数值稳定性至关重要，特别是在有大量非法动作被 Mask 的情况下
            logits, _, raw_power = self.actor(dag_batch, topo_batch, candidates_mask)

            # --- A. 离散动作采样 ---
            # [关键修复] 使用 logits=logits 参数初始化 Categorical
            # Categorical 内部会自动对 logits 进行 Softmax，能正确处理 -1e9 的 Mask 值
            dist_discrete = Categorical(logits=logits)

            action_discrete = dist_discrete.sample()  # [Batch]
            action_discrete_logprob = dist_discrete.log_prob(action_discrete)

            # --- B. 连续动作采样 ---
            # 论文策略: 训练时加噪声，确定性部分由网络输出
            power_std = torch.exp(self.power_log_std)
            dist_continuous = Normal(raw_power, power_std)

            action_power_sample = dist_continuous.sample()
            # 对于多维连续动作，log_prob 需要求和 (虽然这里功率是 1 维)
            action_power_logprob = dist_continuous.log_prob(action_power_sample).sum(dim=-1)

            # --- C. 动作后处理 ---
            # 功率截断: 确保物理意义 (0, 1]
            # 注意: 计算 PPO Loss 时使用的是 raw_sample 和对应的 log_prob，
            # 这里截断后的值仅用于环境执行 (Environment Execution)。
            action_power_clamped = torch.clamp(action_power_sample, 0.01, 1.0)

        return {
            'action_d': action_discrete,  # 离散动作索引
            'action_p': action_power_sample,  # 原始采样功率 (用于 Buffer)
            'power_val': action_power_clamped,  # 截断后功率 (用于 Env 执行)
            'logprob_d': action_discrete_logprob,
            'logprob_p': action_power_logprob
        }

    def get_value(self, dag_list, topo_list):
        """
        [辅助] 获取状态价值 V(s)
        用于计算 Advantage (GAE)。
        """
        dag_batch = Batch.from_data_list(dag_list).to(self.device)
        topo_batch = Batch.from_data_list(topo_list).to(self.device)

        with torch.no_grad():
            values = self.critic(dag_batch, topo_batch).squeeze(-1)
        return values

    def decode_actions(self, action_d_tensor, power_tensor, num_targets):
        """
        [动作解码] 将神经网络输出解码为环境可读的格式。

        Args:
            action_d_tensor: 离散动作索引 [Batch]
            power_tensor: 功率值 [Batch, 1]
            num_targets: 每个子任务的可选目标数 (用于解码 subtask 和 target)
                         注意: 这里需要确认 action space 是怎么定义的。
                         如果 action space 是 flatten 的 (subtask * targets)，则需要 num_targets。
                         如果 policy 直接输出 target index (假设 subtask 固定为当前 ready 任务)，
                         则不需要 subtask 解码。

                         *根据之前的代码逻辑*: select_action 选择的是 target，subtask 由 dag.target_idx 决定。
                         所以这里的 action_d 就是 target_index (含 Local, RSU, Neighbors)。
        """
        """
                将模型输出的扁平动作索引解码为环境可执行的字典。
                假设 Policy 输出的 Target 顺序为: [RSU, Self, Other_0, Other_1, ...] (Self-First Canonical)
        """
        actions_list = []
        act_d = action_d_tensor.cpu().numpy()
        act_p = power_tensor.cpu().numpy()

        for i in range(len(act_d)):
            # 1. 解析 Flatten Index -> (Subtask, Policy_Target)
            # idx = subtask * num_targets + target
            idx = int(act_d[i])
            subtask_idx = idx // num_targets
            policy_target_idx = idx % num_targets

            # 2. 映射 Policy Target -> Env Target
            # Policy View: [0: RSU, 1: Self, 2+: Others (Sorted by ID)]
            # Env View:    [0: Local, 1: RSU, 2+: NeighborID + 2]

            target_env = 0  # Default to Local

            if policy_target_idx == 0:
                target_env = 1  # RSU
            elif policy_target_idx == 1:
                target_env = 0  # Local (Self)
            else:
                # 这里的逻辑对应 data_utils 中的 "Self-First" 构建顺序
                # policy_idx >= 2 对应其他车辆
                # 我们需要找到这个 "其他车辆" 的真实 ID
                # 逻辑: 在所有车辆 ID [0...N-1] 中，去掉了 Self (i)，剩下的按顺序排列

                # 例子: N=5, Self=2. Others=[0, 1, 3, 4]
                # Policy Index: 0(RSU), 1(Self), 2(ID0), 3(ID1), 4(ID3), 5(ID4)

                list_index = policy_target_idx - 2
                if list_index < i:
                    real_veh_id = list_index
                else:
                    real_veh_id = list_index + 1

                # Env 期望的 Neighbor Target 是 ID + 2 (如果 ID 是 int)
                # 或者直接传 int ID，Env 会处理 (Env 代码: target = neighbor_id if ...)
                # 实际上 Env.step 中: neighbor_id = tgt_idx - 2.
                # 所以我们传 real_veh_id + 2
                target_env = real_veh_id + 2

            actions_list.append({
                'subtask': subtask_idx,  # [修复 KeyError]
                'target': target_env,
                'power': float(act_p[i])
            })

        return actions_list

    def update(self, buffer, batch_size=64):
        """
        [PPO 更新] 核心训练循环。

        Args:
            buffer: 存储轨迹数据的 Replay Buffer。
            batch_size: Mini-batch 大小。
        """
        avg_loss = 0

        # 使用 TC 配置的 Epoch 数
        for _ in range(self.K_epochs):
            # 从 Buffer 中获取数据生成器 (Mini-batch iterator)
            data_generator = buffer.get_generator(batch_size)

            for sample in data_generator:
                (mb_dag, mb_topo, mb_masks,  # 数据与掩码
                 mb_act_d, mb_act_c,  # 动作
                 mb_old_logprobs, mb_returns, mb_advs) = sample  # RL 信号

                # [修改位置]: 替换原有的转移代码，确保完全覆盖
                mb_dag = mb_dag.to(self.device)
                mb_topo = mb_topo.to(self.device)

                # 显式处理 mb_masks 的转移
                if mb_masks is not None:
                    mb_masks = mb_masks.to(self.device)

                # 处理 mb_act_d 的维度并转移
                mb_act_d = mb_act_d.to(self.device)
                if mb_act_d.dim() > 1:
                    mb_act_d = mb_act_d.squeeze(-1)  # 安全地压掉最后一维

                # 其他 Tensor 转移
                mb_act_c = mb_act_c.to(self.device)
                mb_old_logprobs = mb_old_logprobs.to(self.device)
                mb_returns = mb_returns.to(self.device)
                mb_advs = mb_advs.to(self.device)

                # --- 1. Forward New Policy (评估新策略) ---
                # 获取 logits (含 mask) 和 raw_power
                logits, _, raw_power = self.actor(mb_dag, mb_topo, mb_masks)

                # 获取新价值
                new_values = self.critic(mb_dag, mb_topo).squeeze(-1)

                # --- 2. Calculate Loss (计算损失) ---

                # A. Discrete Action Loss
                dist_discrete = Categorical(logits=logits)
                new_logprobs_d = dist_discrete.log_prob(mb_act_d)
                entropy_d = dist_discrete.entropy().mean()

                # B. Continuous Action Loss
                power_std = torch.exp(self.power_log_std)
                dist_continuous = Normal(raw_power, power_std)
                # 对于多维连续动作，log_prob 需要 sum
                new_logprobs_p = dist_continuous.log_prob(mb_act_c).sum(dim=-1)
                entropy_p = dist_continuous.entropy().mean()

                # Combine LogProbs (联合概率的对数 = 对数概率之和)
                total_new_logprob = new_logprobs_d + new_logprobs_p

                # 计算概率比率 (Ratio)
                ratios = torch.exp(total_new_logprob - mb_old_logprobs)

                # PPO Clip Loss
                surr1 = ratios * mb_advs
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advs
                loss_actor = -torch.min(surr1, surr2).mean()

                # Critic Loss (MSE)
                loss_critic = self.mse_loss(new_values, mb_returns)

                # Total Loss
                # 熵正则项鼓励探索 (Entropy Bonus)
                loss_entropy = entropy_d + entropy_p
                loss = loss_actor + TC.VF_COEF * loss_critic - self.entropy_coef * loss_entropy

                # --- 3. Backward (反向传播) ---
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping (防止梯度爆炸)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                avg_loss += loss.item()

        return avg_loss / self.K_epochs
