import torch
import numpy as np
from torch_geometric.data import Batch


class RolloutBuffer:
    """
    [PPO 经验回放池]
    支持 PyG 图数据 (DAG & Topology) 和 变长 Mask 的自动 Batch 化。

    修改记录:
    1. [移除] 独立的 mask_list 处理。Mask 已内嵌在 dag_data 中，由 PyG Batch 自动处理变长拼接。
    2. [优化] 强制将 Value/Logprob/Adv/Return 转换为 [Batch, 1] 形状，防止广播错误。
    3. [优化] add 方法增加 detach() 和 cpu() 转换，确保数据从计算图剥离。
    """

    def __init__(self, num_agents, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        # 存储 PyG Data 对象 (List of Objects)
        self.dag_list = []
        self.topo_list = []

        # 存储动作与数值 (List of values/arrays)
        self.act_d = []  # Discrete Action (Target)
        self.act_c = []  # Continuous Action (Power)
        self.logprobs = []
        self.rewards = []  # Raw rewards for GAE
        self.dones = []  # Dones for GAE
        self.values = []  # Critic Values for GAE

        # 计算后的优势和回报
        self.returns = None
        self.advantages = None

    def add(self, dag_list, topo_list, act_d, act_c, logprob, val, rew, done):
        """
        存储单步交互数据 (通常是所有 Agent 的数据列表)。

        Args:
            dag_list: List[Data], 长度为 num_agents
            topo_list: List[HeteroData], 长度为 num_agents
            act_d: Tensor [num_agents] or [num_agents, 1]
            act_c: Tensor [num_agents, 1]
            logprob: Tensor [num_agents]
            val: Tensor [num_agents, 1]
            rew: List/Array [num_agents]
            done: bool or List[bool]
        """
        # 1. 存储图对象 (引用存储，内存消耗较小)
        self.dag_list.extend(dag_list)
        self.topo_list.extend(topo_list)

        # 2. 存储 Tensor 数据 (转为 CPU list 以节省显存)
        # 确保处理掉可能存在的梯度关联
        self.act_d.extend(act_d.detach().cpu().numpy().tolist())
        self.act_c.extend(act_c.detach().cpu().numpy().tolist())
        self.logprobs.extend(logprob.detach().cpu().numpy().flatten().tolist())  # Flatten 保证是 1D list
        self.values.extend(val.detach().cpu().numpy().flatten().tolist())  # Flatten 保证是 1D list

        # 3. 存储 GAE 所需数据 (保持结构 [Step, Agents])
        self.rewards.append(rew)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, done):
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        last_value = last_value.detach().cpu().numpy().flatten()  # [Num_Agents]

        # 转换为 numpy 矩阵 [Steps, Agents]
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # 处理 Values: [Total_Samples] -> [Steps, Agents]
        n_steps = len(rewards)
        n_agents = self.num_agents

        # [安全Reshape] 确保总数匹配
        values = np.array(self.values)
        if len(values) != n_steps * n_agents:
            # 容错: 如果最后一步没存满(极为罕见)，进行截断或报错
            values = values[:n_steps * n_agents]
        values = values.reshape(n_steps, n_agents)

        # 拼接 Last Value 用于计算最后一步的 TD Error
        # values 变为 [Steps + 1, Agents]
        values = np.concatenate([values, np.expand_dims(last_value, 0)], axis=0)

        advs = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(n_steps)):
            if np.isscalar(dones[t]):
                mask = 1.0 - float(dones[t])
            else:
                mask = 1.0 - dones[t]  # 支持多智能体部分 done (虽然在同步 VEC 环境很少见)

            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * mask * last_gae_lam
            advs[t] = last_gae_lam

        # Returns = Advantages + Values
        returns = advs + values[:-1]

        # Flatten 所有的结果，准备 Batch 训练
        self.returns = returns.flatten()
        self.advantages = advs.flatten()

        # [推荐] Advantage Normalization (标准化)，显著提升 PPO 稳定性
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_generator(self, batch_size):
        """
        生成器: 每次 yield 一个 Batch 的数据

        Yields:
            dag_batch: PyG Batch 对象 (含 mask, target_mask)
            topo_batch: PyG Hetero Batch 对象 (含 edge_attr)
            masks: None (已集成在 dag_batch 中)
            act_d: LongTensor [B]
            act_c: FloatTensor [B, 1]
            logprobs: FloatTensor [B, 1]
            returns: FloatTensor [B, 1]
            advs: FloatTensor [B, 1]
        """
        total_samples = len(self.dag_list)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # 1. Graph Batching (PyG 自动处理变长节点和 Mask 拼接)
            mb_dag_list = [self.dag_list[i] for i in batch_idx]
            mb_topo_list = [self.topo_list[i] for i in batch_idx]

            mb_dag = Batch.from_data_list(mb_dag_list)
            mb_topo = Batch.from_data_list(mb_topo_list)

            # 2. Tensors Construction
            # Act Discrete: [B] -> LongTensor
            mb_act_d = torch.tensor([self.act_d[i] for i in batch_idx], dtype=torch.long)

            # Act Continuous: [B] -> [B, 1]
            mb_act_c = torch.tensor([self.act_c[i] for i in batch_idx], dtype=torch.float32)
            if mb_act_c.dim() == 1: mb_act_c = mb_act_c.view(-1, 1)

            # Logprobs: [B] -> [B, 1]
            mb_logprobs = torch.tensor([self.logprobs[i] for i in batch_idx], dtype=torch.float32).view(-1, 1)

            # Returns & Advs: [B] -> [B, 1]
            mb_returns = torch.tensor(self.returns[batch_idx], dtype=torch.float32).view(-1, 1)
            mb_advs = torch.tensor(self.advantages[batch_idx], dtype=torch.float32).view(-1, 1)

            # [关键] masks 返回 None
            # 理由: dag_batch.mask 和 dag_batch.target_mask 已经由 PyG 正确 Batch 化了。
            # 如果在这里手动 stack 变长的 numpy mask，会报错。
            mb_masks = None

            yield (mb_dag, mb_topo, mb_masks, mb_act_d, mb_act_c, mb_logprobs, mb_returns, mb_advs)