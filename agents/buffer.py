import torch
import numpy as np
from torch_geometric.data import Batch


class RolloutBuffer:
    def __init__(self, num_agents, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        # 存储 PyG Data 对象 (无法直接 Tensor 化，用列表存储)
        self.dag_list = []
        self.topo_list = []
        self.mask_list = []  # 可能为 None

        # 存储 Tensor 数据
        self.act_d = []
        self.act_c = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # 计算后的优势和回报
        self.returns = None
        self.advantages = None

    def add(self, dag_list, topo_list, act_d, act_c, logprob, val, rew, done, mask=None):
        """
        存储一步的数据。
        注意: 为了支持多智能体 (MAPPO)，这里的数据通常是 List[Graph] 或 Batch 后的 Tensor。
        train.py 中传入的是 Batch 后的结果还是 List?
        根据 train.py:
            dag_list=dag_graph_list (List of Data)
            act_d=tensor (Batch)
        这里我们统一将数据拆解并存入列表，或者按 Batch 存入。

        策略: 直接存储 train.py 传入的原始对象 (Lists for graphs, Tensors for values)
        """
        # Graph Objects (List of Data)
        # 这里 extend 是因为 train.py 中传入的是 num_vehicles 个图的列表
        # 但 PPO 更新时通常视为一个大的 Batch。
        # 简单起见，我们假设存储的是 "Step" 级别的数据，Generator 里再 Flatten。

        # 修正策略: RolloutBuffer 存储的是 Flatten 后的所有样本
        # train.py 传入的是当前 Step 所有 Agent 的数据列表/Tensor

        self.dag_list.extend(dag_list)
        self.topo_list.extend(topo_list)

        if mask is not None:
            self.mask_list.extend(mask)  # 假设 mask 是 list
        else:
            # 如果是 None，填充 None 以保持长度一致 (或者在 Generator 里处理)
            self.mask_list.extend([None] * len(dag_list))

        # Tensors: 转换为 CPU list 方便处理
        self.act_d.extend(act_d.cpu().numpy().tolist())
        self.act_c.extend(act_c.cpu().numpy().tolist())
        self.logprobs.extend(logprob.cpu().numpy().tolist())
        self.values.extend(val.cpu().numpy().tolist())

        # Rewards & Dones 需要特殊处理用于 GAE
        # 它们是 [Num_Agents] 的形状
        self.rewards.append(rew)  # List of List (Step -> Agents)
        self.dones.append(done)  # List (Step -> Bool/Int)

    def compute_returns_and_advantages(self, last_value, done):
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        last_value = last_value.cpu().numpy().tolist()  # [Num_Agents]

        # 将 rewards 和 dones 转为 numpy [Steps, Agents]
        rewards = np.array(self.rewards)  # [Steps, Agents]
        dones = np.array(self.dones)  # [Steps] or [Steps, Agents]

        # 展平 values
        # self.values 是 Flatten 的 [Steps * Agents]，需要 reshape 回去计算
        # 假设每次 add 存入 num_agents 个数据
        n_steps = len(rewards)
        n_agents = len(rewards[0])

        values = np.array(self.values).reshape(n_steps, n_agents)

        # 加上最后一个 value
        values = np.concatenate([values, np.array([last_value])], axis=0)

        advs = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(n_steps)):
            # 如果 done 为 True，说明这一步结束后 Episode 结束
            # 如果 dones 是标量 (全环境结束)，广播给所有 Agent
            if np.isscalar(dones[t]):
                mask = 1.0 - float(dones[t])
            else:
                mask = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * mask * last_gae_lam
            advs[t] = last_gae_lam

        returns = advs + values[:-1]

        # Flatten everything for training
        self.returns = returns.flatten()
        self.advantages = advs.flatten()

        # 归一化 Advantage (可选，但推荐)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_generator(self, batch_size):
        """
        生成器: 每次 yield 一个 Batch 的数据
        严格对齐 mappo_agent.py 的解包顺序:
        (dag, topo, masks, act_d, act_c, logprobs, returns, advs)
        """
        total_samples = len(self.dag_list)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # 1. Graphs (需要重新 Batch 化)
            # 使用列表推导式提取索引对应的图对象
            mb_dag_list = [self.dag_list[i] for i in batch_idx]
            mb_topo_list = [self.topo_list[i] for i in batch_idx]

            # 转换为 PyG Batch
            mb_dag = Batch.from_data_list(mb_dag_list)
            mb_topo = Batch.from_data_list(mb_topo_list)

            # 2. Masks
            # 检查是否有有效 Mask
            if self.mask_list and self.mask_list[0] is not None:
                mb_mask_list = [self.mask_list[i] for i in batch_idx]
                mb_masks = torch.tensor(np.array(mb_mask_list), dtype=torch.bool)
            else:
                mb_masks = None

            # 3. Tensors (act_d, act_c, logp, ret, adv)
            # 必须转为 Tensor 且类型正确

            # [关键] act_d 必须是 LongTensor (Int)
            mb_act_d = torch.tensor([self.act_d[i] for i in batch_idx], dtype=torch.long)

            # act_c 是 FloatTensor
            mb_act_c = torch.tensor([self.act_c[i] for i in batch_idx], dtype=torch.float32)

            # Logprobs
            mb_logprobs = torch.tensor([self.logprobs[i] for i in batch_idx], dtype=torch.float32)

            # Returns & Advantages
            mb_returns = torch.tensor(self.returns[batch_idx], dtype=torch.float32)
            mb_advs = torch.tensor(self.advantages[batch_idx], dtype=torch.float32)

            # [关键] 必须严格按照此顺序 Yield
            yield (mb_dag, mb_topo, mb_masks, mb_act_d, mb_act_c, mb_logprobs, mb_returns, mb_advs)