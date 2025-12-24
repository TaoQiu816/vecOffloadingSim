import torch
import numpy as np
from torch_geometric.data import Batch


class RolloutBuffer:
    """
    [PPO 经验回放池 - Final Fixed Version]
    修复了 LogProb 维度广播导致的 Loss 爆炸问题，并增强了内存安全性。
    """

    def __init__(self, num_agents, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        # 存储 PyG Data 对象
        # 注意：为了防止内存泄漏，存入的对象应当尽可能不带梯度
        self.dag_list = []
        self.topo_list = []

        # 存储动作与数值
        self.act_d = []  # Discrete Action (Target)
        self.act_c = []  # Continuous Action (Power)

        # [关键] 存储 LogProb Total (Discrete + Continuous)
        self.logprobs = []

        self.rewards = []
        self.dones = []
        self.values = []

        self.returns = None
        self.advantages = None

    def add(self, dag_list, topo_list, act_d, act_c, logprob, val, rew, done):
        """
        存储单步交互数据。

        [关键要求]: logprob 参数必须是 (log_discrete + log_continuous) 的总和！
        """
        # 1. 存储图对象
        self.dag_list.extend(dag_list)
        self.topo_list.extend(topo_list)

        # 2. 存储 Tensor 数据 (转为 CPU list 以节省显存)
        # act_d: [N] -> List[int]
        if isinstance(act_d, torch.Tensor):
            self.act_d.extend(act_d.detach().cpu().numpy().tolist())
        else:
            self.act_d.extend(act_d)

        # act_c: [N, 1] -> List[float]
        if isinstance(act_c, torch.Tensor):
            self.act_c.extend(act_c.detach().cpu().numpy().flatten().tolist())
        else:
            self.act_c.extend(act_c)

        # logprob: [N] -> List[float]
        # [关键修正] 确保它是扁平的 list，不要嵌套
        if isinstance(logprob, torch.Tensor):
            self.logprobs.extend(logprob.detach().cpu().numpy().flatten().tolist())
        else:
            self.logprobs.extend(logprob)

        # values: [N] -> List[float]
        if isinstance(val, torch.Tensor):
            self.values.extend(val.detach().cpu().numpy().flatten().tolist())
        else:
            self.values.extend(val)

        # 3. 存储 GAE 所需数据 (保持结构 [Step, Agents])
        self.rewards.append(rew)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, done):
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.detach().cpu().numpy().flatten()  # [Num_Agents]

        # 转换为 numpy 矩阵 [Steps, Agents]
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # 处理 Values: [Total_Samples] -> [Steps, Agents]
        n_steps = len(rewards)
        n_agents = self.num_agents

        # [安全Reshape] 确保总数匹配
        values = np.array(self.values)

        # 容错: 防止因中途 truncate 导致 values 长度不匹配
        expected_len = n_steps * n_agents
        if len(values) > expected_len:
            values = values[:expected_len]
        elif len(values) < expected_len:
            # 极端情况补零
            values = np.pad(values, (0, expected_len - len(values)))

        values = values.reshape(n_steps, n_agents)

        # 拼接 Last Value
        values = np.concatenate([values, np.expand_dims(last_value, 0)], axis=0)

        advs = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(n_steps)):
            if np.isscalar(dones[t]):
                mask = 1.0 - float(dones[t])
            else:
                mask = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * mask * last_gae_lam
            advs[t] = last_gae_lam

        # Returns = Advantages + Values
        returns = advs + values[:-1]

        # Flatten 所有的结果，准备 Batch 训练
        self.returns = returns.flatten()
        self.advantages = advs.flatten()

        # [推荐] Advantage Normalization (标准化)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_generator(self, batch_size):
        """
        生成器: 每次 yield 一个 Batch 的数据
        """
        total_samples = len(self.dag_list)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # 1. Graph Batching (PyG 自动处理)
            mb_dag_list = [self.dag_list[i] for i in batch_idx]
            mb_topo_list = [self.topo_list[i] for i in batch_idx]

            mb_dag = Batch.from_data_list(mb_dag_list)
            mb_topo = Batch.from_data_list(mb_topo_list)

            # 2. Tensors Construction

            # Act Discrete: [B] (一维)
            mb_act_d = torch.tensor([self.act_d[i] for i in batch_idx], dtype=torch.long)

            # Act Continuous: [B, 1] (二维，因为它是特征)
            mb_act_c = torch.tensor([self.act_c[i] for i in batch_idx], dtype=torch.float32).view(-1, 1)

            # Logprobs: [B] (一维!)
            # [致命错误修正] 去掉了 .view(-1, 1)
            # 必须保持一维 [B]，以便与 PPO Update 中的 total_new_logprob [B] 直接相减
            mb_logprobs = torch.tensor([self.logprobs[i] for i in batch_idx], dtype=torch.float32)

            # Returns & Advs: [B, 1] (二维，用于 MSE Loss 计算方便)
            mb_returns = torch.tensor(self.returns[batch_idx], dtype=torch.float32).view(-1, 1)
            mb_advs = torch.tensor(self.advantages[batch_idx], dtype=torch.float32).view(-1, 1)

            # Masks: None (已集成在 mb_dag 中)
            mb_masks = None

            yield (mb_dag, mb_topo, mb_masks, mb_act_d, mb_act_c, mb_logprobs, mb_returns, mb_advs)

    def clear(self):
        """显式清空内存"""
        self.reset()