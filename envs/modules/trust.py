"""
信誉外生过程 (Reputation / Trust External Process)

每个远端节点 j 拥有隐藏可靠性 p_j（环境私有）：
- 类型混合：以 TRUST_RELIABLE_PROB 概率为可靠节点, 否则为不可靠节点
- 可靠节点 p_j ~ Uniform(TRUST_P_RELIABLE_RANGE)
- 不可靠节点 p_j ~ Uniform(TRUST_P_UNRELIABLE_RANGE)

维护 Beta 后验 (a_j, b_j)：
- hat_rho_j = a_j / (a_j + b_j)
- uncertainty_j = 1 / (a_j + b_j)

证据延迟 tau_rep_steps:
- 成功/失败证据进入 pending 队列，t + tau_rep_steps 后更新 (a_j, b_j)

在远端子任务完成时：
- 采样 y ~ Bernoulli(p_j)
- y=1 正常完成
- y=0 结果不可用 -> 子任务回到 ready 重试
"""
import numpy as np
from collections import deque
from configs.config import SystemConfig as Cfg


class TrustManager:
    """管理所有远端节点的信誉状态"""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        # 隐藏可靠性 p_j（环境私有，agent 不可见）
        self.hidden_reliability = {}  # node_key -> float
        # Beta 后验参数
        self.beta_a = {}  # node_key -> float
        self.beta_b = {}  # node_key -> float
        # 延迟证据队列: (arrival_step, node_key, success_bool)
        self.pending_evidence = deque()
        # 统计
        self.total_attempts = 0
        self.total_failures = 0
        self.retry_events = []  # (step, veh_id, node_key, subtask_idx)

    def reset(self, remote_node_keys):
        """
        每 episode 开始时调用，为所有远端节点采样隐藏可靠性。
        
        Args:
            remote_node_keys: 所有可能的远端节点标识列表 (e.g., RSU ids + vehicle ids)
        """
        self.hidden_reliability.clear()
        self.beta_a.clear()
        self.beta_b.clear()
        self.pending_evidence.clear()
        self.total_attempts = 0
        self.total_failures = 0
        self.retry_events.clear()

        prior_a = getattr(Cfg, 'TRUST_PRIOR_A', 1.0)
        prior_b = getattr(Cfg, 'TRUST_PRIOR_B', 1.0)
        reliable_prob = getattr(Cfg, 'TRUST_RELIABLE_PROB', 0.8)
        p_rel_range = getattr(Cfg, 'TRUST_P_RELIABLE_RANGE', (0.7, 1.0))
        p_unrel_range = getattr(Cfg, 'TRUST_P_UNRELIABLE_RANGE', (0.3, 0.7))

        for key in remote_node_keys:
            if self.rng.random() < reliable_prob:
                p_j = self.rng.uniform(p_rel_range[0], p_rel_range[1])
            else:
                p_j = self.rng.uniform(p_unrel_range[0], p_unrel_range[1])
            self.hidden_reliability[key] = float(p_j)
            self.beta_a[key] = float(prior_a)
            self.beta_b[key] = float(prior_b)

    def sample_outcome(self, node_key):
        """
        在远端子任务完成时采样: y ~ Bernoulli(p_j)
        
        Returns:
            bool: True = 成功, False = 失败(需重试)
        """
        p_j = self.hidden_reliability.get(node_key, 1.0)
        self.total_attempts += 1
        success = self.rng.random() < p_j
        if not success:
            self.total_failures += 1
        return bool(success)

    def submit_evidence(self, current_step, node_key, success):
        """提交延迟证据"""
        delay = getattr(Cfg, 'TRUST_DELAY_STEPS', 3)
        self.pending_evidence.append((current_step + delay, node_key, success))

    def process_pending(self, current_step):
        """处理到期的延迟证据，更新 Beta 后验"""
        while self.pending_evidence and self.pending_evidence[0][0] <= current_step:
            _, node_key, success = self.pending_evidence.popleft()
            if node_key in self.beta_a:
                if success:
                    self.beta_a[node_key] += 1.0
                else:
                    self.beta_b[node_key] += 1.0

    def get_reputation(self, node_key):
        """
        获取节点信誉估计 (hat_rho, uncertainty)
        
        Returns:
            (hat_rho, uncertainty): hat_rho = a/(a+b), uncertainty = 1/(a+b)
        """
        a = self.beta_a.get(node_key, 1.0)
        b = self.beta_b.get(node_key, 1.0)
        total = a + b
        hat_rho = a / max(total, 1e-9)
        uncertainty = 1.0 / max(total, 1e-9)
        return float(hat_rho), float(uncertainty)

    def get_stats(self):
        """获取 episode 统计"""
        return {
            'trust_attempts': self.total_attempts,
            'trust_failures': self.total_failures,
            'trust_failure_rate': self.total_failures / max(self.total_attempts, 1),
            'trust_retry_count': len(self.retry_events),
        }
