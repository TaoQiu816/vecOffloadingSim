"""
信誉外生过程 (Reputation / Trust External Process)

每个远端节点 j 拥有隐藏可靠性 p_j（环境私有）：

[原始模式] MALICIOUS_RATIO=0 或 REP_INIT_MODE="uniform"：
- 类型混合：以 TRUST_RELIABLE_PROB 概率为可靠节点, 否则为不可靠节点
- 可靠节点 p_j ~ Uniform(TRUST_P_RELIABLE_RANGE)
- 不可靠节点 p_j ~ Uniform(TRUST_P_UNRELIABLE_RANGE)

[恶意比例模式] MALICIOUS_RATIO>0 且 REP_INIT_MODE="beta"：
- RSU：p_j = RSU_REPUTATION（固定高信誉）
- 车辆：以 MALICIOUS_RATIO 概率为恶意节点
  - 诚实节点 p_j ~ Beta(REP_HONEST_BETA)，均值≈0.82
  - 恶意节点 p_j ~ Beta(REP_MAL_BETA)，均值≈0.18
  - 轻微重叠扰动 N(0, REP_OVERLAP)，clip[0,1]

维护 Beta 后验 (a_j, b_j)：
- hat_rho_j = a_j / (a_j + b_j)
- uncertainty_j = 1 / (a_j + b_j)

证据延迟 tau_rep_steps:
- 成功/失败证据进入 pending 队列，t + tau_rep_steps 后更新 (a_j, b_j)

TRUST_FAIL_SCOPE = "v2v_only"：RSU 节点 sample_outcome 始终返回 True（不注入失败）
TRUST_FAIL_SCOPE = "all"：所有远端节点均按 Bernoulli(p_j) 采样

在远端子任务完成时：
- 采样 y ~ Bernoulli(p_j)（v2v_only 模式下 RSU 跳过）
- y=1 正常完成
- y=0 结果不可用 -> 子任务回到 ready 重试，illegal_reason 保持 None
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
        # 恶意节点标记（episode 级，仅在 beta 模式下有效）
        self.is_malicious = {}  # node_key -> bool

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
        self.is_malicious.clear()

        prior_a = float(getattr(Cfg, 'TRUST_PRIOR_A', 1.0))
        prior_b = float(getattr(Cfg, 'TRUST_PRIOR_B', 1.0))

        malicious_ratio = float(getattr(Cfg, 'MALICIOUS_RATIO', 0.0))
        rep_init_mode = str(getattr(Cfg, 'REP_INIT_MODE', 'uniform'))
        use_beta_mode = (malicious_ratio > 0.0 and rep_init_mode == 'beta')

        if use_beta_mode:
            rep_honest_beta = getattr(Cfg, 'REP_HONEST_BETA', (18, 4))
            rep_mal_beta = getattr(Cfg, 'REP_MAL_BETA', (4, 18))
            rep_overlap = float(getattr(Cfg, 'REP_OVERLAP', 0.05))
            rsu_reputation = float(getattr(Cfg, 'RSU_REPUTATION', 0.95))

            for key in remote_node_keys:
                node_type = key[0] if isinstance(key, tuple) and len(key) >= 1 else None
                if node_type == 'RSU':
                    p_j = float(np.clip(rsu_reputation, 0.0, 1.0))
                    self.is_malicious[key] = False
                else:  # VEH
                    is_mal = bool(self.rng.random() < malicious_ratio)
                    self.is_malicious[key] = is_mal
                    a, b = rep_mal_beta if is_mal else rep_honest_beta
                    p_j = float(self.rng.beta(float(a), float(b)))
                    if rep_overlap > 0.0:
                        p_j += float(self.rng.normal(0.0, rep_overlap))
                    p_j = float(np.clip(p_j, 0.0, 1.0))

                self.hidden_reliability[key] = p_j
                self.beta_a[key] = prior_a
                self.beta_b[key] = prior_b
        else:
            # 原有逻辑：Uniform 混合分布，MALICIOUS_RATIO=0 时保证行为与原版完全一致
            reliable_prob = float(getattr(Cfg, 'TRUST_RELIABLE_PROB', 0.8))
            p_rel_range = getattr(Cfg, 'TRUST_P_RELIABLE_RANGE', (0.7, 1.0))
            p_unrel_range = getattr(Cfg, 'TRUST_P_UNRELIABLE_RANGE', (0.3, 0.7))
            rsu_range = getattr(Cfg, 'TRUST_P_RSU_RANGE', None)
            veh_range = getattr(Cfg, 'TRUST_P_VEH_RANGE', None)

            for key in remote_node_keys:
                self.is_malicious[key] = False
                node_type = key[0] if isinstance(key, tuple) and len(key) >= 1 else None
                p_j = None
                if node_type == 'RSU' and rsu_range is not None:
                    lo, hi = float(rsu_range[0]), float(rsu_range[1])
                    p_j = self.rng.uniform(min(lo, hi), max(lo, hi))
                elif node_type == 'VEH' and veh_range is not None:
                    lo, hi = float(veh_range[0]), float(veh_range[1])
                    p_j = self.rng.uniform(min(lo, hi), max(lo, hi))

                if p_j is None:
                    if self.rng.random() < reliable_prob:
                        p_j = self.rng.uniform(p_rel_range[0], p_rel_range[1])
                    else:
                        p_j = self.rng.uniform(p_unrel_range[0], p_unrel_range[1])

                self.hidden_reliability[key] = float(p_j)
                self.beta_a[key] = prior_a
                self.beta_b[key] = prior_b

    def sample_outcome(self, node_key):
        """
        在远端子任务完成时采样: y ~ Bernoulli(p_j)

        TRUST_FAIL_SCOPE="v2v_only" 时 RSU 节点始终返回 True（不注入失败），
        以保证 RSU 固定高信誉语义一致。

        Returns:
            bool: True = 成功, False = 失败(需重试)，失败时 illegal_reason 保持 None
        """
        fail_scope = str(getattr(Cfg, 'TRUST_FAIL_SCOPE', 'all'))
        node_type = node_key[0] if isinstance(node_key, tuple) and len(node_key) >= 1 else None
        self.total_attempts += 1
        if fail_scope == 'v2v_only' and node_type == 'RSU':
            return True
        p_j = self.hidden_reliability.get(node_key, 1.0)
        success = bool(self.rng.random() < p_j)
        if not success:
            self.total_failures += 1
        return success

    def submit_evidence(self, current_step, node_key, success, delay_steps=None):
        """提交延迟证据

        delay_steps:
            Optional override for the evidence delay (in env steps). This is useful
            when coupling trust update latency to a chain proxy's confirmation delay.
        """
        if delay_steps is None:
            delay = getattr(Cfg, 'TRUST_DELAY_STEPS', 3)
        else:
            try:
                delay = int(delay_steps)
            except Exception:
                delay = getattr(Cfg, 'TRUST_DELAY_STEPS', 3)
        if delay < 0:
            delay = 0
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

    def get_trust_stats(self, node_key, z: float = None):
        """
        返回 (mean, uncertainty, lcb)。

        lcb 使用 Beta 后验的正态近似下界，供主线的可行性掩码与风险成本使用。
        """
        a = float(self.beta_a.get(node_key, 1.0))
        b = float(self.beta_b.get(node_key, 1.0))
        total = max(a + b, 1e-9)
        mean = a / total
        uncertainty = 1.0 / total
        var = (a * b) / max((total ** 2) * (total + 1.0), 1e-9)
        std = float(np.sqrt(max(var, 0.0)))
        if z is None:
            z = float(getattr(Cfg, "TRUST_LCB_Z", 1.0))
        lcb = float(np.clip(mean - float(z) * std, 0.0, 1.0))
        return float(mean), float(uncertainty), lcb

    def get_stats(self):
        """获取 episode 统计"""
        mal_count = sum(1 for v in self.is_malicious.values() if v)
        return {
            'trust_attempts': self.total_attempts,
            'trust_failures': self.total_failures,
            'trust_failure_rate': self.total_failures / max(self.total_attempts, 1),
            'trust_retry_count': len(self.retry_events),
            'malicious_count': mal_count,
        }
