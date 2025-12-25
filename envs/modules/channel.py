import numpy as np
from configs.config import SystemConfig as Cfg


class ChannelModel:
    """
    [物理信道模型] - Final Revised Version
    负责计算 V2I (Vehicle-to-Infrastructure) 和 V2V (Vehicle-to-Vehicle) 的真实传输速率。

    修改记录:
    1. [V2I Model] 移除了不稳定的线性干扰模型，完全采用带宽分时复用 (Time Division / Bandwidth Sharing) 模型。
    2. [Observation] compute_one_rate 不再硬编码用户数，而是基于系统车辆总数进行动态悲观估算。
    3. [Interface] 支持 curr_time 参数输入。
    """

    def __init__(self):
        # 1. 噪声功率 (Watts)
        self.noise_w = Cfg.dbm2watt(Cfg.NOISE_POWER_DBM)

        # 2. 路径损耗常数 Beta0
        beta0_db = getattr(Cfg, 'BETA_0_DB', -30)
        self.beta0 = 10 ** (beta0_db / 10.0)

        # 3. 莱斯衰落参数 (Rician K-factor)
        k_db = getattr(Cfg, 'RICIAN_K_DB', 6.0)
        k_linear = 10 ** (k_db / 10.0)
        self.mu = np.sqrt(k_linear / (k_linear + 1))  # 主径分量
        self.sigma = np.sqrt(1 / (k_linear + 1))  # 散射分量

    def compute_rates(self, vehicles, rsu_pos):
        """
        [批量计算] 计算当前这一帧所有活跃传输的真实速率
        """
        rates = {}

        # ==========================================
        # A. V2I 通信 (OFDMA / 时分复用模型)
        # ==========================================
        v2i_group = [v for v in vehicles if v.curr_target == 'RSU']

        if v2i_group:
            num_users = len(v2i_group)

            # [关键修改] 拥堵模型: 纯带宽分享
            # 无论用户多少，总带宽 Cfg.BW_V2I 被所有用户均分
            # 这比之前的 "excess * 1e-13" 干扰模型更稳定，且符合调度物理规律
            eff_bandwidth = Cfg.BW_V2I / max(num_users, 1)

            for v in v2i_group:
                # 距离计算
                dist = np.linalg.norm(v.pos - rsu_pos)

                # 信道增益: PathLoss * Rician Fading
                # _rician_fading(1) 返回数组，取 [0]
                h_gain = self._path_loss(dist, Cfg.ALPHA_V2I) * self._rician_fading(1)[0]

                # 接收功率
                p_tx = Cfg.dbm2watt(v.tx_power_dbm)
                p_rx = p_tx * h_gain

                # SINR = P_rx / Noise (V2I 正交，忽略内部干扰)
                sinr = p_rx / self.noise_w

                # Shannon Capacity
                rates[v.id] = eff_bandwidth * np.log2(1 + sinr)

        # ==========================================
        # B. V2V 通信 (全干扰模型)
        # ==========================================
        v2v_senders = [v for v in vehicles if isinstance(v.curr_target, int)]

        # 构建配对列表
        valid_pairs = []
        veh_map = {v.id: v for v in vehicles}

        for tx_veh in v2v_senders:
            rx_id = tx_veh.curr_target
            rx_veh = veh_map.get(rx_id)
            if rx_veh:
                valid_pairs.append((tx_veh, rx_veh))

        if valid_pairs:
            n_pairs = len(valid_pairs)

            # 提取位置与功率向量
            tx_positions = np.array([p[0].pos for p in valid_pairs])  # [N, 2]
            rx_positions = np.array([p[1].pos for p in valid_pairs])  # [N, 2]
            tx_powers = np.array([Cfg.dbm2watt(p[0].tx_power_dbm) for p in valid_pairs])  # [N]

            # 1. 信号强度 (Signal Power)
            sig_dists = np.linalg.norm(tx_positions - rx_positions, axis=1)
            # V2V 使用 Rician 或 Rayleigh (视 Config 而定，此处复用 Rician 逻辑但参数不同)
            # 使用 ALPHA_V2V (通常较大，如 3.0)
            h_sig = self._path_loss(sig_dists, Cfg.ALPHA_V2V) * self._rician_fading(n_pairs)
            signal_powers = tx_powers * h_sig

            # 2. 干扰矩阵 (Interference Matrix)
            # dist_mat[i, j] = dist(Rx[i], Tx[j])
            dist_mat = np.linalg.norm(rx_positions[:, None, :] - tx_positions[None, :, :], axis=2)

            # 路径损耗矩阵
            pl_mat = self._path_loss(dist_mat, Cfg.ALPHA_V2V)

            # 干扰功率矩阵: P_tx[j] * PL[i, j]
            int_power_mat = tx_powers[None, :] * pl_mat

            # 移除信号本身 (对角线置0)
            np.fill_diagonal(int_power_mat, 0.0)

            # 总干扰: 按行求和
            total_interference = np.sum(int_power_mat, axis=1)

            # 3. 计算速率 (V2V 共享带宽，干扰受限)
            for i in range(n_pairs):
                # V2V 带宽通常固定 (因为是空间复用，主要受干扰限制)
                sinr = signal_powers[i] / (self.noise_w + total_interference[i])
                rate = Cfg.BW_V2V * np.log2(1 + sinr)

                sender_id = valid_pairs[i][0].id
                rates[sender_id] = rate

        return rates

    def compute_one_rate(self, vehicle, target_pos, link_type='V2I', curr_time=None):
        """
        [单体预估] 计算假设场景下的速率
        用于 Observation 特征生成和 CFT 估算。
        """
        dist = np.linalg.norm(vehicle.pos - target_pos)
        p_tx = Cfg.dbm2watt(vehicle.tx_power_dbm)

        if link_type == 'V2I':
            # [关键修改] 动态拥堵估算
            # 不再硬编码 est_users=5，而是基于总车辆数的一个比例 (例如 50% 的车在竞争)
            # 这样当 NUM_VEHICLES = 20 时，估算带宽会更接近真实拥堵情况
            est_users = max(Cfg.NUM_VEHICLES // 5, 1)
            bandwidth = Cfg.BW_V2I / est_users
            alpha = Cfg.ALPHA_V2I

            # V2I 主要是带宽受限，干扰通常由正交性解决 (或忽略)
            interference = 0.0
        else:
            # V2V (Vehicle-to-Vehicle)
            bandwidth = Cfg.BW_V2V
            alpha = Cfg.ALPHA_V2V

            # V2V 主要是干扰受限
            # 给定一个悲观的干扰估计 (假设有邻居在同时发)
            interference = self.noise_w * 10.0

        # 计算平均路损 (不加快衰落，作为期望值)
        g = self._path_loss(dist, alpha)

        sinr = (p_tx * g) / (self.noise_w + interference)
        rate = bandwidth * np.log2(1 + sinr)

        return rate

    def _path_loss(self, dists, alpha):
        """
        路径损耗模型: Beta0 * d^(-alpha)
        """
        # 物理保护: 最小距离 1.0m，防止无穷大
        dists = np.maximum(dists, 1.0)
        return self.beta0 * (dists ** (-alpha))

    def _rician_fading(self, n):
        """
        生成 n 个莱斯衰落样本
        """
        # 复高斯噪声
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.7071
        h = self.mu + self.sigma * noise
        return np.abs(h) ** 2