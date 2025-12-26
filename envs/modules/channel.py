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
        # 应修正为标准莱斯分布
        k_linear = 10 ** (k_db / 10.0)
        self.mu = np.sqrt(k_linear)  # 直射路径幅度
        self.sigma = 1.0  # 散射路径标准差

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

    def compute_one_rate(self, vehicle, target_pos, link_type='V2I', curr_time=None, active_tx_vehicles=None):
        """
        [单体预估] 计算假设场景下的速率
        用于 Observation 特征生成和 CFT 估算。

        改进点:
        - 支持传入活跃发射车辆列表，实现动态干扰估算
        - 当无法获取活跃发射列表时，使用保守估计
        """
        dist = np.linalg.norm(vehicle.pos - target_pos)
        p_tx = Cfg.dbm2watt(vehicle.tx_power_dbm)

        if link_type == 'V2I':
            est_users = max(Cfg.NUM_VEHICLES // 5, 1)
            bandwidth = Cfg.BW_V2I / est_users
            alpha = Cfg.ALPHA_V2I
            interference = 0.0
        else:
            bandwidth = Cfg.BW_V2V
            alpha = Cfg.ALPHA_V2V

            if active_tx_vehicles is not None and len(active_tx_vehicles) > 0:
                interference = self._compute_interference(
                    vehicle.pos, target_pos, active_tx_vehicles, p_tx
                )
            else:
                interference = self.noise_w * Cfg.V2V_INTERFERENCE_FACTOR

        g = self._path_loss(dist, alpha)
        sinr = (p_tx * g) / (self.noise_w + interference)
        rate = bandwidth * np.log2(1 + sinr)

        return rate

    def _compute_interference(self, rx_pos, tx_pos, active_tx_vehicles, p_tx_self):
        """
        计算V2V接收端的实际干扰功率

        Args:
            rx_pos: 接收车辆位置
            tx_pos: 期望发送者位置
            active_tx_vehicles: 当前环境中所有活跃的V2V发射车辆列表
            p_tx_self: 自身发射功率(W)

        Returns:
            干扰功率(W)
        """
        total_interference = 0.0

        for other_tx in active_tx_vehicles:
            if other_tx is None:
                continue

            dist_interferer = np.linalg.norm(rx_pos - other_tx.pos)

            if dist_interferer < 1.0:
                continue

            p_tx_other = Cfg.dbm2watt(other_tx.tx_power_dbm)
            g = self._path_loss(dist_interferer, Cfg.ALPHA_V2V)
            interference_power = p_tx_other * g

            if dist_interferer <= Cfg.V2V_RANGE:
                total_interference += interference_power

        return total_interference

    def get_active_v2v_transmitters(self, vehicles):
        """
        获取当前环境中所有活跃的V2V发射车辆列表

        Args:
            vehicles: 环境中的所有车辆

        Returns:
            活跃V2V发射车辆的列表
        """
        active_transmitters = []
        for v in vehicles:
            if isinstance(v.curr_target, int):
                active_transmitters.append(v)
        return active_transmitters

    def compute_v2v_rate_with_interference(self, vehicles, tx_veh, rx_veh):
        """
        计算指定V2V链路的实际速率（考虑所有活跃干扰）

        Args:
            vehicles: 环境中的所有车辆
            tx_veh: 发射车辆
            rx_veh: 接收车辆

        Returns:
            实际传输速率(bps)
        """
        p_tx = Cfg.dbm2watt(tx_veh.tx_power_dbm)

        dist = np.linalg.norm(tx_veh.pos - rx_veh.pos)
        if dist < 1.0:
            dist = 1.0

        h = self._path_loss(dist, Cfg.ALPHA_V2V) * self._rician_fading(1)[0]
        signal_power = p_tx * h

        interference = self._compute_interference(
            rx_veh.pos, tx_veh.pos, vehicles, p_tx
        )

        sinr = signal_power / (self.noise_w + interference)
        rate = Cfg.BW_V2V * np.log2(1 + sinr)

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