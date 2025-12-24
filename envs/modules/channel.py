import numpy as np
from configs.config import SystemConfig as Cfg


class ChannelModel:
    """
    [物理信道模型]
    负责计算 V2I (Vehicle-to-Infrastructure) 和 V2V (Vehicle-to-Vehicle) 的真实传输速率。

    包含特性:
    1. 大尺度衰落 (Path Loss): 基于距离的对数衰减。
    2. 小尺度衰落 (Fading): Rician (V2I) 和 Rayleigh (V2V)。
    3. 干扰模型:
       - V2I: OFDMA 拥堵干扰 (当用户数 > RB 数时)。
       - V2V: 空间复用干扰 (所有同时发射的车辆互为干扰源)。
    """

    def __init__(self):
        # 1. 噪声功率 (Watts)
        self.noise_w = Cfg.dbm2watt(Cfg.NOISE_POWER_DBM)

        # 2. 路径损耗常数 Beta0 (Linear)
        # Beta0_dB 通常定义在 1m 处，例如 -30dB => 0.001
        # 如果 Config 中没有定义 BETA_0_DB，提供默认值
        beta0_db = getattr(Cfg, 'BETA_0_DB', -30)
        self.beta0 = 10 ** (beta0_db / 10.0)

        # 3. 莱斯衰落参数 (Rician K-factor)
        # 用于 V2I (视距概率高)
        k_db = getattr(Cfg, 'RICIAN_K_DB', 6.0)
        k_linear = 10 ** (k_db / 10.0)
        self.mu = np.sqrt(k_linear / (k_linear + 1))  # 主径分量 (LOS)
        self.sigma = np.sqrt(1 / (k_linear + 1))  # 散射分量 (NLOS)

        # 4. 资源块 (Resource Block) 设置
        # 假设每个 RB 带宽为 180 kHz (LTE 标准)
        # 如果 BW_V2I = 10MHz，大约有 50 个 RB
        self.rb_bandwidth = 180e3
        self.num_rbs = int(Cfg.BW_V2I / self.rb_bandwidth)

        # 干扰系数 (当用户过多时的额外惩罚)
        self.interference_factor = getattr(Cfg, 'INTERFERENCE_FACTOR', 1e-13)

    def compute_rates(self, vehicles, rsu_pos):
        """
        [批量计算] 计算当前这一帧所有活跃传输的真实速率
        用于 env.step() 中的状态更新。

        Args:
            vehicles: 车辆列表
            rsu_pos: RSU 坐标

        Returns:
            rates (dict): {vehicle_id: rate_bps}
        """
        rates = {}

        # ==========================================
        # A. V2I 通信 (OFDMA 模型)
        # ==========================================
        # 筛选出当前目标是 RSU 的车辆
        v2i_group = [v for v in vehicles if v.curr_target == 'RSU']

        if v2i_group:
            num_users = len(v2i_group)

            # 1. 资源分配 (带宽均分)
            # 如果用户少于 RB 数，每人分得更多带宽，上限通常受限于设备能力或协议
            # 这里简化模型: 带宽池完全均分
            eff_bandwidth = Cfg.BW_V2I / max(num_users, 1)

            # 2. 干扰计算 (Interference)
            # 如果用户数 > RB 数，发生 RB 碰撞/复用，引入干扰
            interference = 0.0
            if num_users > self.num_rbs:
                excess = num_users - self.num_rbs
                # 简单的线性干扰增长模型
                interference = excess * self.interference_factor

            # 3. 计算每个用户的速率
            for v in v2i_group:
                # 距离计算
                dist = np.linalg.norm(v.pos - rsu_pos)

                # 信道增益: PathLoss * Fading
                # V2I 使用 Rician Fading
                h_gain = self._path_loss(dist, Cfg.ALPHA_V2I) * self._rician_fading(1)[0]

                # 接收功率
                p_tx = Cfg.dbm2watt(v.tx_power_dbm)
                p_rx = p_tx * h_gain

                # SINR = P_rx / (Noise + Interference)
                sinr = p_rx / (self.noise_w + interference)

                # Shannon Capacity: R = B * log2(1 + SINR)
                rates[v.id] = eff_bandwidth * np.log2(1 + sinr)

        # ==========================================
        # B. V2V 通信 (全干扰模型)
        # ==========================================
        # 筛选: 目标是 int 类型 (即 Vehicle ID)
        v2v_senders = [v for v in vehicles if isinstance(v.curr_target, int)]

        # 构建配对列表: [(Sender, Receiver), ...]
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

            # 1. 信号强度 (Signal Power) - 对角线对应
            # 这里的距离是 Sender[i] -> Receiver[i]
            sig_dists = np.linalg.norm(tx_positions - rx_positions, axis=1)
            # V2V 使用 Rayleigh Fading (Rician K=0 时近似) 或更严重的衰落
            # 这里统一用 Rician 但可以通过参数调整，或直接简化为 Rayleigh
            # 假设 V2V 遮挡多，使用 PathLoss * Rayleigh
            h_sig = self._path_loss(sig_dists, Cfg.ALPHA_V2V) * self._rician_fading(n_pairs)
            signal_powers = tx_powers * h_sig

            # 2. 干扰矩阵 (Interference Matrix)
            # 计算所有 Sender[j] 到 Receiver[i] 的距离 (j != i)
            # dist_mat[i, j] = dist(Rx[i], Tx[j])
            dist_mat = np.linalg.norm(rx_positions[:, None, :] - tx_positions[None, :, :], axis=2)

            # 路径损耗矩阵
            pl_mat = self._path_loss(dist_mat, Cfg.ALPHA_V2V)

            # 干扰功率矩阵: P_tx[j] * PL[i, j]
            # 注意: 这里忽略了干扰链路的小尺度衰落，取平均值以节省计算
            int_power_mat = tx_powers[None, :] * pl_mat

            # 移除信号本身 (对角线置0)
            np.fill_diagonal(int_power_mat, 0.0)

            # 总干扰: 按行求和
            total_interference = np.sum(int_power_mat, axis=1)

            # 3. 计算速率
            # V2V 带宽通常是复用的 (Spatial Reuse)，假设独占或干扰受限
            # 这里假设所有 V2V 共享 BW_V2V 频段 (干扰受限系统)
            for i in range(n_pairs):
                sinr = signal_powers[i] / (self.noise_w + total_interference[i])
                rate = Cfg.BW_V2V * np.log2(1 + sinr)

                sender_id = valid_pairs[i][0].id
                rates[sender_id] = rate

        return rates

    def compute_one_rate(self, vehicle, target_pos, link_type='V2I'):
        """
        [单体预估] 计算假设场景下的速率
        用于:
        1. 奖励函数中的 CFT 估算 (Oracle)
        2. 观测空间中的 Rate 特征生成

        注意: 这是一个估计值，假设平均干扰或零干扰。
        """
        dist = np.linalg.norm(vehicle.pos - target_pos)
        p_tx = Cfg.dbm2watt(vehicle.tx_power_dbm)

        if link_type == 'V2I':
            # 假设分配到平均带宽 (例如假设同时有5人在用)
            est_users = 5
            bandwidth = Cfg.BW_V2I / est_users
            alpha = Cfg.ALPHA_V2I
            # 仅计算 Path Loss，忽略快衰落 (期望值)
            g = self._path_loss(dist, alpha)
            interference = 0.0  # 乐观估计
        else:  # V2V
            bandwidth = Cfg.BW_V2V
            alpha = Cfg.ALPHA_V2V
            g = self._path_loss(dist, alpha)
            # V2V 干扰通常较大，加一个底噪估计
            interference = self.noise_w * 10.0

        sinr = (p_tx * g) / (self.noise_w + interference)
        rate = bandwidth * np.log2(1 + sinr)

        return rate

    def _path_loss(self, dists, alpha):
        """
        路径损耗模型: Beta0 * d^(-alpha)
        """
        # 避免距离过近导致奇异值 (最小 1.0 米)
        dists = np.maximum(dists, 1.0)
        return self.beta0 * (dists ** (-alpha))

    def _rician_fading(self, n):
        """
        生成 n 个莱斯衰落样本 (|h|^2)
        H = (mu + sigma * randn_complex)
        Power Gain = |H|^2
        """
        # 复高斯噪声: CN(0, 1) -> 实部虚部各 sigma
        # 注意: random.randn 标准差是 1，这里要乘 sigma
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.7071

        h = self.mu + self.sigma * noise
        return np.abs(h) ** 2