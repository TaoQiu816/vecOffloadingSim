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

        # 3. V2V瑞利衰落参数（用于信号功率计算）
        # 注意：V2I不使用衰落，V2V使用瑞利衰落
        # 瑞利衰落：h ~ CN(0, 1)，|h|^2 服从指数分布
        # 每次计算时重新采样，模拟快变信道

    def compute_rates(self, vehicles, rsu_pos):
        """
        [批量计算] 计算当前这一帧所有活跃传输的真实速率
        """
        rates = {}

        # ==========================================
        # A. V2I 通信 (带宽竞争模型，无衰落)
        # ==========================================
        v2i_group = [v for v in vehicles if isinstance(v.curr_target, tuple) and v.curr_target[0] == 'RSU']
        # 兼容旧代码：单个RSU场景
        if len(v2i_group) == 0:
            v2i_group = [v for v in vehicles if v.curr_target == 'RSU']

        if v2i_group:
            num_users = len(v2i_group)

            # 带宽竞争模型: 总带宽被所有用户均分
            eff_bandwidth = Cfg.BW_V2I / max(num_users, 1)

            for v in v2i_group:
                # 距离计算
                dist = np.linalg.norm(v.pos - rsu_pos)

                # 信道增益: 只有路径损耗，无衰落（LoS链路）
                h_bar = self._path_loss(dist, Cfg.ALPHA_V2I)

                # 接收功率
                p_tx = Cfg.dbm2watt(v.tx_power_dbm)
                p_rx = p_tx * h_bar

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
            # V2V 使用瑞利衰落（每个step重新采样）
            # 大尺度路径损耗
            h_bar_sig = self._path_loss(sig_dists, Cfg.ALPHA_V2V)
            # 小尺度瑞利衰落（每次重新采样）
            h_rayleigh = self._rayleigh_fading(n_pairs)
            # 信号功率 = P_tx * h_bar * |h_rayleigh|^2
            signal_powers = tx_powers * h_bar_sig * h_rayleigh

            # 2. 干扰矩阵 (Interference Matrix)
            # dist_mat[i, j] = dist(Rx[i], Tx[j])
            dist_mat = np.linalg.norm(rx_positions[:, None, :] - tx_positions[None, :, :], axis=2)

            # 路径损耗矩阵（干扰只考虑大尺度路径损耗，无衰落）
            h_bar_interference = self._path_loss(dist_mat, Cfg.ALPHA_V2V)

            # 干扰功率矩阵: P_tx[j] * h_bar[i, j]（只有大尺度路径损耗）
            int_power_mat = tx_powers[None, :] * h_bar_interference

            # 移除信号本身 (对角线置0)
            np.fill_diagonal(int_power_mat, 0.0)

            # 总干扰: 按行求和（期望值E[I]）
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
            # V2I: 带宽竞争模型，无衰落
            est_users = max(Cfg.NUM_VEHICLES // 5, 1)
            bandwidth = Cfg.BW_V2I / est_users
            h_bar = self._path_loss(dist, Cfg.ALPHA_V2I)
            # V2I只有路径损耗，无衰落
            sinr = (p_tx * h_bar) / self.noise_w
            rate = bandwidth * np.log2(1 + sinr)
        else:
            # V2V: 包含瑞利衰落，全干扰模型
            bandwidth = Cfg.BW_V2V
            h_bar = self._path_loss(dist, Cfg.ALPHA_V2V)
            
            # 计算干扰（只考虑大尺度路径损耗）
            if active_tx_vehicles is not None and len(active_tx_vehicles) > 0:
                interference = self._compute_interference(
                    vehicle.pos, target_pos, active_tx_vehicles, p_tx
                )
            else:
                interference = self.noise_w * Cfg.V2V_INTERFERENCE_FACTOR
            
            # 信号功率包含瑞利衰落（每次重新采样）
            h_rayleigh = self._rayleigh_fading(1)[0]
            signal_power = p_tx * h_bar * h_rayleigh
            
            sinr = signal_power / (self.noise_w + interference)
            rate = bandwidth * np.log2(1 + sinr)

        return rate

    def _compute_interference(self, rx_pos, tx_pos, active_tx_vehicles, p_tx_self):
        """
        计算V2V接收端的实际干扰功率（期望值E[I]）
        
        注意：干扰功率只考虑大尺度路径损耗，不包含小尺度衰落

        Args:
            rx_pos: 接收车辆位置
            tx_pos: 期望发送者位置
            active_tx_vehicles: 当前环境中所有活跃的V2V发射车辆列表
            p_tx_self: 自身发射功率(W)

        Returns:
            干扰功率(W) - 期望值E[I]
        """
        total_interference = 0.0

        for other_tx in active_tx_vehicles:
            if other_tx is None:
                continue

            dist_interferer = np.linalg.norm(rx_pos - other_tx.pos)

            if dist_interferer < 1.0:
                continue

            p_tx_other = Cfg.dbm2watt(other_tx.tx_power_dbm)
            # 干扰只考虑大尺度路径损耗（h_bar），不包含衰落
            h_bar = self._path_loss(dist_interferer, Cfg.ALPHA_V2V)
            interference_power = p_tx_other * h_bar

            if dist_interferer <= Cfg.V2V_RANGE:
                total_interference += interference_power

        return total_interference
    
    def compute_reliability(self, vehicle, target_pos, link_type='V2I', active_tx_vehicles=None):
        """
        计算传输可靠性P_succ（仅用于奖励函数）
        
        V2I: P_succ = 1.0（默认，除非超时）
        V2V: P_succ = exp(-γ_th * (N_0 + E[I]) / (P_tx * h_bar))
        
        Args:
            vehicle: 发射车辆
            target_pos: 目标位置
            link_type: 'V2I' 或 'V2V'
            active_tx_vehicles: 活跃V2V发射车辆列表（用于计算干扰）
        
        Returns:
            float: 可靠性概率P_succ
        """
        if link_type == 'V2I':
            return 1.0  # V2I默认可靠
        
        # V2V可靠性计算
        dist = np.linalg.norm(vehicle.pos - target_pos)
        p_tx = Cfg.dbm2watt(vehicle.tx_power_dbm)
        h_bar = self._path_loss(dist, Cfg.ALPHA_V2V)
        
        # 计算干扰（期望值E[I]）
        if active_tx_vehicles is not None and len(active_tx_vehicles) > 0:
            interference = self._compute_interference(
                target_pos, vehicle.pos, active_tx_vehicles, p_tx
            )
        else:
            interference = self.noise_w * Cfg.V2V_INTERFERENCE_FACTOR
        
        # 可靠性公式：P_succ = exp(-γ_th * (N_0 + E[I]) / (P_tx * h_bar))
        gamma_th = getattr(Cfg, 'V2V_GAMMA_TH', 2.0)
        numerator = gamma_th * (self.noise_w + interference)
        denominator = p_tx * h_bar
        
        if denominator <= 0:
            return 0.0
        
        p_succ = np.exp(-numerator / denominator)
        return np.clip(p_succ, 0.0, 1.0)

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

        # V2V信号功率：包含大尺度路径损耗和瑞利衰落
        h_bar = self._path_loss(dist, Cfg.ALPHA_V2V)
        h_rayleigh = self._rayleigh_fading(1)[0]
        signal_power = p_tx * h_bar * h_rayleigh

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

    def _rayleigh_fading(self, n):
        """
        生成 n 个瑞利衰落样本（用于V2V信号功率计算）
        
        瑞利衰落：h ~ CN(0, 1)，|h|^2 服从指数分布
        每次调用都重新采样，模拟快变信道
        
        Args:
            n: 样本数量
        
        Returns:
            np.array: |h|^2 的值（形状为(n,)）
        """
        # 复高斯噪声：实部和虚部都是标准正态分布
        # h = (X + jY) / sqrt(2)，其中X, Y ~ N(0, 1)
        # |h|^2 = (X^2 + Y^2) / 2，服从指数分布
        real_part = np.random.randn(n)
        imag_part = np.random.randn(n)
        h_rayleigh = (real_part + 1j * imag_part) / np.sqrt(2.0)
        return np.abs(h_rayleigh) ** 2