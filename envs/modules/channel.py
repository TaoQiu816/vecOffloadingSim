import numpy as np
from configs.config import SystemConfig as Cfg


class ChannelModel:
    """
    [物理信道模型] - 主线固定:
    - V2I/V2R: RB_SINR（RSU 内正交调度 + 跨 RSU 同 RB 干扰）
    - V2V: resource pool / RB reuse + same-RB interference
    """

    def __init__(self):
        noise_psd_dbm_hz = Cfg.NOISE_POWER_DENSITY_DBM + Cfg.NOISE_FIGURE
        self.noise_psd_w_hz = Cfg.dbm2watt(noise_psd_dbm_hz)

        beta0_db = getattr(Cfg, 'BETA_0_DB', -30)
        self.beta0 = 10 ** (beta0_db / 10.0)

        self.use_block_fading = getattr(Cfg, "USE_BLOCK_FADING", True)
        self.v2v_fading_cache = {}
        self._v2v_cache_slot = None

        # RB 干扰模型参数
        self.num_rb = getattr(Cfg, 'V2V_NUM_RB', 4)
        self.bw_per_rb = getattr(Cfg, 'V2V_BW_PER_RB', Cfg.BW_V2V / self.num_rb)
        self.v2i_num_rb = int(max(getattr(Cfg, "V2I_NUM_RB", 0), 0))
        if self.v2i_num_rb <= 0:
            rb_bw = float(max(getattr(Cfg, "V2I_RB_BW_HZ", 180e3), 1.0))
            self.v2i_num_rb = max(int(round(float(Cfg.BW_V2I) / rb_bw)), 1)
        self.v2i_bw_per_rb = float(Cfg.BW_V2I) / float(max(self.v2i_num_rb, 1))
        self.v2i_ici_enabled = bool(getattr(Cfg, "V2I_ICI_ENABLED", False))
        self.v2i_reuse_factor = int(max(getattr(Cfg, "V2I_FREQ_REUSE_FACTOR", 1), 1))
        self.tie_rng = np.random.default_rng(getattr(Cfg, "SEED", None))

        # 每步干扰统计（供 env 读取）
        self.last_v2v_stats = self._empty_v2v_stats()
        self.last_v2i_stats = self._empty_v2i_stats()

    def _empty_v2v_stats(self):
        return {
            'sinr_values': [],               # 所有 link 的 SINR (linear)
            'i_caused': {},                  # sender_id -> 该 link 对同 RB 其他接收端干扰
            'i_total': {},                   # sender_id -> 该 link 接收端总干扰
            'i_caused_input': {},            # sender_id -> 仅 INPUT_TX 造成的可控干扰
            'i_total_input': {},             # sender_id -> 仅 INPUT_TX 受到的总干扰
            'rb_occupancy': np.zeros(self.num_rb, dtype=int),
            'rb_assignment': {},             # sender_id -> rb_id
        }

    def _empty_v2i_stats(self):
        return {
            'sinr_values': [],
            'i_caused': {},                  # sender_id -> 该链路对其他接收端干扰
            'i_total': {},                   # sender_id -> 该链路接收端总干扰
            'i_caused_input': {},            # sender_id -> 仅 INPUT_TX 造成的可控干扰
            'i_total_input': {},             # sender_id -> 仅 INPUT_TX 受到的总干扰
            'rb_occupancy': np.zeros(self.v2i_num_rb, dtype=int),
            'rb_assignment': {},             # sender_id -> [rb ids]
        }

    # ------------------------------------------------------------------
    # 显式链路列表驱动的 V2V RB-SINR 计算（不依赖 curr_target）
    # ------------------------------------------------------------------
    def compute_v2v_rb_sinr(self, v2v_links):
        """
        对显式 V2V 链路列表做 RB 确定性分配 + per-RB SINR + I_caused。

        Args:
            v2v_links: list of dict, 每条需含:
                sender_id, tx_pos (np.array), rx_pos (np.array), power_w (float)
        Returns:
            v2v_rates: dict  sender_id -> rate (bps)
            （同时更新 self.last_v2v_stats）
        """
        # 重置统计
        self.last_v2v_stats = self._empty_v2v_stats()
        v2v_rates = {}
        if not v2v_links:
            return v2v_rates

        n = len(v2v_links)
        tx_positions = np.array([lk['tx_pos'] for lk in v2v_links])
        rx_positions = np.array([lk['rx_pos'] for lk in v2v_links])
        tx_powers = np.array([lk['power_w'] for lk in v2v_links])

        sig_dists = np.linalg.norm(tx_positions - rx_positions, axis=1)
        h_bar_sig = self._path_loss(sig_dists, Cfg.PL_BETA_V2V)
        if self.use_block_fading:
            signal_powers = tx_powers * h_bar_sig * self._rayleigh_fading(n)
        else:
            signal_powers = tx_powers * h_bar_sig

        # 干扰路径损耗矩阵 G(d_{mj})
        dist_mat = np.linalg.norm(
            rx_positions[:, None, :] - tx_positions[None, :, :], axis=2
        )
        g_mat = self._path_loss(dist_mat, Cfg.PL_BETA_V2V)

        # 子信道分配：
        # 1) 先选当前占用最少的子信道
        # 2) 若并列，选预测增量干扰最小的子信道
        # 3) 若仍并列，随机打破平局
        w = self._path_loss(sig_dists, Cfg.PL_BETA_V2V)
        order = np.argsort(-w)
        rb_assign = np.full(n, -1, dtype=int)
        rb_links_list = [[] for _ in range(self.num_rb)]
        for li in order:
            occupancies = np.array([len(links) for links in rb_links_list], dtype=np.int64)
            min_occ = int(np.min(occupancies)) if occupancies.size > 0 else 0
            candidate_rbs = [c for c in range(self.num_rb) if occupancies[c] == min_occ]
            best_cost = np.inf
            best_rbs = []
            for c in range(self.num_rb):
                if c not in candidate_rbs:
                    continue
                cost = 0.0
                for m in rb_links_list[c]:
                    cost += tx_powers[li] * g_mat[m, li] + tx_powers[m] * g_mat[li, m]
                if cost < best_cost:
                    best_cost = cost
                    best_rbs = [c]
                elif np.isclose(cost, best_cost, rtol=1e-9, atol=1e-12):
                    best_rbs.append(c)
            if not best_rbs:
                best_rbs = candidate_rbs if candidate_rbs else [0]
            best_rb = int(self.tie_rng.choice(best_rbs))
            rb_assign[li] = best_rb
            rb_links_list[best_rb].append(li)

        noise_rb = self._noise_power(self.bw_per_rb)
        sinr_arr = np.zeros(n)
        i_total_arr = np.zeros(n)
        i_caused_arr = np.zeros(n)

        for i in range(n):
            same = [j for j in range(n) if rb_assign[j] == rb_assign[i] and j != i]
            interf = sum(tx_powers[j] * g_mat[i, j] for j in same)
            i_total_arr[i] = interf
            sinr_arr[i] = signal_powers[i] / (noise_rb + interf)
            i_caused_arr[i] = sum(tx_powers[i] * g_mat[j, i] for j in same)

        rates_vec = self.bw_per_rb * np.log2(1 + sinr_arr)

        rb_occ = np.zeros(self.num_rb, dtype=int)
        for i in range(n):
            sid = v2v_links[i]['sender_id']
            tx_kind = str(v2v_links[i].get("tx_kind", "INPUT")).upper()
            rb_occ[rb_assign[i]] += 1
            v2v_rates[sid] = float(v2v_rates.get(sid, 0.0) + rates_vec[i])
            self.last_v2v_stats['rb_assignment'][sid] = int(rb_assign[i])
            self.last_v2v_stats['i_caused'][sid] = float(self.last_v2v_stats['i_caused'].get(sid, 0.0) + i_caused_arr[i])
            self.last_v2v_stats['i_total'][sid] = float(self.last_v2v_stats['i_total'].get(sid, 0.0) + i_total_arr[i])
            if tx_kind == "INPUT":
                self.last_v2v_stats['i_caused_input'][sid] = float(
                    self.last_v2v_stats['i_caused_input'].get(sid, 0.0) + i_caused_arr[i]
                )
                self.last_v2v_stats['i_total_input'][sid] = float(
                    self.last_v2v_stats['i_total_input'].get(sid, 0.0) + i_total_arr[i]
                )
        self.last_v2v_stats['sinr_values'] = sinr_arr.tolist()
        self.last_v2v_stats['rb_occupancy'] = rb_occ
        return v2v_rates

    # ------------------------------------------------------------------
    # 显式链路列表驱动的 V2I 批量速率计算
    # ------------------------------------------------------------------
    def compute_v2i_rates(self, v2i_links, rsu_pos_map=None, rsu_pos_default=None):
        """
        Args:
            v2i_links: list of dict, 含 sender_id, tx_pos, rsu_id, power_w
            rsu_pos_map: {rsu_id: position}
            rsu_pos_default: 默认 RSU 位置
        Returns:
            dict  sender_id -> rate
        """
        rates = {}
        self.last_v2i_stats = self._empty_v2i_stats()
        if not v2i_links:
            return rates
        # RB_SINR: intra-cell orthogonal scheduling, inter-cell ICI on reused RBs.
        groups = {}
        for lk in v2i_links:
            groups.setdefault(int(lk['rsu_id']), []).append(lk)
        num_rb = int(max(getattr(Cfg, "V2I_NUM_RB", self.v2i_num_rb), 1))
        bw_rb = float(Cfg.BW_V2I) / float(num_rb)
        noise_rb = self._noise_power(bw_rb)
        ici_enabled = bool(getattr(Cfg, "V2I_ICI_ENABLED", self.v2i_ici_enabled))
        reuse_factor = int(max(getattr(Cfg, "V2I_FREQ_REUSE_FACTOR", self.v2i_reuse_factor), 1))

        # Deterministic orthogonal scheduler per RSU:
        # - Each (round, RB) has at most one user in the same RSU (intra-cell orthogonal).
        # - All RBs are utilized each round.
        # - Users can receive multiple RBs in one step when num_rb > n_users.
        # - If n_users > num_rb, users are time-shared across rounds.
        sched = []
        rsu_rounds = {}
        for rid, grp in groups.items():
            grp_sorted = sorted(grp, key=lambda x: int(x.get("sender_id", -1)))
            n_users = len(grp_sorted)
            rounds = int(np.ceil(n_users / float(max(num_rb, 1))))
            rounds = max(rounds, 1)
            rsu_rounds[rid] = rounds
            for rnd in range(rounds):
                for rb in range(num_rb):
                    # Round-robin mapping over users; when num_rb > n_users this
                    # naturally allocates multiple RBs per user in a round.
                    user_idx = int((rnd * num_rb + rb) % n_users)
                    lk = grp_sorted[user_idx]
                    sched.append((rid, rb, rnd, lk))
                    self.last_v2i_stats['rb_occupancy'][rb] += 1
                    sid = lk["sender_id"]
                    if sid not in self.last_v2i_stats['rb_assignment']:
                        self.last_v2i_stats['rb_assignment'][sid] = []
                    self.last_v2i_stats['rb_assignment'][sid].append(int(rb))

        # Build fast lookup for cross-cell interferers by (rb, round, reuse_group).
        buckets = {}
        for rid, rb, rnd, lk in sched:
            reuse_group = rid % reuse_factor
            buckets.setdefault((rb, rnd, reuse_group), []).append((rid, lk))

        for rid, rb, rnd, lk in sched:
            rsu_p = (rsu_pos_map or {}).get(rid, rsu_pos_default)
            if rsu_p is None:
                continue
            tx_pos = np.asarray(lk["tx_pos"], dtype=float)
            p_sig = float(lk["power_w"])
            d_sig = np.linalg.norm(tx_pos - np.asarray(rsu_p, dtype=float))
            h_sig = self._path_loss(max(d_sig, 1.0), Cfg.PL_BETA_V2I)
            sig_w = p_sig * h_sig

            interf_w = 0.0
            if ici_enabled:
                reuse_group = rid % reuse_factor
                candidates = buckets.get((rb, rnd, reuse_group), [])
                for other_rid, other_lk in candidates:
                    if other_rid == rid:
                        continue
                    tx_o = np.asarray(other_lk["tx_pos"], dtype=float)
                    p_o = float(other_lk["power_w"])
                    d_o = np.linalg.norm(tx_o - np.asarray(rsu_p, dtype=float))
                    h_o = self._path_loss(max(d_o, 1.0), Cfg.PL_BETA_V2I)
                    interf_w += p_o * h_o

            sinr = sig_w / max(noise_rb + interf_w, 1e-12)
            sender_id = lk["sender_id"]
            tx_kind = str(lk.get("tx_kind", "INPUT")).upper()
            reuse_group = rid % reuse_factor
            caused_w = 0.0
            if ici_enabled:
                candidates = buckets.get((rb, rnd, reuse_group), [])
                for other_rid, other_lk in candidates:
                    if other_rid == rid:
                        continue
                    rsu_o = (rsu_pos_map or {}).get(other_rid, rsu_pos_default)
                    if rsu_o is None:
                        continue
                    d_to_other = np.linalg.norm(tx_pos - np.asarray(rsu_o, dtype=float))
                    h_to_other = self._path_loss(max(d_to_other, 1.0), Cfg.PL_BETA_V2I)
                    caused_w += p_sig * h_to_other

            self.last_v2i_stats['sinr_values'].append(float(sinr))
            self.last_v2i_stats['i_total'][sender_id] = float(
                self.last_v2i_stats['i_total'].get(sender_id, 0.0) + interf_w
            )
            self.last_v2i_stats['i_caused'][sender_id] = float(
                self.last_v2i_stats['i_caused'].get(sender_id, 0.0) + caused_w
            )
            if tx_kind == "INPUT":
                self.last_v2i_stats['i_total_input'][sender_id] = float(
                    self.last_v2i_stats['i_total_input'].get(sender_id, 0.0) + interf_w
                )
                self.last_v2i_stats['i_caused_input'][sender_id] = float(
                    self.last_v2i_stats['i_caused_input'].get(sender_id, 0.0) + caused_w
                )
            # Round-based orthogonal time sharing in overloaded cells.
            time_share = 1.0 / float(max(rsu_rounds.get(rid, 1), 1))
            rates[sender_id] = float(rates.get(sender_id, 0.0) + time_share * bw_rb * np.log2(1.0 + sinr))

        return rates

    # ------------------------------------------------------------------
    # 旧接口保留兼容（通过 curr_target 识别 sender）
    # ------------------------------------------------------------------
    def compute_rates(self, vehicles, rsu_pos, rsus=None):
        """
        [批量计算] 旧接口 — 通过 vehicle.curr_target 识别活跃传输。
        新代码应优先使用 compute_v2v_rb_sinr / compute_v2i_rates。
        路损口径统一沿用 PL_BETA_*（由子接口内部实现）。
        """
        rates = {}
        rsu_pos_map = {}
        if rsus is not None:
            for rsu in rsus:
                rsu_pos_map[rsu.id] = rsu.position

        # A. V2I
        v2i_group = [v for v in vehicles if isinstance(v.curr_target, tuple) and v.curr_target[0] == 'RSU']
        if not v2i_group:
            v2i_group = [v for v in vehicles if v.curr_target == 'RSU']
        if v2i_group:
            v2i_links = []
            for v in v2i_group:
                rid = v.curr_target[1] if isinstance(v.curr_target, tuple) and len(v.curr_target) > 1 else -1
                v2i_links.append({
                    "sender_id": int(v.id),
                    "tx_pos": np.asarray(v.pos, dtype=float),
                    "rsu_id": int(rid),
                    "power_w": Cfg.dbm2watt(v.tx_power_dbm),
                })
            rates.update(self.compute_v2i_rates(v2i_links, rsu_pos_map=rsu_pos_map, rsu_pos_default=rsu_pos))

        # B. V2V — 委托新接口
        v2v_links = []
        veh_map = {v.id: v for v in vehicles}
        for v in vehicles:
            if isinstance(v.curr_target, int):
                rx = veh_map.get(v.curr_target)
                if rx:
                    v2v_links.append({
                        'sender_id': v.id, 'tx_pos': v.pos, 'rx_pos': rx.pos,
                        'power_w': Cfg.dbm2watt(v.tx_power_dbm),
                    })
        v2v_rates = self.compute_v2v_rb_sinr(v2v_links)
        rates.update(v2v_rates)
        return rates

    def _assign_rb_deterministic(self, pairs, tx_powers, g_mat, n_pairs):
        """
        确定性 RB 分配：
        1) 按信号增益 w_ij = G(d_ij) 从大到小排序
        2) 贪心放入使增量干扰代价 ΔC(c) 最小的 RB
        """
        # w_ij = 信号路径增益（越大越优先分配）
        sig_dists = np.array([
            np.linalg.norm(pairs[i][0].pos - pairs[i][1].pos)
            for i in range(n_pairs)
        ])
        w = self._path_loss(sig_dists, Cfg.PL_BETA_V2V)
        order = np.argsort(-w)  # 从大到小

        rb_assign = np.full(n_pairs, -1, dtype=int)
        rb_links = [[] for _ in range(self.num_rb)]  # 每 RB 已分配的 link indices

        for link_i in order:
            best_rb = 0
            best_cost = np.inf
            for c in range(self.num_rb):
                cost = 0.0
                # link_i 加入 RB c 的增量干扰代价
                for m in rb_links[c]:
                    # link_i 对 m 的接收端的干扰
                    cost += tx_powers[link_i] * g_mat[m, link_i]
                    # m 对 link_i 的接收端的干扰
                    cost += tx_powers[m] * g_mat[link_i, m]
                if cost < best_cost:
                    best_cost = cost
                    best_rb = c
            rb_assign[link_i] = best_rb
            rb_links[best_rb].append(link_i)

        return rb_assign

    def compute_one_rate(self, vehicle, target_pos, link_type='V2I', curr_time=None, active_tx_vehicles=None, v2i_user_count=None, power_dbm_override=None):
        """
        [单体预估] 计算假设场景下的速率
        用于 Observation 特征生成和 CFT 估算。

        改进点:
        - 支持传入活跃发射车辆列表，实现动态干扰估算
        - 当无法获取活跃发射列表时，使用保守估计
        - [新增] power_dbm_override: 允许显式覆盖发射功率（用于EDGE传输使用固定最大功率）
        """
        dist = np.linalg.norm(vehicle.pos - target_pos)
        # [功率口径] 优先使用覆盖功率（EDGE用），否则使用vehicle属性（INPUT用）
        p_tx = Cfg.dbm2watt(power_dbm_override if power_dbm_override is not None else vehicle.tx_power_dbm)

        if link_type == 'V2I':
            # RB-SINR estimate for observation / diagnostics:
            # intra-cell orthogonal RB scheduling + probabilistic ICI.
            if v2i_user_count is not None:
                est_users = max(int(v2i_user_count), 1)
            elif active_tx_vehicles is not None:
                est_users = max(len(active_tx_vehicles), 1)
            else:
                est_users = max(Cfg.NUM_VEHICLES // 5, 1)
            num_rb = int(max(getattr(Cfg, "V2I_NUM_RB", self.v2i_num_rb), 1))
            bw_rb = float(Cfg.BW_V2I) / float(num_rb)
            rounds = max(int(np.ceil(est_users / float(max(num_rb, 1)))), 1)
            time_share = 1.0 / float(rounds)
            avg_rb_per_user = float(num_rb) / float(max(est_users, 1))

            noise_w = self._noise_power(bw_rb)
            h_bar = self._path_loss(dist, Cfg.PL_BETA_V2I)
            signal_w = p_tx * h_bar

            interf_w = 0.0
            if bool(getattr(Cfg, "V2I_ICI_ENABLED", self.v2i_ici_enabled)) and active_tx_vehicles is not None:
                reuse_factor = int(max(getattr(Cfg, "V2I_FREQ_REUSE_FACTOR", self.v2i_reuse_factor), 1))
                same_rb_prob = 1.0 / float(max(num_rb, 1))
                same_reuse_prob = 1.0 / float(reuse_factor)
                p_overlap = same_rb_prob * same_reuse_prob
                for veh in active_tx_vehicles:
                    if veh is None or getattr(veh, "id", None) == getattr(vehicle, "id", None):
                        continue
                    d_int = np.linalg.norm(np.asarray(veh.pos, dtype=float) - np.asarray(target_pos, dtype=float))
                    h_int = self._path_loss(max(d_int, 1.0), Cfg.PL_BETA_V2I)
                    interf_w += Cfg.dbm2watt(getattr(veh, "tx_power_dbm", Cfg.TX_POWER_MIN_DBM)) * h_int
                interf_w *= p_overlap

            sinr = signal_w / max(noise_w + interf_w, 1e-12)
            rate = time_share * avg_rb_per_user * bw_rb * np.log2(1 + sinr)
        else:
            # V2V: per-RB SINR（单链路预估使用单 RB 带宽 + 背景干扰）
            bandwidth = self.bw_per_rb
            noise_w = self._noise_power(bandwidth)
            h_bar = self._path_loss(dist, Cfg.PL_BETA_V2V)
            
            if active_tx_vehicles is not None:
                interference = self._compute_interference(
                    vehicle.pos, target_pos, active_tx_vehicles, p_tx
                )
                # 预估时假设干扰均匀分摊到 num_rb 个 RB
                interference = interference / max(self.num_rb, 1)
            else:
                interference = Cfg.dbm2watt(Cfg.V2V_INTERFERENCE_DBM)
            
            if self.use_block_fading:
                if curr_time is not None:
                    slot = int(round(curr_time / Cfg.DT))
                    if slot != self._v2v_cache_slot:
                        self.v2v_fading_cache.clear()
                        self._v2v_cache_slot = slot
                    target_key = tuple(np.round(target_pos, 3).tolist())
                    cache_key = (vehicle.id, target_key)
                    if cache_key not in self.v2v_fading_cache:
                        self.v2v_fading_cache[cache_key] = self._rayleigh_fading(1)[0]
                    h_rayleigh = self.v2v_fading_cache[cache_key]
                else:
                    h_rayleigh = self._rayleigh_fading(1)[0]
                signal_power = p_tx * h_bar * h_rayleigh
            else:
                # 关闭 Rayleigh：只用大尺度路损
                signal_power = p_tx * h_bar
            
            sinr = signal_power / (noise_w + interference)
            rate = bandwidth * np.log2(1 + sinr)

        return rate

    def _compute_interference(self, rx_pos, tx_pos, active_tx_vehicles, p_tx_self):
        """
        计算V2V接收端的实际干扰功率（期望值E[I]）
        
        注意：干扰功率只考虑大尺度路径损耗，不包含小尺度衰落
        
        优化：使用numpy向量化运算替代Python循环，提升性能

        Args:
            rx_pos: 接收车辆位置
            tx_pos: 期望发送者位置
            active_tx_vehicles: 当前环境中所有活跃的V2V发射车辆列表
            p_tx_self: 自身发射功率(W)

        Returns:
            干扰功率(W) - 期望值E[I]
        """
        if active_tx_vehicles is None or len(active_tx_vehicles) == 0:
            return 0.0
        
        # 过滤None值
        valid_vehicles = [v for v in active_tx_vehicles if v is not None]
        if len(valid_vehicles) == 0:
            return 0.0
        
        # 向量化计算：提取所有干扰源的位置和功率
        interferer_positions = np.array([v.pos for v in valid_vehicles])  # [N, 2]
        interferer_powers = np.array([Cfg.dbm2watt(v.tx_power_dbm) for v in valid_vehicles])  # [N]
        
        # 计算距离矩阵（向量化）
        rx_pos_array = np.array(rx_pos)
        distances = np.linalg.norm(interferer_positions - rx_pos_array, axis=1)  # [N]
        
        # 过滤：距离 >= 1.0 且 <= V2V_RANGE
        valid_mask = (distances >= 1.0) & (distances <= Cfg.V2V_RANGE)
        
        if not np.any(valid_mask):
            return 0.0
        
        # 只处理有效的干扰源
        valid_distances = distances[valid_mask]
        valid_powers = interferer_powers[valid_mask]
        
        # 向量化计算路径损耗和干扰功率
        h_bar = self._path_loss(valid_distances, Cfg.PL_BETA_V2V)  # [M], M为有效干扰源数
        interference_powers = valid_powers * h_bar  # [M]
        
        # 求和得到总干扰
        total_interference = np.sum(interference_powers)
        
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
        h_bar = self._path_loss(dist, Cfg.PL_BETA_V2V)
        
        # 计算干扰（期望值E[I]）
        if active_tx_vehicles is not None and len(active_tx_vehicles) > 0:
            interference = self._compute_interference(
                target_pos, vehicle.pos, active_tx_vehicles, p_tx
            )
        else:
            interference = Cfg.dbm2watt(Cfg.V2V_INTERFERENCE_DBM)

        gamma_th = getattr(Cfg, 'V2V_GAMMA_TH', 2.0)
        noise_w = self._noise_power(self.bw_per_rb)
        numerator = gamma_th * (noise_w + interference)
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
        """计算指定V2V链路速率（per-RB 带宽 + 背景干扰预估）"""
        p_tx = Cfg.dbm2watt(tx_veh.tx_power_dbm)
        dist = max(np.linalg.norm(tx_veh.pos - rx_veh.pos), 1.0)

        h_bar = self._path_loss(dist, Cfg.PL_BETA_V2V)
        if self.use_block_fading:
            h_rayleigh = self._rayleigh_fading(1)[0]
            signal_power = p_tx * h_bar * h_rayleigh
        else:
            signal_power = p_tx * h_bar

        interference = self._compute_interference(
            rx_veh.pos, tx_veh.pos, vehicles, p_tx
        )
        interference = interference / max(self.num_rb, 1)

        noise_w = self._noise_power(self.bw_per_rb)
        sinr = signal_power / (noise_w + interference)
        rate = self.bw_per_rb * np.log2(1 + sinr)
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

    def _noise_power(self, bandwidth_hz):
        """
        根据带宽计算等效噪声功率 (W)
        """
        bandwidth_hz = max(bandwidth_hz, 1.0)  # 防止0带宽
        return self.noise_psd_w_hz * bandwidth_hz
