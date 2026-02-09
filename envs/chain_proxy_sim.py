import numpy as np


class ChainProxySim:
    """
    Level-0/Level-1 Chain Proxy Simulator (Settlement Risk Layer)

    Modes:
    - NONE: all-zero chain state
    - CONST: constant chain state (uses LOW parameters)
    - SWITCH: toggles between LOW/HIGH every SWITCH_PERIOD_STEPS
    """

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.mode = getattr(cfg, "CHAIN_MODE", "NONE")
        if seed is None:
            seed = getattr(cfg, "SEED", 0)
        self._rng = np.random.RandomState(int(seed))
        self._step_count = 0

    def reset(self):
        self._step_count = 0

    def _apply_noise(self, value, std):
        if std <= 0:
            return value
        return value + float(self._rng.normal(0.0, std))

    def step(self, tx_arrivals: int) -> dict:
        _ = tx_arrivals  # reserved for future use
        mode = (self.mode or "NONE").upper()
        if mode == "NONE":
            return {
                "p50_confirm": 0.0,
                "p95_confirm": 0.0,
                "p_fail": 0.0,
                "mempool_len": 0.0,
                "rho": 0.0,
            }

        if mode == "SWITCH":
            period = int(getattr(self.cfg, "CHAIN_SWITCH_PERIOD_STEPS", 200))
            if period <= 0:
                phase = 0
            else:
                phase = (self._step_count // period) % 2
        else:
            phase = 0

        if phase == 0:
            p50 = float(getattr(self.cfg, "CHAIN_P50_LOW", 0.0))
            p95 = float(getattr(self.cfg, "CHAIN_P95_LOW", 0.0))
            p_fail = float(getattr(self.cfg, "CHAIN_PFAIL_LOW", 0.0))
        else:
            p50 = float(getattr(self.cfg, "CHAIN_P50_HIGH", 0.0))
            p95 = float(getattr(self.cfg, "CHAIN_P95_HIGH", 0.0))
            p_fail = float(getattr(self.cfg, "CHAIN_PFAIL_HIGH", 0.0))

        noise_std = float(getattr(self.cfg, "CHAIN_NOISE_STD", 0.0))
        p50 = max(0.0, self._apply_noise(p50, noise_std))
        p95 = max(0.0, self._apply_noise(p95, noise_std))
        if p95 < p50:
            p95 = p50
        p_fail = self._apply_noise(p_fail, noise_std)
        p_fail = float(np.clip(p_fail, 0.0, 1.0))

        self._step_count += 1
        return {
            "p50_confirm": p50,
            "p95_confirm": p95,
            "p_fail": p_fail,
            "mempool_len": 0.0,
            "rho": 0.0,
        }
