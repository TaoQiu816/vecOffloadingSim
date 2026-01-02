import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv, compute_absolute_reward


def test_reward_formula_outputs_expected_values():
    dt = Cfg.DT
    t_prev = 5.0
    t_curr = 4.85
    power_ratio_local = 0.3

    dT_rem = t_prev - t_curr
    r_local, comp_local = compute_absolute_reward(
        dT_rem,
        t_tx=0.0,
        power_ratio=power_ratio_local,
        dt=dt,
        p_max_watt=Cfg.dbm2watt(Cfg.TX_POWER_UP_DBM),
        reward_min=Cfg.REWARD_MIN,
        reward_max=Cfg.REWARD_MAX,
        hard_triggered=False,
        illegal_action=False,
    )
    dT_eff_expected = np.clip(dT_rem, Cfg.DELTA_CFT_CLIP_MIN, Cfg.DELTA_CFT_CLIP_MAX) - dt
    expected_energy_norm = power_ratio_local
    assert abs(comp_local["energy_norm"] - expected_energy_norm) < 1e-9
    assert abs(comp_local["dT_eff"] - dT_eff_expected) < 1e-9
    assert abs(r_local - (Cfg.DELTA_CFT_SCALE * np.clip(dT_rem, Cfg.DELTA_CFT_CLIP_MIN, Cfg.DELTA_CFT_CLIP_MAX) - Cfg.DELTA_CFT_ENERGY_WEIGHT * expected_energy_norm)) < 1e-6

    power_ratio_rsu = 0.6
    t_tx = 0.05
    r_rsu, comp_rsu = compute_absolute_reward(
        dT_rem,
        t_tx=t_tx,
        power_ratio=power_ratio_rsu,
        dt=dt,
        p_max_watt=Cfg.dbm2watt(Cfg.TX_POWER_UP_DBM),
        reward_min=Cfg.REWARD_MIN,
        reward_max=Cfg.REWARD_MAX,
        hard_triggered=False,
        illegal_action=False,
    )
    assert 0.0 < comp_rsu["energy_norm"] <= 1.0
    expected_energy_norm = power_ratio_rsu
    assert abs(comp_rsu["energy_norm"] - expected_energy_norm) < 1e-9
    assert abs(comp_rsu["dT_eff"] - dT_eff_expected) < 1e-9
    expected_reward = Cfg.DELTA_CFT_SCALE * np.clip(dT_rem, Cfg.DELTA_CFT_CLIP_MIN, Cfg.DELTA_CFT_CLIP_MAX) - Cfg.DELTA_CFT_ENERGY_WEIGHT * expected_energy_norm
    assert abs(r_rsu - expected_reward) < 1e-6


def test_reward_is_finite_and_handles_penalties():
    r_ok, comp_ok = compute_absolute_reward(
        1.0,
        t_tx=Cfg.DT * 0.5,
        power_ratio=0.5,
        dt=Cfg.DT,
        p_max_watt=Cfg.dbm2watt(Cfg.TX_POWER_UP_DBM),
        reward_min=Cfg.REWARD_MIN,
        reward_max=Cfg.REWARD_MAX,
        hard_triggered=False,
        illegal_action=False,
    )
    assert np.isfinite(r_ok)
    assert np.isfinite(comp_ok["energy_norm"])

    r_penalty, comp_penalty = compute_absolute_reward(
        1.0,
        t_tx=Cfg.DT * 0.5,
        power_ratio=0.5,
        dt=Cfg.DT,
        p_max_watt=Cfg.dbm2watt(Cfg.TX_POWER_V2V_DBM),
        reward_min=Cfg.REWARD_MIN,
        reward_max=Cfg.REWARD_MAX,
        hard_triggered=True,
        illegal_action=False,
    )
    assert r_penalty == Cfg.REWARD_MIN
    assert comp_penalty["t_tx"] <= Cfg.DT
    r_illegal, _ = compute_absolute_reward(
        1.0,
        t_tx=Cfg.DT * 0.5,
        power_ratio=0.5,
        dt=Cfg.DT,
        p_max_watt=Cfg.dbm2watt(Cfg.TX_POWER_V2V_DBM),
        reward_min=Cfg.REWARD_MIN,
        reward_max=Cfg.REWARD_MAX,
        hard_triggered=False,
        illegal_action=True,
    )
    assert r_illegal == Cfg.REWARD_MIN


def test_remaining_time_used_for_dt_eff(monkeypatch):
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.NUM_VEHICLES = 1
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0
        env = VecOffloadingEnv()
        obs_list, _ = env.reset(seed=2)

        def fake_cft(snapshot_time=None, v2i_user_count=None, vehicle_ids=None):
            if snapshot_time is None:
                snapshot_time = env.time
            return snapshot_time + Cfg.DT + 10.0

        monkeypatch.setattr(env, "_compute_mean_cft_pi0", fake_cft)
        actions = [{"target": 0, "power": 1.0} for _ in obs_list]
        env.step(actions)

        bucket = env._reward_stats.metrics.get("dT_eff")
        assert bucket is not None
        dT_eff = bucket.mean()
        assert abs(dT_eff + Cfg.DT) < 1e-6
        dT_bucket = env._reward_stats.metrics.get("delta_cft")
        assert dT_bucket is not None
        assert abs(dT_bucket.mean()) < 1e-6
        dt_used = env._reward_stats.metrics.get("dt_used")
        assert dt_used is not None
        implied = dT_bucket.mean() - bucket.mean()
        assert abs(implied - Cfg.DT) < 1e-6
    finally:
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
