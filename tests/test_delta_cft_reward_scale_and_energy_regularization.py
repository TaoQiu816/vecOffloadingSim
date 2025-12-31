import numpy as np
import pytest

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _setup_env(seed, monkeypatch):
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=seed)
    if not env.rsus:
        pytest.skip("RSU not configured")
    # Place vehicle at RSU to ensure coverage.
    env.vehicles[0].pos = env.rsus[0].position.copy()
    env._dist_matrix_cache = None
    env._rsu_dist_cache.clear()
    obs_list = env._get_obs()
    rsu_id = env._last_rsu_choice.get(env.vehicles[0].id)
    if rsu_id is None:
        pytest.skip("RSU not available for this seed/config")

    # Freeze CFT to isolate energy regularization.
    monkeypatch.setattr(env, "_compute_mean_cft_pi0", lambda *args, **kwargs: 100.0)
    # Constant rate so tx_time does not change with power.
    monkeypatch.setattr(env.channel, "compute_one_rate", lambda *args, **kwargs: 1e6)

    return env, obs_list, rsu_id


def _set_power_dbm(power_ratio):
    raw_power = Cfg.TX_POWER_MIN_DBM + power_ratio * (Cfg.TX_POWER_MAX_DBM - Cfg.TX_POWER_MIN_DBM)
    return float(np.clip(raw_power, Cfg.TX_POWER_MIN_DBM, Cfg.TX_POWER_MAX_DBM))


def test_delta_cft_reward_scale_and_energy_regularization(monkeypatch):
    orig_mode = Cfg.REWARD_MODE
    orig_num = Cfg.NUM_VEHICLES
    orig_energy = Cfg.ENERGY_IN_DELTA_CFT
    try:
        Cfg.REWARD_MODE = "delta_cft"
        Cfg.ENERGY_IN_DELTA_CFT = True
        Cfg.NUM_VEHICLES = 1

        env_low, obs_low, rsu_id = _setup_env(seed=0, monkeypatch=monkeypatch)
        env_high, obs_high, rsu_id_high = _setup_env(seed=0, monkeypatch=monkeypatch)
        assert rsu_id_high == rsu_id

        v_low = env_low.vehicles[0]
        v_high = env_high.vehicles[0]
        subtask_idx = v_low.task_dag.get_top_priority_task()
        task_comp = v_low.task_dag.total_comp[subtask_idx] if subtask_idx is not None else Cfg.MEAN_COMP_LOAD

        v_low.tx_power_dbm = _set_power_dbm(0.1)
        v_high.tx_power_dbm = _set_power_dbm(0.9)
        comp_low = env_low._compute_cost_components(0, ("RSU", rsu_id), subtask_idx, task_comp)
        comp_high = env_high._compute_cost_components(0, ("RSU", rsu_id), subtask_idx, task_comp)
        assert comp_high["energy_norm"] >= comp_low["energy_norm"] - 1e-9

        actions_low = [{
            "target": 1,
            "power": 0.1,
            "obs_stamp": int(obs_low[0]["obs_stamp"]),
        }]
        actions_high = [{
            "target": 1,
            "power": 0.9,
            "obs_stamp": int(obs_high[0]["obs_stamp"]),
        }]
        _, rewards_low, _, _, _ = env_low.step(actions_low)
        _, rewards_high, _, _, _ = env_high.step(actions_high)
        assert rewards_high[0] <= rewards_low[0] + 1e-9
    finally:
        Cfg.REWARD_MODE = orig_mode
        Cfg.NUM_VEHICLES = orig_num
        Cfg.ENERGY_IN_DELTA_CFT = orig_energy
