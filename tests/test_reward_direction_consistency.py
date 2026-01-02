from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import compute_absolute_reward


def test_reward_direction_consistency():
    t_prev = 5.0
    t_curr = 4.8
    p_rsu = Cfg.dbm2watt(Cfg.TX_POWER_UP_DBM)
    p_v2v = Cfg.dbm2watt(Cfg.TX_POWER_V2V_DBM)

    r_local, comp_local = compute_absolute_reward(
        t_prev, t_curr, Cfg.DT, power_ratio=0.1, t_tx=0.0,
        reward_min=Cfg.REWARD_MIN, reward_max=Cfg.REWARD_MAX, p_max_watt=p_rsu, hard_triggered=False, illegal_action=False
    )
    r_rsu, comp_rsu = compute_absolute_reward(
        t_prev, t_curr, Cfg.DT, power_ratio=0.9, t_tx=Cfg.DT,
        reward_min=Cfg.REWARD_MIN, reward_max=Cfg.REWARD_MAX, p_max_watt=p_v2v, hard_triggered=False, illegal_action=False
    )

    assert comp_rsu["energy_norm"] >= comp_local["energy_norm"]
    assert r_local >= r_rsu
