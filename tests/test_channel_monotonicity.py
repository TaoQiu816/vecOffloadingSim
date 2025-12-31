import numpy as np

from envs.modules.channel import ChannelModel
from envs.entities.vehicle import Vehicle
from configs.config import SystemConfig as Cfg


def test_v2i_rate_monotonic_in_power():
    channel = ChannelModel()
    veh = Vehicle(0, np.array([0.0, 0.0]))
    target_pos = np.array([10.0, 0.0])

    veh.tx_power_dbm = Cfg.TX_POWER_MIN_DBM
    rate_low = channel.compute_one_rate(veh, target_pos, "V2I", curr_time=0.0, v2i_user_count=1)

    veh.tx_power_dbm = Cfg.TX_POWER_MAX_DBM
    rate_high = channel.compute_one_rate(veh, target_pos, "V2I", curr_time=0.0, v2i_user_count=1)

    assert rate_high >= rate_low
