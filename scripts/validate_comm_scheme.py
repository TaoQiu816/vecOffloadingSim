import math
import os
import sys
from contextlib import contextmanager

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configs.config import SystemConfig as Cfg
from envs.jobs import TransferJob
from envs.modules.channel import ChannelModel
from envs.vec_offloading_env import VecOffloadingEnv


@contextmanager
def temporary_cfg(**kwargs):
    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = getattr(Cfg, key)
        setattr(Cfg, key, value)
    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(Cfg, key, value)


def validate_v2i_rb_sinr_contention():
    with temporary_cfg(V2I_RATE_MODEL="RB_SINR", V2I_ICI_ENABLED=False, V2I_NUM_RB=1):
        channel = ChannelModel()
        rsu_pos_map = {0: np.array([0.0, 0.0], dtype=float)}
        one_link = [{
            "sender_id": 0,
            "tx_pos": np.array([50.0, 0.0], dtype=float),
            "rsu_id": 0,
            "power_w": Cfg.dbm2watt(20.0),
        }]
        two_links = one_link + [{
            "sender_id": 1,
            "tx_pos": np.array([60.0, 0.0], dtype=float),
            "rsu_id": 0,
            "power_w": Cfg.dbm2watt(20.0),
        }]
        rate_one = channel.compute_v2i_rates(one_link, rsu_pos_map=rsu_pos_map)[0]
        rate_two = channel.compute_v2i_rates(two_links, rsu_pos_map=rsu_pos_map)[0]
        assert rate_two < rate_one, f"Expected RB_SINR V2I rate drop, got {rate_one} -> {rate_two}"
        print(f"[OK] V2I same-RSU contention: {rate_one:.3e} -> {rate_two:.3e}")


def validate_v2v_subchannel_congestion():
    with temporary_cfg(V2V_NUM_RB=1, V2V_BW_PER_RB=Cfg.BW_V2V):
        channel = ChannelModel()
        one_link = [{
            "sender_id": 0,
            "tx_pos": np.array([0.0, 0.0], dtype=float),
            "rx_pos": np.array([20.0, 0.0], dtype=float),
            "power_w": Cfg.dbm2watt(20.0),
        }]
        two_links = one_link + [{
            "sender_id": 1,
            "tx_pos": np.array([0.0, 30.0], dtype=float),
            "rx_pos": np.array([20.0, 30.0], dtype=float),
            "power_w": Cfg.dbm2watt(20.0),
        }]
        rate_one = channel.compute_v2v_rb_sinr(one_link)[0]
        rate_two = channel.compute_v2v_rb_sinr(two_links)[0]
        assert rate_two < rate_one, f"Expected V2V rate drop with shared subchannel, got {rate_one} -> {rate_two}"
        print(f"[OK] V2V same-subchannel congestion: {rate_one:.3e} -> {rate_two:.3e}")


def validate_v2v_power_coupling():
    with temporary_cfg(V2V_NUM_RB=1, V2V_BW_PER_RB=Cfg.BW_V2V):
        channel = ChannelModel()
        base_links = [
            {
                "sender_id": 0,
                "tx_pos": np.array([0.0, 0.0], dtype=float),
                "rx_pos": np.array([20.0, 0.0], dtype=float),
                "power_w": Cfg.dbm2watt(18.0),
            },
            {
                "sender_id": 1,
                "tx_pos": np.array([0.0, 30.0], dtype=float),
                "rx_pos": np.array([20.0, 30.0], dtype=float),
                "power_w": Cfg.dbm2watt(18.0),
            },
        ]
        high_power_links = [dict(base_links[0]), dict(base_links[1])]
        high_power_links[0]["power_w"] = Cfg.dbm2watt(23.0)
        rates_base = channel.compute_v2v_rb_sinr(base_links)
        rates_high = channel.compute_v2v_rb_sinr(high_power_links)
        assert rates_high[0] > rates_base[0], "Expected own V2V rate to increase with higher power"
        assert rates_high[1] < rates_base[1], "Expected co-channel V2V victim rate to decrease with higher interferer power"
        print(
            "[OK] V2V power coupling:"
            f" self {rates_base[0]:.3e}->{rates_high[0]:.3e},"
            f" victim {rates_base[1]:.3e}->{rates_high[1]:.3e}"
        )


def validate_v2i_power_and_interference():
    with temporary_cfg(V2I_RATE_MODEL="RB_SINR", V2I_ICI_ENABLED=True, V2I_NUM_RB=1):
        channel = ChannelModel()
        rsu_pos_map = {
            0: np.array([0.0, 0.0], dtype=float),
            1: np.array([200.0, 0.0], dtype=float),
        }
        low_power = [
            {
                "sender_id": 0,
                "tx_pos": np.array([50.0, 0.0], dtype=float),
                "rsu_id": 0,
                "power_w": Cfg.dbm2watt(13.0),
            },
            {
                "sender_id": 1,
                "tx_pos": np.array([150.0, 0.0], dtype=float),
                "rsu_id": 1,
                "power_w": Cfg.dbm2watt(20.0),
            },
        ]
        high_power = [dict(low_power[0]), dict(low_power[1])]
        high_power[0]["power_w"] = Cfg.dbm2watt(23.0)
        rates_low = channel.compute_v2i_rates(low_power, rsu_pos_map=rsu_pos_map)
        rates_high = channel.compute_v2i_rates(high_power, rsu_pos_map=rsu_pos_map)
        bits = 1.0e6
        energy_low = low_power[0]["power_w"] * bits / max(rates_low[0], 1e-9)
        energy_high = high_power[0]["power_w"] * bits / max(rates_high[0], 1e-9)
        assert rates_high[0] > rates_low[0], "Expected own V2I rate to increase with higher power"
        assert not math.isclose(energy_low, energy_high, rel_tol=1e-6, abs_tol=1e-12), "Expected V2I energy to change with power"
        assert rates_high[1] < rates_low[1], "Expected neighboring RSU uplink to worsen under cross-RSU interference"
        print(
            "[OK] V2I power/interference:"
            f" self {rates_low[0]:.3e}->{rates_high[0]:.3e},"
            f" victim {rates_low[1]:.3e}->{rates_high[1]:.3e},"
            f" energy {energy_low:.3e}->{energy_high:.3e}"
        )


def validate_v2v_contact_break():
    env = VecOffloadingEnv()
    env.reset(seed=Cfg.SEED)
    src = env.vehicles[0]
    dst = env.vehicles[1]
    src.pos = np.array([0.0, 0.0], dtype=float)
    dst.pos = np.array([Cfg.V2V_RANGE + 50.0, 0.0], dtype=float)
    job = TransferJob(
        kind="INPUT",
        src_node=("VEH", src.id),
        dst_node=("VEH", dst.id),
        owner_vehicle_id=src.id,
        subtask_id=0,
        rem_bytes=1.0e6,
        tx_power_dbm=20.0,
        link_type="V2V",
        enqueue_time=env.time,
        dag_uid=id(src.task_dag),
    )
    env.txq_v2v[("VEH", src.id)].append(job)
    env._capture_rate_snapshot([])
    env._phase3_advance_comm_queues()
    assert env._last_aborted_jobs, "Expected V2V job to abort on contact break"
    assert src.task_dag.is_failed, "Expected owner DAG to fail after V2V contact break"
    print(f"[OK] V2V contact break aborted {len(env._last_aborted_jobs)} job(s)")


def main():
    validate_v2i_rb_sinr_contention()
    validate_v2v_subchannel_congestion()
    validate_v2v_power_coupling()
    validate_v2i_power_and_interference()
    validate_v2v_contact_break()
    print("All communication-scheme checks passed.")


if __name__ == "__main__":
    main()
