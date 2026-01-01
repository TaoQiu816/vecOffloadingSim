import json
import io
from contextlib import redirect_stdout

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


def test_mean_cft_nan_on_empty():
    env = VecOffloadingEnv()
    # 清空车辆与CFT，模拟无估计的截断
    env.vehicles = []
    env.vehicle_cfts = []
    env._episode_steps = 300
    env.time = 15.0

    buf = io.StringIO()
    prev_flag = Cfg.EPISODE_JSONL_STDOUT
    Cfg.EPISODE_JSONL_STDOUT = True
    try:
        with redirect_stdout(buf):
            env._log_episode_stats(terminated=False, truncated=True)
    finally:
        Cfg.EPISODE_JSONL_STDOUT = prev_flag
    output = buf.getvalue().strip().splitlines()
    assert output, "no json output captured"
    record = json.loads(output[-1])
    # 当 vehicle_cfts 为空时，估计值应为 NaN
    assert record.get("mean_cft_est") != 15.0
    assert str(record.get("mean_cft_est")).lower() == "nan"
    assert str(record.get("mean_cft_completed")).lower() == "nan"
