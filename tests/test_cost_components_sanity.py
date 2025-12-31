import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _calc_times(env, v, target, task_idx):
    task_comp = v.task_dag.total_comp[task_idx]
    task_data = v.task_dag.total_data[task_idx]
    tx_time = 0.0
    queue_wait = 0.0
    cpu_freq = v.cpu_freq
    if target == "Local":
        queue_wait = v.task_queue.get_estimated_wait_time(v.cpu_freq)
        cpu_freq = v.cpu_freq
    elif env._is_rsu_location(target):
        rsu_id = env._get_rsu_id_from_location(target)
        if rsu_id is not None and 0 <= rsu_id < len(env.rsus):
            rsu = env.rsus[rsu_id]
            queue_wait = rsu.get_estimated_wait_time()
            cpu_freq = rsu.cpu_freq
            rate = env.channel.compute_one_rate(
                v, rsu.position, "V2I", env.time, v2i_user_count=env._estimate_v2i_users()
            )
            rate = max(rate, 1e-6)
            tx_time = task_data / rate if task_data > 0 else 0.0
    elif isinstance(target, int):
        t_veh = env._get_vehicle_by_id(target)
        if t_veh is not None:
            queue_wait = t_veh.task_queue.get_estimated_wait_time(t_veh.cpu_freq)
            cpu_freq = t_veh.cpu_freq
            rate = env.channel.compute_one_rate(v, t_veh.pos, "V2V", env.time)
            rate = max(rate, 1e-6)
            tx_time = task_data / rate if task_data > 0 else 0.0
    comp_time = task_comp / max(cpu_freq, 1e-6)
    return tx_time, queue_wait, comp_time


def test_cost_components_sanity():
    env = VecOffloadingEnv()
    env.reset(seed=123)

    episodes = 2
    steps_per_episode = 20
    for _ in range(episodes):
        for _ in range(steps_per_episode):
            for v in env.vehicles:
                task_idx = v.task_dag.get_top_priority_task()
                if task_idx is None:
                    continue

                targets = ["Local"]
                if env.rsus:
                    targets.append(("RSU", 0))
                if len(env.vehicles) > 1:
                    neighbor_id = next(other.id for other in env.vehicles if other.id != v.id)
                    targets.append(neighbor_id)

                for target in targets:
                    tx_time, wait_time, comp_time = _calc_times(env, v, target, task_idx)
                    assert np.isfinite(tx_time) and tx_time >= 0.0
                    assert np.isfinite(wait_time) and wait_time >= 0.0
                    assert np.isfinite(comp_time) and comp_time >= 0.0

                    components = env._compute_cost_components(v.id, target, task_idx, v.task_dag.total_comp[task_idx])
                    for key in ("delay_norm", "energy_norm", "r_soft_pen", "r_timeout"):
                        val = components[key]
                        assert np.isfinite(val)
                        if key in ("delay_norm", "energy_norm"):
                            assert val >= 0.0

            actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
            _, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                env.reset()
                break
