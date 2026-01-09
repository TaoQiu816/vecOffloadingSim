"""
CommWait 分布脚本已废弃：CommWait特征不再包含在resource_raw中。
保留此文件以提示调用者，直接退出。
"""

import argparse
import numpy as np
import torch

raise SystemExit("CommWait features have been removed from observations; commwait_distribution_report is obsolete.")


def collect_commwait(episodes: int, steps_per_ep: int, seed: int, stress: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if stress:
        from configs.config import SystemConfig as Cfg
        Cfg.NUM_VEHICLES = 6
        Cfg.NUM_RSU = 1
        Cfg.VEHICLE_ARRIVAL_RATE = 0
        Cfg.BW_V2I = 20e6
        Cfg.VEL_MEAN = 0.0
        Cfg.VEL_MAX = 0.0
        Cfg.MAX_STEPS = steps_per_ep
        env = VecOffloadingEnv(config=Cfg)
        def stress_reset(ep_seed: int):
            obs_list, _ = env.reset(seed=ep_seed)
            # 固定在RSU附近并构造大数据量DAG（跨多步积压）
            if env.rsus:
                rsu_pos = env.rsus[0].position
                for v in env.vehicles:
                    v.pos = rsu_pos + np.array([50.0, 0.0])
                    v.vel = np.array([0.0, 0.0])
            sample_v = env.vehicles[0]
            rate = env.channel.compute_one_rate(
                sample_v, env.rsus[0].position, "V2I", curr_time=env.time,
                v2i_user_count=max(len(env.vehicles), 1),
            )
            step_capacity = rate * env.config.DT
            big_data = max(step_capacity * 10.0, 1e9)
            big_comp = 5e9
            from scripts.audit.run_audit_scenarios import build_dag, reset_dag_fields
            for v in env.vehicles:
                dag = build_dag(num_nodes=1, edges=[], comp=[big_comp], input_data=[big_data], deadline=50.0)
                reset_dag_fields(dag)
                v.task_dag = dag
            return obs_list
    else:
        env = VecOffloadingEnv()
        def stress_reset(ep_seed: int):
            return env.reset(seed=ep_seed)
    net = OffloadingPolicyNetwork().eval()

    comm_values = []  # [N, 4]
    samples_print = []
    rr_shape = None

    total_steps = 0
    for ep in range(episodes):
        obs_list = stress_reset(seed + ep)
        for t in range(steps_per_ep):
            # 收集当前 step 的 CommWait 4 维
            if obs_list:
                for obs in obs_list:
                    rr = np.asarray(obs["resource_raw"], dtype=np.float32)
                    rr_shape = rr.shape
                    comm_slice = rr[:, -4:]  # 末尾4维
                    # 记录平均（按资源）后的4维
                    comm_mean = comm_slice.mean(axis=0)
                    comm_values.append(comm_mean)
                    if len(samples_print) < 50:
                        samples_print.append(comm_mean.tolist())
            else:
                # 如果观测为空，直接使用 _compute_comm_wait 计算并归一化
                norm_max = getattr(env.config, "NORM_MAX_COMM_WAIT", 2.0)
                for v in env.vehicles:
                    cw = env._compute_comm_wait(v.id)
                    raw = np.array([cw["total_v2i"], cw["edge_v2i"], cw["total_v2v"], cw["edge_v2v"]], dtype=np.float32)
                    comm_norm = np.clip(np.log1p(raw) / np.log1p(norm_max), 0, 1)
                    comm_values.append(comm_norm)
                    if rr_shape is None:
                        rr_shape = (env.config.MAX_TARGETS, env.config.RESOURCE_RAW_DIM)
                    if len(samples_print) < 50:
                        samples_print.append(comm_norm.tolist())

            # 简单策略：50% 概率选 RSU(1)，否则 Local
            actions = []
            target_len = len(obs_list) if obs_list else len(env.vehicles)
            for _ in range(target_len):
                if stress:
                    actions.append({"target": 1, "power": 1.0})
                else:
                    if np.random.rand() < 0.5:
                        actions.append({"target": 1, "power": 1.0})
                    else:
                        actions.append({"target": 0, "power": 0.5})
            obs_list, rewards, terminated, truncated, _ = env.step(actions)
            total_steps += 1
            if terminated or truncated:
                break
        if total_steps >= steps_per_ep * episodes:
            break
    env.close()
    return np.array(comm_values, dtype=np.float32), samples_print, rr_shape


def calc_stats(arr):
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def run_report(episodes, steps, seed, stress: bool = False):
    comm, samples_print, rr_shape = collect_commwait(episodes, steps, seed, stress=stress)
    # comm shape [N,4]
    if comm.size == 0:
        print("No CommWait samples collected; check environment configuration.")
        return
    agg = comm.mean(axis=1)
    stats_per_dim = [calc_stats(comm[:, i]) for i in range(4)]
    stats_agg = calc_stats(agg)

    # 非零比例与最大连续非零段
    nonzero_mask = np.any(comm > 1e-6, axis=1)
    nonzero_ratio = float(nonzero_mask.mean())
    max_run = 0
    cur = 0
    for v in nonzero_mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    p99_agg = stats_agg["p99"]
    factor = 0.8 / max(p99_agg, 1e-9)

    print(f"resource_raw shape example: {rr_shape}, CommWait slice = resource_raw[:, -4:]")
    print(f"Collected samples: {len(comm)}")
    for i, st in enumerate(stats_per_dim):
        print(f"Dim {i} stats: {st}")
    print(f"Aggregate (mean over 4 dims) stats: {stats_agg}")
    print(f"Nonzero ratio: {nonzero_ratio:.4f}, max consecutive nonzero run: {max_run}")
    print(f"0.8 vs p99 (agg): factor={factor:.2f} (p99={p99_agg:.4f})")
    print("Suggested perturbation set (agg):", {
        "p90": stats_agg["p90"],
        "p95": stats_agg["p95"],
        "p99": stats_agg["p99"],
    })
    print("First 50 averaged CommWait samples:")
    for s in samples_print:
        print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stress", action="store_true", help="使用压力场景（多车RSU卸载、大数据量）")
    args = parser.parse_args()
    run_report(args.episodes, args.steps, args.seed, stress=args.stress)
