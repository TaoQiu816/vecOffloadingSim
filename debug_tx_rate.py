"""诊断传输速率"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("传输速率诊断")
print("="*80)

# 执行RSU动作
actions = [{"target": 1, "power": 1.0}] * env.config.NUM_VEHICLES
next_obs, rewards, terminated, truncated, info = env.step(actions)

# 检查第一个传输任务
v0_key = ('VEH', 0)
if v0_key in env.txq_v2i and len(env.txq_v2i[v0_key]) > 0:
    job = env.txq_v2i[v0_key][0]
    print(f"\nVehicle 0的传输任务:")
    print(f"  rem_bytes: {job.rem_bytes/1e6:.3f}MB")
    print(f"  dst_node: {job.dst_node}")
    print(f"  link_type: {job.link_type}")
    print(f"  tx_power_dbm: {job.tx_power_dbm}")
    print(f"  kind: {job.kind}")
    
    # 手动计算速率
    rate = env._compute_job_rate(job, v0_key)
    print(f"\n  计算的速率: {rate/1e6:.3f} Mbps")
    
    # 检查车辆和RSU位置
    v0 = env.vehicles[0]
    rsu0 = env.rsus[0]
    dist = np.linalg.norm(v0.pos - rsu0.position)
    print(f"\n  Vehicle 0位置: {v0.pos}")
    print(f"  RSU 0位置: {rsu0.position}")
    print(f"  距离: {dist:.2f}m")
    print(f"  在RSU范围内: {dist <= env.config.RSU_RANGE}")
    
    # 检查channel配置
    print(f"\n  Channel配置:")
    print(f"    BW_V2I: {env.config.BW_V2I/1e6:.1f} MHz")
    print(f"    NOISE_POWER_DBM: {env.config.NOISE_POWER_DBM}")
    
    # 尝试直接调用channel计算速率
    print(f"\n  直接调用channel.compute_one_rate:")
    rate_direct = env.channel.compute_one_rate(
        v0, rsu0.position, "V2I", env.time,
        power_dbm_override=job.tx_power_dbm
    )
    print(f"    速率: {rate_direct/1e6:.3f} Mbps")

print("\n" + "="*80)

