"""详细诊断channel计算"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg

np.random.seed(42)
env = VecOffloadingEnv()
obs_list = env.reset(seed=42)

print("="*80)
print("Channel计算详细诊断")
print("="*80)

v0 = env.vehicles[0]
rsu0 = env.rsus[0]

print(f"\n基本信息:")
print(f"  Vehicle 0位置: {v0.pos}")
print(f"  RSU 0位置: {rsu0.position}")

dist = np.linalg.norm(v0.pos - rsu0.position)
print(f"  距离: {dist:.2f}m")

# 手动计算V2I速率
print(f"\n手动计算V2I速率:")
power_dbm = 23.0
p_tx = Cfg.dbm2watt(power_dbm)
print(f"  power_dbm: {power_dbm}")
print(f"  p_tx: {p_tx:.6e} W")

est_users = max(Cfg.NUM_VEHICLES // 5, 1)
print(f"  est_users: {est_users}")

bandwidth = Cfg.BW_V2I / est_users
print(f"  BW_V2I: {Cfg.BW_V2I/1e6:.1f} MHz")
print(f"  bandwidth: {bandwidth/1e6:.3f} MHz")

# 计算噪声功率
noise_dbm = Cfg.NOISE_POWER_DBM
noise_w = Cfg.dbm2watt(noise_dbm)
print(f"  noise_dbm: {noise_dbm}")
print(f"  noise_w: {noise_w:.6e} W")

# 计算路径损耗
print(f"\n路径损耗计算:")
print(f"  PL_ALPHA_V2I: {Cfg.PL_ALPHA_V2I}")

# 检查_path_loss方法
h_bar = env.channel._path_loss(dist, Cfg.PL_ALPHA_V2I)
print(f"  h_bar (path loss gain): {h_bar:.6e}")

# 计算SINR
sinr = (p_tx * h_bar) / noise_w
print(f"\nSINR计算:")
print(f"  p_tx * h_bar: {p_tx * h_bar:.6e}")
print(f"  SINR: {sinr:.6e}")
print(f"  SINR (dB): {10*np.log10(sinr):.2f} dB")

# 计算速率
rate = bandwidth * np.log2(1 + sinr)
print(f"\n速率计算:")
print(f"  log2(1 + sinr): {np.log2(1 + sinr):.6f}")
print(f"  rate: {rate/1e6:.3f} Mbps")

# 调用channel方法验证
print(f"\n调用channel.compute_one_rate验证:")
rate_channel = env.channel.compute_one_rate(
    v0, rsu0.position, "V2I", env.time,
    power_dbm_override=power_dbm
)
print(f"  返回速率: {rate_channel/1e6:.3f} Mbps")

# 检查vehicle的tx_power_dbm属性
print(f"\n检查vehicle属性:")
print(f"  v0.tx_power_dbm: {v0.tx_power_dbm if hasattr(v0, 'tx_power_dbm') else 'NOT SET'}")

print("\n" + "="*80)

