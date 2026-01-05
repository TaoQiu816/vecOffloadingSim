"""分析奖励分布和归一化"""
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.train_config import TrainConfig as TC
from agents.mappo_agent import MAPPOAgent
from models.offloading_policy import OffloadingPolicyNetwork

np.random.seed(42)

# 创建环境和agent
env = VecOffloadingEnv()
network = OffloadingPolicyNetwork(
    d_model=TC.EMBED_DIM,
    num_heads=TC.NUM_HEADS,
    num_layers=TC.NUM_LAYERS
)
agent = MAPPOAgent(network, device='cpu')

print("="*80)
print("奖励分布分析")
print("="*80)

# 收集统计数据
dT_values = []
energy_norm_values = []
reward_values = []
reward_components = []

obs_list = env.reset(seed=42)

# 运行50步收集数据
for step in range(50):
    actions, _, _ = agent.select_action(obs_list, deterministic=False)
    next_obs_list, rewards, terminated, truncated, info = env.step(actions)
    
    # 收集奖励值
    reward_values.extend(rewards)
    
    # 尝试获取奖励组件（如果有的话）
    if hasattr(env, '_last_reward_components'):
        for comp in env._last_reward_components:
            if comp:
                if 'dT' in comp:
                    dT_values.append(comp['dT'])
                if 'energy_norm' in comp:
                    energy_norm_values.append(comp['energy_norm'])
                reward_components.append(comp)
    
    obs_list = next_obs_list
    if terminated or truncated:
        obs_list = env.reset(seed=42+step)

print(f"\n收集了 {len(reward_values)} 个奖励样本")

# 统计分析
print(f"\n{'='*80}")
print("奖励统计")
print(f"{'='*80}")
print(f"  均值: {np.mean(reward_values):.4f}")
print(f"  标准差: {np.std(reward_values):.4f}")
print(f"  最小值: {np.min(reward_values):.4f}")
print(f"  最大值: {np.max(reward_values):.4f}")
print(f"  中位数: {np.median(reward_values):.4f}")
print(f"  P5: {np.percentile(reward_values, 5):.4f}")
print(f"  P95: {np.percentile(reward_values, 95):.4f}")

if dT_values:
    print(f"\n{'='*80}")
    print("时间变化 (dT) 统计")
    print(f"{'='*80}")
    print(f"  均值: {np.mean(dT_values):.4f} 秒")
    print(f"  标准差: {np.std(dT_values):.4f} 秒")
    print(f"  最小值: {np.min(dT_values):.4f} 秒")
    print(f"  最大值: {np.max(dT_values):.4f} 秒")
    print(f"  中位数: {np.median(dT_values):.4f} 秒")
    
    # 检查裁剪
    clipped_min = np.sum(np.array(dT_values) <= -1.0)
    clipped_max = np.sum(np.array(dT_values) >= 1.0)
    print(f"\n  触及下界(-1.0)次数: {clipped_min} ({clipped_min/len(dT_values)*100:.1f}%)")
    print(f"  触及上界(1.0)次数: {clipped_max} ({clipped_max/len(dT_values)*100:.1f}%)")

if energy_norm_values:
    print(f"\n{'='*80}")
    print("归一化能耗 (energy_norm) 统计")
    print(f"{'='*80}")
    print(f"  均值: {np.mean(energy_norm_values):.4f}")
    print(f"  标准差: {np.std(energy_norm_values):.4f}")
    print(f"  最小值: {np.min(energy_norm_values):.4f}")
    print(f"  最大值: {np.max(energy_norm_values):.4f}")
    print(f"  中位数: {np.median(energy_norm_values):.4f}")

# 分析奖励组成
if dT_values and energy_norm_values:
    print(f"\n{'='*80}")
    print("奖励组成分析")
    print(f"{'='*80}")
    
    from configs.config import SystemConfig as Cfg
    
    # 计算各部分的贡献
    time_contributions = np.array(dT_values) * Cfg.DELTA_CFT_SCALE
    energy_contributions = np.array(energy_norm_values) * Cfg.DELTA_CFT_ENERGY_WEIGHT
    
    print(f"\n时间收益部分 (DELTA_CFT_SCALE * dT):")
    print(f"  DELTA_CFT_SCALE = {Cfg.DELTA_CFT_SCALE}")
    print(f"  均值贡献: {np.mean(time_contributions):.4f}")
    print(f"  标准差: {np.std(time_contributions):.4f}")
    print(f"  范围: [{np.min(time_contributions):.4f}, {np.max(time_contributions):.4f}]")
    
    print(f"\n能耗惩罚部分 (DELTA_CFT_ENERGY_WEIGHT * energy_norm):")
    print(f"  DELTA_CFT_ENERGY_WEIGHT = {Cfg.DELTA_CFT_ENERGY_WEIGHT}")
    print(f"  均值贡献: {np.mean(energy_contributions):.4f}")
    print(f"  标准差: {np.std(energy_contributions):.4f}")
    print(f"  范围: [{np.min(energy_contributions):.4f}, {np.max(energy_contributions):.4f}]")
    
    # 计算比例
    avg_time_abs = np.mean(np.abs(time_contributions))
    avg_energy_abs = np.mean(np.abs(energy_contributions))
    if avg_energy_abs > 0:
        ratio = avg_time_abs / avg_energy_abs
        print(f"\n时间/能耗平均贡献比例: {ratio:.1f}:1")
        print(f"  (时间部分平均绝对值: {avg_time_abs:.4f})")
        print(f"  (能耗部分平均绝对值: {avg_energy_abs:.4f})")

print(f"\n{'='*80}")
print("配置参数")
print(f"{'='*80}")
from configs.config import SystemConfig as Cfg
print(f"  DELTA_CFT_SCALE: {Cfg.DELTA_CFT_SCALE}")
print(f"  DELTA_CFT_ENERGY_WEIGHT: {Cfg.DELTA_CFT_ENERGY_WEIGHT}")
print(f"  DELTA_CFT_CLIP_MIN: {Cfg.DELTA_CFT_CLIP_MIN}")
print(f"  DELTA_CFT_CLIP_MAX: {Cfg.DELTA_CFT_CLIP_MAX}")
print(f"  REWARD_MIN: {Cfg.REWARD_MIN}")
print(f"  REWARD_MAX: {Cfg.REWARD_MAX}")

print(f"\n{'='*80}")

