#!/usr/bin/env python3
"""诊断V2V可用性问题"""
import sys
sys.path.insert(0, '.')
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg

np.random.seed(42)
env = VecOffloadingEnv()
obs, _ = env.reset(seed=42)

print("=" * 70)
print("V2V可用性诊断")
print("=" * 70)

# 1. 检查参数
print("\n[参数检查]")
print(f"  MAP_SIZE = {Cfg.MAP_SIZE}m")
print(f"  V2V_RANGE = {Cfg.V2V_RANGE}m")
print(f"  NUM_VEHICLES = {Cfg.NUM_VEHICLES}")
print(f"  VEL_MIN/MAX = {Cfg.VEL_MIN}/{Cfg.VEL_MAX} m/s")
print(f"  MAX_NEIGHBORS = {Cfg.MAX_NEIGHBORS}")

# 2. 检查车辆位置分布
positions = np.array([v.pos for v in env.vehicles])
print("\n[车辆位置分布]")
print(f"  X范围: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]m")
print(f"  Y范围: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]m")

# 3. 检查距离矩阵
dist_matrix = env._get_dist_matrix()
dm_copy = dist_matrix.copy()
np.fill_diagonal(dm_copy, np.inf)
print("\n[车辆间距离]")
print(f"  最近邻距离: {dm_copy.min():.1f}m")
print(f"  平均最近邻距离: {dm_copy.min(axis=1).mean():.1f}m")
in_range_count = (dm_copy < Cfg.V2V_RANGE).sum(axis=1).mean()
print(f"  在V2V_RANGE内的邻居数(每车): {in_range_count:.1f}")

# 4. 检查典型任务参数
v = env.vehicles[0]
task_comp = v.task_dag.total_comp.mean()
task_data = v.task_dag.total_data.mean()
print("\n[典型任务参数]")
print(f"  平均计算量: {task_comp:.2e} cycles")
print(f"  平均数据量: {task_data:.2e} bits")
print(f"  DAG deadline: {v.task_dag.deadline:.2f}s")

# 5. 检查T_finish_est各分量
print("\n[T_finish_est分解]")

# CommWait (通常初始为0)
comm_wait = env._compute_comm_wait(v.id)
print(f"  CommWait_v2v: {comm_wait['total_v2v']:.3f}s")

# 计算典型传输时间
other = env.vehicles[1]
dist = np.linalg.norm(v.pos - other.pos)
rate = env.channel.compute_one_rate(v, other.pos, "V2V", env.time)
trans_time = task_data / max(rate, 1e-6) if task_data > 0 else 0.0
print(f"  Trans_time (到veh1): {trans_time:.3f}s (dist={dist:.1f}m, rate={rate:.2e}bps)")

# 计算队列等待时间
queue_wait = env._get_node_delay(other)
print(f"  Queue_wait (veh1): {queue_wait:.3f}s")

# 计算执行时间
comp_time = task_comp / max(other.cpu_freq, 1e-6)
print(f"  Comp_time: {comp_time:.3f}s (cpu_freq={other.cpu_freq:.2e}Hz)")

t_finish_est = comm_wait['total_v2v'] + trans_time + queue_wait + comp_time
print(f"  ============")
print(f"  T_finish_est总计: {t_finish_est:.3f}s")

# 6. 检查contact time
rel_vel = other.vel - v.vel
pos_diff = other.pos - v.pos
pos_diff_norm = np.linalg.norm(pos_diff)
if pos_diff_norm > 1e-6:
    rel_vel_proj = np.dot(rel_vel, pos_diff) / pos_diff_norm
    if rel_vel_proj > 0.1:
        time_to_break = (Cfg.V2V_RANGE - dist) / rel_vel_proj
    else:
        time_to_break = Cfg.V2V_RANGE / max(Cfg.VEL_MIN, 1e-6)
else:
    time_to_break = Cfg.V2V_RANGE / max(Cfg.VEL_MIN, 1e-6)

print("\n[Contact Time估计]")
print(f"  相对速度向量: ({rel_vel[0]:.2f}, {rel_vel[1]:.2f}) m/s")
print(f"  相对速度投影(远离方向): {rel_vel_proj:.2f} m/s")
print(f"  当前距离: {dist:.1f}m")
print(f"  估计接触时间: {time_to_break:.2f}s")

# 7. 结论
print("\n[诊断结论]")
if t_finish_est > time_to_break:
    print(f"  ⚠️  T_finish_est ({t_finish_est:.2f}s) > contact_time ({time_to_break:.2f}s)")
    print(f"  → V2V被过滤！原因:")
    if comp_time > time_to_break:
        print(f"     - 计算时间({comp_time:.2f}s)就已超过接触时间")
    if trans_time > time_to_break * 0.5:
        print(f"     - 传输时间({trans_time:.2f}s)占比过大")
    if rel_vel_proj > 5:
        print(f"     - 车辆正在快速远离(相对速度{rel_vel_proj:.1f}m/s)")
else:
    print(f"  ✓ T_finish_est ({t_finish_est:.2f}s) <= contact_time ({time_to_break:.2f}s)")
    print(f"  → V2V应该可用")

# 8. 实际mask检查
print("\n[实际Mask状态]")
for i, o in enumerate(obs[:3]):  # 检查前3辆车
    mask = o.get("target_mask", [])
    v2v_avail = int(np.sum(mask[2:])) if len(mask) > 2 else 0
    print(f"  Vehicle {i}: Local={mask[0]}, RSU={mask[1]}, V2V可用数={v2v_avail}")

# 9. 深入分析：为什么V2V被过滤？
print("\n" + "=" * 70)
print("深入分析：每对车辆的V2V可行性")
print("=" * 70)

v = env.vehicles[0]
task_comp_size = v.task_dag.total_comp.mean()
task_data_size = v.task_dag.total_data.mean()

rejected_reasons = {'dist': 0, 'queue_full': 0, 'contact_time': 0, 'accepted': 0}

for j, other in enumerate(env.vehicles):
    if v.id == other.id:
        continue

    dist = np.linalg.norm(v.pos - other.pos)

    # 检查距离
    if dist > Cfg.V2V_RANGE:
        rejected_reasons['dist'] += 1
        continue

    # 检查队列
    if env._is_veh_queue_full(other.id, task_comp_size):
        rejected_reasons['queue_full'] += 1
        continue

    # 计算T_finish_est
    rate = env.channel.compute_one_rate(v, other.pos, "V2V", env.time, power_dbm_override=Cfg.TX_POWER_MAX_DBM)
    rate = max(rate, 1e-6)
    trans_time = task_data_size / rate if task_data_size > 0 else 0.0
    queue_wait = env._get_node_delay(other)
    comp_time = task_comp_size / max(other.cpu_freq, 1e-6)
    comm_wait = env._compute_comm_wait(v.id)
    t_finish_est = comm_wait['total_v2v'] + trans_time + queue_wait + comp_time

    # 计算contact time
    rel_vel = other.vel - v.vel
    pos_diff = other.pos - v.pos
    pos_diff_norm = np.linalg.norm(pos_diff)
    if pos_diff_norm > 1e-6:
        rel_vel_proj = np.dot(rel_vel, pos_diff) / pos_diff_norm
        if rel_vel_proj > 0.1:
            time_to_break = (Cfg.V2V_RANGE - dist) / rel_vel_proj
        else:
            time_to_break = Cfg.V2V_RANGE / max(Cfg.VEL_MIN, 1e-6)
    else:
        time_to_break = Cfg.V2V_RANGE / max(Cfg.VEL_MIN, 1e-6)

    if t_finish_est > time_to_break:
        rejected_reasons['contact_time'] += 1
        print(f"  Veh{j}: T_finish={t_finish_est:.2f}s > contact={time_to_break:.2f}s (dist={dist:.0f}m, rel_v={rel_vel_proj:.1f}m/s)")
    else:
        rejected_reasons['accepted'] += 1

print(f"\n[过滤原因统计]")
print(f"  距离超限: {rejected_reasons['dist']}")
print(f"  队列已满: {rejected_reasons['queue_full']}")
print(f"  接触时间不足: {rejected_reasons['contact_time']}")
print(f"  可接受: {rejected_reasons['accepted']}")
