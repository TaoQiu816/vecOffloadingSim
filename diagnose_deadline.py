"""
深度诊断：Deadline合理性分析

分析为什么Task SR = 0%，Subtask SR = 10%
"""
import sys
import numpy as np
from configs.config import SystemConfig as Cfg

print("="*80)
print("Deadline合理性深度诊断")
print("="*80)

print("\n【当前参数配置】")
print(f"MIN_COMP: {Cfg.MIN_COMP:.2e} cycles ({Cfg.MIN_COMP/1e9:.3f}G)")
print(f"MAX_COMP: {Cfg.MAX_COMP:.2e} cycles ({Cfg.MAX_COMP/1e9:.3f}G)")
print(f"MIN_DATA: {Cfg.MIN_DATA:.2e} bits ({Cfg.MIN_DATA/8/1e3:.0f}KB)")
print(f"MAX_DATA: {Cfg.MAX_DATA:.2e} bits ({Cfg.MAX_DATA/8/1e3:.0f}KB)")
print(f"MIN_EDGE_DATA: {Cfg.MIN_EDGE_DATA:.2e} bits ({Cfg.MIN_EDGE_DATA/8/1e3:.0f}KB)")
print(f"MAX_EDGE_DATA: {Cfg.MAX_EDGE_DATA:.2e} bits ({Cfg.MAX_EDGE_DATA/8/1e3:.0f}KB)")
print(f"")
print(f"MIN_CPU: {Cfg.MIN_VEHICLE_CPU_FREQ/1e9:.1f}GHz")
print(f"MAX_CPU: {Cfg.MAX_VEHICLE_CPU_FREQ/1e9:.1f}GHz")
print(f"RSU_CPU: {Cfg.F_RSU/1e9:.1f}GHz")
print(f"BW_V2I: {Cfg.BW_V2I/1e6:.0f}Mbps")
print(f"BW_V2V: {Cfg.BW_V2V/1e6:.0f}Mbps")
print(f"")
print(f"DEADLINE_TIGHTENING: [{Cfg.DEADLINE_TIGHTENING_MIN}, {Cfg.DEADLINE_TIGHTENING_MAX}]")
print(f"DT: {Cfg.DT}s")

# 计算平均值
avg_comp = (Cfg.MIN_COMP + Cfg.MAX_COMP) / 2
avg_data = (Cfg.MIN_DATA + Cfg.MAX_DATA) / 2
avg_edge_data = (Cfg.MIN_EDGE_DATA + Cfg.MAX_EDGE_DATA) / 2
avg_cpu = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2

print("\n" + "="*80)
print("【场景1: 平均DAG (10节点)】")
print("="*80)

n_nodes = 10
total_comp_avg = avg_comp * n_nodes
total_input_data_avg = avg_data * n_nodes
# 假设每个节点平均2条入边
total_edge_data_avg = avg_edge_data * n_nodes * 2

print(f"\n总计算量: {total_comp_avg/1e9:.2f}G cycles")
print(f"总输入数据: {total_input_data_avg/8/1e6:.2f}MB")
print(f"总边数据: {total_edge_data_avg/8/1e6:.2f}MB")
print(f"总数据量: {(total_input_data_avg + total_edge_data_avg)/8/1e6:.2f}MB")

print(f"\n--- Deadline计算（dag_generator逻辑）---")
f_median = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2
base_time = total_comp_avg / f_median
deadline_min = Cfg.DEADLINE_TIGHTENING_MIN * base_time
deadline_max = Cfg.DEADLINE_TIGHTENING_MAX * base_time

print(f"CPU中位数: {f_median/1e9:.1f}GHz")
print(f"Base Time: {base_time:.3f}s (仅计算，不含传输)")
print(f"Deadline范围: [{deadline_min:.3f}s, {deadline_max:.3f}s]")
print(f"Deadline平均: {(deadline_min + deadline_max)/2:.3f}s")

print(f"\n--- 本地执行（单车，顺序执行）---")
# 最快车
local_time_fast = total_comp_avg / Cfg.MAX_VEHICLE_CPU_FREQ
print(f"最快车(3GHz): {local_time_fast:.3f}s ({local_time_fast/0.05:.0f}步)")

# 平均车
local_time_avg = total_comp_avg / avg_cpu
print(f"平均车(2GHz): {local_time_avg:.3f}s ({local_time_avg/0.05:.0f}步)")

# 最慢车
local_time_slow = total_comp_avg / Cfg.MIN_VEHICLE_CPU_FREQ
print(f"最慢车(1GHz): {local_time_slow:.3f}s ({local_time_slow/0.05:.0f}步)")

# 与deadline对比
print(f"\n【本地执行 vs Deadline】")
print(f"最快车: {local_time_fast:.3f}s < Deadline {deadline_min:.3f}s ✓ (余量{(deadline_min - local_time_fast)/deadline_min*100:.0f}%)")
print(f"平均车: {local_time_avg:.3f}s < Deadline {(deadline_min + deadline_max)/2:.3f}s ✓ (余量{((deadline_min + deadline_max)/2 - local_time_avg)/((deadline_min + deadline_max)/2)*100:.0f}%)")
print(f"最慢车: {local_time_slow:.3f}s vs Deadline {deadline_max:.3f}s ", end='')
if local_time_slow < deadline_max:
    print(f"✓ (余量{(deadline_max - local_time_slow)/deadline_max*100:.0f}%)")
else:
    print(f"❌ (超时{(local_time_slow - deadline_max)/deadline_max*100:.0f}%)")

print(f"\n--- RSU执行（单车独占50Mbps）---")
# 上传输入数据
tx_input = total_input_data_avg / Cfg.BW_V2I
print(f"上传输入数据: {tx_input:.3f}s ({tx_input/0.05:.0f}步)")

# RSU计算
rsu_comp = total_comp_avg / Cfg.F_RSU
print(f"RSU计算: {rsu_comp:.3f}s ({rsu_comp/0.05:.0f}步)")

# 下载结果（假设10KB）
rx_result = 10 * 8 * 1e3 / Cfg.BW_V2I
print(f"下载结果: {rx_result:.3f}s ({rx_result/0.05:.0f}步)")

# 总时间
rsu_total = tx_input + rsu_comp + rx_result
print(f"RSU总时间: {rsu_total:.3f}s ({rsu_total/0.05:.0f}步)")

print(f"\n【RSU vs Deadline】")
print(f"RSU总时间: {rsu_total:.3f}s < Deadline {deadline_min:.3f}s ", end='')
if rsu_total < deadline_min:
    print(f"✓ (余量{(deadline_min - rsu_total)/deadline_min*100:.0f}%)")
    print(f"RSU相比本地(平均车)加速: {(1 - rsu_total/local_time_avg)*100:.0f}%")
else:
    print(f"❌ (超时{(rsu_total - deadline_min)/deadline_min*100:.0f}%)")

print(f"\n--- RSU执行（12车共享50Mbps，排队等待）---")
# 12车按顺序排队上传
n_vehicles = 12
tx_input_queued = (total_input_data_avg * n_vehicles) / Cfg.BW_V2I
print(f"12车排队上传: {tx_input_queued:.3f}s ({tx_input_queued/0.05:.0f}步)")

# 假设RSU按FIFO处理，每个任务需要rsu_comp时间
rsu_comp_queued = rsu_comp * n_vehicles
print(f"12车RSU计算(串行): {rsu_comp_queued:.3f}s ({rsu_comp_queued/0.05:.0f}步)")

# 最后一车完成时间（最坏情况）
rsu_total_worst = tx_input_queued + rsu_comp_queued
print(f"最后一车完成: {rsu_total_worst:.3f}s ({rsu_total_worst/0.05:.0f}步)")

print(f"\n【12车竞争场景】")
episode_duration = 10.0  # 假设episode时长10秒
print(f"Episode时长: {episode_duration}s ({episode_duration/0.05:.0f}步)")
print(f"最后一车完成: {rsu_total_worst:.3f}s ", end='')
if rsu_total_worst < episode_duration:
    print(f"✓ (余量{(episode_duration - rsu_total_worst)/episode_duration*100:.0f}%)")
else:
    print(f"❌ (超时{(rsu_total_worst - episode_duration)/episode_duration*100:.0f}%)")

print(f"\n第1车Deadline: {deadline_min:.3f}s, 实际完成: {tx_input + rsu_comp:.3f}s ", end='')
if tx_input + rsu_comp < deadline_min:
    print("✓")
else:
    print("❌")

print(f"第12车Deadline: {deadline_max:.3f}s, 实际完成: {rsu_total_worst:.3f}s ", end='')
if rsu_total_worst < deadline_max:
    print("✓")
else:
    print(f"❌ (超时{(rsu_total_worst - deadline_max):.1f}s)")

print("\n" + "="*80)
print("【关键发现】")
print("="*80)

# 诊断1：单车能否完成？
single_ok = (local_time_avg < deadline_max and rsu_total < deadline_min)
if single_ok:
    print("✓ 单车场景：本地和RSU都能在deadline内完成")
else:
    print("❌ 单车场景：无法在deadline内完成")

# 诊断2：12车竞争是否导致问题？
queue_problem = (rsu_total_worst > deadline_max)
if queue_problem:
    print("❌ 队列拥塞：12车全部卸载RSU会导致大量超时")
    print(f"   → 第{int(deadline_max / (tx_input + rsu_comp)) + 1}车开始超deadline")

# 诊断3：Deadline是否考虑传输？
deadline_includes_tx = (deadline_max > local_time_avg * 1.5)
if deadline_includes_tx:
    print(f"✓ Deadline留有余量: {(deadline_max - local_time_avg)/local_time_avg*100:.0f}% buffer")
else:
    print(f"⚠️ Deadline余量不足: 仅{(deadline_max - local_time_avg)/local_time_avg*100:.0f}% buffer")

# 诊断4：最坏情况
print(f"\n【最坏情况分析】")
print(f"最慢车(1GHz) + 最大DAG(12节点) + 最大计算量:")
worst_comp = Cfg.MAX_COMP * 12
worst_local = worst_comp / Cfg.MIN_VEHICLE_CPU_FREQ
worst_deadline = Cfg.DEADLINE_TIGHTENING_MAX * (worst_comp / f_median)
print(f"  本地执行: {worst_local:.3f}s")
print(f"  Deadline: {worst_deadline:.3f}s")
print(f"  结果: ", end='')
if worst_local < worst_deadline:
    print(f"✓ 可完成 (余量{(worst_deadline - worst_local)/worst_deadline*100:.0f}%)")
elif worst_local < episode_duration:
    print(f"⚠️ 超deadline但未超episode (超时{(worst_local - worst_deadline):.1f}s)")
else:
    print(f"❌ 超episode (超时{(worst_local - episode_duration):.1f}s)")

print("\n" + "="*80)
print("【根因分析】")
print("="*80)
print("""
可能原因：
1. ❌ Deadline基于CPU中位数(2GHz)计算，但有1GHz车辆
   → 慢车的base_time被低估50%
   
2. ❌ Deadline仅考虑计算时间，未考虑传输时间
   → RSU卸载需要上传时间，但deadline未预留
   
3. ❌ 12车竞争导致队列等待时间远超预期
   → 即使单车可完成，多车共享资源时大量超时
   
4. ⚠️ DAG依赖导致串行执行
   → 10节点DAG可能有6-8层，无法并行
   
5. ⚠️ 数据传输被低估
   → Edge数据(100-500KB)可能导致额外传输延迟
""")

print("\n【修复建议】")
print("="*80)
print("""
方案A: 放宽Deadline（治标）
  - DEADLINE_TIGHTENING_MIN: 2.0 → 4.0
  - DEADLINE_TIGHTENING_MAX: 3.0 → 6.0
  - 原因：给传输和队列等待留出时间
  
方案B: 修改Deadline计算基准（治本）
  - 使用最慢车CPU(1GHz)而非中位数(2GHz)
  - 或使用最慢车的1.5倍作为安全buffer
  - 原因：确保所有车辆都能完成
  
方案C: 减少车辆数量（降低竞争）
  - NUM_VEHICLES: 12 → 6
  - 原因：减少队列拥塞
  
方案D: 增加带宽（减少传输瓶颈）
  - BW_V2I: 50Mbps → 100Mbps
  - 原因：支持更多车辆同时卸载
  
推荐组合:
  1. 短期：方案A（放宽到4-6倍）
  2. 长期：方案B（修改计算基准）+ 方案D（增加带宽到100Mbps）
""")

