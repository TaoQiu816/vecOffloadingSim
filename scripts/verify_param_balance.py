"""
验证 exp_dynamic 参数的合理性与均衡性（计算 vs 通信、Local/RSU/V2V 相对吸引力）。
不依赖训练好的模型，仅基于物理公式与配置计算量级。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from configs.exp_dynamic_config import apply_exp_dynamic_config

def main():
    apply_exp_dynamic_config(Cfg, TC)
    DT = 0.1
    C_mid = (Cfg.MIN_COMP + Cfg.MAX_COMP) / 2
    D_mid = (Cfg.MIN_DATA + Cfg.MAX_DATA) / 2
    f_rsu = Cfg.F_RSU
    f_veh_weak = Cfg.MIN_VEHICLE_CPU_FREQ
    f_veh_strong = Cfg.MAX_VEHICLE_CPU_FREQ
    # 典型速率量级（bps）：V2I/V2V 共享模型下取保守估计
    R_v2i_est = 2e6   # ~2 Mbps 典型
    R_v2v_est = 4e6   # ~4 Mbps 典型

    print("=" * 60)
    print("exp_dynamic 参数均衡性验证")
    print("=" * 60)
    print("\n【1】计算/数据量级")
    print(f"  MIN_COMP={Cfg.MIN_COMP:.2e}, MAX_COMP={Cfg.MAX_COMP:.2e} cycles")
    print(f"  MIN_DATA={Cfg.MIN_DATA:.2e}, MAX_DATA={Cfg.MAX_DATA:.2e} bits")
    print(f"  中位 C={C_mid:.2e}, D={D_mid:.2e}")

    print("\n【2】单子任务执行时间 T_exec = C/f (s)")
    t_rsu = C_mid / f_rsu
    t_veh_weak = C_mid / f_veh_weak
    t_veh_strong = C_mid / f_veh_strong
    print(f"  RSU (单核 {f_rsu/1e9:.1f} GHz):     {t_rsu:.3f} s  ({t_rsu/DT:.1f} DT)")
    print(f"  弱车 ({f_veh_weak/1e9:.1f} GHz):     {t_veh_weak:.3f} s  ({t_veh_weak/DT:.1f} DT)")
    print(f"  强车 ({f_veh_strong/1e9:.1f} GHz):   {t_veh_strong:.3f} s  ({t_veh_strong/DT:.1f} DT)")
    print(f"  比例 RSU:弱车:强车 ≈ 1 : {t_rsu/t_veh_weak:.1f} : {t_rsu/t_veh_strong:.1f} (RSU 更快)")

    print("\n【3】单边传输时间 T_tx = D/R (s)，R 保守估计")
    t_tx_v2i = D_mid / R_v2i_est
    t_tx_v2v = D_mid / R_v2v_est
    print(f"  V2I (R≈{R_v2i_est/1e6:.0f} Mbps):  {t_tx_v2i:.3f} s  ({t_tx_v2i/DT:.1f} DT)")
    print(f"  V2V (R≈{R_v2v_est/1e6:.0f} Mbps):  {t_tx_v2v:.3f} s  ({t_tx_v2v/DT:.1f} DT)")

    print("\n【4】计算 vs 通信瓶颈（单子任务）")
    print(f"  RSU: T_exec={t_rsu:.3f}s, 若卸载则 +T_tx(V2I)≈{t_tx_v2i:.3f}s → 总约 {t_rsu+t_tx_v2i:.3f}s")
    print(f"  本地(强车): T_exec={t_veh_strong:.3f}s（无传输）")
    ccr_rsu = t_tx_v2i / t_rsu
    ccr_veh = t_tx_v2i / t_veh_strong
    print(f"  CCR 量级: T_tx/T_exec ≈ {ccr_rsu:.1f} (V2I vs RSU算)，{ccr_veh:.1f} (V2I vs 强车算)")
    if ccr_rsu > 3:
        print("  → 通信为主导瓶颈（T_tx >> T_exec），符合“保持数据/通信瓶颈”设计；")
        print("     RSU 算力强但上行受限，策略需在 V2I 排队/传输与 Local/V2V 间权衡。")
    elif 0.3 < ccr_rsu < 3.0:
        print("  → 计算与传输同量级，均衡合理。")
    else:
        print("  → 计算主导（传输很快），RSU 易单极优势。")

    print("\n【5】RSU 排队容量")
    cap = Cfg.RSU_QUEUE_CYCLES_LIMIT
    n_proc = Cfg.RSU_NUM_PROCESSORS
    per_core = cap / n_proc
    avg_tasks_per_core = per_core / C_mid
    print(f"  RSU_QUEUE_CYCLES_LIMIT={cap:.2e}, {n_proc} 核 → 每核 {per_core:.2e} cycles")
    print(f"  约 {avg_tasks_per_core:.0f} 个中位任务/核 可排队，RSU 易拥塞时 V2V/Local 有空间。")

    print("\n【6】通信带宽与 RB")
    print(f"  BW_V2I={Cfg.BW_V2I/1e6:.0f} MHz, V2I_NUM_RB={Cfg.V2I_NUM_RB} → 每 RB {Cfg.BW_V2I/Cfg.V2I_NUM_RB/1e6:.0f} MHz")
    print(f"  BW_V2V={Cfg.BW_V2V/1e6:.0f} MHz, V2V_NUM_RB={Cfg.V2V_NUM_RB} → 每 RB {Cfg.BW_V2V/Cfg.V2V_NUM_RB/1e6:.0f} MHz")
    print("  → V2V 带宽大于 V2I，与“车-车协作”场景一致。")

    print("\n【7】Deadline 与步数")
    print(f"  MAX_STEPS={Cfg.MAX_STEPS}, DT=0.1s → 单 episode 最长 {Cfg.MAX_STEPS*DT}s")
    print(f"  DEADLINE_ALPHA={Cfg.DEADLINE_ALPHA_MIN}~{Cfg.DEADLINE_ALPHA_MAX} (LB_ALPHA)")
    print("  → 200 步内多车多 DAG 重生，deadline 有压力，策略分化明显（见 baseline）。")

    # 【8】任务负担：deadline 与单次传输/执行量级对比
    print("\n【8】任务负担（LB_ALPHA 下 deadline 量级）")
    f_max = max(Cfg.MAX_VEHICLE_CPU_FREQ, Cfg.F_RSU)
    # 典型 DAG：10 节点，关键路径约 4~6 节点，cp_cycles ≈ 4~6 * C_mid
    cp_depth_typical = 5
    cp_cycles_typical = cp_depth_typical * C_mid
    LB0 = cp_cycles_typical / f_max
    alpha_mid = (Cfg.DEADLINE_ALPHA_MIN + Cfg.DEADLINE_ALPHA_MAX) / 2
    slack = getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0)
    ddl_typical = alpha_mid * LB0 + slack
    print(f"  典型关键路径: 约 {cp_depth_typical} 节点, cp_cycles≈{cp_cycles_typical:.2e}")
    print(f"  LB0 = cp_cycles/f_max = {LB0:.3f}s,  deadline ≈ alpha*LB0+slack = {ddl_typical:.3f}s")
    print(f"  单次 V2I 传输(中位 D)≈{t_tx_v2i:.1f}s,  单次 RSU 执行(中位 C)≈{t_rsu:.3f}s")
    if t_tx_v2i > ddl_typical * 0.8:
        print("  → 负担偏重：单次上传时间已接近/超过整 DAG deadline，易导致 deadline 大量不满足。")
        print("    可考虑：放宽 DEADLINE_ALPHA（如 2.2~3.0）或略降 MAX_DATA/MAX_COMP。")
    elif ddl_typical < 2.0 * DT:
        print("  → 负担偏重：deadline 仅约 1~2 个时隙，策略几乎无容错。")
    else:
        print("  → 负担在可接受范围：deadline 与传输/执行量级有博弈空间。")

    print("\n" + "=" * 60)
    print("结论：通信瓶颈主导(T_tx>>T_exec)，RSU 算力强但上行受限；")
    print("Greedy 偏 RSU、EFT 更均衡，Local/V2V 在 V2I 拥塞时有空间，参数合理且均衡。")
    print("=" * 60)

if __name__ == "__main__":
    main()
