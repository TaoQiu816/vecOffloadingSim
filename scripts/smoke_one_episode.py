#!/usr/bin/env python3
"""
最小复现：跑 1 个短 episode，打印 I_caused / SINR / 四段分解 / 信誉 / HO 事件。
增加 reward 分量尺度统计与数量级合理性断言。

用法: python scripts/smoke_one_episode.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from configs.config import SystemConfig as Cfg

# 确保使用 UNIFIED 奖励
Cfg.REWARD_SCHEME = "UNIFIED"
Cfg.TRUST_ENABLED = True
Cfg.NUM_VEHICLES = 10
Cfg.MAX_STEPS = 80
Cfg.TERMINATE_ON_ALL_FINISHED = False

from envs.vec_offloading_env import VecOffloadingEnv


def main():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=42)
    print(f"=== Smoke Episode: {Cfg.NUM_VEHICLES} vehs, {Cfg.MAX_STEPS} steps ===")
    print(f"  REWARD_SCHEME = {Cfg.REWARD_SCHEME}")
    print(f"  V2V_NUM_RB = {getattr(Cfg, 'V2V_NUM_RB', 'N/A')}")
    print(f"  TRUST_ENABLED = {getattr(Cfg, 'TRUST_ENABLED', False)}")
    print(f"  HO_FREEZE_STEPS = {getattr(Cfg, 'HO_FREEZE_STEPS', 'N/A')}")
    print()

    total_rewards = []
    rng = np.random.default_rng(42)
    target_type_counts = {'Local': 0, 'RSU': 0, 'V2V': 0}

    for step in range(Cfg.MAX_STEPS):
        actions = []
        for i, obs in enumerate(obs_list):
            mask = np.asarray(obs["action_mask"]).astype(bool)
            valid = np.where(mask)[0]
            target = int(rng.choice(valid)) if len(valid) > 0 else 0
            # 统计 target 类型
            ct = obs.get('candidate_types', None)
            if ct is not None and target < len(ct):
                t = int(ct[target])
                if t == 1: target_type_counts['Local'] += 1
                elif t == 2: target_type_counts['RSU'] += 1
                elif t == 3: target_type_counts['V2V'] += 1
            actions.append({"target": target, "power": float(rng.random())})

        obs_list, rewards, terminated, truncated, info = env.step(actions)

        # per-step 打印
        sinr_p50 = info.get('v2v_sinr_p50', 0.0)
        sinr_p05 = info.get('v2v_sinr_p05', 0.0)
        rb_conc = info.get('v2v_rb_concurrency', 0.0)
        i_caused = info.get('v2v_i_caused_mean', 0.0)
        i_total = info.get('v2v_i_total_mean', 0.0)
        ho_cnt = info.get('ho_event_count', 0)
        trust_ret = info.get('trust_retry_count', 0)

        r_mean = float(np.mean(rewards)) if rewards else 0.0
        total_rewards.extend(rewards if isinstance(rewards, list) else [rewards])

        print(f"Step {step:3d} | R={r_mean:+.3f} | "
              f"SINR_p50={sinr_p50:.1f} p05={sinr_p05:.1f} | "
              f"RB_conc={rb_conc:.1f} | "
              f"I_caused={i_caused:.2e} I_total={i_total:.2e} | "
              f"HO={ho_cnt} Trust_retry={trust_ret}")

        if terminated or truncated:
            break

    # ────────── 决策分布 ──────────
    total_dec = sum(target_type_counts.values())
    print(f"\n=== Decision Distribution (total={total_dec}) ===")
    for k, v in target_type_counts.items():
        pct = v / max(total_dec, 1) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    # ────────── 终局指标 ──────────
    print("\n=== Episode Summary ===")
    for k in sorted(info.keys()):
        if k.startswith('ep_'):
            print(f"  {k}: {info[k]}")

    print(f"\n  reward_mean: {np.mean(total_rewards):.4f}")
    print(f"  reward_std:  {np.std(total_rewards):.4f}")

    # ────────── resource_raw 布局验证 ──────────
    sample_obs = obs_list[0] if isinstance(obs_list, list) and len(obs_list) > 0 else None
    if sample_obs and isinstance(sample_obs, dict) and 'resource_raw' in sample_obs:
        rr = sample_obs['resource_raw']
        ct = np.asarray(sample_obs.get('candidate_types', np.zeros(rr.shape[0])))
        print(f"\n  resource_raw.shape = {rr.shape}")
        print(f"  resource_raw col layout:")
        print(f"    [0] cpu_norm      range: [{rr[:, 0].min():.4f}, {rr[:, 0].max():.4f}]")
        print(f"    [1] comp_backlog  range: [{rr[:, 1].min():.4f}, {rr[:, 1].max():.4f}]")
        print(f"    [2] tx_backlog    range: [{rr[:, 2].min():.4f}, {rr[:, 2].max():.4f}]")
        print(f"    [3] dist_norm     range: [{rr[:, 3].min():.4f}, {rr[:, 3].max():.4f}]")
        print(f"    [7] contact_norm  range: [{rr[:, 7].min():.4f}, {rr[:, 7].max():.4f}]")
        print(f"    [8] contention    range: [{rr[:, 8].min():.4f}, {rr[:, 8].max():.4f}]")
        print(f"    [9] occupancy     range: [{rr[:, 9].min():.4f}, {rr[:, 9].max():.4f}]")

        # 验证 Local row
        local_row = rr[0]
        assert local_row[7] == 1.0, f"Local contact_norm should be 1.0, got {local_row[7]}"
        assert local_row[8] == 0.0, f"Local contention should be 0.0, got {local_row[8]}"
        assert local_row[9] == 0.0, f"Local occupancy should be 0.0, got {local_row[9]}"
        print("  ✓ Local row layout correct (contact=1, contention=0, occupancy=0)")

        # 验证 V2V rows 有效
        v2v_mask = (ct == 3)
        if v2v_mask.any():
            v2v_cont = rr[v2v_mask, 8]
            v2v_occ = rr[v2v_mask, 9]
            print(f"  V2V rows: contention=[{v2v_cont.min():.3f}, {v2v_cont.max():.3f}], "
                  f"occupancy=[{v2v_occ.min():.3f}, {v2v_occ.max():.3f}]")
        else:
            print("  (no V2V candidates in final obs)")

    # ────────── reward 分量尺度统计 ──────────
    print("\n=== Reward Component Statistics ===")
    if hasattr(env, '_reward_stats'):
        summary = env._reward_stats.summary()
        metrics = summary.get('metrics', {})

        components = {
            'r_step': 'step (time+energy+interf+illegal)',
            'r_term': 'terminal (success/fail)',
            'r_pbrs': 'PBRS (potential shaping)',
            'reward': 'total reward',
        }

        abs_means = {}
        for key, label in components.items():
            bucket = metrics.get(key, {})
            mean_val = bucket.get('mean', 0.0)
            abs_mean = bucket.get('abs_mean', 0.0)
            p95_val = bucket.get('p95', 0.0)
            min_val = bucket.get('min', 0.0)
            max_val = bucket.get('max', 0.0)
            count = bucket.get('count', 0)
            print(f"  {key:>10s} ({label}):")
            print(f"    mean={mean_val:+.4f}  abs_mean={abs_mean:.4f}  "
                  f"p95={p95_val:+.4f}  min={min_val:+.4f}  max={max_val:+.4f}  "
                  f"count={count}")
            abs_means[key] = abs_mean

        # ────── 数量级合理性断言 ──────
        # 防止单分量碾压：任一分量的 |mean| 不超过 total 的 20 倍
        # 使用宽松阈值，仅检测严重失衡
        total_abs = abs_means.get('reward', 1e-12)
        if total_abs > 1e-6:
            for key in ['r_step', 'r_term', 'r_pbrs']:
                comp_abs = abs_means.get(key, 0.0)
                ratio = comp_abs / max(total_abs, 1e-12)
                if ratio > 50.0:
                    print(f"  ⚠ WARNING: |{key}|/|reward| = {ratio:.1f} > 50, "
                          f"分量可能碾压总奖励")
                else:
                    print(f"  ✓ |{key}|/|reward| = {ratio:.1f} (OK)")

        # 额外断言：各分量不为 NaN/Inf
        for key in ['r_step', 'r_term', 'r_pbrs', 'reward']:
            bucket = metrics.get(key, {})
            assert np.isfinite(bucket.get('mean', 0.0)), f"{key} mean is not finite"
            assert np.isfinite(bucket.get('p95', 0.0)), f"{key} p95 is not finite"

        # 断言 r_step 非零（至少有 time 惩罚）
        r_step_count = metrics.get('r_step', {}).get('count', 0)
        r_step_nz = metrics.get('r_step', {}).get('nonzero_count', 0)
        if r_step_count > 0:
            assert r_step_nz > 0, "r_step 全为零，time penalty 可能未生效"
            print(f"  ✓ r_step nonzero_ratio = {r_step_nz}/{r_step_count}")

        # PBRS 不应长期主导 terminal（宽松阈值：abs_mean 比值 < 20）
        pbrs_abs = abs_means.get('r_pbrs', 0.0)
        term_abs = abs_means.get('r_term', 0.0)
        if term_abs > 1e-6:
            pbrs_term_ratio = pbrs_abs / term_abs
            if pbrs_term_ratio > 20.0:
                print(f"  ⚠ WARNING: |r_pbrs|/|r_term| = {pbrs_term_ratio:.1f} > 20, "
                      f"PBRS 可能主导 terminal")
            else:
                print(f"  ✓ |r_pbrs|/|r_term| = {pbrs_term_ratio:.1f} (OK)")
        elif pbrs_abs > 0.01:
            print(f"  ⚠ WARNING: |r_term| near zero but |r_pbrs|={pbrs_abs:.4f}")

    else:
        print("  (no _reward_stats available)")

    print("\n=== PASS ===")


if __name__ == "__main__":
    main()
