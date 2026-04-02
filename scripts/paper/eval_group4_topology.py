#!/usr/bin/env python3
"""
Group 4 标准评估脚本
加载 best_model.pth，deterministic=True 推理，200轮评估取均值
对 TERA-MAPPO、F-MAPPO、IPPO 三种算法 × 三种拓扑 进行独立评估
"""
import os
import sys
import json
import random
import numpy as np
import torch
from pathlib import Path

# 项目根路径
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.agent_factory import build_agent

# ── 评估配置 ──────────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 10       # 每个(算法×拓扑)评估轮数
SEEDS            = [42]    # 单种子快速运行
DEVICE           = "cpu"

# ── 待评估的 runs ──────────────────────────────────────────────────────────────
RUNS = {
    "TERA-MAPPO": {
        "balanced": "runs/rc1_batch1_part1_topology_20260323_182712/topology_balanced/full",
        "deep":     "runs/rc1_batch1_part1_topology_20260323_182712/topology_deep/full",
        "parallel": "runs/rc1_batch1_part1_topology_20260323_182712/topology_parallel/full",
    },
    "F-MAPPO": {
        "balanced": "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_balanced/fmappo_flat",
        "deep":     "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_deep/fmappo_flat",
        "parallel": "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_parallel/fmappo_flat",
    },
    "IPPO": {
        "balanced": "runs/rc1_batch1_part1_topology_20260323_182712/ippo_main",
        "deep":     "runs/rc1_batch1_part1_topology_20260323_182712/ippo_main",
        "parallel": "runs/rc1_batch1_part1_topology_20260323_182712/ippo_main",
    },
}

# ── 拓扑配置覆盖 ──────────────────────────────────────────────────────────────
TOPO_CONFIGS = {
    "balanced": {
        "TOPOLOGY_MODE": "balanced",
        "RSU_COUNT": 3,
    },
    "deep": {
        "TOPOLOGY_MODE": "deep",
        "RSU_COUNT": 3,
    },
    "parallel": {
        "TOPOLOGY_MODE": "parallel",
        "RSU_COUNT": 3,
    },
}


def _load_config_snapshot(run_dir: str) -> bool:
    """加载run目录的config_snapshot，更新Cfg和TC。"""
    snap_path = os.path.join(run_dir, "logs", "config_snapshot.json")
    if not os.path.exists(snap_path):
        snap_path = os.path.join(run_dir, "config_snapshot.json")
    if not os.path.exists(snap_path):
        print(f"  [警告] 未找到config_snapshot: {snap_path}")
        return False
    with open(snap_path, "r") as f:
        data = json.load(f)
    sc = data.get("system_config", {})
    tc = data.get("train_config", {})
    for k, v in sc.items():
        if hasattr(Cfg, k):
            setattr(Cfg, k, v)
    for k, v in tc.items():
        if hasattr(TC, k):
            setattr(TC, k, v)
    return True


def _apply_topo_override(topo: str):
    """覆盖拓扑相关配置。"""
    overrides = TOPO_CONFIGS.get(topo, {})
    for k, v in overrides.items():
        if hasattr(Cfg, k):
            setattr(Cfg, k, v)


def _build_ctde_global_state(env, obs_list):
    if hasattr(env, "_build_ctde_global_state"):
        g = np.asarray(env._build_ctde_global_state(), dtype=np.float32).reshape(-1)
    elif obs_list and isinstance(obs_list[0], dict) and obs_list[0].get("global_state") is not None:
        g = np.asarray(obs_list[0]["global_state"], dtype=np.float32).reshape(-1)
    else:
        g = np.zeros(int(getattr(TC, "CTDE_GLOBAL_DIM", 30)), dtype=np.float32)
    gdim = int(getattr(TC, "CTDE_GLOBAL_DIM", g.shape[0]))
    if g.shape[0] < gdim:
        g = np.pad(g, (0, gdim - g.shape[0]))
    elif g.shape[0] > gdim:
        g = g[:gdim]
    return g.astype(np.float32)


def _attach_global_state(obs_list, global_state):
    g = np.asarray(global_state, dtype=np.float32).reshape(-1)
    for obs in obs_list:
        obs["global_state"] = g.copy()


def evaluate_one_run(run_dir: str, topo: str, algo: str, n_episodes: int, seeds: list) -> dict:
    """
    加载 best_model.pth，deterministic=True 运行 n_episodes，返回均值指标。
    多seed取均值以减少随机性。
    """
    model_path = os.path.join(run_dir, "models", "best_model.pth")
    if not os.path.exists(model_path):
        # 退化为 last_model.pth
        model_path = os.path.join(run_dir, "models", "last_model.pth")
    if not os.path.exists(model_path):
        print(f"  [错误] 未找到模型: {run_dir}/models/")
        return {}

    print(f"  加载模型: {model_path}")

    # 加载配置
    _load_config_snapshot(run_dir)
    _apply_topo_override(topo)

    # 构建网络
    network = OffloadingPolicyNetwork(
        d_model=TC.EMBED_DIM,
        num_heads=TC.NUM_HEADS,
        num_layers=TC.NUM_LAYERS,
    )
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # 支持直接state_dict或包含state_dict的dict
    if isinstance(state, dict) and "network" in state:
        network.load_state_dict(state["network"])
    elif isinstance(state, dict) and "model_state_dict" in state:
        network.load_state_dict(state["model_state_dict"])
    else:
        try:
            network.load_state_dict(state)
        except Exception:
            # 可能是整个checkpoint dict
            for key in ["state_dict", "policy", "actor"]:
                if key in state:
                    network.load_state_dict(state[key])
                    break
    network.eval()

    algo_mode = getattr(TC, "ALGO_MODE", "mappo")
    agent = build_agent(network, device=DEVICE, algo_mode=algo_mode)

    all_sr, all_cft_c, all_cft_rem = [], [], []
    all_cwt, all_lbr_c, all_lbr_r = [], [], []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = VecOffloadingEnv()
        eps_per_seed = n_episodes // len(seeds)

        for ep in range(eps_per_seed):
            obs_list, info = env.reset()
            if not obs_list:
                continue

            # 附加全局状态
            global_state = _build_ctde_global_state(env, obs_list)
            _attach_global_state(obs_list, global_state)

            done = False
            ep_task_done = 0
            ep_task_total = 0
            ep_cft_sum = 0.0
            ep_cft_rem_sum = 0.0

            while not done:
                action_dict = agent.select_action(obs_list, deterministic=True)
                actions_list = action_dict["actions"] if isinstance(action_dict, dict) and "actions" in action_dict else action_dict
                obs_list, rewards, terminated, truncated, step_info = env.step(actions_list)
                done = terminated or truncated

                if obs_list:
                    global_state = _build_ctde_global_state(env, obs_list)
                    _attach_global_state(obs_list, global_state)

            # 从env获取episode统计（使用 _last_episode_metrics）
            ep_info = getattr(env, "_last_episode_metrics", {})
            if not ep_info:
                ep_info = getattr(env, "episode_info", {})
            if not ep_info:
                ep_info = getattr(env, "_last_episode_info", {}) or {}

            # 提取指标
            sr = ep_info.get("task_success_rate", ep_info.get("success_rate", np.nan))
            cft_c = ep_info.get("mean_cft_completed", ep_info.get("mean_cft", np.nan))
            cft_rem = ep_info.get("mean_cft_rem", 0.0)
            cwt = ep_info.get("dT_eff_mean", np.nan)
            lbr_c = ep_info.get("lbr_compute", np.nan)
            lbr_r = ep_info.get("lbr_radio", np.nan)

            if not np.isnan(sr):
                all_sr.append(sr)
            if not np.isnan(cft_c):
                all_cft_c.append(cft_c)
            all_cft_rem.append(cft_rem if not np.isnan(cft_rem) else 0.0)
            if not np.isnan(cwt):
                all_cwt.append(cwt)
            if not np.isnan(lbr_c):
                all_lbr_c.append(lbr_c)
            if not np.isnan(lbr_r):
                all_lbr_r.append(lbr_r)

        env.close() if hasattr(env, "close") else None

    result = {
        "sr_mean":       float(np.mean(all_sr))   if all_sr   else np.nan,
        "sr_std":        float(np.std(all_sr))    if all_sr   else np.nan,
        "cft_c_mean":    float(np.mean(all_cft_c)) if all_cft_c else np.nan,
        "cft_rem_mean":  float(np.mean(all_cft_rem)) if all_cft_rem else 0.0,
        "cft_all_mean":  float(np.mean(all_cft_c) + np.mean(all_cft_rem)) if (all_cft_c and all_cft_rem) else np.nan,
        "cwt_mean":      float(np.mean(all_cwt))  if all_cwt  else np.nan,
        "lbr_c_mean":    float(np.mean(all_lbr_c)) if all_lbr_c else np.nan,
        "lbr_r_mean":    float(np.mean(all_lbr_r)) if all_lbr_r else np.nan,
        "n_episodes":    len(all_sr),
    }
    cwt_str  = f"{result['cwt_mean']:.4f}"   if np.isfinite(result.get('cwt_mean',  float('nan'))) else "N/A"
    lbrc_str = f"{result['lbr_c_mean']:.4f}" if np.isfinite(result.get('lbr_c_mean',float('nan'))) else "N/A"
    lbrr_str = f"{result['lbr_r_mean']:.4f}" if np.isfinite(result.get('lbr_r_mean',float('nan'))) else "N/A"
    print(f"    SR={result['sr_mean']:.4f}±{result['sr_std']:.4f}  "
          f"CFT_c={result['cft_c_mean']:.4f}  CFT_all={result['cft_all_mean']:.4f}  "
          f"CWT={cwt_str}  LBR(c)={lbrc_str}  LBR(r)={lbrr_str}  "
          f"(n={result['n_episodes']})")
    return result


def main():
    os.chdir(ROOT)
    results = {}

    for algo, topo_runs in RUNS.items():
        results[algo] = {}
        for topo, run_dir in topo_runs.items():
            print(f"\n>>> 评估 {algo} / {topo}")
            r = evaluate_one_run(run_dir, topo, algo, N_EVAL_EPISODES, SEEDS)
            results[algo][topo] = r

    # 保存JSON结果
    out_dir = ROOT / "runs/paper_final_results_20260327/group4_topology_comparison"
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    # 打印汇总表
    print("\n========== 评估结果汇总 ==========")
    print(f"{'算法':<14} {'拓扑':<12} {'SR':>8} {'CFT_all':>10}")
    print("-" * 48)
    for algo in results:
        for topo in ["balanced", "deep", "parallel"]:
            r = results[algo].get(topo, {})
            sr  = r.get("sr_mean",  float("nan"))
            cft = r.get("cft_all_mean", float("nan"))
            print(f"{algo:<14} {topo:<12} {sr:>8.4f} {cft:>10.4f}")


if __name__ == "__main__":
    main()
