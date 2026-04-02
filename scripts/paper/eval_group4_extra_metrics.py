#!/usr/bin/env python3
"""
Group 4 附加指标评估脚本：CWT、LBR(c)、FoV

对 TERA-MAPPO、F-MAPPO、IPPO（RL方法）和 Greedy、Local-Only（基线）
在三种拓扑（Balanced/Deep/Parallel）下评估三个指标：
  - CWT  : 计算等待时间均值  (env._last_episode_metrics['dT_eff_mean'])
  - LBR(c): 计算节点负载均衡率 (Jain公平指数，env._last_episode_metrics['lbr_compute'])
  - FoV   : 车辆公平性指数 (Jain公平指数，env._last_episode_metrics['fov'])

拓扑通过 DAG 参数 (DAG_FAT, DAG_DENSITY, DAG_REGULAR) 实现差异化。
"""
import os
import sys
import json
import random
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.agent_factory import build_agent
from baselines.greedy_policy import GreedyPolicy
from baselines.local_only_policy import LocalOnlyPolicy

# ── 评估配置 ──────────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 10
# 每个拓扑使用不同种子，确保不同 DAG 实例；同一拓扑内各算法共享种子以保证公平比较
TOPO_SEEDS = {
    "balanced": [42],
    "deep":     [43],
    "parallel": [44],
}
DEVICE           = "cpu"

# ── RL runs（同 eval_group4_topology.py） ─────────────────────────────────────
RL_RUNS = {
    "TERA-MAPPO": {
        # 拓扑专用run不存在，使用 ablation full run + _apply_topo 覆盖拓扑
        "balanced": "runs/rc1_ablation_1500ep_20260322_180707/full",
        "deep":     "runs/rc1_ablation_1500ep_20260322_180707/full",
        "parallel": "runs/rc1_ablation_1500ep_20260322_180707/full",
    },
    "F-MAPPO": {
        "balanced": "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_balanced/fmappo_flat",
        "deep":     "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_deep/fmappo_flat",
        "parallel": "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_parallel/fmappo_flat",
    },
    "IPPO": {
        # 拓扑专用run不存在，使用 vehicle_20 ippo run + _apply_topo 覆盖拓扑
        "balanced": "runs/rc1_batch2_vehicle_20260324_181254/vehicle_20/ippo",
        "deep":     "runs/rc1_batch2_vehicle_20260324_181254/vehicle_20/ippo",
        "parallel": "runs/rc1_batch2_vehicle_20260324_181254/vehicle_20/ippo",
    },
}

# ── 拓扑配置覆盖（基于 F-MAPPO 训练配置的 DAG 参数） ────────────────────────
# 来源: rc1_batch1_topology_fmappo_20260328_224844/topology_*/fmappo_flat/logs/config_snapshot.json
TOPO_CONFIGS = {
    "balanced": {
        "DAG_FAT": 1.0,
        "DAG_DENSITY": 0.24,
        "DAG_REGULAR": 0.5,
        "DAG_CCR": 0.2,
        # 禁用 DAG 域随机化，确保评估时拓扑一致
        "DR_DAG_FAT_MIN": None,
        "DR_DAG_FAT_MAX": None,
        "DR_DAG_DENSITY_MIN": None,
        "DR_DAG_DENSITY_MAX": None,
    },
    "deep": {
        "DAG_FAT": 0.35,
        "DAG_DENSITY": 0.36,
        "DAG_REGULAR": 0.2,
        "DAG_CCR": 0.2,
        "DR_DAG_FAT_MIN": None,
        "DR_DAG_FAT_MAX": None,
        "DR_DAG_DENSITY_MIN": None,
        "DR_DAG_DENSITY_MAX": None,
    },
    "parallel": {
        "DAG_FAT": 1.0,
        "DAG_DENSITY": 0.12,
        "DAG_REGULAR": 0.8,
        "DAG_CCR": 0.2,
        "DR_DAG_FAT_MIN": None,
        "DR_DAG_FAT_MAX": None,
        "DR_DAG_DENSITY_MIN": None,
        "DR_DAG_DENSITY_MAX": None,
    },
}

# 使用 ablation full run 作为 baseline 配置初始化
BASELINE_CONFIG_RUN = "runs/rc1_ablation_1500ep_20260322_180707/full"


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _load_config_snapshot(run_dir: str) -> bool:
    """加载 run 目录的 config_snapshot / config.json，更新 Cfg 和 TC。"""
    # 优先尝试 config_snapshot.json（嵌套格式）
    for path in [
        os.path.join(run_dir, "logs", "config_snapshot.json"),
        os.path.join(run_dir, "config_snapshot.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for k, v in data.get("system_config", {}).items():
                if hasattr(Cfg, k):
                    setattr(Cfg, k, v)
            for k, v in data.get("train_config", {}).items():
                if hasattr(TC, k):
                    setattr(TC, k, v)
            return True
    # 尝试 config.json（直接键值对格式）
    cfg_json = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_json):
        with open(cfg_json) as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(Cfg, k):
                setattr(Cfg, k, v)
            if hasattr(TC, k):
                setattr(TC, k, v)
        return True
    print(f"  [警告] 未找到config_snapshot/config.json: {run_dir}")
    return False


def _apply_topo(topo_name: str):
    """覆盖 DAG 拓扑相关 Cfg 配置。"""
    cfg = TOPO_CONFIGS[topo_name]
    for k, v in cfg.items():
        setattr(Cfg, k, v)
    print(f"  [拓扑] {topo_name}: DAG_FAT={Cfg.DAG_FAT}, DAG_DENSITY={Cfg.DAG_DENSITY}, DAG_REGULAR={Cfg.DAG_REGULAR}")


def _build_ctde_global_state(env, obs_list):
    """拼接全局状态向量（简化版，与 eval_group4_topology.py 保持一致）。"""
    parts = []
    for obs in obs_list:
        for key in ["position", "velocity", "obs_vec"]:
            v = obs.get(key)
            if v is not None:
                parts.append(np.asarray(v, dtype=np.float32).flatten())
    return np.concatenate(parts, axis=0) if parts else np.zeros(1, dtype=np.float32)


def _attach_global_state(obs_list, global_state):
    """将全局状态附加到每个 obs 中。"""
    for obs in obs_list:
        obs["global_state"] = global_state


def _collect_episode_metrics(env):
    """从 env._last_episode_metrics 提取 CWT（全流程等待）、CWT分量 和 FoV。
    CWT = cwt_full_mean: 所有子任务的 cpu_start - ready_time 均值，
    含通信等待 + 传输时间 + CPU 排队等待。
    cwt_comm_mean: 通信等待（含传输），cwt_cpu_queue_mean: 计算排队等待。
    """
    ep = getattr(env, "_last_episode_metrics", {}) or {}
    # 优先使用全流程等待时间，回退到旧的 dT_eff_mean
    cwt = ep.get("cwt_full_mean", ep.get("dT_eff_mean", np.nan))
    cwt_comm = ep.get("cwt_comm_mean", np.nan)
    cwt_cpu_q = ep.get("cwt_cpu_queue_mean", np.nan)
    fov = ep.get("fov", np.nan)
    return cwt, cwt_comm, cwt_cpu_q, fov


# ─────────────────────────────────────────────────────────────────────────────
# RL 方法评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rl(run_dir: str, topo: str, algo: str,
                n_episodes: int = N_EVAL_EPISODES,
                seeds=None, topo_key: str = "balanced") -> dict:
    """评估单个 RL run 的 CWT、LBR(c)、FoV。"""
    if seeds is None:
        seeds = TOPO_SEEDS.get(topo_key, [42])

    _load_config_snapshot(run_dir)
    _apply_topo(topo)

    # 载入模型
    ckpt_paths = [
        os.path.join(run_dir, "models", "best_model.pth"),
        os.path.join(run_dir, "best_model.pth"),
        os.path.join(run_dir, "checkpoints", "best_model.pth"),
    ]
    ckpt = None
    for p in ckpt_paths:
        if os.path.exists(p):
            ckpt = p
            break
    if ckpt is None:
        print(f"  [警告] 未找到模型 checkpoint: {run_dir}")
        return {"cwt": np.nan, "lbr_c": np.nan, "fov": np.nan, "n": 0}

    raw_ckpt = torch.load(ckpt, map_location=DEVICE)
    # 支持多种 checkpoint 格式
    if isinstance(raw_ckpt, dict):
        if "network_state_dict" in raw_ckpt:
            state_dict = raw_ckpt["network_state_dict"]
        elif "network" in raw_ckpt:
            state_dict = raw_ckpt["network"]
        elif "model_state_dict" in raw_ckpt:
            state_dict = raw_ckpt["model_state_dict"]
        else:
            state_dict = raw_ckpt
        # 从 checkpoint 中提取 algo_mode（比 TC 更可靠）
        ckpt_algo = raw_ckpt.get("algo_mode", None)
    else:
        state_dict = raw_ckpt
        ckpt_algo = None

    network = OffloadingPolicyNetwork(
        d_model=getattr(TC, "EMBED_DIM", 128),
        num_heads=getattr(TC, "NUM_HEADS", 4),
        num_layers=getattr(TC, "NUM_LAYERS", 3),
    ).to(DEVICE)
    try:
        network.load_state_dict(state_dict, strict=True)
        print(f"  [OK] 模型权重严格加载成功")
    except RuntimeError as e:
        has_flat_proj = any("flat_node_proj" in k for k in state_dict.keys())
        if has_flat_proj:
            print(f"  [信息] 检测到 Flat 类型权重，尝试非严格加载")
        else:
            print(f"  [警告] strict=True 失败，改用 strict=False: {e}")
        network.load_state_dict(state_dict, strict=False)
    network.eval()

    # 优先使用 checkpoint 中记录的 algo_mode，其次使用 TC，最后默认 mappo
    algo_mode = ckpt_algo or getattr(TC, "ALGO_MODE", "mappo")
    print(f"  algo_mode={algo_mode}")
    agent = build_agent(network, device=DEVICE, algo_mode=algo_mode)

    all_cwt, all_cwt_comm, all_cwt_cpu_q, all_fov = [], [], [], []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = VecOffloadingEnv()
        for ep in range(n_episodes // len(seeds)):
            obs_list, _ = env.reset()
            if not obs_list:
                continue
            gs = _build_ctde_global_state(env, obs_list)
            _attach_global_state(obs_list, gs)
            done = False
            while not done:
                action_dict = agent.select_action(obs_list, deterministic=True)
                actions = (action_dict["actions"]
                           if isinstance(action_dict, dict) and "actions" in action_dict
                           else action_dict)
                obs_list, _, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated
                if obs_list:
                    gs = _build_ctde_global_state(env, obs_list)
                    _attach_global_state(obs_list, gs)
            cwt, cwt_comm, cwt_cpu_q, fov = _collect_episode_metrics(env)
            if not np.isnan(cwt): all_cwt.append(cwt)
            if not np.isnan(cwt_comm): all_cwt_comm.append(cwt_comm)
            if not np.isnan(cwt_cpu_q): all_cwt_cpu_q.append(cwt_cpu_q)
            if not np.isnan(fov): all_fov.append(fov)
        env.close() if hasattr(env, "close") else None

    return {
        "cwt":         float(np.mean(all_cwt)) if all_cwt else np.nan,
        "cwt_comm":    float(np.mean(all_cwt_comm)) if all_cwt_comm else np.nan,
        "cwt_cpu_q":   float(np.mean(all_cwt_cpu_q)) if all_cwt_cpu_q else np.nan,
        "fov":         float(np.mean(all_fov)) if all_fov else np.nan,
        "n":           len(all_cwt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 方法评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(policy_name: str, topo: str,
                      n_episodes: int = N_EVAL_EPISODES,
                      seeds=None, topo_key: str = "balanced") -> dict:
    """评估 Greedy / Local-Only 基线的 CWT（全流程等待）和 FoV。"""
    if seeds is None:
        seeds = TOPO_SEEDS.get(topo_key, [42])

    # 使用 TERA-MAPPO balanced config 初始化环境（保持配置一致）
    _load_config_snapshot(BASELINE_CONFIG_RUN)
    _apply_topo(topo)

    all_cwt, all_cwt_comm, all_cwt_cpu_q, all_fov = [], [], [], []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        env = VecOffloadingEnv()

        if policy_name == "Greedy":
            policy = GreedyPolicy(env)
        elif policy_name == "Local-Only":
            policy = LocalOnlyPolicy()
        else:
            raise ValueError(f"未知 baseline: {policy_name}")

        for ep in range(n_episodes // len(seeds)):
            obs_list, _ = env.reset()
            if not obs_list:
                continue

            if policy_name == "Greedy":
                policy.env = env  # reset 后重新绑定 env

            done = False
            while not done:
                actions = policy.select_action(obs_list)
                obs_list, _, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated

            cwt, cwt_comm, cwt_cpu_q, fov = _collect_episode_metrics(env)
            if not np.isnan(cwt): all_cwt.append(cwt)
            if not np.isnan(cwt_comm): all_cwt_comm.append(cwt_comm)
            if not np.isnan(cwt_cpu_q): all_cwt_cpu_q.append(cwt_cpu_q)
            if not np.isnan(fov): all_fov.append(fov)
        env.close() if hasattr(env, "close") else None

    return {
        "cwt":         float(np.mean(all_cwt)) if all_cwt else np.nan,
        "cwt_comm":    float(np.mean(all_cwt_comm)) if all_cwt_comm else np.nan,
        "cwt_cpu_q":   float(np.mean(all_cwt_cpu_q)) if all_cwt_cpu_q else np.nan,
        "fov":         float(np.mean(all_fov)) if all_fov else np.nan,
        "n":   len(all_cwt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}  # results[method][topo] = {cwt, cwt_comm, cwt_cpu_q, fov, n}
    topos = ["balanced", "deep", "parallel"]
    topo_labels = {"balanced": "Balanced", "deep": "Deep", "parallel": "Parallel"}

    # ── RL 方法 ──────────────────────────────────────────────────────────────
    for algo, topo_runs in RL_RUNS.items():
        results[algo] = {}
        for topo, run_dir in topo_runs.items():
            print(f"\n>>> 评估 {algo} / {topo_labels[topo]}")
            r = evaluate_rl(run_dir, topo, algo, topo_key=topo)
            results[algo][topo] = r
            fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else "N/A"
            print(f"    CWT={fmt(r['cwt'])}  comm={fmt(r['cwt_comm'])}  cpu_q={fmt(r['cwt_cpu_q'])}  FoV={fmt(r['fov'])}  (n={r['n']})")

    # ── Baseline 方法 ─────────────────────────────────────────────────────────
    for baseline in ["Greedy", "Local-Only"]:
        results[baseline] = {}
        for topo in topos:
            print(f"\n>>> 评估 {baseline} / {topo_labels[topo]}")
            r = evaluate_baseline(baseline, topo, topo_key=topo)
            results[baseline][topo] = r
            fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else "N/A"
            print(f"    CWT={fmt(r['cwt'])}  comm={fmt(r['cwt_comm'])}  cpu_q={fmt(r['cwt_cpu_q'])}  FoV={fmt(r['fov'])}  (n={r['n']})")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out_dir = ROOT / "runs/paper_final_results_20260327/group4_topology_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "extra_metrics.json"
    # 转换 nan 为 null 以便 JSON 合法序列化
    def _clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    clean_results = {
        m: {t: {k: _clean(v) for k, v in d.items()} for t, d in td.items()}
        for m, td in results.items()
    }
    with open(out_json, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n[完成] 结果已保存: {out_json}")

    # ── 打印汇总表 ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'方法':<14} {'拓扑':<10} {'CWT':>8} {'comm':>8} {'cpu_q':>8} {'FoV':>8}")
    print("-"*70)
    method_order = ["TERA-MAPPO", "F-MAPPO", "IPPO", "Greedy", "Local-Only"]
    for m in method_order:
        if m not in results:
            continue
        for topo in topos:
            r = results[m].get(topo, {})
            fmt = lambda v: f"{v:.4f}" if v is not None and np.isfinite(v) else "  N/A  "
            print(f"{m:<14} {topo_labels[topo]:<10} {fmt(r.get('cwt')):>8} {fmt(r.get('cwt_comm')):>8} {fmt(r.get('cwt_cpu_q')):>8} {fmt(r.get('fov')):>8}")
    print("="*70)


if __name__ == "__main__":
    main()
