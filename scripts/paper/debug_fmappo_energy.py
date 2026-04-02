#!/usr/bin/env python3
"""
调试 F-MAPPO lbr_radio 问题：检查 E_tx_input_cost 为何全零
"""
import os, sys, json, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.agent_factory import build_agent

RUN_DIR = "runs/rc1_batch1_topology_fmappo_20260328_224844/topology_balanced/fmappo_flat"
DEVICE = "cpu"

# 1. 加载 config_snapshot
snap_path = os.path.join(RUN_DIR, "logs", "config_snapshot.json")
print(f"Loading snapshot: {snap_path}")
with open(snap_path) as f:
    data = json.load(f)
sc = data.get("system_config", {})
tc = data.get("train_config", {})
for k, v in sc.items():
    if hasattr(Cfg, k): setattr(Cfg, k, v)
for k, v in tc.items():
    if hasattr(TC, k): setattr(TC, k, v)

# 拓扑覆盖（balanced）
Cfg.N_RSUS = 3
Cfg.RSU_TOPOLOGY = "balanced"

print(f"TX_POWER_MIN_DBM = {Cfg.TX_POWER_MIN_DBM}")
print(f"TX_POWER_DEFAULT_DBM = {getattr(Cfg, 'TX_POWER_DEFAULT_DBM', 'N/A')}")
print(f"ALGO_MODE = {getattr(Cfg, 'ALGO_MODE', 'N/A')}")
print(f"ABLATION_MODE = {getattr(Cfg, 'ABLATION_MODE', 'N/A')}")

# 2. 找 checkpoint
ckpt_paths = [
    os.path.join(RUN_DIR, "models", "best_model.pth"),
    os.path.join(RUN_DIR, "best_model.pth"),
]
ckpt = None
for p in ckpt_paths:
    if os.path.exists(p):
        ckpt = p
        break
if ckpt is None:
    print("ERROR: No checkpoint found!")
    sys.exit(1)
print(f"Found checkpoint: {ckpt}")

# 3. 加载模型
raw_ckpt = torch.load(ckpt, map_location=DEVICE)
print(f"Checkpoint keys: {list(raw_ckpt.keys())}")
state_dict = None
for key in ["network_state_dict", "network", "model_state_dict", "state_dict"]:
    if key in raw_ckpt:
        state_dict = raw_ckpt[key]
        print(f"Using key: '{key}'")
        break
if state_dict is None:
    if isinstance(raw_ckpt, dict) and all(isinstance(k, str) for k in raw_ckpt):
        state_dict = raw_ckpt
        print("Using raw checkpoint as state_dict")
    else:
        print(f"ERROR: Cannot extract state_dict. Keys: {list(raw_ckpt.keys())}")
        sys.exit(1)

# 4. 构建环境和 agent
env = VecOffloadingEnv(config=Cfg)
obs_list, _ = env.reset(seed=42)
n_agents = len(obs_list)

agent = build_agent(
    algo="mappo",
    obs_list=obs_list,
    env=env,
    device=DEVICE,
)
agent.network.load_state_dict(state_dict, strict=False)
agent.network.eval()

print(f"n_agents={n_agents}")

def _build_global_state(obs_list):
    parts = []
    for obs in obs_list:
        for key in ["position", "velocity", "obs_vec"]:
            v = obs.get(key)
            if v is not None:
                parts.append(np.asarray(v, dtype=np.float32).flatten())
    return np.concatenate(parts, axis=0) if parts else np.zeros(1, dtype=np.float32)

# 5. 运行一个 episode，收集详细信息
for obs in obs_list:
    obs["global_state"] = _build_global_state(obs_list)

done = False
step = 0
decision_counts = {"local": 0, "rsu": 0, "v2v": 0}

while not done:
    with torch.no_grad():
        actions, _ = agent.act(obs_list)
    if actions is None:
        actions = [0] * n_agents
    
    # 检查前几步的车辆功率状态
    if step < 3:
        for v in env.vehicles[:3]:
            print(f"  [step {step}] veh{v.id} tx_power_dbm={v.tx_power_dbm:.1f}")
    
    obs_list, _, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    # 检查 comm_result（无法直接访问，但可以看 E_tx_input_cost 增量）
    if step < 5:
        e_input = dict(env.E_tx_input_cost)
        total_e = sum(e_input.values())
        print(f"  [step {step}] E_tx_input_cost total={total_e:.6f}, n_veh_with_tx={sum(1 for v in e_input.values() if v > 0)}")
    
    # 统计决策
    for v in env.vehicles:
        mode = getattr(v, 'last_mode', None) or getattr(v, '_last_decision', None)
        if mode is not None:
            decision_counts[str(mode)] = decision_counts.get(str(mode), 0) + 1
    
    if obs_list:
        gs = _build_global_state(obs_list)
        for obs in obs_list:
            obs["global_state"] = gs
    
    step += 1
    if step > 3:
        break  # 只看前几步

# 6. 打印 episode 结果
print(f"\nAfter {step} steps:")
print(f"E_tx_input_cost = {dict(env.E_tx_input_cost)}")
print(f"E_tx_edge_record total = {sum(env.E_tx_edge_record.values()):.6f}")
print(f"txq_v2i keys = {list(env.txq_v2i.keys())}")
print(f"txq_v2v keys = {list(env.txq_v2v.keys())}")

# 7. 检查 last_episode_metrics（只有完整 episode 才有）
if hasattr(env, '_last_episode_metrics') and env._last_episode_metrics:
    m = env._last_episode_metrics
    print(f"\n_last_episode_metrics:")
    for k in ['dT_eff_mean', 'lbr_compute', 'lbr_radio']:
        print(f"  {k} = {m.get(k, 'N/A')}")
else:
    print("\n_last_episode_metrics not available (episode not finished)")

print("\n=== 检查 agent action 结构 ===")
with torch.no_grad():
    env2 = VecOffloadingEnv(config=Cfg)
    obs2, _ = env2.reset(seed=42)
    for o in obs2:
        o["global_state"] = _build_global_state(obs2)
    actions2, extras = agent.act(obs2)
    print(f"actions type={type(actions2)}, value={actions2[:3] if actions2 is not None else None}")
    if isinstance(extras, dict):
        for k, v in extras.items():
            print(f"  extras[{k}] = {v[:3] if hasattr(v, '__len__') else v}")
