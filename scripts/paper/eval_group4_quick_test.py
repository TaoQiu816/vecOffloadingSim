#!/usr/bin/env python3
"""
快速测试：仅评估 TERA-MAPPO balanced，5 episodes，验证 eval_group4_topology.py 逻辑
"""
import os, sys, json, random, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.agent_factory import build_agent

RUN_DIR = "runs/rc1_batch1_part1_topology_20260323_182712/topology_balanced/full"
N_EPS = 5
DEVICE = "cpu"

# 加载 config_snapshot
snap = os.path.join(RUN_DIR, "logs", "config_snapshot.json")
with open(snap) as f:
    data = json.load(f)
sc = data.get("system_config", {})
tc = data.get("train_config", {})
for k, v in sc.items():
    if hasattr(Cfg, k): setattr(Cfg, k, v)
for k, v in tc.items():
    if hasattr(TC, k): setattr(TC, k, v)
# 拓扑覆盖
Cfg.TOPOLOGY_MODE = "balanced"
print(f"Config: TOPOLOGY_MODE={Cfg.TOPOLOGY_MODE}, ALGO_MODE={getattr(TC,'ALGO_MODE','N/A')}")
print(f"N_AGENTS={getattr(Cfg,'N_AGENTS','N/A')}")

# 加载模型
model_path = os.path.join(RUN_DIR, "models", "best_model.pth")
from models.offloading_policy import OffloadingPolicyNetwork
try:
    net = OffloadingPolicyNetwork(
        d_model=getattr(TC, 'EMBED_DIM', 128),
        num_heads=getattr(TC, 'NUM_HEADS', 4),
        num_layers=getattr(TC, 'NUM_LAYERS', 3),
    )
except Exception as e:
    print(f"[ERR] build network: {e}")
    sys.exit(1)

state = torch.load(model_path, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and "network" in state:
    net.load_state_dict(state["network"])
elif isinstance(state, dict) and "model_state_dict" in state:
    net.load_state_dict(state["model_state_dict"])
else:
    try: net.load_state_dict(state)
    except: pass
net.eval()
print("Model loaded OK")

agent = build_agent(net, device=DEVICE, algo_mode=getattr(TC, "ALGO_MODE", "mappo"))
print(f"Agent type: {type(agent).__name__}")

# 运行几个 episodes
all_sr, all_cft = [], []
env = VecOffloadingEnv()
for ep in range(N_EPS):
    obs_list, info = env.reset()
    if not obs_list:
        print(f"  ep{ep}: empty obs")
        continue
    done = False
    while not done:
        action_dict = agent.select_action(obs_list, deterministic=True)
        actions_list = action_dict["actions"] if isinstance(action_dict, dict) and "actions" in action_dict else action_dict
        obs_list, rewards, terminated, truncated, step_info = env.step(actions_list)
        done = terminated or truncated
    ep_info = getattr(env, "_last_episode_metrics", {})
    sr  = ep_info.get("task_success_rate", ep_info.get("success_rate", float("nan")))
    cft = ep_info.get("mean_cft_completed", ep_info.get("mean_cft", float("nan")))
    print(f"  ep{ep+1}: SR={sr:.4f}  CFT={cft:.4f}")
    if not np.isnan(sr):  all_sr.append(sr)
    if cft is not None and not np.isnan(cft): all_cft.append(cft)

print(f"\n=== 均值 SR={np.mean(all_sr) if all_sr else 'N/A':.4f}  CFT={np.mean(all_cft) if all_cft else 'N/A':.4f} ===")
