import copy
import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork


def _add_padding_noise(obs, rng):
    obs2 = copy.deepcopy(obs)
    task_mask = obs2["task_mask"].astype(bool)
    pad_idx = np.where(~task_mask)[0]
    if pad_idx.size == 0:
        return obs2

    # Node features
    node_x = obs2["node_x"].copy()
    node_x[pad_idx, :] = rng.normal(0.0, 1.0, size=node_x[pad_idx, :].shape)
    obs2["node_x"] = node_x

    # Adjacency-like matrices
    for key in ("adj", "data_matrix", "Delta"):
        mat = obs2[key].copy()
        for idx in pad_idx:
            mat[idx, :] = rng.normal(0.0, 1.0, size=mat[idx, :].shape)
            mat[:, idx] = rng.normal(0.0, 1.0, size=mat[:, idx].shape)
        obs2[key] = mat

    # Topology and priority
    for key in ("L_fwd", "L_bwd"):
        arr = obs2[key].copy()
        arr[pad_idx] = rng.integers(0, 4, size=pad_idx.shape[0], dtype=arr.dtype)
        obs2[key] = arr
    priority = obs2["priority"].copy()
    priority[pad_idx] = rng.random(size=pad_idx.shape[0])
    obs2["priority"] = priority
    location = obs2["location"].copy()
    location[pad_idx] = rng.integers(0, 4, size=pad_idx.shape[0], dtype=location.dtype)
    obs2["location"] = location

    return obs2


def main():
    parser = argparse.ArgumentParser(description="Test DAG mask invariance.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = VecOffloadingEnv(config=Cfg)
    obs_list, _ = env.reset(seed=args.seed)
    if not obs_list:
        raise RuntimeError("Empty observation list")
    obs = obs_list[0]

    rng = np.random.default_rng(args.seed + 999)
    obs2 = _add_padding_noise(obs, rng)

    policy = OffloadingPolicyNetwork()
    policy.eval()
    with torch.no_grad():
        inputs1 = policy.prepare_inputs([obs], device="cpu")
        logits1, _, _, value1 = policy.forward(
            node_x=inputs1["node_x"],
            adj=inputs1["adj"],
            status=inputs1["status"],
            location=inputs1["location"],
            L_fwd=inputs1["L_fwd"],
            L_bwd=inputs1["L_bwd"],
            data_matrix=inputs1["data_matrix"],
            delta=inputs1["delta"],
            resource_ids=inputs1["resource_ids"],
            resource_raw=inputs1["resource_raw"],
            subtask_index=inputs1["subtask_index"],
            action_mask=inputs1["action_mask"],
            task_mask=inputs1["task_mask"],
            priority=inputs1["priority"],
        )
        inputs2 = policy.prepare_inputs([obs2], device="cpu")
        logits2, _, _, value2 = policy.forward(
            node_x=inputs2["node_x"],
            adj=inputs2["adj"],
            status=inputs2["status"],
            location=inputs2["location"],
            L_fwd=inputs2["L_fwd"],
            L_bwd=inputs2["L_bwd"],
            data_matrix=inputs2["data_matrix"],
            delta=inputs2["delta"],
            resource_ids=inputs2["resource_ids"],
            resource_raw=inputs2["resource_raw"],
            subtask_index=inputs2["subtask_index"],
            action_mask=inputs2["action_mask"],
            task_mask=inputs2["task_mask"],
            priority=inputs2["priority"],
        )

    logit_diff = torch.max(torch.abs(logits1 - logits2)).item()
    value_diff = torch.max(torch.abs(value1 - value2)).item()
    tol = float(args.tol)
    if logit_diff > tol or value_diff > tol:
        raise AssertionError(
            f"Mask invariance FAIL: logit_diff={logit_diff:.6g}, value_diff={value_diff:.6g}, tol={tol}"
        )
    print(f"PASS: logit_diff={logit_diff:.6g}, value_diff={value_diff:.6g}, tol={tol}")


if __name__ == "__main__":
    main()
