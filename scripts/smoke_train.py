import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _set_cfg(**kwargs):
    orig = {k: getattr(Cfg, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(Cfg, k, v)
    return orig


def _restore_cfg(orig):
    for k, v in orig.items():
        setattr(Cfg, k, v)


def _obs_finite(obs: Dict[str, Any]) -> bool:
    for val in obs.values():
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                return False
    return True


def _normalize_flags(flags):
    if isinstance(flags, dict):
        return [bool(v) for _, v in sorted(flags.items())]
    if isinstance(flags, (list, np.ndarray)):
        return [bool(v) for v in flags]
    return [bool(flags)]


def main():
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    orig = _set_cfg(
        NUM_VEHICLES=3,
        NUM_RSU=1,
        VEHICLE_ARRIVAL_RATE=0,
        MAX_STEPS=120,
        BW_V2I=20e6,
        REWARD_SCHEME="LEGACY_CFT",
    )

    env = None
    try:
        env = VecOffloadingEnv(config=Cfg)
        total_steps = 0
        max_steps = 300
        max_episodes = 3
        episode = 0

        reward_sum = 0.0
        reward_count = 0
        reward_finite = 0
        mask_valid_counts = []
        done_count = 0
        trunc_count = 0

        while total_steps < max_steps and episode < max_episodes:
            obs_list, _ = env.reset(seed=seed + episode)
            if len(obs_list) != Cfg.NUM_VEHICLES:
                raise RuntimeError("obs_list length mismatch")
            if not all(_obs_finite(obs) for obs in obs_list):
                raise RuntimeError("obs contains NaN/inf at reset")

            while True:
                actions = []
                for obs in obs_list:
                    mask = obs["action_mask"]
                    valid = np.where(mask)[0]
                    mask_valid_counts.append(int(len(valid)))
                    target = int(rng.choice(valid)) if len(valid) else 0
                    actions.append({"target": target, "power": float(rng.random())})

                obs_list, rewards, dones, truncs, infos = env.step(actions)
                total_steps += 1

                rew_arr = np.asarray(rewards, dtype=np.float64)
                reward_sum += float(np.sum(rew_arr))
                reward_count += int(rew_arr.size)
                reward_finite += int(np.isfinite(rew_arr).sum())

                if not all(_obs_finite(obs) for obs in obs_list):
                    raise RuntimeError("obs contains NaN/inf after step")

                done_flags = _normalize_flags(dones)
                trunc_flags = _normalize_flags(truncs)

                if all(done_flags):
                    done_count += 1
                if all(trunc_flags):
                    trunc_count += 1

                if total_steps >= max_steps:
                    break
                if all(done_flags) or all(trunc_flags):
                    break

            episode += 1

        mean_reward = reward_sum / max(reward_count, 1)
        finite_rate = reward_finite / max(reward_count, 1)
        mask_min = int(np.min(mask_valid_counts)) if mask_valid_counts else 0
        mask_max = int(np.max(mask_valid_counts)) if mask_valid_counts else 0
        mask_mean = float(np.mean(mask_valid_counts)) if mask_valid_counts else 0.0

        print("=== Smoke Train Summary ===")
        print(f"seed={seed} steps={total_steps} episodes={episode}")
        print(f"mean_reward={mean_reward:.6f}")
        print(f"reward_finite_rate={finite_rate:.6f}")
        print(f"action_mask_valid_count[min/mean/max]={mask_min}/{mask_mean:.2f}/{mask_max}")
        print(f"done_count={done_count} trunc_count={trunc_count}")
    finally:
        if env is not None:
            env.close()
        _restore_cfg(orig)


if __name__ == "__main__":
    main()
