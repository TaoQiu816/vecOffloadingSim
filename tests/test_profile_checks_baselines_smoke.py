import os

from envs.vec_offloading_env import VecOffloadingEnv
from baselines import LocalOnlyPolicy, GreedyPolicy
from configs.config import SystemConfig as Cfg


def _run_policy(tmp_path, name, policy, env, episodes=2, steps=10):
    jsonl_path = tmp_path / f"{name}.jsonl"
    os.environ["REWARD_JSONL_PATH"] = str(jsonl_path)
    os.environ["MAX_EPISODES"] = str(episodes)
    if hasattr(env, "_jsonl_path"):
        env._jsonl_path = str(jsonl_path)

    for ep in range(episodes):
        obs_list, _ = env.reset(seed=ep)
        # Force truncation on the next step so JSONL is always emitted.
        env.steps = max(Cfg.MAX_STEPS - 1, 0)
        policy.reset()
        for _ in range(steps):
            actions = policy.select_action(obs_list)
            for i, obs in enumerate(obs_list):
                if "obs_stamp" in obs and "obs_stamp" not in actions[i]:
                    actions[i]["obs_stamp"] = int(obs["obs_stamp"])
            obs_list, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

    assert jsonl_path.exists()
    assert jsonl_path.read_text(encoding="utf-8").strip()


def test_profile_checks_baselines_smoke(tmp_path):
    env_local = VecOffloadingEnv()
    _run_policy(tmp_path, "local_only", LocalOnlyPolicy(), env_local)

    env_greedy = VecOffloadingEnv()
    _run_policy(tmp_path, "greedy", GreedyPolicy(env_greedy), env_greedy)
