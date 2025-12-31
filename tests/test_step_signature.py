import numpy as np

from envs.vec_offloading_env import VecOffloadingEnv
from baselines.random_policy import RandomPolicy
from configs.train_config import TrainConfig as TC
import eval_baselines


def _make_actions(obs_list):
    actions = []
    for obs in obs_list:
        valid = np.where(obs["action_mask"])[0]
        target = int(valid[0]) if len(valid) > 0 else 0
        actions.append({"target": target, "power": 0.5})
    return actions


def test_env_step_returns_five():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)
    actions = _make_actions(obs_list)
    result = env.step(actions)
    assert len(result) == 5


def test_eval_baselines_signature_compatible(monkeypatch):
    monkeypatch.setattr(TC, "MAX_STEPS", 1, raising=False)
    env = VecOffloadingEnv()
    policy = RandomPolicy(seed=0)
    eval_baselines.evaluate_policy(env, policy, "Random", num_episodes=1)
