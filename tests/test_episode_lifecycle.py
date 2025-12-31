from envs.vec_offloading_env import VecOffloadingEnv


def test_episode_counters_reset_and_increment():
    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=0)
    episode_id = env._episode_id
    assert env.time == 0.0
    assert env.steps == 0
    assert env._episode_steps == 0

    actions = [{"target": 0, "power": 0.5} for _ in obs]
    env.step(actions)
    assert env.time > 0.0
    assert env.steps == 1

    env.reset(seed=1)
    assert env._episode_id == episode_id + 1
    assert env.time == 0.0
    assert env.steps == 0
    assert env._episode_steps == 0
    assert env._reward_stats.counters == {}
