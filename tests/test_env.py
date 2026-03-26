import gymnasium as gym
import numpy as np
import pytest


import dmcrs  # noqa
from dmcrs.scheduling.environment import Action


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


@pytest.fixture
def simple2r1t():
    env = gym.make("dmcrs-8x8-2r-1t-2d-v3")
    env.reset()

    env.unwrapped.field[:] = 0

    env.unwrapped.field[4, 4] = 2
    env.unwrapped._task_spawned = env.field.sum()

    env.unwrapped.resources[0].position = (4, 3)
    env.unwrapped.resources[1].position = (4, 5)

    env.unwrapped.resources[0].level = 2
    env.unwrapped.resources[1].level = 2
    assert env.unwrapped.test_gen_valid_moves()

    return env


@pytest.fixture
def simple2r1t_sight1():
    env = gym.make("dmcrs-8x8-2r-1t-2d-v3", sight=1)
    env.reset()

    env.unwrapped.field[:] = 0

    env.unwrapped.field[4, 4] = 2
    env.unwrapped._task_spawned = env.field.sum()

    env.unwrapped.resources[0].position = (4, 3)
    env.unwrapped.resources[1].position = (4, 5)

    env.unwrapped.resources[0].level = 2
    env.unwrapped.resources[1].level = 2
    assert env.unwrapped.test_gen_valid_moves()
    return env


@pytest.fixture
def simple2r1t_sight2():
    env = gym.make("dmcrs-8x8-2r-1t-2d-v3", sight=2)
    env.reset()

    env.unwrapped.field[:] = 0

    env.unwrapped.field[4, 4] = 2
    env.unwrapped._task_spawned = env.field.sum()

    env.unwrapped.resources[0].position = (4, 3)
    env.unwrapped.resources[1].position = (4, 5)

    env.unwrapped.resources[0].level = 2
    env.unwrapped.resources[1].level = 2
    assert env.unwrapped.test_gen_valid_moves()
    return env


def test_make():
    names = [
        "dmcrs-8x8-2r-1t-2d-v3",
        "dmcrs-5x5-2r-1t-2d-v3",
        "dmcrs-8x8-3r-1t-2d-v3",
        "dmcrs-8x8-3r-1t-coop-2d-v3",
    ]
    for name in names:
        env = gym.make(name)
        assert env is not None
        env.reset()


def test_spaces():
    pass


def test_seed():
    env = gym.make("dmcrs-8x8-2r-2t-2d-v3")
    for seed in range(10):
        obs1 = []
        obs2 = []
        env.seed(seed)
        for r in range(10):
            obs, _ = env.reset()
            obs1.append(obs)
        env.seed(seed)
        for r in range(10):
            obs, _ = env.reset()
            obs2.append(obs)

    for o1, o2 in zip(obs1, obs2):
        assert np.array_equal(o1, o2)


def test_task_spawning_0():
    env = gym.make("dmcrs-6x6-2r-2t-2d-v3")

    for i in range(1000):
        env.reset()

        tasks = [np.array(f) for f in zip(*env.field.nonzero())]
        # we should have 2 tasks
        assert len(tasks) == 2

        # tasks must not be within 2 steps of each other
        assert manhattan_distance(tasks[0], tasks[1]) > 2

        # task cannot be placed in first or last col/row
        assert tasks[0][0] not in [0, 7]
        assert tasks[0][1] not in [0, 7]
        assert tasks[1][0] not in [0, 7]
        assert tasks[1][1] not in [0, 7]


def test_task_spawning_1():
    env = gym.make("dmcrs-8x8-2r-3t-2d-v3")

    for i in range(1000):
        env.reset()

        tasks = [np.array(f) for f in zip(*env.field.nonzero())]
        # we should have 3 tasks
        assert len(tasks) == 3

        # tasks must not be within 2 steps of each other
        assert manhattan_distance(tasks[0], tasks[1]) > 2
        assert manhattan_distance(tasks[0], tasks[2]) > 2
        assert manhattan_distance(tasks[1], tasks[2]) > 2


def test_reward_0(simple2r1t):
    _, rewards, _, _, _ = simple2r1t.step([5, 5])
    assert rewards[0] == 0.5
    assert rewards[1] == 0.5


def test_reward_1(simple2r1t):
    _, rewards, _, _, _ = simple2r1t.step([0, 5])
    assert rewards[0] == 0
    assert rewards[1] == 1


def test_reward_2(simple2r1t):
    _, rewards, _, _, _ = simple2r1t.step([5, 0])
    assert rewards[0] == 1
    assert rewards[1] == 0


def test_partial_obs_1(simple2r1t_sight1):
    env = simple2r1t_sight1
    obs = env.unwrapped.test_make_gym_obs()

    assert obs[0][-2] == -1
    assert obs[1][-2] == -1


def test_partial_obs_2(simple2r1t_sight2):
    env = simple2r1t_sight2
    obs = env.unwrapped.test_make_gym_obs()

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1

    obs, _, _, _, _ = env.step([Action.WEST, Action.NONE])

    assert obs[0][-2] == -1
    assert obs[1][-2] == -1


def test_partial_obs_3(simple2r1t):
    env = simple2r1t
    obs = env.unwrapped.test_make_gym_obs()

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1

    obs, _, _, _, _ = env.step([Action.WEST, Action.NONE])

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1


def test_reproducibility(simple2r1t):
    env = simple2r1t
    episodes_per_seed = 5
    for seed in range(5):
        obss1 = []
        field1 = []
        resource_positions1 = []
        resource_levels1 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss1.append(np.array(obss).copy())
            field1.append(env.unwrapped.field.copy())
            resource_positions1.append([r.position for r in env.unwrapped.resources])
            resource_levels1.append([r.level for r in env.unwrapped.resources])

        obss2 = []
        field2 = []
        resource_positions2 = []
        resource_levels2 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss2.append(np.array(obss).copy())
            field2.append(env.unwrapped.field.copy())
            resource_positions2.append([r.position for r in env.unwrapped.resources])
            resource_levels2.append([r.level for r in env.unwrapped.resources])

        print("Seed: ", seed)
        for obs1, obs2 in zip(obss1, obss2):
            print(obs1)
            print(obs2)
            print(np.array_equal(obs1, obs2))
            print()

        for i, (obs1, obs2) in enumerate(zip(obss1, obss2)):
            assert np.array_equal(
                obs1, obs2
            ), f"Observations of env not identical for episode {i} with seed {seed}"
        for i, (field1, field2) in enumerate(zip(field1, field2)):
            assert np.array_equal(
                field1, field2
            ), f"Fields of env not identical for episode {i} with seed {seed}"
        for i, (resource_positions1, resource_positions2) in enumerate(
            zip(resource_positions1, resource_positions2)
        ):
            assert (
                resource_positions1 == resource_positions2
            ), f"Resource positions of env not identical for episode {i} with seed {seed}"
        for i, (resource_levels1, resource_levels2) in enumerate(
            zip(resource_levels1, resource_levels2)
        ):
            assert (
                resource_levels1 == resource_levels2
            ), f"Resource levels of env not identical for episode {i} with seed {seed}"
