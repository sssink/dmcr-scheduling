import gymnasium as gym
import numpy as np
import pytest


import dmcrs  # noqa
from dmcrs.scheduling.environment import Action, EPSILON, Task, TaskDynamicType


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

    env.unwrapped.resources[0].capability = 2
    env.unwrapped.resources[1].capability = 2
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

    env.unwrapped.resources[0].capability = 2
    env.unwrapped.resources[1].capability = 2
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

    env.unwrapped.resources[0].capability = 2
    env.unwrapped.resources[1].capability = 2
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


def _setup_single_task_env(
    dynamic_type,
    initial_capability,
    current_capability,
    spawn_time=0,
    current_step=0,
    growth_reward_base_multiplier=1.05,
    growth_reward_time_decay_rate=0.03,
):
    env = gym.make(
        "dmcrs-8x8-2r-2t-2d-v3",
        normalize_reward=False,
        enable_task_dynamic_capability=False,
        growth_reward_base_multiplier=growth_reward_base_multiplier,
        growth_reward_time_decay_rate=growth_reward_time_decay_rate,
    )
    env.reset()
    unwrapped = env.unwrapped
    unwrapped.tasks = {}
    task = Task()
    task.setup(
        position=(3, 3),
        capability=np.array(initial_capability, dtype=np.float32),
        spawn_time=spawn_time,
        dynamic_type=dynamic_type,
        dynamic_params={},
        base_value=unwrapped._compute_task_base_value(np.array(initial_capability, dtype=np.float32)),
    )
    task.capability = np.array(current_capability, dtype=np.float32)
    unwrapped.tasks[(3, 3)] = task
    unwrapped.current_step = current_step
    unwrapped.resources[0].position = (3, 2)
    unwrapped.resources[1].position = (0, 0)
    unwrapped.resources[0].capability = np.array([10.0, 10.0], dtype=np.float32)
    unwrapped.resources[1].capability = np.array([1.0, 1.0], dtype=np.float32)
    unwrapped._sync_fields_from_tasks()
    unwrapped._task_base_value_spawned = float(
        np.sum(
            [
                unwrapped._compute_task_nominal_base_value(existing_task)
                for existing_task in unwrapped.tasks.values()
            ]
        )
    )
    unwrapped._gen_valid_moves()
    return env


def test_reward_decay_task_uses_current_requirement():
    env = _setup_single_task_env(
        dynamic_type=TaskDynamicType.LINEAR_DECAY,
        initial_capability=[4.0, 4.0],
        current_capability=[1.0, 2.0],
    )
    _, rewards, _, _, _ = env.step([Action.LOAD, Action.NONE])
    assert np.isclose(rewards[0], 3.0)


def test_reward_growth_task_uses_initial_base_with_time_decay():
    env = _setup_single_task_env(
        dynamic_type=TaskDynamicType.LINEAR_GROWTH,
        initial_capability=[2.0, 2.0],
        current_capability=[6.0, 6.0],
        spawn_time=0,
        current_step=4,
        growth_reward_base_multiplier=1.2,
        growth_reward_time_decay_rate=0.1,
    )
    _, rewards, _, _, _ = env.step([Action.LOAD, Action.NONE])
    expected = 4.0 * 1.2 * np.exp(-0.1 * 5)
    assert np.isclose(rewards[0], expected)


def test_reward_random_task_uses_static_base_value():
    env = _setup_single_task_env(
        dynamic_type=TaskDynamicType.RANDOM_FLUCTUATE,
        initial_capability=[3.0, 1.0],
        current_capability=[9.0, 9.0],
    )
    _, rewards, _, _, _ = env.step([Action.LOAD, Action.NONE])
    assert np.isclose(rewards[0], 4.0)


def test_task_observation_upper_bound_uses_resource_team_ceiling():
    env = gym.make(
        "dmcrs-8x8-2r-1t-2d-v3",
        min_resource_capability=[[1.0, 1.0], [1.0, 1.0]],
        max_resource_capability=[[2.0, 3.0], [4.0, 5.0]],
        task_capability_ceiling_slack=0.25,
    )
    unwrapped = env.unwrapped
    ceiling = unwrapped._get_task_capability_ceiling_per_dim()
    assert np.allclose(ceiling, np.array([6.25, 8.25], dtype=np.float32))
    assert np.allclose(unwrapped.observation_space[0].high[4:6], ceiling)


def test_growth_task_capability_is_clipped_by_ceiling():
    env = gym.make(
        "dmcrs-8x8-2r-1t-2d-v3",
        min_resource_capability=[[1.0, 1.0], [1.0, 1.0]],
        max_resource_capability=[[2.0, 2.0], [2.0, 2.0]],
        task_capability_ceiling_slack=0.0,
        enable_task_dynamic_capability=True,
    )
    env.reset()
    unwrapped = env.unwrapped
    task = Task()
    task.setup(
        position=(3, 3),
        capability=np.array([3.9, 3.95], dtype=np.float32),
        spawn_time=0,
        dynamic_type=TaskDynamicType.LINEAR_GROWTH,
        dynamic_params={"dynamic_rate": np.array([0.5, 0.5], dtype=np.float32)},
        base_value=1.0,
    )
    unwrapped.tasks = {(3, 3): task}
    unwrapped.current_step = 1
    unwrapped._update_task_capabilities()
    assert np.all(unwrapped.tasks[(3, 3)].capability <= np.array([4.0, 4.0], dtype=np.float32) + EPSILON)


def test_reset_task_generation_upper_bound_uses_episode_resource_sum_ratio():
    env = gym.make(
        "dmcrs-8x8-2r-3t-2d-v3",
        min_resource_capability=[[2.0, 3.0], [4.0, 1.0]],
        max_resource_capability=[[2.0, 3.0], [4.0, 1.0]],
        min_task_capability=[0.0, 0.0],
        max_task_capability=None,
        task_generation_coop_ratio=0.5,
        enable_task_dynamic_capability=False,
    )
    env.reset()
    unwrapped = env.unwrapped
    episode_resource_sum = np.sum(
        np.array([resource.capability for resource in unwrapped.resources]),
        axis=0,
    )
    expected_upper = episode_resource_sum * 0.5
    for task in unwrapped.tasks.values():
        assert np.all(task.initial_capability <= expected_upper + EPSILON)


def test_force_coop_tasks_have_bottleneck_dimension_above_single_peak():
    env = gym.make(
        "dmcrs-8x8-2r-3t-2d-v3",
        min_resource_capability=[[2.0, 1.5], [1.0, 2.5]],
        max_resource_capability=[[2.0, 1.5], [1.0, 2.5]],
        min_task_capability=[0.0, 0.0],
        max_task_capability=None,
        force_coop=True,
        task_generation_coop_ratio=1.0,
        enable_task_dynamic_capability=False,
    )
    for _ in range(10):
        env.reset()
        unwrapped = env.unwrapped
        single_peak = np.max(
            np.array([resource.capability for resource in unwrapped.resources]),
            axis=0,
        )
        for task in unwrapped.tasks.values():
            assert np.any(task.initial_capability > single_peak + EPSILON)


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
        resource_capabilities1 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss1.append(np.array(obss).copy())
            field1.append(env.unwrapped.field.copy())
            resource_positions1.append([r.position for r in env.unwrapped.resources])
            resource_capabilities1.append([r.capability for r in env.unwrapped.resources])

        obss2 = []
        field2 = []
        resource_positions2 = []
        resource_capabilities2 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss2.append(np.array(obss).copy())
            field2.append(env.unwrapped.field.copy())
            resource_positions2.append([r.position for r in env.unwrapped.resources])
            resource_capabilities2.append([r.capability for r in env.unwrapped.resources])

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
        for i, (resource_capabilities1, resource_capabilities2) in enumerate(
            zip(resource_capabilities1, resource_capabilities2)
        ):
            assert (
                resource_capabilities1 == resource_capabilities2
            ), f"Resource capabilities of env not identical for episode {i} with seed {seed}"
