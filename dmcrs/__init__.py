import os
from itertools import product

from gymnasium import register
from gymnasium.envs.registration import registry


sizes = range(5, 15)
resources = range(2, 10)
tasks = range(2, 10)
max_task_capability = [None]  # [None, 1]
coop = [True, False]
partial_obs = [True, False]
partial_obs_size = range(2, 8)
pens = [False]  # [True, False]
capability_dim = [2, 3]

minimal_env_specs = [
    (8, 3, 2, None, False, False, 2, False, 2),
    (8, 3, 2, None, False, True, 2, False, 2),
    (8, 3, 2, None, False, True, 2, False, 3),
    (8, 3, 2, None, True, True, 2, False, 2),
    (8, 3, 2, None, True, True, 2, False, 3),
    (9, 3, 2, None, False, True, 2, False, 2),
    (9, 3, 2, None, False, True, 2, False, 3),
    (9, 3, 2, None, True, True, 2, False, 2),
    (9, 3, 2, None, True, True, 2, False, 3),
    (11, 2, 2, None, False, True, 2, False, 2),
    (11, 2, 2, None, False, True, 2, False, 3),
    (11, 2, 2, None, True, True, 2, False, 2),
    (11, 2, 2, None, True, True, 2, False, 3), 
    (11, 3, 2, None, False, True, 4, False, 2),
    (11, 3, 2, None, False, True, 4, False, 3),
    (11, 3, 2, None, True, True, 4, False, 2),
    (11, 3, 2, None, True, True, 4, False, 3),
    (11, 3, 2, None, False, True, 7, False, 2),
    (11, 3, 2, None, False, True, 7, False, 3),
    (11, 3, 2, None, True, True, 7, False, 2),
    (11, 3, 2, None, True, True, 7, False, 3),
    (20, 5, 3, None, False, True, 2, False, 2),
    (20, 5, 3, None, False, True, 2, False, 3),
    (20, 5, 3, None, True, True, 2, False, 2),
    (20, 5, 3, None, True, True, 2, False, 3),
]


def _register_env(s, r, t, mtc, c, po, pos, pen, dim):
    env_id = "dmcrs{4}-{0}x{0}-{1}r-{2}t-{7}d{3}{5}{6}-v3".format(
        s,
        r,
        t,
        "-coop" if c else "",
        f"-{pos}s" if po else "",
        "-ind" if mtc else "",
        "-pen" if pen else "",
        dim,
    )
    if env_id in registry:
        return
    register(
        id=env_id,
        entry_point="dmcrs.scheduling:SchedulingEnv",
        kwargs={
            "resources": r,
            "min_resource_capability": [1] * dim,
            "max_resource_capability": [2] * dim,
            "field_size": (s, s),
            "min_task_capability": [1] * dim,
            "max_task_capability": mtc,
            "max_num_tasks": t,
            "sight": pos if po else s,
            "max_episode_steps": 50,
            "force_coop": c,
            "grid_observation": False,
            "penalty": 0.1 if pen else 0.0,
            "capability_dim": dim,
        },
    )


def register_minimal_envs():
    for s, r, t, mtc, c, po, pos, pen, dim in minimal_env_specs:
        _register_env(s, r, t, mtc, c, po, pos, pen, dim)


def register_full_envs():
    for s, r, t, mtc, c, po, pos, pen, dim in product(
        sizes, resources, tasks, max_task_capability, coop, partial_obs, partial_obs_size, pens, capability_dim
    ):
        if not po and pos != partial_obs_size[0]:
            continue
        _register_env(s, r, t, mtc, c, po, pos, pen, dim)


def register_grid_envs():
    for s, r, t, mtc, c, dim in product(sizes, resources, tasks, max_task_capability, coop, capability_dim):
        for sight in range(1, s + 1):
            env_id = "dmcrs-grid{4}-{0}x{0}-{1}r-{2}t-{6}d{3}{5}-v3".format(
                s,
                r,
                t,
                "-coop" if c else "",
                "" if sight == s else f"-{sight}s",
                "-ind" if mtc else "",
                dim,
            )
            if env_id in registry:
                continue
            register(
                id=env_id,
                entry_point="dmcrs.scheduling:SchedulingEnv",
                kwargs={
                    "resources": r,
                    "min_resource_capability": [1] * dim,
                    "max_resource_capability": [2] * dim,
                    "field_size": (s, s),
                    "min_task_capability": [1] * dim,
                    "max_task_capability": mtc,
                    "max_num_tasks": t,
                    "sight": sight,
                    "max_episode_steps": 50,
                    "force_coop": c,
                    "grid_observation": True,
                    "capability_dim": dim,
                },
            )


if os.getenv("DMCRS_REGISTRATION_MODE", "minimal").lower() == "full":
    register_full_envs()
else:
    register_minimal_envs()
