from itertools import product

from gymnasium import register


sizes = range(5, 15)
resources = range(2, 10)
tasks = range(2, 10)
max_task_level = [None]  # [None, 1]
coop = [False] # [True, False]
partial_obs = [True, False]
partial_obs_size = range(2, 8)
pens = [False]  # [True, False]
level_dim = [2, 3]

for s, r, t, mfl, c, po, pos, pen, dim in product(
    sizes, resources, tasks, max_task_level, coop, partial_obs, partial_obs_size, pens, level_dim
):
    if not po and pos != partial_obs_size[0]:
        continue
    register(
        id="dmcrs{4}-{0}x{0}-{1}r-{2}t-{7}d{3}{5}{6}-v3".format(
            s,
            r,
            t, 
            "-coop" if c else "",
            f"-{pos}s" if po else "",
            "-ind" if mfl else "",
            "-pen" if pen else "",
            dim,
        ),
        entry_point="dmcrs.scheduling:SchedulingEnv",
        kwargs={
            "resources": r,
            "min_resource_level": [1] * dim,
            "max_resource_level": [2] * dim,
            "field_size": (s, s),
            "min_task_level": [1] * dim,
            "max_task_level": mfl,
            "max_num_tasks": t,
            "sight": pos if po else s,
            "max_episode_steps": 50,
            "force_coop": c,
            "grid_observation": False,
            "penalty": 0.1 if pen else 0.0,
            "level_dim": dim,
        },
    )


def register_grid_envs(): # no test
    for s, r, t, mfl, c, dim in product(sizes, resources, tasks, max_task_level, coop, level_dim):
        for sight in range(1, s + 1):
            register(
                id="dmcrs-grid{4}-{0}x{0}-{1}r-{2}t-{6}d{3}{5}-v3".format(
                    s,
                    r,
                    t, 
                    "-coop" if c else "",
                    "" if sight == s else f"-{sight}s",
                    "-ind" if mfl else "",
                    dim,
                ),
                entry_point="dmcrs.scheduling:SchedulingEnv",
                kwargs={
                    "resources": r,
                    "min_resource_level": [1] * dim,
                    "max_resource_level": [2] * dim,
                    "field_size": (s, s),
                    "min_task_level": [1] * dim,
                    "max_task_level": mfl,
                    "max_num_tasks": t,
                    "sight": sight,
                    "max_episode_steps": 50,
                    "force_coop": c,
                    "grid_observation": True,
                    "level_dim": dim,
                },
            )
