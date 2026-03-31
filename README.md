<p align="center">
 <img width="350px" src="docs/img/logo_dmcrs.png" align="center" alt="Dynamic Multi-dimensional Capability Resource Scheduling (DMCRS)" />
 <p align="center">A multi-agent reinforcement learning environment for dynamic resource scheduling</p>
</p>

<!-- TABLE OF CONTENTS -->
<h1> Table of Contents </h1>

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
  - [Observation Space](#observation-space)
  - [Action space](#action-space)
  - [Rewards](#rewards)
- [Human Play](#human-play)
- [Please Cite](#please-cite)
- [Contributing](#contributing)
- [Contact](#contact)




<!-- ABOUT THE PROJECT -->
# About The Project

This environment is a mixed cooperative-competitive game that simulates a **dynamic resource scheduling scenario**. Resources navigate a grid world and handle tasks by cooperating with other resources if needed.

<p align="center">
 <img width="450px" src="docs/img/lbf.gif" align="center" alt="Dynamic Multi-dimensional Capability Resource Scheduling (DMCRS) illustration" />
</p>

## Core Concepts

### Resources
Resources are entities in the environment that can move and execute tasks. Each resource has a **multi-dimensional capability vector** (formerly a scalar capability), representing its abilities across different dimensions. Resources can navigate the environment and attempt to execute tasks placed next to them.

### Tasks
Tasks are randomly scattered in the grid world, each having a **multi-dimensional requirement vector**. A task can be successfully executed only if the sum of the capability vectors of the resources involved meets or exceeds the task's requirement vector in **each dimension**. This creates a collaborative execution mechanism where resources must coordinate to complete tasks that exceed their individual capabilities.

When `force_coop=True`, task generation injects at least one random **bottleneck dimension** per task, and the requirement on that dimension is pushed above the single-resource peak capability while staying within team-achievable range. This guarantees that one resource alone cannot complete those tasks.

### Dynamic Task Evolution
The environment supports dynamic changes to task requirements over time, including:
- **Linear Decay/Growth**: Task requirements decrease or increase linearly over time
- **Exponential Decay/Growth**: Task requirements change exponentially over time
- **Random Fluctuation**: Task requirements fluctuate randomly within a specified range

This feature introduces temporal dynamics, requiring resources to adapt their strategies based on the changing nature of tasks.

To avoid unbounded growth creating impossible tasks, the environment uses a hard capability ceiling for tasks:
- Task capability ceiling per dimension = **sum of all resources' theoretical max capabilities** + `task_capability_ceiling_slack`
- Growth-type tasks are clipped to this ceiling during updates

### Task Visibility
Tasks have spawn times that determine when they become visible to resources. Tasks that have been spawned but not yet visible are not included in the resource's observations. This adds a planning dimension where resources must consider both current and future task availability.

## Game Mechanics

While it may appear simple, this is a very challenging environment, requiring the cooperation of multiple resources while being competitive at the same time. The discount factor necessitates speed for the maximization of rewards. Each resource is only awarded points if it participates in task execution, and it must balance between:
- Executing low-requirement tasks independently
- Cooperating to acquire higher rewards from complex tasks

In situations with three or more resources, highly strategic decisions can be required, involving resources needing to choose with whom to cooperate. Another significant difficulty for RL algorithms is the sparsity of rewards, which causes slower learning.

This is a Python simulator for dynamic resource scheduling. It is based on OpenAI's RL framework, with modifications for the multi-agent domain. The efficient implementation allows for thousands of simulation steps per second on a single thread, while the rendering capabilities allow humans to visualize resource actions. Our implementation can support different grid sizes or resource/task count. Also, game variants are implemented, such as cooperative mode (resources always need to cooperate) and shared reward (all resources always get the same reward), which is attractive as a credit assignment problem.



<!-- GETTING STARTED -->
# Getting Started

## Installation

Install using pip
```sh
pip install dmcrs
```
Or to ensure that you have the latest version:
```sh
git clone https://github.com/sssink/dmlb-foraging.git
cd dmlb-foraging
pip install -e .
```


<!-- USAGE EXAMPLES -->
# Usage

Create environments with the gym framework.
First import
```python
import dmcrs
```

Then create an environment:
```python
env = gym.make("dmcrs-8x8-2r-1t-2d-v3")
```

We offer a variety of environments using this template:
```
"dmcrs{-{SIGHT_SIZE}s- IF PARTIAL OBS MODE}-{GRID_SIZE}x{GRID_SIZE}-{RESOURCE COUNT}r-{TASK COUNT}t-{DIMENSION}d{-coop IF COOPERATIVE MODE}-v0"
```

But you can register your own variation using (change parameters as needed):
```python
from gymnasium.envs.registration import register

register(
    id="dmcrs-{0}x{0}-{1}r-{2}t-{3}d{4}-v3".format(s, r, t, d, "-coop" if c else ""),
    entry_point="dmcrs.scheduling:SchedulingEnv",
    kwargs={
        "resources": r,
        "min_resource_capability": [1] * d,
        "max_resource_capability": [2] * d,
        "field_size": (s, s),
        "min_task_capability": [1] * d,
        "max_task_capability": None,
        "max_num_tasks": t,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
        "capability_dim": d,
    },
)
```

Registration mode:
- Default import mode is **minimal registration** for faster startup.
- Set `DMCRS_REGISTRATION_MODE=full` to enable full combinational registration.

Similarly to Gym, but adapted to multi-agent settings step() function is defined as
```python
nobs, nreward, ndone, ninfo = env.step(actions)
```

Where n-obs, n-rewards, n-done and n-info are LISTS of N items (where N is the number of resources). The i'th element of each list should be assigned to the i'th resource.



## Observation Space

The observation space consists of the following components:

1. **Task Field**: A grid representation of the environment showing the position and capability of tasks that are visible to the resource. 
2. **Resource Positions**: The positions of all resources in the environment. 
3. **Resource Capability**: The capability vectors of all resources (if `observe_agent_capabilities` is True).

With the task spawn time modification, **only tasks that have reached their spawn time (i.e., are visible) are included in the observation space**. Tasks that have been spawned but not yet visible are not included in the resource's observations.

## Action space

actions is a LIST of N INTEGERS (one of each resource) that should be executed in that step. The integers should correspond to the Enum below:

```python
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5
```
Valid actions can always be sampled like in a gym environment, using:
```python
env.action_space.sample() # [2, 3, 0, 1]
```
Also, ALL actions are valid. If a resource cannot move to a location or load, its action will be replaced with `NONE` automatically.

## Rewards

The rewards are calculated as follows. When one or more resources execute a task, the task's requirement vector is considered. The execution is successful only if the sum of the resources' capability vectors meets or exceeds the task's requirement vector in each dimension. 

For the rewards distribution:
1. Each resource's contribution is determined by its capability vector components relative to the sum of all participating resources' capability vectors
2. The task reward base depends on the task dynamic type:
   - **Linear/Exponential Decay**: use the **current** task requirement vector at execution time
   - **Linear/Exponential Growth**: use an initial-base reward multiplied by a time-decay coefficient based on `(current_step - spawn_time)`
   - **Random Fluctuation / None**: use the task's static base reward
3. The resource reward is `task_reward_base * contribution`
4. If enabled, the reward is normalized so that the sum of rewards (if all tasks have been completed) is one

If you prefer code:

```python
for a in adj_resources: # the resources that participated in executing the task
    contribution = 0.0
    for i in range(self.capability_dim):
        if adj_resource_capability[i] > EPSILON:
            contribution += float(a.capability[i] / adj_resource_capability[i])
    contribution = contribution / self.capability_dim
    a.reward += task_base_value * contribution
    if self._normalize_reward:
        a.reward = a.reward / max(self._task_base_value_spawned, EPSILON)
```

## Feature: Dynamic Task Capability Changes

The environment supports dynamic changes to task requirements over time. This feature can be enabled by setting `enable_task_dynamic_capability=True` when creating the environment. Each task is assigned a random dynamic type at spawn time, which determines how its requirement vector changes over time. The following dynamic types are supported:

1. **None (NONE)**: Task requirements remain constant over time.
2. **Linear Decay (LINEAR_DECAY)**: Task requirements decrease linearly over time based on a decay rate.
3. **Linear Growth (LINEAR_GROWTH)**: Task requirements increase linearly over time based on a growth rate.
4. **Exponential Decay (EXPONENTIAL_DECAY)**: Task requirements decrease exponentially over time based on a decay factor.
5. **Exponential Growth (EXPONENTIAL_GROWTH)**: Task requirements increase exponentially over time based on a growth factor.
6. **Random Fluctuation (RANDOM_FLUCTUATE)**: Task requirements fluctuate randomly within a specified range.

Key parameters for configuring task dynamic capability changes:

- `task_dynamic_rates`: A list of two values specifying the minimum and maximum rates for linear decay/growth.
- `task_dynamic_factors`: A list of two values specifying the minimum and maximum factors for exponential decay/growth.
- `task_fluctuation_range`: A list of two values specifying the minimum and maximum fluctuations for random fluctuation.
- `growth_reward_base_multiplier`: Multiplier applied to growth-task initial base reward.
- `growth_reward_time_decay_rate`: Exponential decay rate used on growth-task reward over alive time.
- `task_generation_coop_ratio`: In each episode, task generation upper bound uses `(sum of spawned resource capabilities) * ratio`.
- `task_capability_ceiling_slack`: Small additive slack on top of the hard task capability ceiling.

Task generation upper bound in each episode:
- `upper_per_dim = min((sum of current spawned resource capabilities) * task_generation_coop_ratio, capability_ceiling_per_dim)`
- This upper bound is then used to sample each task's initial requirement vector.

When a task's requirement in all dimensions drops below a small epsilon value (1e-6), it is removed from the environment.

This feature introduces temporal dynamics to the environment, requiring resources to make strategic decisions about when to execute tasks based on their changing requirements. For example, resources might need to prioritize executing tasks that are decaying rapidly or wait for tasks that are fluctuating to reach a lower requirement.

<!-- HUMAN PLAY SCRIPT -->
# Human Play

We also provide a simple script that allows you to play the environment as a human. This is useful for debugging and understanding the environment dynamics. To play the environment, run the following command:
```sh
python human_play.py --env <env_name>
```
where `<env_name>` is the name of the environment you want to play. For example, to play a DMCRS task with two resources and one task in a 8x8 grid, run:
```sh
python human_play.py --env dmcrs-8x8-2r-1t-2d-v3
```

Within the script, you can control a single resource at a time using the following keys:
- Arrow keys: move current resource up/ down/ left/ right
- L: execute task
- K: execute task and let resource keep loading (even if resource is swapped)
- SPACE: do nothing
- TAB: change the current resource (rotates through all resources)
- R: reset the environment and start a new episode
- H: show help
- D: display resource info (at every time step)
- ESC: exit


<!-- CITATION -->
# Please Cite
1. The paper that first uses this implementation of Level-based Foraging (LBF) and achieves state-of-the-art results:
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Schäfer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
2. A comparative evaluation of cooperative MARL algorithms and includes an introduction to this environment:
```
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
}
```

<!-- CONTRIBUTING -->
# Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
# Contact

sink - sssssink@163.com

This project is a fork of the original lb-foraging project. The original project is available at [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging).
