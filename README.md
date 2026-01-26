
<p align="center">
 <img width="350px" src="docs/img/logo.png" align="center" alt="Dynamic Multi-Level-based Foraging (DMLBF)" />
 <p align="center">A multi-agent reinforcement learning environment</p>
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


> [!CAUTION]
> The LBF environment was updated to support the new [Gymnasium](https://gymnasium.farama.org/) interface in replacement of the deprecated `gym=0.21` dependency (many thanks @LukasSchaefer). For backwards compatibility, please see [Gymnasium compatibility documentation](https://gymnasium.farama.org/content/gym_compatibility/) or use version v1.1.1 of the repository. The main changes to the interface are as follows:
> - `obss = env.reset()` --> `obss, info = env.reset()`
> - `obss, rewards, dones, info = env.step(actions)` --> `obss, rewards, done, truncated, info = env.step(actions)`
> - The `done` flag is now given as a single boolean value instead of a list of booleans.
> - You can give the reset function a particular seed with `obss, info = env.reset(seed=42)` to initialise a particular episode.


<!-- ABOUT THE PROJECT -->
# About The Project

This environment is a mixed cooperative-competitive game, which focuses on the coordination of the agents involved. Agents navigate a grid world and collect food by cooperating with other agents if needed.

<p align="center">
 <img width="450px" src="docs/img/lbf.gif" align="center" alt="Level Based Foraging (LBF) illustration" />
</p>

More specifically, agents are placed in the grid world, and each is assigned a level vector (formerly a scalar level). Food is also randomly scattered, each having a level vector on its own. Agents can navigate the environment and can attempt to collect food placed next to them. The collection of food is successful only if the sum of the levels of the agents involved in loading meets or exceeds the food's level vector in each dimension. Finally, agents are awarded points based on the food's level vector and their contribution proportional to their level vector components. The figures below show two states of the game, one that requires cooperation, and one more competitive.

The environment now supports dynamic food spawning with visibility delays. When initializing the environment, each food item is assigned a random spawn time between these two values (inclusive). This feature adds a new dimension to the environment, requiring agents to plan their actions based on both current and future food availability.

Additionally, the environment now supports dynamic changes to food levels over time, including linear decay, exponential decay, and random fluctuations. This feature introduces temporal dynamics to the environment, requiring agents to adapt their strategies based on the changing value of food resources.

While it may appear simple, this is a very challenging environment, requiring the cooperation of multiple agents while being competitive at the same time. In addition, the discount factor also necessitates speed for the maximisation of rewards. Each agent is only awarded points if it participates in the collection of food, and it has to balance between collecting low-levelled food on his own or cooperating in acquiring higher rewards. In situations with three or more agents, highly strategic decisions can be required, involving agents needing to choose with whom to cooperate. Another significant difficulty for RL algorithms is the sparsity of rewards, which causes slower learning.

This is a Python simulator for level based foraging. It is based on OpenAI's RL framework, with modifications for the multi-agent domain. The efficient implementation allows for thousands of simulation steps per second on a single thread, while the rendering capabilities allows humans to visualise agent actions. Our implementation can support different grid sizes or agent/food count. Also, game variants are implemented, such as cooperative mode (agents always need to cooperate) and shared reward (all agents always get the same reward), which is attractive as a credit assignment problem.



<!-- GETTING STARTED -->
# Getting Started

## Installation

Install using pip
```sh
pip install dmlbforaging
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
import lbforaging
```

Then create an environment:
```python
env = gym.make("Foraging-8x8-2p-1f-2d-v3")
```

We offer a variety of environments using this template:
```
"Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f-{DIMENSION}d{-coop IF COOPERATIVE MODE}-v0"
```

But you can register your own variation using (change parameters as needed):
```python
from gym.envs.registration register

register(
    id="Foraging-{0}x{0}-{1}p-{2}f-{3}d{4}-v3".format(s, p, f, d, "-coop" if c else ""),
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": p,
        "max_player_level": 3,
        "field_size": (s, s),
        "max_food": f,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
        "level_dim": d,
    },
)
```

Similarly to Gym, but adapted to multi-agent settings step() function is defined as
```python
nobs, nreward, ndone, ninfo = env.step(actions)
```

Where n-obs, n-rewards, n-done and n-info are LISTS of N items (where N is the number of agents). The i'th element of each list should be assigned to the i'th agent.



## Observation Space

The observation space consists of the following components:

1. **Food Field**: A grid representation of the environment showing the position and level of food items that are visible to the agent. 
2. **Agent Positions**: The positions of all agents in the environment. 
3. **Agent Levels**: The level vectors of all agents (if `observe_agent_levels` is True).

With the food spawn time modification, **only foods that have reached their spawn time (i.e., are visible) are included in the observation space**. Foods that have been spawned but not yet visible (translucent) are not included in the agent's observations.

## Action space

actions is a LIST of N INTEGERS (one of each agent) that should be executed in that step. The integers should correspond to the Enum below:

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
Also, ALL actions are valid. If an agent cannot move to a location or load, his action will be replaced with `NONE` automatically.

## Rewards

The rewards are calculated as follows. When one or more agents load a food, the food level vector is considered. The collection is successful only if the sum of the agents' level vectors meets or exceeds the food's level vector in each dimension. 

For the rewards distribution:
1. Each agent's contribution is determined by their level vector components relative to the sum of all participating agents' level vectors
2. The reward for each agent is calculated based on this contribution proportion and the food's level vector
3. If enabled, the reward is normalized so that the sum of rewards (if all foods have been picked-up) is one

If you prefer code:

```python
for a in adj_players: # the players that participated in loading the food
    a.reward = float(a.level * food) # higher-leveled agents contribute more and are rewarded more. 
    if self._normalize_reward:
        a.reward = a.reward / float(
            adj_player_level * self._food_spawned
        )  # normalize reward so that the final sum of rewards is one.
```
## New Feature: Food Dynamic Level Changes

The environment now supports dynamic changes to food levels over time. This feature can be enabled by setting `enable_food_dynamic_level=True` when creating the environment. Each food item is assigned a random decay type at spawn time, which determines how its level vector changes over time. The following decay types are supported:

1. **None (NONE)**: Food levels remain constant over time.
2. **Linear Decay (LINEAR_DECAY)**: Food levels decrease linearly over time based on a decay rate.
3. **Exponential Decay (EXPONENTIAL_DECAY)**: Food levels decrease exponentially over time based on a decay factor.
4. **Random Fluctuation (RANDOM_FLUCTUATE)**: Food levels fluctuate randomly within a specified range.

Key parameters for configuring food dynamic level changes:

- `food_decay_rates`: A list of two values specifying the minimum and maximum decay rates for linear decay.
- `food_decay_factors`: A list of two values specifying the minimum and maximum decay factors for exponential decay.
- `food_fluctuation_range`: A list of two values specifying the minimum and maximum fluctuations for random fluctuation.

When a food's level in all dimensions drops below a small epsilon value (1e-6), it is removed from the environment.

This feature introduces temporal dynamics to the environment, requiring agents to make strategic decisions about when to collect food based on its changing value. For example, agents might need to prioritize collecting food that is decaying rapidly or wait for food that is fluctuating to reach a higher value.

<!-- HUMAN PLAY SCRIPT -->
# Human Play

We also provide a simple script that allows you to play the environment as a human. This is useful for debugging and understanding the environment dynamics. To play the environment, run the following command:
```sh
python human_play.py --env <env_name>
```
where `<env_name>` is the name of the environment you want to play. For example, to play an LBF task with two agents and one food in a 8x8 grid, run:
```sh
python human_play.py --env Foraging-8x8-2p-1f-2d-v3
```

Within the script, you can control a single agent at the time using the following keys:
- Arrow keys: move current agent up/ down/ left/ right
- L: load food
- K: load food and let agent keep loading (even if agent is swapped)
- SPACE: do nothing
- TAB: change the current agent (rotates through all agents)
- R: reset the environment and start a new episode
- H: show help
- D: display agent info (at every time step)
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
2. A comperative evaluation of cooperative MARL algorithms and includes an introduction to this environment:
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

