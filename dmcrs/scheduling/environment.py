from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
import logging
from typing import Iterable

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

EPSILON = 1e-6
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    TASK = 2
    AGENT = 3

class TaskDynamicType(Enum):
    NONE = 0 # no decay
    LINEAR_DECAY = 1
    EXPONENTIAL_DECAY = 2
    LINEAR_GROWTH = 3
    EXPONENTIAL_GROWTH = 4
    RANDOM_FLUCTUATE = 5


class Resource:
    def __init__(self):
        self.controller = None # agent for the resource
        self.position = None
        self.capability = None # multi-dimensional capability
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, capability, field_size):
        self.history = []
        self.position = position
        self.capability = capability
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Resource"

class Task:
    def __init__(self):
        self.position = None
        self.capability = None
        self.initial_capability = None
        self.base_value = 0.0
        self.spawn_time = 0
        self.dynamic_type = TaskDynamicType.NONE
        self.dynamic_params = {}

    def setup(
        self,
        position,
        capability,
        spawn_time,
        dynamic_type,
        dynamic_params,
        base_value,
    ):
        self.position = position
        self.capability = np.array(capability, dtype=np.float32)
        self.initial_capability = np.array(capability, dtype=np.float32)
        self.base_value = float(base_value)
        self.spawn_time = int(spawn_time)
        self.dynamic_type = dynamic_type
        self.dynamic_params = dynamic_params

    def is_visible(self, current_step):
        return self.spawn_time <= current_step and np.any(self.capability > EPSILON)

    def is_active(self):
        return np.any(self.capability > EPSILON)

    def apply_dynamic(self, current_step, np_random):
        if not self.is_visible(current_step):
            return
        if self.dynamic_type == TaskDynamicType.LINEAR_DECAY:
            for dim in range(len(self.capability)):
                if self.capability[dim] > EPSILON:
                    decay = self.initial_capability[dim] * self.dynamic_params["dynamic_rate"][dim]
                    self.capability[dim] = max(0.0, self.capability[dim] - decay)
        elif self.dynamic_type == TaskDynamicType.EXPONENTIAL_DECAY:
            for dim in range(len(self.capability)):
                if self.capability[dim] > EPSILON:
                    self.capability[dim] = max(
                        0.0,
                        self.capability[dim] * self.dynamic_params["dynamic_factor"][dim],
                    )
        elif self.dynamic_type == TaskDynamicType.LINEAR_GROWTH:
            for dim in range(len(self.capability)):
                growth = self.initial_capability[dim] * self.dynamic_params["dynamic_rate"][dim]
                self.capability[dim] = self.capability[dim] + growth
        elif self.dynamic_type == TaskDynamicType.EXPONENTIAL_GROWTH:
            for dim in range(len(self.capability)):
                self.capability[dim] = self.capability[dim] * (
                    2 - self.dynamic_params["dynamic_factor"][dim]
                )
        elif self.dynamic_type == TaskDynamicType.RANDOM_FLUCTUATE:
            for dim in range(len(self.capability)):
                if self.capability[dim] > EPSILON:
                    delta = np_random.uniform(
                        self.dynamic_params["min_delta"],
                        self.dynamic_params["max_delta"],
                    )
                    self.capability[dim] = max(0.0, self.capability[dim] + delta)


class SchedulingEnv(gym.Env):
    """
    A class that contains rules/actions for the game Dynamic Multi-dimensional Capability Resource Scheduling (DMCRS).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "resources", "game_over", "self_position", "sight", "current_step"],
    )
    ResourceObservation = namedtuple(
        "ResourceObservation", ["position", "capability", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        resources,
        min_resource_capability,
        max_resource_capability,
        min_task_capability,
        max_task_capability,
        field_size,
        max_num_tasks,
        sight, # sight of the agent
        max_episode_steps,
        force_coop, # whether to force cooperation
        capability_dim, # dimension of the capability
        task_spawn_min_time=0, # minimum time between task spawns
        task_spawn_max_time=15, # maximum time between task spawns
        normalize_reward=True,
        grid_observation=False,
        observe_agent_capabilities=True,
        penalty=0.0,
        render_mode=None,
        enable_task_dynamic_capability=True,
        task_dynamic_rates=[0.04, 0.1], # can be adjusted
        task_dynamic_factors=[0.8, 0.95],
        task_fluctuation_range=[-0.2, 0.2],
        growth_reward_base_multiplier=1.05,
        growth_reward_time_decay_rate=0.03,
        task_generation_coop_ratio=1.0,
        task_capability_ceiling_slack=0.5,
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.resources = [Resource() for _ in range(resources)]
        self.capability_dim = capability_dim # save dimension info

        self.field = np.zeros(field_size+(capability_dim,), np.float32) # can modify to 1D or not
        self.visible_field = np.zeros(field_size+(capability_dim,), np.float32)
        self.tasks = {}

        self.enable_task_dynamic_capability = enable_task_dynamic_capability
        self.task_dynamic_rates = task_dynamic_rates
        self.task_dynamic_factors = task_dynamic_factors
        self.task_fluctuation_range = task_fluctuation_range
        self.growth_reward_base_multiplier = growth_reward_base_multiplier
        self.growth_reward_time_decay_rate = growth_reward_time_decay_rate
        self.task_generation_coop_ratio = task_generation_coop_ratio
        self.task_capability_ceiling_slack = task_capability_ceiling_slack

        self.penalty = penalty

        # modify min_task_capability to be a 2D array
        if isinstance(min_task_capability, Iterable):
            if isinstance(min_task_capability[0], Iterable):
                assert (
                    len(min_task_capability) == max_num_tasks
                ), "min_task_capability must be a list of length max_num_tasks"
                self.min_task_capability = np.array(min_task_capability)
            else:
                self.min_task_capability = np.array([min_task_capability] * max_num_tasks)
        else:
            self.min_task_capability = np.array([[min_task_capability] * capability_dim] * max_num_tasks)

        # modify max_task_capability to be a 2D array
        if max_task_capability is None:
            self.max_task_capability = None
        elif isinstance(max_task_capability, Iterable):
            if isinstance(max_task_capability[0], Iterable):
                assert (
                    len(max_task_capability) == max_num_tasks
                ), "max_task_capability must be a list of length max_num_tasks"
                self.max_task_capability = np.array(max_task_capability)
            else:
                self.max_task_capability = np.array([max_task_capability] * max_num_tasks)
        else:
            self.max_task_capability = np.array([[max_task_capability] * capability_dim] * max_num_tasks)

        # verify that min_task_capability <= max_task_capability for each task
        if self.max_task_capability is not None:
            # check if min_task_capability is less than max_task_capability
            for min_task_capability, max_task_capability in zip(
                self.min_task_capability, self.max_task_capability
            ):
                assert (
                    (min_task_capability <= max_task_capability).all()
                ), "min_task_capability must be less than or equal to max_task_capability for each task"
        # set max_num_tasks and the spawned task counter
        self.max_num_tasks = max_num_tasks
        self._task_spawned = np.zeros(capability_dim)
        self._task_base_value_spawned = 0.0

        # modify min_resource_capability to be a 2D array
        if isinstance(min_resource_capability, Iterable):
            if isinstance(min_resource_capability[0], Iterable):
                assert (
                    len(min_resource_capability) == resources
                ), "min_resource_capability must be a list of length resources"
                self.min_resource_capability = np.array(min_resource_capability)
            else:
                self.min_resource_capability = np.array([min_resource_capability] * resources)
        else:
            self.min_resource_capability = np.array([[min_resource_capability] * capability_dim] * resources)

        # modify max_resource_capability to be a 2D array
        if isinstance(max_resource_capability, Iterable):
            if isinstance(max_resource_capability[0], Iterable):
                assert (
                    len(max_resource_capability) == resources
                ), "max_resource_capability must be a list of length resources"
                self.max_resource_capability = np.array(max_resource_capability)
            else:
                self.max_resource_capability = np.array([max_resource_capability] * resources)
        else:
            self.max_resource_capability = np.array([[max_resource_capability] * capability_dim] * resources)

        # verfify that min_resource_capability <= max_resource_capability for each resource
        if self.max_resource_capability is not None:
            # check if min_resource_capability is less than max_resource_capability for each resource
            for i, (min_resource_capability, max_resource_capability) in enumerate(
                zip(self.min_resource_capability, self.max_resource_capability)
            ):
                assert (
                    (min_resource_capability <= max_resource_capability).all()
                ), f"min_resource_capability must be less than or equal to max_resource_capability for each resource but was {min_resource_capability} > {max_resource_capability} for resource {i}"
        
        self.task_spawn_min_time = task_spawn_min_time
        self.task_spawn_max_time = task_spawn_max_time
        
        # set other basic attributes of the environment
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        self._observe_agent_capabilities = observe_agent_capabilities

        # set action space and observation space
        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.resources))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.resources))
        )

        self.viewer = None

        self.n_agents = len(self.resources)
        self.current_step = 0

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with tasks / task description (x, y, global_x, global_y, capability_vector)*task_count
        - resource description (x, y, capability_vector)*resource_count
        """
        capability_dim = self.capability_dim

        max_resource_capability_per_dim = self.max_resource_capability.max(axis=0)
        print(self.max_task_capability)
        max_task_capability_per_dim = (
            self.max_task_capability.max(axis=0)
            if self.max_task_capability is not None
            else self._get_task_capability_ceiling_per_dim()
        )

        # no grid observation
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_num_tasks = self.max_num_tasks
            # observation space with agent capabilities
            if self._observe_agent_capabilities:
                min_obs = ([-1, -1, -1, -1] + [0] * capability_dim) * max_num_tasks + ([-1, -1] + [0] * capability_dim) * len(self.resources)
                max_obs = [field_x - 1, field_y - 1, field_x - 1, field_y - 1, *max_task_capability_per_dim] * max_num_tasks + [
                    field_x - 1,
                    field_y - 1,
                    *max_resource_capability_per_dim,
                ] * len(self.resources)
            else: # observation space without agent capabilities
                min_obs = ([-1, -1, -1, -1] + [0] * capability_dim) * max_num_tasks + [-1, -1] * len(self.resources)
                max_obs = [field_x - 1, field_y - 1, field_x - 1, field_y - 1, *max_task_capability_per_dim] * max_num_tasks + [
                    field_x - 1,
                    field_y - 1,
                ] * len(self.resources)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            min_obs = []
            max_obs = []
            # agents layer: agent capabilities
            if self._observe_agent_capabilities:
                for dim in range(capability_dim):
                    min_obs.append(np.zeros(grid_shape, dtype=np.float32))
                    max_obs.append(np.ones(grid_shape, dtype=np.float32) * max_resource_capability_per_dim[dim])
            else: # Indicate whether there is an agent
                min_obs.append(np.zeros(grid_shape, dtype=np.float32))
                max_obs.append(np.ones(grid_shape, dtype=np.float32))

            # tasks layer: tasks capability
            for dim in range(capability_dim):
                min_obs.append(np.zeros(grid_shape, dtype=np.float32))
                max_obs.append(np.ones(grid_shape, dtype=np.float32) * max_task_capability_per_dim[dim])            

            # access layer: i the cell available
            min_obs.append(np.zeros(grid_shape, dtype=np.float32))
            max_obs.append(np.ones(grid_shape, dtype=np.float32))

            # total layer
            min_obs = np.stack(min_obs)
            max_obs = np.stack(max_obs)

        low_obs = np.array(min_obs)
        high_obs = np.array(max_obs)

        assert low_obs.shape == high_obs.shape
        return gym.spaces.Box(
            low=low_obs, high=high_obs, shape=[len(low_obs)], dtype=np.float32
        )

    @classmethod
    def from_obs(cls, obs): #(todo)
        resources = []
        for r in obs.resources:
            resource = Resource()
            resource.setup(r.position, r.capability, obs.field.shape)
            resource.score = r.score if r.score else 0
            resources.append(resource)
        
        capability_dim = len(resources[0].capability)

        env = cls(
            resources,
            min_resource_capability=[1] * capability_dim,
            max_resource_capability=[2] * capability_dim,
            min_task_capability=[1] * capability_dim,
            max_task_capability=None,
            field_size=None,
            max_num_tasks=None,
            sight=None,
            max_episode_steps=50,
            force_coop=False,
            capability_dim=capability_dim,
        )

        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape[:2]

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _compute_task_base_value(self, initial_task_capability):
        return float(np.sum(initial_task_capability))

    def _get_task_capability_ceiling_per_dim(self):
        return (
            np.sum(self.max_resource_capability, axis=0).astype(np.float32)
            + self.task_capability_ceiling_slack
        )

    def _compute_task_nominal_base_value(self, task):
        if task.dynamic_type in (
            TaskDynamicType.LINEAR_GROWTH,
            TaskDynamicType.EXPONENTIAL_GROWTH,
        ):
            return task.base_value * self.growth_reward_base_multiplier
        return task.base_value

    def _compute_task_value_at_current_step(self, task):
        if task.dynamic_type in (
            TaskDynamicType.LINEAR_DECAY,
            TaskDynamicType.EXPONENTIAL_DECAY,
        ):
            return float(np.sum(task.capability))
        if task.dynamic_type in (
            TaskDynamicType.LINEAR_GROWTH,
            TaskDynamicType.EXPONENTIAL_GROWTH,
        ):
            alive_steps = max(0, self.current_step - task.spawn_time)
            time_decay = float(
                np.exp(-self.growth_reward_time_decay_rate * alive_steps)
            )
            return task.base_value * self.growth_reward_base_multiplier * time_decay
        return task.base_value

    def _sample_task_dynamic_spec(self):
        if not self.enable_task_dynamic_capability:
            return TaskDynamicType.NONE, {}

        dynamic_type = self.np_random.choice(list(TaskDynamicType))
        params = {}
        if dynamic_type in (
            TaskDynamicType.LINEAR_DECAY,
            TaskDynamicType.LINEAR_GROWTH,
        ):
            dynamic_rate = self.np_random.uniform(
                self.task_dynamic_rates[0], self.task_dynamic_rates[1]
            )
            params["dynamic_rate"] = np.full(self.capability_dim, dynamic_rate, dtype=np.float32)
        elif dynamic_type in (
            TaskDynamicType.EXPONENTIAL_DECAY,
            TaskDynamicType.EXPONENTIAL_GROWTH,
        ):
            params["dynamic_factor"] = np.array(
                [
                    self.np_random.uniform(
                        self.task_dynamic_factors[0], self.task_dynamic_factors[1]
                    )
                    for _ in range(self.capability_dim)
                ],
                dtype=np.float32,
            )
        elif dynamic_type == TaskDynamicType.RANDOM_FLUCTUATE:
            params["min_delta"] = self.task_fluctuation_range[0]
            params["max_delta"] = self.task_fluctuation_range[1]
        return dynamic_type, params

    def _sync_fields_from_tasks(self):
        self.field.fill(0)
        self.visible_field.fill(0)
        for task in self.tasks.values():
            if not task.is_active():
                continue
            row, col = task.position
            self.field[row, col] = task.capability
            if task.is_visible(self.current_step):
                self.visible_field[row, col] = task.capability

    def get_task(self, row, col):
        return self.tasks.get((row, col))

    def _gen_valid_moves(self): # generate the valid moves for each resource
        self._valid_actions = {
            resource: [
                action for action in Action if self._is_valid_action(resource, action)
            ]
            for resource in self.resources
        }

    def neighborhood(self, field, row, col, distance=1, ignore_diag=False): # return the neighborhood of the cell (row, col)
        if not ignore_diag:
            return field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_task(self, row, col): # return the sum of capabilities to determine whether there is task in the adjacent cells
        return (
            self.visible_field[max(row - 1, 0), col]
            + self.visible_field[min(row + 1, self.rows - 1), col]
            + self.visible_field[row, max(col - 1, 0)]
            + self.visible_field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_task_location(self, row, col): # return the location of the task in the adjacent cells
        if row > 1 and np.any(self.visible_field[row - 1, col] > EPSILON):
            return row - 1, col
        elif row < self.rows - 1 and np.any(self.visible_field[row + 1, col] > EPSILON):
            return row + 1, col
        elif col > 1 and np.any(self.visible_field[row, col - 1] > EPSILON):
            return row, col - 1
        elif col < self.cols - 1 and np.any(self.visible_field[row, col + 1] > EPSILON):
            return row, col + 1

    def adjacent_resources(self, row, col): # return the resources in the adjacent cells
        return [
            resource
            for resource in self.resources
            if abs(resource.position[0] - row) == 1
            and resource.position[1] == col
            or abs(resource.position[1] - col) == 1
            and resource.position[0] == row
        ]

    def spawn_tasks(
        self,
        max_num_tasks,
        min_capabilities,
        max_capabilities,
        coop_single_capability_peak_per_dim=None,
        coop_team_capability_sum_per_dim=None,
    ): # spawn the tasks in the environment randomly
        task_count = 0
        attempts = 0

        # permute task capabilities
        task_permutation = self.np_random.permutation(max_num_tasks)
        min_capabilities = min_capabilities[task_permutation]
        max_capabilities = max_capabilities[task_permutation]

        while task_count < max_num_tasks and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(self.field, row, col).sum() > EPSILON # (todo:task can overlap with task that has not yet been generated at time 0)
                or self.neighborhood(self.field, row, col, distance=2, ignore_diag=True) > EPSILON
                or not self.is_empty_location(row, col)
            ):
                continue
            
            task_capability = np.zeros(self.capability_dim, dtype=np.float32)
            for i in range(self.capability_dim):
                if min_capabilities[task_count][i] == max_capabilities[task_count][i]:
                    task_capability[i] = min_capabilities[task_count][i]
                else:
                    task_capability[i] = self.np_random.uniform(
                        min_capabilities[task_count][i], max_capabilities[task_count][i]
                    )

            if self.force_coop:
                single_capability_peak = (
                    coop_single_capability_peak_per_dim
                    if coop_single_capability_peak_per_dim is not None
                    else np.max(self.max_resource_capability, axis=0)
                )
                team_capability_sum = (
                    coop_team_capability_sum_per_dim
                    if coop_team_capability_sum_per_dim is not None
                    else np.sum(self.max_resource_capability, axis=0)
                )
                coop_dims = np.where(
                    (task_capability > single_capability_peak + EPSILON)
                    & (task_capability <= team_capability_sum + EPSILON)
                )[0]
                if coop_dims.size == 0:
                    candidate_dims = np.where(
                        np.minimum(max_capabilities[task_count], team_capability_sum)
                        > single_capability_peak + EPSILON
                    )[0]
                    if candidate_dims.size > 0:
                        bottleneck_dim = int(
                            self.np_random.choice(candidate_dims)
                        )
                        bottleneck_lower = max(
                            float(min_capabilities[task_count][bottleneck_dim]),
                            float(single_capability_peak[bottleneck_dim] + EPSILON),
                        )
                        bottleneck_upper = min(
                            float(max_capabilities[task_count][bottleneck_dim]),
                            float(team_capability_sum[bottleneck_dim]),
                        )
                        if bottleneck_upper <= bottleneck_lower:
                            task_capability[bottleneck_dim] = bottleneck_upper
                        else:
                            task_capability[bottleneck_dim] = self.np_random.uniform(
                                bottleneck_lower,
                                bottleneck_upper,
                            )
            
            spawn_time = self.np_random.integers(
                self.task_spawn_min_time, self.task_spawn_max_time + 1
            )
            dynamic_type, dynamic_params = self._sample_task_dynamic_spec()
            task = Task()
            task.setup(
                position=(row, col),
                capability=task_capability,
                spawn_time=spawn_time,
                dynamic_type=dynamic_type,
                dynamic_params=dynamic_params,
                base_value=self._compute_task_base_value(task_capability),
            )
            self.tasks[(row, col)] = task
            self.field[row, col] = task_capability

            task_count += 1
        self._sync_fields_from_tasks()
        self._task_spawned = np.sum(
            [task.initial_capability for task in self.tasks.values()], axis=0
        ) if self.tasks else np.zeros(self.capability_dim, dtype=np.float32)
        self._task_base_value_spawned = float(
            np.sum([self._compute_task_nominal_base_value(task) for task in self.tasks.values()])
        )

    # check if the task is visible
    def is_task_visible(self, row, col):
        task = self.get_task(row, col)
        return task is not None and task.is_visible(self.current_step)

    def _update_visible_field(self):
        self._sync_fields_from_tasks()

    def _update_task_capabilities(self):
        if not self.enable_task_dynamic_capability:
            return

        task_capability_ceiling_per_dim = self._get_task_capability_ceiling_per_dim()
        expired_positions = []
        for position, task in self.tasks.items():
            task.apply_dynamic(self.current_step, self.np_random)
            if task.dynamic_type in (
                TaskDynamicType.LINEAR_GROWTH,
                TaskDynamicType.EXPONENTIAL_GROWTH,
            ):
                task.capability = np.minimum(
                    task.capability,
                    task_capability_ceiling_per_dim,
                )
            if not task.is_active():
                expired_positions.append(position)
        for position in expired_positions:
            del self.tasks[position]
        self._sync_fields_from_tasks()

    # check if the location is empty
    def is_empty_location(self, row, col):
        if np.any(self.field[row, col] > EPSILON): # (todo:resources can overlap with task that has not yet been generated at time 0 and fix field)
            return False
        for a in self.resources:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_resources(self, min_resource_capabilities, max_resource_capabilities):
        # permute resource capabilities
        resource_permutation = self.np_random.permutation(len(self.resources))
        min_resource_capabilities = min_resource_capabilities[resource_permutation]
        max_resource_capabilities = max_resource_capabilities[resource_permutation]
        for resource, min_resource_capability, max_resource_capability in zip(
            self.resources, min_resource_capabilities, max_resource_capabilities
        ):
            attempts = 0
            resource.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self.is_empty_location(row, col):
                    resource.capability = np.zeros(self.capability_dim, dtype=np.float32)
                    for i in range(self.capability_dim):
                        resource.capability[i] = self.np_random.uniform(
                            min_resource_capability[i], max_resource_capability[i]
                        )
                    resource.setup(
                        (row, col),
                        resource.capability,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, resource, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                resource.position[0] > 0
                and np.all(self.visible_field[resource.position[0] - 1, resource.position[1]] <= EPSILON)
            )
        elif action == Action.SOUTH:
            return (
                resource.position[0] < self.rows - 1
                and np.all(self.visible_field[resource.position[0] + 1, resource.position[1]] <= EPSILON)
            )
        elif action == Action.WEST:
            return (
                resource.position[1] > 0
                and np.all(self.visible_field[resource.position[0], resource.position[1] - 1] <= EPSILON)
            )
        elif action == Action.EAST:
            return (
                resource.position[1] < self.cols - 1
                and np.all(self.visible_field[resource.position[0], resource.position[1] + 1] <= EPSILON)
            )
        elif action == Action.LOAD:
            return np.any(self.adjacent_task(*resource.position) > EPSILON)

        self.logger.error("Undefined action {} from {}".format(action, resource.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position): # Convert global coordinates to local coordinates relative to the center of the agent's field of view
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[resource] for resource in self.resources]))

    def _make_obs(self, resource):
        return self.Observation(
            actions=self._valid_actions[resource],
            resources=[
                self.ResourceObservation(
                    position=self._transform_to_neighborhood(
                        resource.position, self.sight, a.position
                    ),
                    capability=a.capability,
                    is_self=a == resource,
                    history=a.history,
                    reward=a.reward if a == resource else None,
                )
                for a in self.resources
                if (
                    min(
                        self._transform_to_neighborhood(
                            resource.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        resource.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(self.visible_field, *resource.position, self.sight)),
            game_over=self.game_over,
            self_position=resource.position,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation): # no grid observation
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self resource is always first
            seen_resources = [r for r in observation.resources if r.is_self] + [
                r for r in observation.resources if not r.is_self
            ]

            task_obs_len = 4 + self.capability_dim
            # task
            for i in range(self.max_num_tasks):
                obs[task_obs_len * i] = -1
                obs[task_obs_len * i + 1] = -1
                obs[task_obs_len * i + 2] = -1
                obs[task_obs_len * i + 3] = -1
                for dim in range(self.capability_dim):
                    obs[task_obs_len * i + 4 + dim] = 0
            for i, (y, x) in enumerate(zip(*np.where(np.any(observation.field > EPSILON, axis = 2)))):
                obs[task_obs_len * i] = y
                obs[task_obs_len * i + 1] = x
                obs[task_obs_len * i + 2] = y + max(observation.self_position[0] - self.sight, 0)
                obs[task_obs_len * i + 3] = x + max(observation.self_position[1] - self.sight, 0)            
                for dim in range(self.capability_dim):
                    obs[task_obs_len * i + 4 + dim] = observation.field[y, x][dim]

            # resource
            resource_obs_len = 2 + self.capability_dim if self._observe_agent_capabilities else 2
            for i in range(len(self.resources)):
                obs[self.max_num_tasks * task_obs_len + resource_obs_len * i] = -1
                obs[self.max_num_tasks * task_obs_len + resource_obs_len * i + 1] = -1
                if self._observe_agent_capabilities:
                    for dim in range(self.capability_dim):
                        obs[self.max_num_tasks * task_obs_len + resource_obs_len * i + 2 + dim] = 0

            for i, r in enumerate(seen_resources):
                obs[self.max_num_tasks * task_obs_len + resource_obs_len * i] = r.position[0]
                obs[self.max_num_tasks * task_obs_len + resource_obs_len * i + 1] = r.position[1]
                if self._observe_agent_capabilities:
                    for dim in range(self.capability_dim):
                        obs[self.max_num_tasks * task_obs_len + resource_obs_len * i + 2 + dim] = r.capability[dim]

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size[0:2]
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)
            
            all_layers = []

            # agent layer
            if self._observe_agent_capabilities:
                for dim in range(self.capability_dim):
                    agents_layer = np.zeros(grid_shape, dtype=np.float32)
                    for resource in self.resources:
                        resource_x, resource_y = resource.position
                        agents_layer[resource_x + self.sight, resource_y + self.sight] = resource.capability[dim]
                    all_layers.append(agents_layer)
            else:
                agents_layer = np.zeros(grid_shape, dtype=np.float32)
                for resource in self.resources:
                    resource_x, resource_y = resource.position
                    agents_layer[resource_x + self.sight, resource_y + self.sight] = 1
                all_layers.append(agents_layer)
            
            # task_layer
            for dim in range(self.capability_dim):
                tasks_layer = np.zeros(grid_shape, dtype=np.float32)
                tasks_layer[self.sight : -self.sight, self.sight : -self.sight] = self.visible_field[:,:,dim]
                all_layers.append(tasks_layer)

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[: self.sight, :] = 0.0
            access_layer[-self.sight :, :] = 0.0
            access_layer[:, : self.sight] = 0.0
            access_layer[:, -self.sight :] = 0.0
            # agent locations are not accessible
            for resource in self.resources:
                resource_x, resource_y = resource.position
                access_layer[resource_x + self.sight, resource_y + self.sight] = 0.0
            # task locations are not accessible
            tasks_x, tasks_y = np.where(np.any(self.field > EPSILON, axis = 2))
            for x, y in zip(tasks_x, tasks_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            all_layers.append(access_layer)

            return np.stack(all_layers)

        def get_agent_grid_bounds(agent_x, agent_y):
            return (
                agent_x,
                agent_x + 2 * self.sight + 1,
                agent_y,
                agent_y + 2 * self.sight + 1,
            )

        # generate basic observation data for each agent
        observations = [self._make_obs(resource) for resource in self.resources]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*resource.position) for resource in self.resources
            ] # Calculate the visual field boundary for each agent
            nobs = tuple(
                [
                    layers[:, start_x:end_x, start_y:end_y]
                    for start_x, end_x, start_y, end_y in agents_bounds
                ]
            )
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])

        # check the space of obs
        for i, obs in enumerate(nobs):
            assert self.observation_space[i].contains(
                obs
            ), f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"

        return nobs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        self.field = np.zeros(self.field_size+(self.capability_dim,), np.float32)
        self.visible_field = np.zeros(self.field_size+(self.capability_dim,), np.float32)
        self.tasks = {}
        self.current_step = 0
        
        self.spawn_resources(self.min_resource_capability, self.max_resource_capability)
        episode_resource_capability_max = np.max(
            np.array([resource.capability for resource in self.resources]),
            axis=0,
        )
        episode_resource_capability_sum = np.sum(
            np.array([resource.capability for resource in self.resources]),
            axis=0,
        )
        episode_task_capability_upper_per_dim = (
            episode_resource_capability_sum * self.task_generation_coop_ratio
        )
        episode_task_capability_upper_per_dim = np.minimum(
            episode_task_capability_upper_per_dim,
            self._get_task_capability_ceiling_per_dim(),
        )
        episode_task_capability_upper_per_task = np.tile(
            episode_task_capability_upper_per_dim,
            (self.max_num_tasks, 1),
        )
        episode_task_capability_upper_per_task = np.maximum(
            episode_task_capability_upper_per_task,
            self.min_task_capability,
        )

        self.spawn_tasks(
            self.max_num_tasks,
            min_capabilities=self.min_task_capability,
            max_capabilities=np.minimum(
                self.max_task_capability,
                episode_task_capability_upper_per_task,
            )
            if self.max_task_capability is not None
            else episode_task_capability_upper_per_task,
            coop_single_capability_peak_per_dim=episode_resource_capability_max,
            coop_team_capability_sum_per_dim=episode_resource_capability_sum,
        )
        self._update_visible_field()

        self._game_over = False
        self._gen_valid_moves()

        nobs = self._make_gym_obs()
        for r in self.resources:
            print(r.position)
        for t in self.tasks.values():
            print(t.position, t.capability, t.base_value)
        return nobs, self._get_info()

    def step(self, actions):
        self.current_step += 1

        for r in self.resources:
            r.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[r] else Action.NONE
            for r, a in zip(self.resources, actions)
        ]
        
        # check if actions are valid
        for i, (resource, action) in enumerate(zip(self.resources, actions)):
            if action not in self._valid_actions[resource]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        resource.name, resource.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_resources = set()

        # move resources
        # if two or more resources try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for resource, action in zip(self.resources, actions):
            if action == Action.NONE:
                collisions[resource.position].append(resource)
            elif action == Action.NORTH:
                collisions[(resource.position[0] - 1, resource.position[1])].append(resource)
            elif action == Action.SOUTH:
                collisions[(resource.position[0] + 1, resource.position[1])].append(resource)
            elif action == Action.WEST:
                collisions[(resource.position[0], resource.position[1] - 1)].append(resource)
            elif action == Action.EAST:
                collisions[(resource.position[0], resource.position[1] + 1)].append(resource)
            elif action == Action.LOAD:
                collisions[resource.position].append(resource)
                loading_resources.add(resource)

        # and do movements for non colliding resources
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an resource will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_resources:
            # find adjacent task
            resource = loading_resources.pop()
            trow, tcol = self.adjacent_task_location(*resource.position)
            task = self.get_task(trow, tcol)
            if task is None:
                continue
            task_capability = task.capability 

            adj_resources = self.adjacent_resources(trow, tcol)
            adj_resources = [
                r for r in adj_resources if r in loading_resources or r is resource
            ]

            adj_resource_capability = np.sum([a.capability for a in adj_resources], axis=0) # sum of capabilities of adjacent resources
            loading_resources = loading_resources - set(adj_resources)

            if not (adj_resource_capability >= task_capability).all():
                # failed to load
                for a in adj_resources:
                    a.reward -= self.penalty
                continue

            # else the task was loaded and each resource scores points according to their capability on each dimension
            task_value_at_current_step = self._compute_task_value_at_current_step(task)
            for a in adj_resources:
                contribution = 0.0
                for i in range(self.capability_dim):
                    if adj_resource_capability[i] > EPSILON:
                        contribution += float(a.capability[i] / adj_resource_capability[i])
                contribution = contribution / self.capability_dim
                a.reward += task_value_at_current_step * contribution
                if self._normalize_reward:
                    a.reward = a.reward / max(self._task_base_value_spawned, EPSILON)

            # and the task is removed
            del self.tasks[(trow, tcol)]

        self._update_task_capabilities()
        self._update_visible_field()

        self._game_over = (
            len(self.tasks) == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for r in self.resources:
            r.score += r.reward

        for r in self.resources:
            print(r.position)
        
        for t in self.tasks.values():
            print(t.position, t.capability, t.base_value)
            

        rewards = [r.reward for r in self.resources]
        done = self._game_over
        truncated = False
        info = self._get_info()

        return self._make_gym_obs(), rewards, done, truncated, info

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()

    def test_make_gym_obs(self):
        """Test wrapper to test the current observation in a public manner."""
        return self._make_gym_obs()

    def test_gen_valid_moves(self):
        """Wrapper around a private method to test if the generated moves are valid."""
        try:
            self._gen_valid_moves()
        except Exception as _:
            return False
        return True
