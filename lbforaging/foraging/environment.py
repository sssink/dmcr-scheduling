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


class Player:
    def __init__(self):
        self.controller = None # agent for the player
        self.position = None
        self.level = None # multi-dimensional level
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
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
            return "Player"

class Task:
    def __init__(self):
        self.position = None
        self.level = None
        self.initial_level = None
        self.base_value = 0.0
        self.spawn_time = 0
        self.dynamic_type = TaskDynamicType.NONE
        self.dynamic_params = {}

    def setup(
        self,
        position,
        level,
        spawn_time,
        dynamic_type,
        dynamic_params,
        base_value,
    ):
        self.position = position
        self.level = np.array(level, dtype=np.float32)
        self.initial_level = np.array(level, dtype=np.float32)
        self.base_value = float(base_value)
        self.spawn_time = int(spawn_time)
        self.dynamic_type = dynamic_type
        self.dynamic_params = dynamic_params

    def is_visible(self, current_step):
        return self.spawn_time <= current_step and np.any(self.level > EPSILON)

    def is_active(self):
        return np.any(self.level > EPSILON)

    def apply_dynamic(self, current_step, np_random):
        if not self.is_visible(current_step):
            return
        if self.dynamic_type == TaskDynamicType.LINEAR_DECAY:
            for dim in range(len(self.level)):
                if self.level[dim] > EPSILON:
                    decay = self.initial_level[dim] * self.dynamic_params["dynamic_rate"][dim]
                    self.level[dim] = max(0.0, self.level[dim] - decay)
        elif self.dynamic_type == TaskDynamicType.EXPONENTIAL_DECAY:
            for dim in range(len(self.level)):
                if self.level[dim] > EPSILON:
                    self.level[dim] = max(
                        0.0,
                        self.level[dim] * self.dynamic_params["dynamic_factor"][dim],
                    )
        elif self.dynamic_type == TaskDynamicType.LINEAR_GROWTH:
            for dim in range(len(self.level)):
                growth = self.initial_level[dim] * self.dynamic_params["dynamic_rate"][dim]
                self.level[dim] = self.level[dim] + growth
        elif self.dynamic_type == TaskDynamicType.EXPONENTIAL_GROWTH:
            for dim in range(len(self.level)):
                self.level[dim] = self.level[dim] * (
                    2 - self.dynamic_params["dynamic_factor"][dim]
                )
        elif self.dynamic_type == TaskDynamicType.RANDOM_FLUCTUATE:
            for dim in range(len(self.level)):
                if self.level[dim] > EPSILON:
                    delta = np_random.uniform(
                        self.dynamic_params["min_delta"],
                        self.dynamic_params["max_delta"],
                    )
                    self.level[dim] = max(0.0, self.level[dim] + delta)


class ForagingEnv(gym.Env):
    """
    A class that contains rules/actions for the game Dynamic Multi-Level-based Foraging (DMLBF).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "self_position", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        min_player_level,
        max_player_level,
        min_task_level,
        max_task_level,
        field_size,
        max_num_tasks,
        sight, # sight of the agent
        max_episode_steps,
        force_coop, # whether to force cooperation
        level_dim, # dimension of the level
        task_spawn_min_time=0, # minimum time between task spawns
        task_spawn_max_time=15, # maximum time between task spawns
        normalize_reward=True,
        grid_observation=False,
        observe_agent_levels=True,
        penalty=0.0,
        render_mode=None,
        enable_task_dynamic_level=True,
        task_dynamic_rates=[0.04, 0.1], # can be adjusted
        task_dynamic_factors=[0.8, 0.95],
        task_fluctuation_range=[-0.2, 0.2],
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.players = [Player() for _ in range(players)]
        self.level_dim = level_dim # save dimension info

        self.field = np.zeros(field_size+(level_dim,), np.float32) # can modify to 1D or not
        self.visible_field = np.zeros(field_size+(level_dim,), np.float32)
        self.tasks = {}

        self.enable_task_dynamic_level = enable_task_dynamic_level
        self.task_dynamic_rates = task_dynamic_rates
        self.task_dynamic_factors = task_dynamic_factors
        self.task_fluctuation_range = task_fluctuation_range

        self.penalty = penalty

        # modify min_task_level to be a 2D array
        if isinstance(min_task_level, Iterable):
            if isinstance(min_task_level[0], Iterable):
                assert (
                    len(min_task_level) == max_num_tasks
                ), "min_task_level must be a list of length max_num_tasks"
                self.min_task_level = np.array(min_task_level)
            else:
                self.min_task_level = np.array([min_task_level] * max_num_tasks)
        else:
            self.min_task_level = np.array([[min_task_level] * level_dim] * max_num_tasks)

        # modify max_task_level to be a 2D array
        if max_task_level is None:
            self.max_task_level = None
        elif isinstance(max_task_level, Iterable):
            if isinstance(max_task_level[0], Iterable):
                assert (
                    len(max_task_level) == max_num_tasks
                ), "max_task_level must be a list of length max_num_tasks"
                self.max_task_level = np.array(max_task_level)
            else:
                self.max_task_level = np.array([max_task_level] * max_num_tasks)
        else:
            self.max_task_level = np.array([[max_task_level] * level_dim] * max_num_tasks)

        # verify that min_task_level <= max_task_level for each task
        if self.max_task_level is not None:
            # check if min_task_level is less than max_task_level
            for min_task_level, max_task_level in zip(
                self.min_task_level, self.max_task_level
            ):
                assert (
                    (min_task_level <= max_task_level).all()
                ), "min_task_level must be less than or equal to max_task_level for each task"
        # set max_num_tasks and the spawned task counter
        self.max_num_tasks = max_num_tasks
        self._task_spawned = np.zeros(level_dim)
        self._task_base_value_spawned = 0.0

        # modify min_player_level to be a 2D array
        if isinstance(min_player_level, Iterable):
            if isinstance(min_player_level[0], Iterable):
                assert (
                    len(min_player_level) == players
                ), "min_player_level must be a list of length players"
                self.min_player_level = np.array(min_player_level)
            else:
                self.min_player_level = np.array([min_player_level] * players)
        else:
            self.min_player_level = np.array([[min_player_level] * level_dim] * players)

        # modify max_player_level to be a 2D array
        if isinstance(max_player_level, Iterable):
            if isinstance(max_player_level[0], Iterable):
                assert (
                    len(max_player_level) == players
                ), "max_player_level must be a list of length players"
                self.max_player_level = np.array(max_player_level)
            else:
                self.max_player_level = np.array([max_player_level] * players)
        else:
            self.max_player_level = np.array([[max_player_level] * level_dim] * players)

        # verfify that min_player_level <= max_player_level for each player
        if self.max_player_level is not None:
            # check if min_player_level is less than max_player_level for each player
            for i, (min_player_level, max_player_level) in enumerate(
                zip(self.min_player_level, self.max_player_level)
            ):
                assert (
                    (min_player_level <= max_player_level).all()
                ), f"min_player_level must be less than or equal to max_player_level for each player but was {min_player_level} > {max_player_level} for player {i}"
        
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
        self._observe_agent_levels = observe_agent_levels

        # set action space and observation space
        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players))
        )

        self.viewer = None

        self.n_agents = len(self.players)
        self.current_step = 0

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with tasks / task description (x, y, global_x, global_y, level_vector)*task_count
        - player description (x, y, level_vector)*player_count
        """
        level_dim = self.level_dim

        max_player_level_per_dim = self.max_player_level.max(axis=0)
        max_task_level_per_dim = (
            self.max_task_level.max(axis=0)
            if self.max_task_level is not None
            else max_player_level_per_dim * 2 # (todo) can be adjusted
        )

        # no grid observation
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_num_tasks = self.max_num_tasks
            # observation space with agent levels
            if self._observe_agent_levels:
                min_obs = ([-1, -1, -1, -1] + [0] * level_dim) * max_num_tasks + ([-1, -1] + [0] * level_dim) * len(self.players)
                max_obs = [field_x - 1, field_y - 1, field_x - 1, field_y - 1, *max_task_level_per_dim] * max_num_tasks + [
                    field_x - 1,
                    field_y - 1,
                    *max_player_level_per_dim,
                ] * len(self.players)
            else: # observation space without agent levels
                min_obs = ([-1, -1, -1, -1] + [0] * level_dim) * max_num_tasks + [-1, -1] * len(self.players)
                max_obs = [field_x - 1, field_y - 1, field_x - 1, field_y - 1, *max_task_level_per_dim] * max_num_tasks + [
                    field_x - 1,
                    field_y - 1,
                ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            min_obs = []
            max_obs = []
            # agents layer: agent levels
            if self._observe_agent_levels:
                for dim in range(level_dim):
                    min_obs.append(np.zeros(grid_shape, dtype=np.float32))
                    max_obs.append(np.ones(grid_shape, dtype=np.float32) * max_player_level_per_dim[dim])
            else: # Indicate whether there is an agent
                min_obs.append(np.zeros(grid_shape, dtype=np.float32))
                max_obs.append(np.ones(grid_shape, dtype=np.float32))

            # tasks layer: tasks level
            for dim in range(level_dim):
                min_obs.append(np.zeros(grid_shape, dtype=np.float32))
                max_obs.append(np.ones(grid_shape, dtype=np.float32) * max_task_level_per_dim[dim])            

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
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)
        
        level_dim = len(players[0].level)

        env = cls(
            players,
            min_player_level=[1] * level_dim,
            max_player_level=[2] * level_dim,
            min_task_level=[1] * level_dim,
            max_task_level=None,
            field_size=None,
            max_num_tasks=None,
            sight=None,
            max_episode_steps=50,
            force_coop=False,
            level_dim=level_dim,
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

    def _compute_task_base_value(self, initial_task_level):
        return float(np.sum(initial_task_level))

    def _sample_task_dynamic_spec(self):
        if not self.enable_task_dynamic_level:
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
            params["dynamic_rate"] = np.full(self.level_dim, dynamic_rate, dtype=np.float32)
        elif dynamic_type in (
            TaskDynamicType.EXPONENTIAL_DECAY,
            TaskDynamicType.EXPONENTIAL_GROWTH,
        ):
            params["dynamic_factor"] = np.array(
                [
                    self.np_random.uniform(
                        self.task_dynamic_factors[0], self.task_dynamic_factors[1]
                    )
                    for _ in range(self.level_dim)
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
            self.field[row, col] = task.level
            if task.is_visible(self.current_step):
                self.visible_field[row, col] = task.level

    def get_task(self, row, col):
        return self.tasks.get((row, col))

    def _gen_valid_moves(self): # generate the valid moves for each player
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
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

    def adjacent_task(self, row, col): # return the sum of levels to determine whether there is task in the adjacent cells
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

    def adjacent_players(self, row, col): # return the players in the adjacent cells
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_tasks(self, max_num_tasks, min_levels, max_levels): # spawn the tasks in the environment randomly
        task_count = 0
        attempts = 0
        min_levels = max_levels if self.force_coop else min_levels

        # permute task levels
        task_permutation = self.np_random.permutation(max_num_tasks)
        min_levels = min_levels[task_permutation]
        max_levels = max_levels[task_permutation]

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
            
            task_level = np.zeros(self.level_dim, dtype=np.float32)
            for i in range(self.level_dim):
                if min_levels[task_count][i] == max_levels[task_count][i]:
                    task_level[i] = min_levels[task_count][i]
                else:
                    task_level[i] = self.np_random.uniform(
                        min_levels[task_count][i], max_levels[task_count][i]
                    )
            
            spawn_time = self.np_random.integers(
                self.task_spawn_min_time, self.task_spawn_max_time + 1
            )
            dynamic_type, dynamic_params = self._sample_task_dynamic_spec()
            task = Task()
            task.setup(
                position=(row, col),
                level=task_level,
                spawn_time=spawn_time,
                dynamic_type=dynamic_type,
                dynamic_params=dynamic_params,
                base_value=self._compute_task_base_value(task_level),
            )
            self.tasks[(row, col)] = task
            self.field[row, col] = task_level

            task_count += 1
        self._sync_fields_from_tasks()
        self._task_spawned = np.sum(
            [task.initial_level for task in self.tasks.values()], axis=0
        ) if self.tasks else np.zeros(self.level_dim, dtype=np.float32)
        self._task_base_value_spawned = float(
            np.sum([task.base_value for task in self.tasks.values()])
        )

    # check if the task is visible
    def is_task_visible(self, row, col):
        task = self.get_task(row, col)
        return task is not None and task.is_visible(self.current_step)

    def _update_visible_field(self):
        self._sync_fields_from_tasks()

    def _update_task_levels(self):
        if not self.enable_task_dynamic_level:
            return

        expired_positions = []
        for position, task in self.tasks.items():
            task.apply_dynamic(self.current_step, self.np_random)
            if not task.is_active():
                expired_positions.append(position)
        for position in expired_positions:
            del self.tasks[position]
        self._sync_fields_from_tasks()

    # check if the location is empty
    def is_empty_location(self, row, col):
        if np.any(self.field[row, col] > EPSILON): # (todo:players can overlap with task that has not yet been generated at time 0 and fix field)
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, min_player_levels, max_player_levels):
        # permute player levels
        player_permutation = self.np_random.permutation(len(self.players))
        min_player_levels = min_player_levels[player_permutation]
        max_player_levels = max_player_levels[player_permutation]
        for player, min_player_level, max_player_level in zip(
            self.players, min_player_levels, max_player_levels
        ):
            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self.is_empty_location(row, col):
                    player.level = np.zeros(self.level_dim, dtype=np.float32)
                    for i in range(self.level_dim):
                        player.level[i] = self.np_random.uniform(
                            min_player_level[i], max_player_level[i]
                        )
                    player.setup(
                        (row, col),
                        player.level,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and np.all(self.visible_field[player.position[0] - 1, player.position[1]] <= EPSILON)
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and np.all(self.visible_field[player.position[0] + 1, player.position[1]] <= EPSILON)
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and np.all(self.visible_field[player.position[0], player.position[1] - 1] <= EPSILON)
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and np.all(self.visible_field[player.position[0], player.position[1] + 1] <= EPSILON)
            )
        elif action == Action.LOAD:
            return np.any(self.adjacent_task(*player.position) > EPSILON)

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position): # Convert global coordinates to local coordinates relative to the center of the agent's field of view
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(self.visible_field, *player.position, self.sight)),
            game_over=self.game_over,
            self_position=player.position,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation): # no grid observation
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            task_obs_len = 4 + self.level_dim
            # task
            for i in range(self.max_num_tasks):
                obs[task_obs_len * i] = -1
                obs[task_obs_len * i + 1] = -1
                obs[task_obs_len * i + 2] = -1
                obs[task_obs_len * i + 3] = -1
                for dim in range(self.level_dim):
                    obs[task_obs_len * i + 4 + dim] = 0
            for i, (y, x) in enumerate(zip(*np.where(np.any(observation.field > EPSILON, axis = 2)))):
                obs[task_obs_len * i] = y
                obs[task_obs_len * i + 1] = x
                obs[task_obs_len * i + 2] = y + max(observation.self_position[0] - self.sight, 0)
                obs[task_obs_len * i + 3] = x + max(observation.self_position[1] - self.sight, 0)            
                for dim in range(self.level_dim):
                    obs[task_obs_len * i + 4 + dim] = observation.field[y, x][dim]

            # player
            player_obs_len = 2 + self.level_dim if self._observe_agent_levels else 2
            for i in range(len(self.players)):
                obs[self.max_num_tasks * task_obs_len + player_obs_len * i] = -1
                obs[self.max_num_tasks * task_obs_len + player_obs_len * i + 1] = -1
                if self._observe_agent_levels:
                    for dim in range(self.level_dim):
                        obs[self.max_num_tasks * task_obs_len + player_obs_len * i + 2 + dim] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_num_tasks * task_obs_len + player_obs_len * i] = p.position[0]
                obs[self.max_num_tasks * task_obs_len + player_obs_len * i + 1] = p.position[1]
                if self._observe_agent_levels:
                    for dim in range(self.level_dim):
                        obs[self.max_num_tasks * task_obs_len + player_obs_len * i + 2 + dim] = p.level[dim]

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
            if self._observe_agent_levels:
                for dim in range(self.level_dim):
                    agents_layer = np.zeros(grid_shape, dtype=np.float32)
                    for player in self.players:
                        player_x, player_y = player.position
                        agents_layer[player_x + self.sight, player_y + self.sight] = player.level[dim]
                    all_layers.append(agents_layer)
            else:
                agents_layer = np.zeros(grid_shape, dtype=np.float32)
                for player in self.players:
                    player_x, player_y = player.position
                    agents_layer[player_x + self.sight, player_y + self.sight] = 1
                all_layers.append(agents_layer)
            
            # task_layer
            for dim in range(self.level_dim):
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
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
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
        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*player.position) for player in self.players
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

        self.field = np.zeros(self.field_size+(self.level_dim,), np.float32)
        self.visible_field = np.zeros(self.field_size+(self.level_dim,), np.float32)
        self.tasks = {}
        self.current_step = 0
        
        self.spawn_players(self.min_player_level, self.max_player_level)
        max_player_levels = np.array([player.level for player in self.players]).max(axis=0)

        self.spawn_tasks(
            self.max_num_tasks,
            min_levels=self.min_task_level,
            max_levels=self.max_task_level
            if self.max_task_level is not None
            else np.array([max_player_levels * 2] * self.max_num_tasks), # (todo) can be adjusted
        )
        self._update_visible_field()

        self._game_over = False
        self._gen_valid_moves()

        nobs = self._make_gym_obs()
        for p in self.players:
            print(p.position)
        for t in self.tasks.values():
            print(t.position, t.level, t.base_value)
        return nobs, self._get_info()

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]
        
        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent task
            player = loading_players.pop()
            trow, tcol = self.adjacent_task_location(*player.position)
            task = self.get_task(trow, tcol)
            if task is None:
                continue
            task_level = task.level 

            adj_players = self.adjacent_players(trow, tcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = np.sum([a.level for a in adj_players], axis=0) # sum of levels of adjacent players
            loading_players = loading_players - set(adj_players)

            if not (adj_player_level >= task_level).all():
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the task was loaded and each player scores points according to their level on each dimension
            task_base_value = task.base_value
            for a in adj_players:
                contribution = 0.0
                for i in range(self.level_dim):
                    if adj_player_level[i] > EPSILON:
                        contribution += float(a.level[i] / adj_player_level[i])
                contribution = contribution / self.level_dim
                a.reward += task_base_value * contribution
                if self._normalize_reward:
                    a.reward = a.reward / max(self._task_base_value_spawned, EPSILON)

            # and the task is removed
            del self.tasks[(trow, tcol)]

        self._update_task_levels()
        self._update_visible_field()

        self._game_over = (
            len(self.tasks) == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        for p in self.players:
            print(p.position)
        
        for t in self.tasks.values():
            print(t.position, t.level, t.base_value)
            

        rewards = [p.reward for p in self.players]
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
