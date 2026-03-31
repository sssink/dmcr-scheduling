import random

import numpy as np

from dmcrs.agents.agent import BaseAgent
from dmcrs.scheduling.environment import Action


class HeuristicAgent(BaseAgent):
    name = "Heuristic Agent"

    def _center_of_resources(self, resources):
        coords = np.array([resource.position for resource in resources])
        return np.rint(coords.mean(axis=0))

    def _move_towards(self, target, allowed):
        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplementedError("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
    H1 agent always goes to the closest task
    """

    name = "H1"

    def step(self, obs):
        try:
            r, c = self._closest_task(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
    H2 Agent goes to the one visible task which is closest to the centre of visible resources
    """

    name = "H2"

    def step(self, obs):
        resources_center = self._center_of_resources(obs.resources)

        try:
            r, c = self._closest_task(obs, None, resources_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
    H3 Agent always goes to the closest task with compatible capability
    """

    name = "H3"

    def step(self, obs):
        try:
            r, c = self._closest_task(obs, self.capability)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    """
    H4 Agent goes to the one visible task which is closest to all visible resources
     such that the sum of their and H4's capability is sufficient to load the task
    """

    name = "H4"

    def step(self, obs):
        resources_center = self._center_of_resources(obs.resources)
        resources_sum_capability = sum([a.capability for a in obs.resources])

        try:
            r, c = self._closest_task(obs, resources_sum_capability, resources_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)
