import random

from dmcrs.agents import BaseAgent


class RandomAgent(BaseAgent):
    name = "Random Agent"

    def step(self, obs):
        return random.choice(obs.actions)
