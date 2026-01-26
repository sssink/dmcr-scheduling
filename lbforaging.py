import argparse
import logging
import time
import sys

import gymnasium as gym
import numpy as np

import lbforaging  # noqa
from lbforaging.agents.heuristic_agent import H3, H4

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    obss, _ = env.reset()
    done = False

    returns = np.zeros(env.unwrapped.n_agents)

    # for player in env.unwrapped.players:
    #     player.set_controller(H4(player))

    if render:
        env.render()
        # time.sleep(0.5)
        print("press 'Enter' to continue.(input 'q' to exit)")
        user_input = input().strip().lower()
        if user_input == 'q':
            sys.exit(0)

    while not done:
        actions = env.action_space.sample()
        # print(obss)
        # actions = [player.controller.step(obs) for player, obs in zip(env.unwrapped.players, obss)]

        obss, rewards, done, _, _ = env.step(actions)
        returns += rewards

        if render:
            env.render()
            # time.sleep(0.5)
            print("press 'Enter' to continue.(input 'q' to exit)")
            user_input = input().strip().lower()
            if user_input == 'q':
                sys.exit(0)

    print("Returns: ", returns)


def main(episodes=1, render=False):
    env = gym.make("Foraging-2s-9x9-3p-2f-2d-v3")
    for episode in range(episodes):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to run"
    )

    args = parser.parse_args()
    main(args.episodes, args.render)
