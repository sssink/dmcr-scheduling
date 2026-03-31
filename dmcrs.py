import argparse
import logging
import time
import sys

import gymnasium as gym
import numpy as np

import dmcrs  # noqa
from dmcrs.agents.heuristic_agent import H3, H4

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    obss, _ = env.reset()
    done = False

    returns = np.zeros(env.unwrapped.n_agents)

    # for resource in env.unwrapped.resources:
    #     resource.set_controller(H4(resource))

    if render:
        env.render()
        # time.sleep(0.5)
        print("OBS", obss)
        print("press 'Enter' to continue.(input 'q' to exit)")
        user_input = input().strip().lower()
        if user_input == 'q':
            sys.exit(0)

    while not done:
        actions = env.action_space.sample()
        # print(obss)
        # actions = [resource.controller.step(obs) for resource, obs in zip(env.unwrapped.resources, obss)]
        obss, rewards, done, _, _ = env.step(actions)
        returns += rewards

        if render:
            env.render()
            # time.sleep(0.5)
            print("OBS", obss)
            print("press 'Enter' to continue.(input 'q' to exit)")
            user_input = input().strip().lower()
            if user_input == 'q':
                sys.exit(0)

    print("Returns: ", returns)


def main(episodes=1, render=False):
    env = gym.make("dmcrs-2s-9x9-3r-2t-3d-coop-v3")
    for episode in range(episodes):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the Dynamic Multi-dimensional Capability Resource Scheduling game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to run"
    )

    args = parser.parse_args()
    main(args.episodes, args.render)
