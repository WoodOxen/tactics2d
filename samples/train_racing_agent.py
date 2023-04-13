import sys

sys.path.append(".")
sys.path.append("..")

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

from collections import deque, defaultdict

import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FlattenObservation
import torch

from rllib.algorithms.sac import SAC
from tactics2d.envs import RacingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env = FlattenObservation(
        GrayScaleObservation(
            RacingEnv(render_mode="rgb_array", render_fps=60)
        )
    )

    agent_configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
    }
    agent = SAC(agent_configs)

    accumulated_rewards = deque([], maxlen=10)
    step_cnt = 0
    episode_cnt = 0
    print_interval = 10

    stop_reason_cnt = defaultdict(int)

    while step_cnt < env.max_step:
        accumulated_reward = 0
        state, info = env.reset()
        done = False

        while not done:
            # env.render()
            action = agent.get_action(state)
            processed_action = [action[0] * 0.5, action[1]]
            next_state, reward, _, done, info = env.step(processed_action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, done_mask))

            state = next_state
            accumulated_reward += reward
            step_cnt += 1

        accumulated_rewards.append(accumulated_reward)
        episode_cnt += 1
        if episode_cnt % print_interval == 0:
            print(
                "Episode: %d, score: %f, buffer capacity: %d, stop_reason: %s"
                % (episode_cnt, accumulated_reward, len(agent.buffer), info["status"].name)
            )
