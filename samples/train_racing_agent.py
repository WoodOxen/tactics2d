import sys

sys.path.append(".")
sys.path.append("..")

import logging

logging.basicConfig(level=logging.ERROR)

# import os

# os.environ["SDL_VIDEODRIVER"] = "dummy"

from collections import deque, defaultdict

import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FlattenObservation
import torch

from rllib.algorithms.sac import SAC
from tactics2d.envs import RacingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    render_mode = "human"
    env = FlattenObservation(
        GrayScaleObservation(RacingEnv(render_mode=render_mode, render_fps=60))
    )

    agent_configs = {
        "state_space": env.observation_space,
        "state_dim": env.observation_space.shape[0] + 4,
        "action_space": env.action_space,
    }
    agent = SAC(agent_configs)

    accumulated_rewards = deque([], maxlen=10)
    step_cnt = 0
    n_step = int(1e5)
    episode_cnt = 0
    print_interval = 10

    stop_reason_cnt = defaultdict(int)

    while step_cnt < n_step:
        accumulated_reward = 0
        average_action = np.array([0.0, 0.0])
        state, info = env.reset()
        done = False
        iter_cnt = 0

        while not done:
            if render_mode == "human":
                env.render()

            processed_state = np.concatenate(
                [
                    state,
                    np.array(
                        [
                            info["velocity"][0],
                            info["velocity"][1],
                            info["acceleration"],
                            info["heading"],
                        ]
                    ),
                ]
            )

            action = agent.get_action(processed_state)

            processed_action = np.array([action[0] * 0.5, action[1] * 2.0], dtype=np.float32)
            average_action += processed_action
            next_state, reward, _, done, info = env.step(processed_action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((processed_state, action, reward, done_mask))
            agent.train()

            state = next_state
            accumulated_reward += reward
            step_cnt += 1
            iter_cnt += 1

        accumulated_rewards.append(accumulated_reward)
        stop_reason_cnt[info["status"].name] += 1
        episode_cnt += 1
        print(
            "Stop reason: %s, average action: (%f, %f)"
            % (
                info["status"].name,
                average_action[0] / iter_cnt,
                average_action[1] / iter_cnt,
            )
        )
        if episode_cnt % print_interval == 0:
            print(
                "Episode: %d, score: %f, buffer capacity: %d"
                % (episode_cnt, accumulated_reward, len(agent.buffer))
            )
