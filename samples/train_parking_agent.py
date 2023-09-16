import sys

sys.path.append(".")
sys.path.append("..")

import logging

logging.basicConfig(level=logging.ERROR)

from collections import deque, defaultdict

import numpy as np
from gymnasium import spaces

from rllib.algorithms.sac import SAC
from tactics2d.envs import ParkingEnv


def process_state(info: dict):
    lidar = info["lidar"]
    lidar[lidar == float("inf")] = -1.0
    state = np.concatenate(
        [
            lidar,
            [info["velocity"][0]],
            [info["velocity"][1]],
            [info["acceleration"]],
            [info["heading"]],
        ]
    )
    return state


if __name__ == "__main__":
    render_mode = "human"
    env = ParkingEnv(render_mode=render_mode, render_fps=60)

    agent_configs = {
        "state_space": spaces.Box(low=-1.0, high=12.0, shape=(504, 1)),
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

            state = process_state(info)
            action = agent.get_action(state)
            processed_action = np.array([action[0] * 0.75, action[1] * 1.0], dtype=np.float32)
            average_action += processed_action

            observation, reward, _, done, info = env.step(processed_action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, done_mask))
            agent.train()

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

    env.close()
