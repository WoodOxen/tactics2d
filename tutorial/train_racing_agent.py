##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: train_racing_agent.py
# @Description: This script gives an example on how to train a PPO model in tactics2d's racing environment.
# @Author: Yueyuan Li
# @Version: 1.0.0


import sys

sys.path.append(".")
sys.path.append("./rllib")
sys.path.append("..")

from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from rllib.algorithms.ppo import *

from tactics2d.envs import RacingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="tactics2d-racing")


# =========================
# Define the network
# =========================


def orthogonal_init(layer, gain: float = np.sqrt(2), constant: float = 0.0):
    nn.init.orthogonal_(layer.weight.data, gain)
    nn.init.constant_(layer.bias.data, constant)
    return layer


class ImageEncoder(nn.Module):
    def __init__(self, channels: list) -> None:
        super().__init__()
        in_channels, out_channels = channels[:-1], channels[1:]
        self.net = nn.Sequential()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.net.append(orthogonal_init(nn.Conv2d(in_channel, out_channel, 3, 1)))
            self.net.append(nn.Tanh())
            self.net.append(nn.MaxPool2d(2))

        self.net.append(nn.Flatten())

    def forward(self, x):
        return self.net(x)


class AgentActor(PPOActor):
    def __init__(self, state_dim, action_dim, hidden_size, continuous):
        super().__init__(state_dim, action_dim, hidden_size, continuous)

        self.encoder = ImageEncoder([12, 4, 16, 64])

        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(6400, 640)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(640, 128)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(128, 32)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(32, action_dim)),
            nn.Tanh(),
        )

    def forward(self, state):
        state = self.encoder(state)
        return self.net(state)


class AgentCritic(PPOCritic):
    def __init__(self, state_dim, hidden_size):
        super().__init__(state_dim, hidden_size)

        self.encoder = ImageEncoder([12, 4, 16, 64])

        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(6400, 640)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(640, 128)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(128, 32)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(32, 1)),
        )

    def forward(self, state):
        state = self.encoder(state)
        return self.net(state)


# =========================
# Define the environment wrapper
# =========================


class RacingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.history_states = [np.zeros((1, 3, 96, 96))] * 4

        self.prev_position = (0, 0)
        self.track_len = 0
        self.next_closet_point = 0

    def _process_action(self, action):
        # action = action.cpu().detach().numpy()[0]
        return [action[0], 0 if action[1] < 0 else action[1], 0 if action[1] > 0 else action[1]]

    def _process_observation(self, state):
        state = state[None, ...].transpose((0, 3, 1, 2)) / 255.0
        self.history_states.pop(0)
        self.history_states.append(state)

    def is_pass(self, x, y):
        _, beta, ckpt_x, ckpt_y = self.env.track[self.next_closet_point]
        ckpt_line = (ckpt_x + np.cos(beta), ckpt_y + np.sin(beta))
        if (ckpt_x - x) * (ckpt_line[1] - y) - (ckpt_y - y) * (ckpt_line[0] - x) >= 0:
            return True
        else:
            return False

    def _process_reward(self, reward):
        reward = max(0, reward)
        # car position
        x, y = self.env.unwrapped.car.hull.position

        if (self.prev_position[0] - x) ** 2 + (self.prev_position[1] - y) ** 2 < 0.0001:
            reward -= 0.1

        if (self.prev_position[0] - x) ** 2 + (self.prev_position[1] - y) ** 2 > 0.8:
            reward -= 0.2

        self.prev_position = (x, y)
        if self.is_pass(x, y):
            self.next_closet_point += 1

        if self.next_closet_point == self.track_len:
            self.next_closet_point = self.track_len - 1

        _, _, x1, y1 = self.env.track[self.next_closet_point]
        if self.next_closet_point == 0:
            _, _, x2, y2 = self.env.track[self.next_closet_point + 1]
        else:
            _, _, x2, y2 = self.env.track[self.next_closet_point - 1]

        dist = (
            (x - x1) ** 2
            + (y - y1) ** 2
            - ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        )
        if dist < 0:
            dist = 1
        dist = np.sqrt(dist)

        if dist > 40 / 6:
            return -20

        return min(reward, 1)

    def step(self, action):
        action = self._process_action(action)
        state, reward, terminated, truncated, info = self.env.step(action)
        self._process_observation(state)
        processed_state = np.concatenate(self.history_states, axis=1)

        reward = self._process_reward(reward)
        if reward < -10:
            truncated = True

        return processed_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        self._process_observation(state)
        processed_state = np.concatenate(self.history_states, axis=1)

        x, y = self.env.unwrapped.car.hull.position
        self.prev_position = (x, y)
        waypoints = np.array([[point[2], point[3]] for point in self.env.track])
        self.track_len = len(waypoints)
        distance = np.sqrt((waypoints[:, 0] - x) ** 2 + (waypoints[:, 1] - y) ** 2)
        self.next_closet_point = np.argsort(distance)[0]

        return processed_state, info


def trainer():
    num_epoch = 100

    env = gym.make("CarRacing-v2", render_mode="human")
    env = RacingWrapper(env)
    state, info = env.reset()
    done = False
    total_reward = 0
    rewards = deque(maxlen=100)

    agent_configs = PPOConfig(
        {
            "debug": True,
            "state_space": env.observation_space,
            "action_dim": 2,
            "actor_net": AgentActor,
            "actor_kwargs": {
                "state_dim": 6400,
                "action_dim": 2,
                "hidden_size": 32,
                "continuous": True,
            },
            "critic_net": AgentCritic,
            "critic_kwargs": {
                "state_dim": 6400,
                "hidden_size": 32,
            },
            "vf_coef": 1,
            "gae_lambda": 0.97,
            "adv_norm": False,
        }
    )
    agent = PPO(agent_configs, device)

    wandb.config.update(agent_configs.__dict__)
    for i in range(num_epoch):
        for t in tqdm.tqdm(range(2048)):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action[0])
            env.render()

            transition = (
                (next_state, [reward], [terminated], [truncated], [info]),
                state,
                action,
                log_prob,
                value,
            )
            agent.push(transition)

            state = next_state
            done = terminated or truncated
            total_reward += reward

            train_result = agent.train()
            if not train_result is None:
                loss, loss_clip, loss_vf, loss_entropy = train_result
                wandb.log(
                    {
                        "loss": loss,
                        "loss_clip": loss_clip,
                        "loss_vf": loss_vf,
                        "loss_entropy": loss_entropy,
                    }
                )

            if done:
                rewards.append(total_reward)
                wandb.log({"mean_reward": np.mean(rewards), "reward": total_reward})
                state, info = env.reset()
                done = False
                total_reward = 0

            wandb.log(
                {
                    "reward": reward,
                    "value": value,
                    "log_prob_0": log_prob[0][0],
                    "log_prob_1": log_prob[0][1],
                }
            )

        print(f"epoch {i}, mean reward: {np.mean(rewards)}")


if __name__ == "__main__":
    trainer()
