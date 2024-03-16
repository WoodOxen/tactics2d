import os

os.path.append(".")
os.path.append("..")
os.path.append("./rllib")

import gymnasium as gym
import torch
import torch.nn as nn
from rllib.algorihms import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgentActor(nn.Module):
    def __init__(self):
        super().__init__()

        pass


def trainer():
    total_epochs = 300
    cnt_epoch = 0

    env = gym.make("CarRacing-v2")

    agent_configs = PPOConfig(
        {
            "vf_coef": 1,
            "gae_lambda": 0.97,
            "adv_norm": False,
        }
    )
    agent = PPO(agent_configs, device)

    for epoch in range(total_epochs):
        observation, infos = env.reset()
