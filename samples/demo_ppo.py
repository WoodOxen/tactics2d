import sys

sys.path.append(".")
sys.path.append("./rllib")

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from rllib.algorithms.ppo import *

OBS_SHAPE = {"state": (120 + 42 + 6,)}

agent_config = PPOConfig(
    {
        "state_space": None,
        "state_dim": 120 + 42 + 6,
        "action_space": None,
        "action_dim": 2,
        "gamma": 0.95,
        "lr_actor": 2e-6,
        "actor_kwargs": {
            "state_dim": 120 + 42 + 6,
            "action_dim": 2,
            "hidden_size": 256,
            "continuous": True,
        },
        "critic_kwargs": {
            "state_dim": 120 + 42 + 6,
            "hidden_size": 256,
        },
        "lr_critic": 2e-6 * 5,
        "horizon": 8192,
        "batch_size": 32,
        "adam_epsilon": 1e-8,
        "hidden_size": 256,
    }
)

VALID_STEER = [-0.75, 0.75]
MAX_SPEED = 2
PRECISION = 10
discrete_actions = []
for i in np.arange(
    VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1] / PRECISION), -VALID_STEER[-1] / PRECISION
):
    discrete_actions.append([i, MAX_SPEED])
for i in np.arange(
    VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1] / PRECISION), -VALID_STEER[-1] / PRECISION
):
    discrete_actions.append([i, -MAX_SPEED])


def choose_action(action_mean, action_std, action_mask):
    action_space = discrete_actions

    if isinstance(action_mean, torch.Tensor):
        action_mean = action_mean.cpu().numpy()
        action_std = action_std.cpu().numpy()
    if isinstance(action_mask, torch.Tensor):
        action_mask = action_mask.cpu().numpy()
    if len(action_mean.shape) == 2:
        action_mean = action_mean.squeeze(0)
        action_std = action_std.squeeze(0)
    if len(action_mask.shape) == 2:
        action_mask = action_mask.squeeze(0)

    def calculate_probability(mean, std, values):
        z_scores = (values - mean) / std
        log_probabilities = -0.5 * z_scores**2 - np.log((np.sqrt(2 * np.pi) * std))
        return np.sum(np.clip(log_probabilities, -10, 10), axis=1)

    possible_actions = np.array(action_space)
    # deal the scaling
    action_mean[1] = 1 if action_mean[1] > 0 else -1
    scale_steer = VALID_STEER[1]
    scale_speed = 1
    possible_actions = possible_actions / np.array([scale_steer, scale_speed])
    prob = calculate_probability(action_mean, action_std, possible_actions)
    exp_prob = np.exp(prob) * action_mask
    prob_softmax = exp_prob / np.sum(exp_prob)
    actions = np.arange(len(possible_actions))
    action_chosen = np.random.choice(actions, p=prob_softmax)

    return possible_actions[action_chosen]


class DemoPPO(PPO):
    def __init__(self, configs=None, device=None) -> None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        configs = agent_config
        super().__init__(configs, device)

    def choose_action(self, obs: np.ndarray, info: dict):
        observation = deepcopy(obs)
        observation = torch.FloatTensor(observation).to(self.device)

        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            mean = torch.clamp(policy_dist, -1, 1)
            log_std = self.actor_net.log_std.expand_as(mean)
            std = torch.exp(log_std)
            dist = Normal(mean, std)

        # action = dist.sample()
        action_mask = info["action_mask"]
        action = choose_action(mean, std, action_mask)
        action = torch.FloatTensor(action).to(self.device)
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)
        value = self.critic_net(observation)

        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()

        return action, log_prob, value

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray):
        """get the log probability for given action based on current policy

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            log_prob(np.ndarray): the log probability of taken action.
        """
        observation = deepcopy(obs)
        observation = torch.FloatTensor(observation).to(self.device)

        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            mean = torch.clamp(policy_dist, -1, 1)
            log_std = self.actor_net.log_std.expand_as(
                mean
            )  # To make 'log_std' have the same dimension as 'mean'
            std = torch.exp(log_std)
            dist = Normal(mean, std)

        action = torch.FloatTensor(action).to(self.device)
        log_prob = dist.log_prob(action)
        value = self.critic_net(observation)
        log_prob = log_prob.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()

        return log_prob, value
