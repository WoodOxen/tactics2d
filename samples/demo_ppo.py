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
from rllib.buffer import RandomReplayBuffer

OBS_SHAPE = {"state": (120 + 42 + 6,)}

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
        action_mean = action_mean.cpu().detach().numpy()
        action_std = action_std.cpu().detach().numpy()
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


class ParkingActor(PPOActor):
    def get_dist(self, state: torch.Tensor):
        policy_dist = self.forward(state)
        mean = torch.clamp(policy_dist, -1, 1)
        std = self.log_std.expand_as(mean).exp()
        dist = Normal(mean, std)

        return dist

    def action(self, state: torch.Tensor, action_mask):
        policy_dist = self.forward(state)
        mean = torch.clamp(policy_dist, -1, 1)
        std = self.log_std.expand_as(mean).exp()
        dist = Normal(mean, std)

        action = choose_action(mean, std, action_mask)
        action = np.clip(action, -1, 1)
        log_prob = dist.log_prob(torch.Tensor(action).to(torch.device("cuda")))
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob


agent_config = PPOConfig(
    {
        "state_space": None,
        "state_dim": 120 + 42 + 6,
        "action_space": None,
        "action_dim": 2,
        "gamma": 0.95,
        "lr": 2e-6,
        "actor_net": ParkingActor,
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
        "horizon": 8192,
        "batch_size": 32,
        "adam_epsilon": 1e-8,
        "hidden_size": 256,
    }
)


class DemoPPO(PPO):
    def __init__(self, configs=None, device=None) -> None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        configs = agent_config
        super().__init__(configs, device)

        # the networks
        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(),
            self.configs.lr,
            eps=self.configs.adam_epsilon,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), self.configs.lr, eps=self.configs.adam_epsilon
        )
        self.critic_target = deepcopy(self.critic_net)

        # As a on-policy RL algorithm, PPO does not have memory, the self.buffer represents
        # the buffer
        self.buffer = RandomReplayBuffer(
            self.configs.horizon + 1, extra_items=["log_prob", "value"]
        )

    def get_action(self, obs: np.ndarray, info: dict):
        observation = deepcopy(obs)
        observation = torch.FloatTensor(observation).to(self.device)

        action, log_prob = self.actor_net.action(observation, info["action_mask"])
        value = self.critic_net(observation)
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

        action = torch.FloatTensor(action).to(self.device)
        log_prob, _ = self.actor_net.evaluate(observation, action)
        log_prob = log_prob.detach().cpu().numpy().flatten()
        value = self.critic_net(observation)
        value = value.detach().cpu().numpy().flatten()

        return log_prob, value

    def push(self, observations):
        """
        Args:
            observations(tuple): (obs, action, reward, done, log_prob, next_obs)
        """
        self.buffer.push(observations)

    def train(self):
        if len(self.buffer) < self.configs.horizon + 1:
            return

        batches = self.buffer.all()

        adv, v_target = self._compute_gae(batches["reward"], batches["value"], batches["done"])
        if self.configs.adv_norm:  # advantage normalization
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        state_batch = torch.FloatTensor(batches["state"]).to(self.device)
        action_batch = torch.FloatTensor(batches["action"]).to(self.device)
        old_log_prob_batch = torch.FloatTensor(batches["log_prob"]).to(self.device)
        adv = torch.FloatTensor(adv).to(self.device)
        v_target = torch.FloatTensor(v_target).to(self.device)

        # apply multi update epoch
        for _ in range(self.configs.num_epochs):
            # use mini batch and shuffle data
            mini_batch = self.configs.batch_size
            batchsize = self.configs.horizon
            train_times = (
                batchsize // mini_batch
                if batchsize % mini_batch == 0
                else batchsize // mini_batch + 1
            )
            random_idx = np.arange(batchsize)
            np.random.shuffle(random_idx)
            for i in range(train_times):
                if i == batchsize // mini_batch:
                    ri = random_idx[i * mini_batch :]
                else:
                    ri = random_idx[i * mini_batch : (i + 1) * mini_batch]
                state = state_batch[ri]
                dist = self.actor_net.get_dist(state)

                log_prob = dist.log_prob(action_batch[ri])
                log_prob = torch.sum(log_prob, dim=1, keepdim=True)
                old_log_prob = torch.sum(old_log_prob_batch[ri], dim=1, keepdim=True)
                prob_ratio = (log_prob - old_log_prob).exp()

                loss1 = prob_ratio * adv[ri]
                loss2 = (
                    torch.clamp(
                        prob_ratio, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon
                    )
                    * adv[ri]
                )

                actor_loss = -torch.min(loss1, loss2)
                critic_loss = F.mse_loss(v_target[ri], self.critic_net(state))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.mean().backward()
                critic_loss.mean().backward()

                if self.configs.gradient_clip:  # gradient clip
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.buffer.clear()
