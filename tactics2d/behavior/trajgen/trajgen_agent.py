##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: agent.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9


from copy import deepcopy

import torch
import torch.nn.functional as F

from .attention_td3 import ActorNet, CriticNet
from .config import TRAJGEN_CONFIG


class TrajGenAgent:
    def __init__(self, device="cpu"):
        for key, value in TRAJGEN_CONFIG.items():
            setattr(self, key, value)

        self.device = device

        self.actor_net = ActorNet().to(self.device)
        self.actor_target_net = ActorNet().to(self.device)

        self.critic_net1 = CriticNet().to(self.device)
        self.critic_target_net1 = deepcopy(CriticNet()).to(self.device)
        self.critic_net2 = CriticNet().to(self.device)
        self.critic_target_net2 = deepcopy(CriticNet()).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_optimizer1 = torch.optim.Adam(self.critic_net1.parameters(), self.lr_critic)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), self.lr_critic)

    def soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.tau + target.data * (1.0 - self.tau))

    def get_action(self, state, epsilon, is_eval=False):
        if is_eval:
            action = self.actor_net.get_action(state, noise=0)
        else:
            action = self.actor_net.get_action(state, noise=epsilon)

        return action

    def train(self, buffer):
        batches = buffer.sample(self.batch_size)
        state = torch.FloatTensor(batches["state"]).to(self.device)
        action = torch.FloatTensor(batches["action"]).to(self.device)
        next_state = torch.FloatTensor(batches["next_state"]).to(self.device)
        reward = torch.FloatTensor(batches["reward"]).to(self.device)
        done = torch.FloatTensor(batches["done"]).to(self.device)

        next_action = self.actor_target_net.get_action(next_state)

        # critic loss
        q1_target = self.critic_target_net1(next_state, next_action)
        q2_target = self.critic_target_net2(next_state, next_action)
        q_target = reward + (1 - done) * self.configs.gamma * torch.min(q1_target, q2_target)

        current_q1 = self.critic_net1(state, action)
        current_q2 = self.critic_net2(state, action)
        q1_loss = F.mse_loss(current_q1, q_target.detach())
        q2_loss = F.mse_loss(current_q2, q_target.detach())

        # update the critic networks
        self.critic_optimizer1.zero_grad()
        q1_loss.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.zero_grad()
        q2_loss.backward()
        self.critic_optimizer2.step()

        # delayed policy updates
        self.update_cnt += 1
        if self.update_cnt % self.configs.target_update_freq == 0:
            action_ = self.actor_net.get_action(state)
            q1_value = self.critic_net1(state, action_)
            actor_loss = -q1_value.mean()
            # update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # soft update target networks
            self.soft_update(self.actor_target_net, self.actor_net)
            self.soft_update(self.critic_target_net1, self.critic_net1)
            self.soft_update(self.critic_target_net2, self.critic_net2)

    def save(self, path: str):
        torch.save(
            {
                "actor_net": self.actor_net.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_net1": self.critic_net1.state_dict(),
                "critic_optimizer1": self.critic_optimizer1.state_dict(),
                "critic_net2": self.critic_net2.state_dict(),
                "critic_optimizer2": self.critic_optimizer2.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])

        self.actor_target_net = deepcopy(self.actor_net)
        self.critic_target_net1 = deepcopy(self.critic_net1)
        self.critic_target_net2 = deepcopy(self.critic_net2)
