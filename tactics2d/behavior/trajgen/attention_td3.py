##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: attention_td3.py
# @Description: The RL network of TrajGen. Original code from https://github.com/gaoyinfeng/TrajGen/blob/main/python/interaction_rl/networks/rl_networks.py.
# @Author: Tactics2D Team
# @Version: 0.1.9


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TRAJGEN_CONFIG


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()

        for key, value in TRAJGEN_CONFIG.items():
            setattr(self, key, value)

        self.mask_num = self.npc_num + 1
        self.state_size = (
            self.route_feature_num
            + self.ego_feature_num
            + self.npc_num * self.npc_feature_num
            + self.mask_num
        )

        self.kqv_size = self.features_per_head * self.feature_head

        # Encoders
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_feature_num, self.encoder_size[0]),
            nn.Tanh(),
            nn.Linear(self.encoder_size[0], self.encoder_size[1]),
            nn.Tanh(),
        )
        self.npc_encoder = nn.Sequential(
            nn.Linear(self.npc_feature_num, self.encoder_size[0]),
            nn.Tanh(),
            nn.Linear(self.encoder_size[0], self.encoder_size[1]),
            nn.Tanh(),
        )

        # Attention projections
        self.query_ego = nn.Linear(self.encoder_size[1], self.kqv_size, bias=False)
        self.key_all = nn.Linear(self.encoder_size[1], self.kqv_size, bias=False)
        self.value_all = nn.Linear(self.encoder_size[1], self.kqv_size, bias=False)

        # Route encoder
        if self.control_steering:
            self.route_encoder = nn.Sequential(
                nn.Linear(self.route_feature_num, self.encoder_size[0]),
                nn.Tanh(),
                nn.Linear(self.encoder_size[0], self.encoder_size[1]),
                nn.Tanh(),
            )
            route_out_dim = self.encoder_size[1]
        else:
            self.route_encoder = nn.Sequential(
                nn.Linear(self.route_feature_num, self.encoder_size[0] // 2),
                nn.Tanh(),
                nn.Linear(self.encoder_size[0] // 2, self.encoder_size[1] // 2),
                nn.Tanh(),
            )
            route_out_dim = self.encoder_size[1] // 2

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(
                self.features_per_head * self.feature_head + route_out_dim, self.decoder_size[0]
            ),
            nn.Tanh(),
            nn.Linear(self.decoder_size[0], self.decoder_size[1]),
            nn.Tanh(),
        )

        if self.control_steering:
            self.acc_head = nn.Linear(self.decoder_size[1], 1)
            self.steer_head = nn.Linear(self.decoder_size[1], 1)
        else:
            self.target_speed_head = nn.Linear(self.decoder_size[1], 1)

    def split_input(self, state):
        # state: [B, state_size]
        route_state = state[:, : self.route_feature_num]
        vehicle_state = state[
            :,
            self.route_feature_num : self.route_feature_num
            + self.ego_feature_num
            + self.npc_num * self.npc_feature_num,
        ]
        ego_state = vehicle_state[:, : self.ego_feature_num].reshape(-1, 1, self.ego_feature_num)
        npcs_state = vehicle_state[:, self.ego_feature_num :].reshape(
            -1, self.npc_num, self.npc_feature_num
        )
        mask = state[:, -self.mask_num :] < 0.5  # [B, npc_num+1]
        return ego_state, npcs_state, route_state, mask

    @staticmethod
    def ou_noise(action, mu=0.0, theta=0.15, sigma=0.3, dt=0.1):
        noise = theta * (mu - action) + sigma * np.sqrt(dt) * np.random.randn(*action.shape)
        return noise

    def attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.features_per_head)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        att_output = torch.matmul(p_attn, value)
        return att_output, p_attn

    def forward(self, state):
        ego_state, npcs_state, route_state, mask = self.split_input(state)
        B = state.shape[0]

        # Encode ego and npcs
        ego_enc = self.ego_encoder(ego_state).unsqueeze(1)  # [B, 1, F]
        npcs_enc = self.npc_encoder(npcs_state)  # [B, N, F]
        concat_enc = torch.cat([ego_enc, npcs_enc], dim=1)  # [B, N+1, F]

        # Attention projections
        query = self.query_ego(ego_enc)  # [B, 1, D]
        key = self.key_all(concat_enc)  # [B, N+1, D]
        value = self.value_all(concat_enc)  # [B, N+1, D]

        # Reshape for multi-head (here only 1 head)
        query = query.view(B, 1, self.feature_head, self.features_per_head).transpose(
            1, 2
        )  # [B, head, 1, F]
        key = key.view(B, self.mask_num, self.feature_head, self.features_per_head).transpose(
            1, 2
        )  # [B, head, N+1, F]
        value = value.view(B, self.mask_num, self.feature_head, self.features_per_head).transpose(
            1, 2
        )  # [B, head, N+1, F]

        # Mask shape: [B, 1, 1, npc_num+1] -> [B, head, 1, npc_num+1]
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.feature_head, 1, self.mask_num)

        # Attention
        att_result, att_matrix = self.attention(query, key, value, mask)
        att_result = att_result.view(B, -1)  # [B, features_per_head * feature_head]

        # Encode route
        route_enc = self.route_encoder(route_state)

        # Concatenate and decode
        concat_result = torch.cat([att_result, route_enc], dim=1)
        dec = self.decoder(concat_result)

        if self.control_steering:
            acc = torch.tanh(self.acc_head(dec))
            steer = torch.tanh(self.steer_head(dec))
            action = torch.cat([acc, steer], dim=1)
        else:
            target_speed = torch.sigmoid(self.target_speed_head(dec))
            action = target_speed
        return action, att_matrix

    def get_action(self, state, noise=1.0):
        self.eval()
        with torch.no_grad():
            action, _ = self.forward(torch.FloatTensor(state))
            action = action.cpu().numpy().squeeze()
        if self.control_steering:
            acc_noised = action[0] + self.ou_noise(action[0], mu=0, theta=0.3, sigma=0.45) * noise
            steer_noised = action[1] + self.ou_noise(action[0], mu=0, theta=0.3, sigma=0.45) * noise
            action_noise = np.squeeze(
                np.array([np.clip(acc_noised, -1, 1), np.clip(steer_noised, -1, 1)])
            )
        else:
            target_speed_noised = (
                action + self.ou_noise(action, mu=0.5, theta=0.3, sigma=0.45) * noise
            )
            action_noise = np.clip(target_speed_noised, 0, 1)

        return action_noise


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()

        for key, value in TRAJGEN_CONFIG.items():
            setattr(self, key, value)

        self.mask_num = self.npc_num + 1
        self.state_size = (
            self.route_feature_num
            + self.ego_feature_num
            + self.npc_num * self.npc_feature_num
            + self.mask_num
        )

        # Encoders
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(
                self.ego_feature_num + self.npc_num * self.npc_feature_num, self.encoder_size[0]
            ),
            nn.Tanh(),
            nn.Linear(self.encoder_size[0], self.encoder_size[1]),
            nn.Tanh(),
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(self.route_feature_num, self.encoder_size[0]),
            nn.Tanh(),
            nn.Linear(self.encoder_size[0], self.encoder_size[1]),
            nn.Tanh(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder_size[1] * 2 + self.action_size, self.decoder_size[0]),
            nn.Tanh(),
            nn.Linear(self.decoder_size[0], self.decoder_size[1]),
            nn.Tanh(),
            nn.Linear(self.decoder_size[1], 1),
        )

    def split_input(self, state):
        route_state = state[:, : self.route_feature_num]
        vehicle_state = state[
            :,
            self.route_feature_num : self.route_feature_num
            + self.ego_feature_num
            + self.npc_num * self.npc_feature_num,
        ]
        return vehicle_state, route_state

    def forward(self, state, action):
        vehicle_state, route_state = self.split_input(state)
        vehicle_enc = self.vehicle_encoder(vehicle_state)
        route_enc = self.route_encoder(route_state)
        concat = torch.cat([vehicle_enc, route_enc, action], dim=1)
        q_value = self.decoder(concat)
        return q_value.squeeze(-1)
