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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, features_per_head):
        super().__init__()
        self.scale = np.sqrt(features_per_head)

    def forward(self, query, key, value, mask=None):
        # query: [B, head, 1, F], key/value: [B, head, N, F]
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [B, head, 1, N]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, value)  # [B, head, 1, F]
        return out, attn


class ActorNet(nn.Module):
    def __init__(self, param_dict, control_steering=True):
        super().__init__()
        self.route_feature_num = param_dict["route_feature_num"]
        self.ego_feature_num = param_dict["ego_feature_num"]
        self.npc_feature_num = param_dict["npc_feature_num"]
        self.npc_num = param_dict["npc_num"]
        self.state_size = param_dict["state_size"]
        self.action_size = param_dict["action_size"] if control_steering else 1
        self.control_steering = control_steering

        self.encoder_size = [64, 64]
        self.decoder_size = [256, 256]
        self.features_per_head = 64
        self.feature_head = 1
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
        self.attention = ScaledDotProductAttention(self.features_per_head)

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
        ego_state = vehicle_state[:, : self.ego_feature_num]
        npcs_state = vehicle_state[:, self.ego_feature_num :].reshape(
            -1, self.npc_num, self.npc_feature_num
        )
        mask = state[:, -(self.npc_num + 1) :] < 0.5  # [B, npc_num+1]
        return ego_state, npcs_state, route_state, mask

    @staticmethod
    def ou_noise(action, mu=0.0, theta=0.15, sigma=0.3, dt=0.1):
        noise = theta * (mu - action) + sigma * np.sqrt(dt) * np.random.randn(*action.shape)
        return noise

    def forward(self, state, noise_std=0.0):
        ego_state, npcs_state, route_state, mask = self.split_input(state)
        B = state.shape[0]

        # Encode ego and npcs
        ego_enc = self.ego_encoder(ego_state).unsqueeze(1)  # [B, 1, F]
        npcs_enc = self.npc_encoder(npcs_state.view(-1, self.npc_feature_num)).view(
            B, self.npc_num, -1
        )  # [B, npc_num, F]
        concat_enc = torch.cat([ego_enc, npcs_enc], dim=1)  # [B, npc_num+1, F]

        # Attention projections
        query = self.query_ego(ego_enc)  # [B, 1, kqv]
        key = self.key_all(concat_enc)  # [B, npc_num+1, kqv]
        value = self.value_all(concat_enc)  # [B, npc_num+1, kqv]

        # Reshape for multi-head (here only 1 head)
        query = query.view(B, self.feature_head, 1, self.features_per_head)
        key = key.view(B, self.feature_head, self.npc_num + 1, self.features_per_head)
        value = value.view(B, self.feature_head, self.npc_num + 1, self.features_per_head)

        # Mask shape: [B, 1, 1, npc_num+1] -> [B, head, 1, npc_num+1]
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.feature_head, 1, -1)

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
            if noise_std > 0:
                action = action + torch.randn_like(action) * noise_std
            action = torch.clamp(action, -1, 1)
        else:
            target_speed = torch.sigmoid(self.target_speed_head(dec))
            if noise_std > 0:
                target_speed = target_speed + torch.randn_like(target_speed) * noise_std
            action = torch.clamp(target_speed, 0, 1)
        return action, att_matrix


class CriticNet(nn.Module):
    def __init__(self, param_dict, control_steering=True):
        super().__init__()
        self.route_feature_num = param_dict["route_feature_num"]
        self.ego_feature_num = param_dict["ego_feature_num"]
        self.npc_feature_num = param_dict["npc_feature_num"]
        self.npc_num = param_dict["npc_num"]
        self.state_size = param_dict["state_size"]
        self.action_size = param_dict["action_size"] if control_steering else 1
        self.control_steering = control_steering

        self.encoder_size = [64, 64]
        self.decoder_size = [256, 256]

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
