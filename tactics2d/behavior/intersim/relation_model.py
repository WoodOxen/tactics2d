# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tactics2D-native VectorNet relation predictor implementation."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab. Minimal relation-only VectorNet whose
# parameter names align with the downloaded relation checkpoint; the raster
# CNN and non-relation decoder branches are not needed by relation inference.

_HIDDEN = 128


class _LayerNorm(nn.Module):
    """Layer normalization with ``weight``/``bias`` parameters."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class _MLP(nn.Module):
    """Linear + layer norm + ReLU."""

    def __init__(self, in_features, out_features=None):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = _LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return F.relu(hidden_states)


class _SubGraph(nn.Module):
    """VectorNet sub-graph over one polyline (three 128->64 MLPs + pooling)."""

    def __init__(self, depth=3, hidden_size=_HIDDEN):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([_MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

    def forward(self, hidden_states: Tensor, vector_num: Optional[List[int]] = None):
        batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        if vector_num is None:
            vector_num = [max_vector_num] * batch_size
        hidden_size = hidden_states.shape[2]
        device = hidden_states.device

        attention_mask = torch.zeros([batch_size, max_vector_num, hidden_size // 2], device=device)
        zeros = torch.zeros([hidden_size // 2], device=device)
        for i in range(batch_size):
            attention_mask[i][vector_num[i]:max_vector_num].fill_(-10000.0)
        for layer in self.layers:
            new_hidden_states = torch.zeros(
                [batch_size, max_vector_num, hidden_size], device=device
            )
            encoded_hidden_states = layer(hidden_states)
            for j in range(max_vector_num):
                attention_mask[:, j] += -10000.0
                max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
                max_hidden = torch.max(max_hidden, zeros)
                attention_mask[:, j] += 10000.0
                new_hidden_states[:, j] = torch.cat(
                    (encoded_hidden_states[:, j], max_hidden), dim=-1
                )
            hidden_states = new_hidden_states
        return torch.max(hidden_states, dim=1)[0]


class _GlobalGraph(nn.Module):
    """Single-head self attention (global graph)."""

    def __init__(self, hidden_size, attention_head_size=None):
        super().__init__()
        if attention_head_size is None:
            attention_head_size = hidden_size
        self.attention_head_size = attention_head_size
        self.num_attention_heads = 1
        self.all_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        if attention_mask is not None:
            extended = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + extended
        attention_probs = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, value_layer)


class _GlobalGraphRes(nn.Module):
    """Concatenation of two global graphs (residual-style)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.global_graph = _GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = _GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None):
        return torch.cat(
            [
                self.global_graph(hidden_states, attention_mask),
                self.global_graph2(hidden_states, attention_mask),
            ],
            dim=-1,
        )


class _CrossAttention(nn.Module):
    """Query/key/value cross attention (laneGCN_A2L)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key_value):
        query_layer = self.query(query)
        key_layer = self.key(key_value)
        value_layer = self.value(key_value)
        scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, value_layer)


class _DecoderResCat(nn.Module):
    """Residual-cat MLP decoder used by the relation head."""

    def __init__(self, in_features, out_features=2):
        super().__init__()
        self.mlp = _MLP(in_features, _HIDDEN)
        self.fc = nn.Linear(_HIDDEN + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        return self.fc(hidden_states)


def _merge_tensors(tensors: List[Tensor], device) -> Tuple[Tensor, List[int]]:
    lengths = [tensor.shape[0] for tensor in tensors]
    res = torch.zeros([len(tensors), max(lengths), _HIDDEN], device=device)
    for i, tensor in enumerate(tensors):
        res[i][: tensor.shape[0]] = tensor
    return res, lengths


def _merge_sub_graph(tensor_list_list, module, device):
    """Run the sub-graph over every polyline of every sample."""

    output_tensor_list = []
    for tensor_list in tensor_list_list:
        inputs, lengths = _merge_tensors(tensor_list, device)
        outputs = module(inputs, lengths)
        output_tensor_list.append(outputs)
    return output_tensor_list


class RelationVectorNet(nn.Module):
    """Relation-only VectorNet matching the relation checkpoint structure.

    Sub-graph encodes each agent/road polyline; ``laneGCN_A2L`` refreshes the
    lane states with the reactor; a global graph produces per-polyline states;
    the relation head scores the reactor/influencer pair on the first two
    polyline states.
    """

    def __init__(self):
        super().__init__()
        self.sub_graph = _SubGraph(depth=3, hidden_size=_HIDDEN)
        self.global_graph = _GlobalGraphRes(_HIDDEN)
        self.laneGCN_A2L = _CrossAttention(_HIDDEN)
        self.decoder = nn.Module()
        self.decoder.inf_r_decoder = _DecoderResCat(in_features=_HIDDEN * 2, out_features=2)

    def forward(self, matrix: np.ndarray, polyline_spans: List[slice],
                map_start_polyline_idx: int, device) -> np.ndarray:
        """Score one reactor/influencer pair.

        Args:
            matrix: ``(R, 128)`` feature rows (agents then roads).
            polyline_spans: per-polyline row slices.
            map_start_polyline_idx: number of agent polylines (road spans
                follow the agent spans).
            device: torch device.

        Returns:
            Two-class softmax scores (influencer index 0, reactor index 1).
        """

        tensor_list = [
            torch.tensor(matrix[span], device=device, dtype=torch.float32)
            for span in polyline_spans
        ]
        element_states = _merge_sub_graph([tensor_list], self.sub_graph, device)[0]

        agent_states = element_states[:map_start_polyline_idx]
        lane_states = element_states[map_start_polyline_idx:]
        lanes = lane_states.unsqueeze(0)
        lane_query = torch.cat([lanes, agent_states[0:1].unsqueeze(0)], dim=1)
        lanes = lanes + self.laneGCN_A2L(lanes, lane_query)
        element_states = torch.cat([agent_states, lanes.squeeze(0)])

        inputs, lengths = _merge_tensors([element_states], device)
        poly_num = inputs.shape[1]
        attention_mask = torch.zeros([1, poly_num, poly_num], device=device)
        attention_mask[0][: lengths[0]][: lengths[0]].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask)  # [1, P, 128]
        pair_hidden = hidden_states[:, :2].reshape(1, -1)  # [1, 256]
        confidences = self.decoder.inf_r_decoder(pair_hidden)
        scores = torch.exp(F.log_softmax(confidences, dim=-1))
        return scores.detach().cpu().numpy()

    @classmethod
    def from_checkpoint(cls, bin_path: str, device: str = "cpu") -> "RelationVectorNet":
        """Load a relation checkpoint into the relation-only VectorNet."""

        model = cls()
        state_dict = torch.load(bin_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        missing = [key for key in model.state_dict() if key not in state_dict]
        if missing:
            raise ValueError(f"Relation checkpoint is missing parameters: {missing}")
        model.eval()
        return model
