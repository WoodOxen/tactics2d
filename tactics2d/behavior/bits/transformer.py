# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transformer encoder with multi-head attention."""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with TBSIM-compatible parameter names."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self.a_2 * (inputs - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection followed by TBSIM-style layer normalization."""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, sublayer) -> torch.Tensor:
        return inputs + self.dropout(sublayer(self.norm(inputs)))


class PositionwiseFeedForward(nn.Module):
    """Feed-forward block with official TBSIM parameter names w_1/w_2."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))


class _MultiHeadAttention(nn.Module):
    """TBSIM SimpleTransformer attention block."""

    def __init__(self, head_count: int, d_model: int, dropout: float = 0.1, pooling_dim=None):
        super().__init__()
        if d_model % head_count != 0:
            raise ValueError("d_model must be divisible by head_count.")
        self.d_k = d_model // head_count
        self.h = head_count
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.pooling_dim = pooling_dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooling_dim = -2 if self.pooling_dim is None else self.pooling_dim
        if mask is not None:
            if mask.ndim == query.ndim - 1:
                mask = mask.view([*mask.shape, 1, 1]).transpose(-1, pooling_dim - 1)
            elif mask.ndim == query.ndim:
                mask = mask.unsqueeze(-2).transpose(-2, pooling_dim - 1)
            else:
                raise ValueError("mask dimension mismatch")
        query, key, value = (
            layer(inputs).view(*inputs.shape[:-1], self.h, self.d_k)
            for layer, inputs in zip(self.linears, (query, key, value))
        )
        attended, self.attn = scaled_dot_product_attention(
            query.transpose(-2, pooling_dim - 1),
            key.transpose(-2, pooling_dim - 1),
            value.transpose(-2, pooling_dim - 1),
            mask,
            dropout=self.dropout,
        )
        attended = attended.transpose(-2, pooling_dim - 1).contiguous()
        attended = attended.view(*attended.shape[:-2], self.h * self.d_k)
        return self.linears[-1](attended)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention = dropout(attention)
    return torch.matmul(attention, value), attention


class _EncoderLayer(nn.Module):
    """Single TBSIM StaticEncoder layer."""

    def __init__(
        self,
        size: int,
        self_attn: _MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        inputs = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](inputs, self.feed_forward)


class _BitsPositionalEncodingNd(nn.Module):
    """Official XY positional encoding for SimpleTransformer."""

    def __init__(self, dim: int, dropout: float, step_size=(0.1, 0.1)):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4.")
        self.dropout = nn.Dropout(p=dropout)
        self.dim = int(dim)
        self.step_size = tuple(float(value) for value in step_size)
        axis_dim = dim // 2
        self.div_term = torch.exp(torch.arange(0, axis_dim, 2) * -(math.log(10000.0) / axis_dim))

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        encoded = torch.zeros(
            *inputs.shape[:-1], self.dim, dtype=inputs.dtype, device=inputs.device
        )
        div_term = self.div_term.to(device=inputs.device, dtype=inputs.dtype)
        axis_dim = self.dim // 2
        for axis, step in enumerate(self.step_size):
            phase = position[..., axis : axis + 1] / step * div_term
            start = axis * axis_dim
            axis_encoded = encoded[..., start : start + axis_dim]
            axis_encoded[..., 0::2] = torch.sin(phase)
            axis_encoded[..., 1::2] = torch.cos(phase)
        return self.dropout(encoded)


class _BitsStaticEncoder(nn.Module):
    """Agent-axis transformer encoder with official state_dict names."""

    def __init__(
        self, agent_enc: _EncoderLayer, xy_pe: _BitsPositionalEncodingNd, layer_count: int = 1
    ):
        super().__init__()
        self.N_layer = int(layer_count)
        self.agent_encs = nn.ModuleList([copy.deepcopy(agent_enc) for _ in range(self.N_layer)])
        self.XY_pe = xy_pe

    def forward(
        self,
        inputs: torch.Tensor,
        source_mask: torch.Tensor,
        source_position: torch.Tensor,
        map_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = [inputs, self.XY_pe(inputs, source_position)]
        if map_embedding is not None:
            pieces.append(map_embedding)
        encoded = torch.cat(pieces, dim=-1) * source_mask.unsqueeze(-1)
        for layer in self.agent_encs:
            encoded = layer(encoded, source_mask)
        return encoded


class Transformer(nn.Module):
    """Official SimpleTransformer used by the released BITS predictor checkpoint."""

    def __init__(
        self,
        src_dim: int,
        N_a: int = 3,
        d_model: int = 384,
        XY_pe_dim: int = 64,
        d_ff: int = 2048,
        head: int = 8,
        dropout: float = 0.1,
        step_size=(0.1, 0.1),
    ):
        super().__init__()
        agent_attn = _MultiHeadAttention(head, d_model, pooling_dim=-3)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        xy_pe = _BitsPositionalEncodingNd(XY_pe_dim, dropout, step_size=step_size)
        self.agent_enc = _BitsStaticEncoder(
            _EncoderLayer(d_model, copy.deepcopy(agent_attn), copy.deepcopy(feed_forward), dropout),
            xy_pe,
            N_a,
        )
        self.pre_emb = nn.Linear(src_dim, d_model - XY_pe_dim)
        self.post_emb = nn.Linear(d_model, src_dim)

    def forward(
        self, features: torch.Tensor, availability: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.pre_emb(features)
        encoded = self.agent_enc(embedded, availability, positions)
        return self.post_emb(encoded)
