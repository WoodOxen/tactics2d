# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Layer primitives shared by the SMART decoders."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .primitives import segment_softmax


def angle_between_2d_vectors(ctr_vector: torch.Tensor, nbr_vector: torch.Tensor) -> torch.Tensor:
    """Return the signed angle from one 2-D vector to another.

    Args:
        ctr_vector (torch.Tensor): Reference vectors of shape ``(..., 2)``.
        nbr_vector (torch.Tensor): Compared vectors of shape ``(..., 2)``.

    Returns:
        Angles in radians in ``[-pi, pi]``, shape ``(...)``.
    """

    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    """Wrap an angle into a half-open interval.

    Args:
        angle (torch.Tensor): Angles in radians.
        min_val (float, optional): Lower bound. Defaults to ``-pi``.
        max_val (float, optional): Upper bound. Defaults to ``pi``.

    Returns:
        The wrapped angles.
    """

    return min_val + (angle + max_val) % (max_val - min_val)


def weight_init(module: nn.Module) -> None:
    """Initialize a module's parameters in place.

    Args:
        module (nn.Module): Module to initialize.
    """

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class FourierEmbedding(nn.Module):
    """Embed continuous features with a learnable Fourier basis.

    Each column gets its own frequency basis, is expanded into ``[cos, sin, raw]``, pushed
    through a private MLP and summed; categorical embeddings are added afterwards.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int):
        """Initialize the embedding.

        Args:
            input_dim (int): Number of continuous columns.
            hidden_dim (int): Output width.
            num_freq_bands (int): Number of learnable frequency bands.
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Embed continuous inputs and optional categorical additions.

        Args:
            continuous_inputs (Optional[torch.Tensor], optional): Values of
                shape ``(N, input_dim)``. Defaults to None.
            categorical_embs (Optional[List[torch.Tensor]], optional):
                Addends of shape ``(N, hidden_dim)``. Defaults to None.

        Returns:
            Embeddings of shape ``(N, hidden_dim)``.

        Raises:
            ValueError: If neither input kind is supplied.
        """

        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)


class MLPEmbedding(nn.Module):
    """Embed a flat vector through a two-hidden-layer MLP."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize the embedding.

        Args:
            input_dim (int): Input width.
            hidden_dim (int): Output width.
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Embed the inputs and add optional categorical terms.

        Args:
            continuous_inputs (Optional[torch.Tensor], optional): Vectors of
                shape ``(N, input_dim)``. Defaults to None.
            categorical_embs (Optional[List[torch.Tensor]], optional):
                Addends of shape ``(N, hidden_dim)``. Defaults to None.

        Returns:
            Embeddings of shape ``(N, hidden_dim)``.

        Raises:
            ValueError: If neither input kind is supplied.
        """

        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = self.mlp(continuous_inputs)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return x


class MLPLayer(nn.Module):
    """A linear, layer-normed, activated and projected block."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the block.

        Args:
            input_dim (int): Input width.
            hidden_dim (int): Hidden width.
            output_dim (int): Output width.
        """

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x (torch.Tensor): Inputs of shape ``(N, input_dim)``.

        Returns:
            Outputs of shape ``(N, output_dim)``.
        """

        return self.mlp(x)


class AttentionLayer(nn.Module):
    """A gated attention block over edge lists.

    The query is gathered at ``edge_index[1]``, the keys, values and positional features at
    ``edge_index[0]``; ``attn_prenorm_x_dst`` aliases ``attn_prenorm_x_src`` when not bipartite.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
    ):
        """Initialize the block.

        Args:
            hidden_dim (int): Node feature width.
            num_heads (int): Number of attention heads.
            head_dim (int): Width of one head.
            dropout (float): Dropout probability on the attention weights.
            bipartite (bool): Whether source and target nodes are distinct.
            has_pos_emb (bool): Whether edges carry positional features.
        """

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run one attention and feed-forward block.

        Args:
            x (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Node features of
                shape ``(N, hidden_dim)``, or a ``(source, target)`` pair when bipartite.
            r (Optional[torch.Tensor]): Edge features of shape ``(E, hidden_dim)``. None
                disables positional attention.
            edge_index (torch.Tensor): ``(2, E)`` index tensor, row 0 the source and row 1
                the target of each edge.

        Returns:
            Updated target features of shape ``(N_dst, hidden_dim)``.
        """

        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def _attn_block(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Project, propagate and project back.

        Args:
            x_src (torch.Tensor): Source features ``(N_src, hidden_dim)``.
            x_dst (torch.Tensor): Target features ``(N_dst, hidden_dim)``.
            r (Optional[torch.Tensor]): Edge features ``(E, hidden_dim)``.
            edge_index (torch.Tensor): ``(2, E)`` source-to-target indices.

        Returns:
            Attention output of shape ``(N_dst, hidden_dim)``.
        """

        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self._propagate(edge_index, x_dst, q, k, v, r)
        return self.to_out(agg)

    def _propagate(
        self,
        edge_index: torch.Tensor,
        x_dst: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gather, message, aggregate and update.

        Args:
            edge_index (torch.Tensor): ``(2, E)`` source-to-target indices.
            x_dst (torch.Tensor): Target features ``(N_dst, hidden_dim)``.
            q (torch.Tensor): Queries ``(N_dst, num_heads, head_dim)``.
            k (torch.Tensor): Keys ``(N_src, num_heads, head_dim)``.
            v (torch.Tensor): Values ``(N_src, num_heads, head_dim)``.
            r (Optional[torch.Tensor]): Edge features ``(E, hidden_dim)``.

        Returns:
            Updated target features of shape ``(N_dst, hidden_dim)``.
        """

        source, target = edge_index[0], edge_index[1]
        num_nodes = q.shape[0]
        messages = self._message(
            q_i=q.index_select(0, target),
            k_j=k.index_select(0, source),
            v_j=v.index_select(0, source),
            r=r,
            index=target,
            num_nodes=num_nodes,
        )
        aggregated = torch.zeros(
            (num_nodes, self.num_heads, self.head_dim), dtype=messages.dtype, device=messages.device
        )
        aggregated = aggregated.index_add(0, target, messages)
        return self._update(aggregated, x_dst)

    def _message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r: Optional[torch.Tensor],
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Score, normalize and weight the messages of every edge.

        Args:
            q_i (torch.Tensor): Per-edge queries ``(E, num_heads, head_dim)``.
            k_j (torch.Tensor): Per-edge keys of the same shape.
            v_j (torch.Tensor): Per-edge values of the same shape.
            r (Optional[torch.Tensor]): Edge features of shape ``(E, hidden_dim)``.
            index (torch.Tensor): Per-edge target node, shape ``(E,)``.
            num_nodes (int): Number of target nodes.

        Returns:
            Weighted values of shape ``(E, num_heads, head_dim)``.
        """

        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = segment_softmax(sim, index, num_nodes)
        self.attention_weight = attn.sum(-1).detach()
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def _update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        """Gate the aggregated messages against the target features.

        Args:
            inputs (torch.Tensor): Aggregated messages ``(N_dst, num_heads, head_dim)``.
            x_dst (torch.Tensor): Target features ``(N_dst, hidden_dim)``.

        Returns:
            Updated features of shape ``(N_dst, head_dim * num_heads)``.
        """

        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward half of the block.

        Args:
            x (torch.Tensor): Features of shape ``(N, hidden_dim)``.

        Returns:
            Transformed features of the same shape.
        """

        return self.ff_mlp(x)
