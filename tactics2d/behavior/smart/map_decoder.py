# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Map-token decoder for the SMART network."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import SmartConfig
from .layers import (
    AttentionLayer,
    FourierEmbedding,
    MLPEmbedding,
    MLPLayer,
    angle_between_2d_vectors,
    weight_init,
    wrap_angle,
)
from .primitives import radius_graph
from .schema import LightType, PointType, PolygonType, SmartMapTokens


class SmartMapDecoder(nn.Module):
    """Encode tokenized map polylines into per-token features.

    Attributes:
        token_emb (MLPEmbedding): Embedding of a token's own 22-d geometry.
        pt2pt_layers (nn.ModuleList): Map self-attention stack.
    """

    def __init__(self, config: SmartConfig, map_token: Dict[str, torch.Tensor]):
        """Initialize the decoder.

        Args:
            config (SmartConfig): Model configuration; dimensions must match the checkpoint.
            map_token (Dict[str, torch.Tensor]): Map codebook; only ``traj_src`` is read.

        Raises:
            ValueError: If the configured input dimension is unsupported.
        """

        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = 2
        self.pl2pl_radius = config.pl2pl_radius
        self.num_layers = config.num_map_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_num_neighbors = config.max_pt2pt_neighbors

        if self.input_dim == 2:
            input_dim_r_pt2pt = 3
        else:
            raise ValueError("{} is not a valid dimension".format(self.input_dim))

        input_dim_token = map_token["traj_src"].shape[1] * 2

        self.type_pt_emb = nn.Embedding(len(PointType), config.hidden_dim)
        self.side_pt_emb = nn.Embedding(4, config.hidden_dim)
        self.polygon_type_emb = nn.Embedding(len(PolygonType), config.hidden_dim)
        self.light_pl_emb = nn.Embedding(len(LightType), config.hidden_dim)

        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pt,
            hidden_dim=config.hidden_dim,
            num_freq_bands=config.num_freq_bands,
        )
        self.pt2pt_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    dropout=config.dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(config.num_map_layers)
            ]
        )
        self.token_size = config.map_token_size
        self.token_predict_head = MLPLayer(
            input_dim=config.hidden_dim, hidden_dim=config.hidden_dim, output_dim=self.token_size
        )
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=config.hidden_dim)
        self.map_token = map_token
        self.apply(weight_init)
        self.mask_pt = False

    def forward(
        self, tokens: SmartMapTokens, predict_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode every map token.

        Args:
            tokens (SmartMapTokens): Tokenized map.
            predict_mask (Optional[torch.Tensor], optional): Mask of point tokens the map
                prediction head scores. Defaults to None.

        Returns:
            A dict holding ``x_pt`` of shape ``(N, hidden_dim)`` and, when *predict_mask* is
            given, the head's top-10 indices and raw logits.
        """

        device = next(self.parameters()).device
        pos_pt = tokens.pt_position[:, : self.input_dim].to(device).contiguous()
        orient_pt = tokens.pt_orientation.to(device).contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        token_sample_pt = self.map_token["traj_src"].to(device).to(torch.float)
        pt_token_emb_src = self.token_emb(token_sample_pt.view(token_sample_pt.shape[0], -1))
        x_pt = pt_token_emb_src[tokens.pt_token_idx.to(device)]

        token2pl = tokens.token2pl.to(device)
        token_light_type = tokens.light_type.to(device)[token2pl[1]]
        x_pt = x_pt + torch.stack(
            [
                self.type_pt_emb(tokens.pt_type.to(device).long()),
                self.polygon_type_emb(tokens.pl_type.to(device).long()),
                self.light_pl_emb(token_light_type.long()),
            ]
        ).sum(dim=0)

        edge_index_pt2pt = radius_graph(
            x=pos_pt[:, :2],
            r=self.pl2pl_radius,
            batch=None,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
        )
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(
            orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]]
        )
        r_pt2pt = torch.stack(
            [
                torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                    nbr_vector=rel_pos_pt2pt[:, :2],
                ),
                rel_orient_pt2pt,
            ],
            dim=-1,
        )
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        result: Dict[str, torch.Tensor] = {"x_pt": x_pt}
        if predict_mask is not None:
            next_token_prob = self.token_predict_head(x_pt[predict_mask])
            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
            _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)
            result["map_next_token_idx"] = next_token_idx
            result["map_next_token_prob"] = next_token_prob
        return result
