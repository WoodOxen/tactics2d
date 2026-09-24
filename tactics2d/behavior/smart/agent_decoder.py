# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Agent-token decoder for the SMART network."""

from typing import Dict, Tuple

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
from .primitives import dense_to_sparse, radius, radius_graph, subgraph
from .tokens import SmartAgentTokens, SmartMapTokens


class SmartAgentDecoder(nn.Module):
    """Decode the joint future of every modelled agent.

    Output is written into a slot array of ``history_slots + future_slots`` slots, the
    first ``history_slots`` of which hold the tokenized history.

    Attributes:
        token_predict_head (MLPLayer): Per-slot distribution over the motion codebook.
        t_attn_layers (nn.ModuleList): Temporal attention stack.
        pt2a_attn_layers (nn.ModuleList): Map-to-agent attention stack.
        a2a_attn_layers (nn.ModuleList): Agent-to-agent attention stack.
    """

    def __init__(self, config: SmartConfig, token_data: Dict[str, object]):
        """Initialize the decoder.

        Args:
            config (SmartConfig): Model configuration; dimensions must match the checkpoint.
            token_data (Dict[str, object]): Codebook with keys ``token`` and
                ``token_all``, each mapping ``veh``/``ped``/``cyc`` to an array.

        Raises:
            ValueError: If the configured input dimension is unsupported.
        """

        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = 2
        self.num_historical_steps = config.history_steps
        self.time_span = config.time_span
        self.pl2a_radius = config.pl2a_radius
        self.a2a_radius = config.a2a_radius
        self.num_layers = config.num_agent_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_pl2a_neighbors = config.max_pl2a_neighbors
        self.max_a2a_neighbors = config.max_a2a_neighbors
        self.shift = config.shift
        self.beam_size = config.beam_size
        self.history_slots = config.history_token_slots
        self.future_slots = config.future_token_slots
        self.future_steps = config.future_steps

        if self.input_dim != 2:
            raise ValueError("{} is not a valid dimension".format(self.input_dim))

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(4, config.hidden_dim)
        self.shape_emb = MLPLayer(3, config.hidden_dim, config.hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=config.hidden_dim,
            num_freq_bands=config.num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=config.hidden_dim,
            num_freq_bands=config.num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=config.hidden_dim,
            num_freq_bands=config.num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=config.hidden_dim,
            num_freq_bands=config.num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=config.hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=config.hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=config.hidden_dim)
        self.fusion_emb = MLPEmbedding(
            input_dim=config.hidden_dim * 2, hidden_dim=config.hidden_dim
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    dropout=config.dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(config.num_agent_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    dropout=config.dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(config.num_agent_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    dropout=config.dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(config.num_agent_layers)
            ]
        )
        self.token_size = config.token_size
        self.token_predict_head = MLPLayer(
            input_dim=config.hidden_dim, hidden_dim=config.hidden_dim, output_dim=self.token_size
        )
        self.trajectory_token = token_data["token"]
        self.trajectory_token_all = token_data["token_all"]
        self.apply(weight_init)
        self.hist_mask = True

    def _agent_types(
        self, agent: SmartAgentTokens, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the per-category agent masks.

        Args:
            agent (SmartAgentTokens): Tokenized agents.
            device (torch.device): Device to place the masks on.

        Returns:
            A tuple of boolean masks of shape ``(A,)`` for vehicles, pedestrians and cyclists.
        """

        agent_type = torch.as_tensor(agent.agent_type, dtype=torch.long, device=device)
        return agent_type == 0, agent_type == 1, agent_type == 2

    def agent_token_embedding(
        self,
        agent: SmartAgentTokens,
        pos_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        agent_token_index: torch.Tensor,
        inference: bool = False,
    ):
        """Embed the per-slot motion token and pose of every agent.

        Args:
            agent (SmartAgentTokens): Tokenized agents.
            pos_a (torch.Tensor): Slot positions of shape ``(A, S, 2)``.
            head_vector_a (torch.Tensor): Slot heading unit vectors ``(A, S, 2)``.
            agent_token_index (torch.Tensor): Slot token indices ``(A, S)``, dtype ``int64``.
            inference (bool, optional): Whether to also build the ``(token, 6, 4, 2)``
                contour table the rollout samples from. Defaults to False.

        Returns:
            A tuple of:
                - feat_a (torch.Tensor): Fused features ``(A, S, hidden_dim)``.
                - agent_token_traj (torch.Tensor): Token contours, built when *inference*
                  is False.
                - agent_token_traj_all (torch.Tensor): Contour table
                  ``(A, token_size, 6, 4, 2)``, built when *inference* is True.
                - agent_token_emb (torch.Tensor): Token embeddings ``(A, S, hidden_dim)``.
                - categorical_embs (List[torch.Tensor]): Type and shape embedding addends.
        """

        num_agent, num_step, _ = pos_a.shape
        motion_vector_a = torch.cat(
            [pos_a.new_zeros(num_agent, 1, self.input_dim), pos_a[:, 1:] - pos_a[:, :-1]], dim=1
        )

        veh_mask, ped_mask, cyc_mask = self._agent_types(agent, pos_a.device)
        trajectory_token_veh = (
            torch.from_numpy(self.trajectory_token["veh"]).clone().to(pos_a.device).to(torch.float)
        )
        self.agent_token_emb_veh = self.token_emb_veh(
            trajectory_token_veh.view(trajectory_token_veh.shape[0], -1)
        )
        trajectory_token_ped = (
            torch.from_numpy(self.trajectory_token["ped"]).clone().to(pos_a.device).to(torch.float)
        )
        self.agent_token_emb_ped = self.token_emb_ped(
            trajectory_token_ped.view(trajectory_token_ped.shape[0], -1)
        )
        trajectory_token_cyc = (
            torch.from_numpy(self.trajectory_token["cyc"]).clone().to(pos_a.device).to(torch.float)
        )
        self.agent_token_emb_cyc = self.token_emb_cyc(
            trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1)
        )

        agent_token_traj_all = None
        if inference:
            agent_token_traj_all = torch.zeros(
                (num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device
            )
            tail = {}
            for key, token_all in self.trajectory_token_all.items():
                tail[key] = (
                    torch.from_numpy(token_all)
                    .clone()
                    .to(pos_a.device)
                    .to(torch.float)[:, : self.shift]
                )
            agent_token_traj_all[veh_mask] = torch.cat(
                [tail["veh"], trajectory_token_veh[:, None, ...]], dim=1
            )
            agent_token_traj_all[ped_mask] = torch.cat(
                [tail["ped"], trajectory_token_ped[:, None, ...]], dim=1
            )
            agent_token_traj_all[cyc_mask] = torch.cat(
                [tail["cyc"], trajectory_token_cyc[:, None, ...]], dim=1
            )

        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        agent_token_traj = None
        if not inference:
            agent_token_traj = torch.zeros(
                (num_agent, num_step, self.token_size, 4, 2), device=pos_a.device
            )
            agent_token_traj[veh_mask] = trajectory_token_veh
            agent_token_traj[ped_mask] = trajectory_token_ped
            agent_token_traj[cyc_mask] = trajectory_token_cyc

        agent_type = torch.as_tensor(agent.agent_type, dtype=torch.long, device=pos_a.device)
        # Tokens already carry the newest history frame, so no extent index is needed.
        agent_shape = agent.agent_shape.to(pos_a.device)
        categorical_embs = [
            self.type_a_emb(agent_type).repeat_interleave(repeats=num_step, dim=0),
            self.shape_emb(agent_shape).repeat_interleave(repeats=num_step, dim=0),
        ]

        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )
        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=categorical_embs,
        )
        x_a = x_a.view(-1, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)
        return feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs

    def build_temporal_edge(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        mask: torch.Tensor,
        inference_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Connect each slot to the earlier slots of the same agent.

        Args:
            pos_a (torch.Tensor): Slot positions of shape ``(A, S, 2)``.
            head_a (torch.Tensor): Slot headings of shape ``(A, S)``.
            head_vector_a (torch.Tensor): Slot heading unit vectors ``(A, S, 2)``.
            mask (torch.Tensor): Availability of the source slots, ``(A, S)``.
            inference_mask (torch.Tensor): Availability of the target slots, ``(A, S)``.

        Returns:
            A tuple of:
                - edge_index_t (torch.Tensor): ``(2, E)`` flattened slot indices, source
                  before target.
                - r_t (torch.Tensor): Edge features of shape ``(E, hidden_dim)``.
        """

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)

        mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        batch_s: torch.Tensor,
        mask_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Connect every slot to the neighbouring agents' slots.

        Args:
            pos_a (torch.Tensor): Slot positions of shape ``(A, S, 2)``.
            head_a (torch.Tensor): Slot headings of shape ``(A, S)``.
            head_vector_a (torch.Tensor): Slot heading unit vectors ``(A, S, 2)``.
            batch_s (torch.Tensor): Time-major segment assignment of the slots.
            mask_s (torch.Tensor): Per-slot availability, time-major.

        Returns:
            A tuple of:
                - edge_index_a2a (torch.Tensor): ``(2, E)`` slot indices.
                - r_a2a (torch.Tensor): Edge features of shape ``(E, hidden_dim)``.
        """

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=self.max_a2a_neighbors,
        )
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        tokens: SmartMapTokens,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        head_vector_a: torch.Tensor,
        mask: torch.Tensor,
        batch_s: torch.Tensor,
        batch_pl: torch.Tensor,
        num_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Connect every map token to the agent slots within its radius.

        Args:
            tokens (SmartMapTokens): Tokenized map.
            pos_a (torch.Tensor): Slot positions of shape ``(A, S, 2)``.
            head_a (torch.Tensor): Slot headings of shape ``(A, S)``.
            head_vector_a (torch.Tensor): Slot heading unit vectors ``(A, S, 2)``.
            mask (torch.Tensor): Availability of the target slots, ``(A, S)``.
            batch_s (torch.Tensor): Time-major segment assignment of the slots.
            batch_pl (torch.Tensor): Time-major segment assignment of the map tokens.
            num_step (int): Number of slots ``S``.

        Returns:
            A tuple of:
                - edge_index_pl2a (torch.Tensor): ``(2, E)`` index tensor, row 0 the map
                  token and row 1 the agent slot.
                - r_pl2a (torch.Tensor): Edge features of shape ``(E, hidden_dim)``.
        """

        mask_pl2a = mask.clone().transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        device = pos_a.device
        pos_pl = tokens.pt_position[:, : self.input_dim].to(device).contiguous()
        orient_pl = tokens.pt_orientation.to(device).contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)
        orient_pl = orient_pl.repeat(num_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=self.max_pl2a_neighbors,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def inference(
        self, agent: SmartAgentTokens, tokens: SmartMapTokens, map_enc: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Decode the joint future of every modelled agent, token by token.

        Each step attends over the history and the committed tokens, samples one token per
        agent out of the ``beam_size`` most likely ones and writes it into the slot arrays.

        Args:
            agent (SmartAgentTokens): Tokenized agent history, carrying at least
                ``history_slots`` slots.
            tokens (SmartMapTokens): Tokenized map.
            map_enc (Dict[str, torch.Tensor]): Output of :meth:`SmartMapDecoder.forward`,
                read for ``x_pt``.

        Returns:
            A dict holding ``pred_traj`` of shape ``(A, future_steps, 2)`` in the world
            frame, ``pred_head`` ``(A, future_steps)``, ``pred_prob`` ``(A, future_slots)``,
            the sampled ``next_token_idx`` ``(A, future_slots)`` and the slot arrays
            ``pos_a`` and ``head_a``.
        """

        total_slots = self.history_slots + self.future_slots
        device = map_enc["x_pt"].device

        pos_a = self._pad_slots(agent.token_pos.to(device), total_slots, 2)
        head_a = self._pad_slots(agent.token_heading.to(device), total_slots, 1)
        agent_token_index = self._pad_slots(agent.token_idx.to(device).long(), total_slots, 1)
        agent_valid_mask = self._pad_slots(agent.agent_valid_mask.to(device), total_slots, 1)

        num_agent = pos_a.shape[0]
        num_step = total_slots
        pos_a[:, self.history_slots :] = 0
        head_a[:, self.history_slots :] = 0

        eval_mask = torch.as_tensor(agent.eval_mask, dtype=torch.bool, device=device)
        agent_valid_mask[:, self.history_slots :] = True
        agent_valid_mask[~eval_mask] = False
        mask = agent_valid_mask.clone()

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        feat_a, _, agent_token_traj_all, agent_token_emb, categorical_embs = (
            self.agent_token_embedding(
                agent, pos_a, head_vector_a, agent_token_index, inference=True
            )
        )

        veh_mask, ped_mask, cyc_mask = self._agent_types(agent, device)

        pred_traj = torch.zeros(num_agent, self.future_steps, 2, device=device)
        pred_head = torch.zeros(num_agent, self.future_steps, device=device)
        pred_prob = torch.zeros(num_agent, self.future_slots, device=device)
        next_token_idx_list = []
        feat_a_t_dict: Dict[int, torch.Tensor] = {}
        batch_s = torch.arange(num_step, device=device).repeat_interleave(num_agent)
        batch_pl = torch.arange(num_step, device=device).repeat_interleave(
            tokens.pt_position.shape[0]
        )
        map_x_pt = (
            map_enc["x_pt"]
            .repeat_interleave(repeats=num_step, dim=0)
            .reshape(-1, num_step, self.hidden_dim)
            .transpose(0, 1)
            .reshape(-1, self.hidden_dim)
        )

        for t in range(self.future_slots):
            if t == 0:
                inference_mask = mask.clone()
                inference_mask[:, self.history_slots + t :] = False
            else:
                inference_mask = torch.zeros_like(mask)
                inference_mask[:, self.history_slots + t - 1] = True
            edge_index_t, r_t = self.build_temporal_edge(
                pos_a, head_a, head_vector_a, mask, inference_mask
            )
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                tokens, pos_a, head_a, head_vector_a, inference_mask, batch_s, batch_pl, num_step
            )
            mask_s = inference_mask.transpose(0, 1).reshape(-1)
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a, head_a, head_vector_a, batch_s, mask_s
            )

            for i in range(self.num_layers):
                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]
                feat_a = feat_a.reshape(-1, self.hidden_dim)
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
                feat_a = (
                    feat_a.reshape(-1, num_step, self.hidden_dim)
                    .transpose(0, 1)
                    .reshape(-1, self.hidden_dim)
                )
                feat_a = self.pt2a_attn_layers[i]((map_x_pt, feat_a), r_pl2a, edge_index_pl2a)
                feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

                if i + 1 not in feat_a_t_dict:
                    feat_a_t_dict[i + 1] = feat_a
                else:
                    slot = self.history_slots - 1 + t
                    feat_a_t_dict[i + 1][:, slot] = feat_a[:, slot]

            next_token_prob = self.token_predict_head(feat_a[:, self.history_slots - 1 + t])
            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
            topk_prob, next_token_idx = torch.topk(
                next_token_prob_softmax, k=self.beam_size, dim=-1
            )

            expanded_index = next_token_idx[..., None, None, None].expand(-1, -1, 6, 4, 2)
            next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_index)

            theta = head_a[:, self.history_slots - 1 + t]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros((num_agent, 2, 2), device=device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_diff_rel = torch.bmm(
                next_token_traj.view(-1, 4, 2),
                rot_mat[:, None, None, ...]
                .repeat(1, self.beam_size, self.shift + 1, 1, 1)
                .view(-1, 2, 2),
            ).view(num_agent, self.beam_size, self.shift + 1, 4, 2)
            agent_pred_rel = (
                agent_diff_rel + pos_a[:, self.history_slots - 1 + t, :][:, None, None, None, ...]
            )

            sample_index = torch.multinomial(topk_prob, 1).to(agent_pred_rel.device)
            agent_pred_rel = agent_pred_rel.gather(
                dim=1, index=sample_index[..., None, None, None].expand(-1, -1, 6, 4, 2)
            )[:, 0, ...]
            pred_prob[:, t] = topk_prob.gather(dim=-1, index=sample_index)[:, 0]
            pred_traj[:, t * self.shift : (t + 1) * self.shift] = (
                agent_pred_rel[:, 1:, ...].clone().mean(dim=2)
            )
            diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
            pred_head[:, t * self.shift : (t + 1) * self.shift] = torch.arctan2(
                diff_xy[:, :, 1], diff_xy[:, :, 0]
            )

            pos_a[:, self.history_slots + t] = agent_pred_rel[:, -1, ...].clone().mean(dim=1)
            diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
            head_a[:, self.history_slots + t] = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            next_token_idx = next_token_idx.gather(dim=1, index=sample_index).squeeze(-1)
            next_token_idx_list.append(next_token_idx[:, None])
            agent_token_emb[veh_mask, self.history_slots + t] = self.agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb[ped_mask, self.history_slots + t] = self.agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb[cyc_mask, self.history_slots + t] = self.agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            motion_vector_a = torch.cat(
                [pos_a.new_zeros(num_agent, 1, self.input_dim), pos_a[:, 1:] - pos_a[:, :-1]], dim=1
            )
            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            motion_vector_a[:, self.history_slots + 1 + t :] = 0
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                    ),
                ],
                dim=-1,
            )
            x_a = self.x_a_emb(
                continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs
            )
            x_a = x_a.view(-1, num_step, self.hidden_dim)
            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)

        return {
            "pred_traj": pred_traj,
            "pred_head": pred_head,
            "pred_prob": pred_prob,
            "next_token_idx": torch.cat(next_token_idx_list, dim=-1),
            "pos_a": pos_a,
            "head_a": head_a,
            "eval_mask": eval_mask,
        }

    @staticmethod
    def _pad_slots(tensor: torch.Tensor, total_slots: int, feature_dim: int) -> torch.Tensor:
        """Clone a slot tensor, extending it with empty future slots.

        Args:
            tensor (torch.Tensor): Slot array ``(A, S)`` when *feature_dim* is 1, else
                ``(A, S, d)``.
            total_slots (int): Slot count the rollout needs.
            feature_dim (int): Trailing dimension count of *tensor*.

        Returns:
            A contiguous copy with ``total_slots`` slots.

        Raises:
            ValueError: If *tensor* already has more slots than requested.
        """

        slots = tensor.shape[1]
        if slots > total_slots:
            raise ValueError(
                f"agent history carries {slots} slots, more than the {total_slots} the "
                "rollout allocates."
            )
        if slots == total_slots:
            return tensor.clone()
        shape = (tensor.shape[0], total_slots - slots) + tuple(tensor.shape[2:])
        return torch.cat([tensor, tensor.new_zeros(shape)], dim=1)
