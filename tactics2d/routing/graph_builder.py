# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build lane-level routing graphs from map data."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from tactics2d.map.element import Map

from .cost import RoutingCostFunction, build_cost_function


@dataclass
class RoutingGraph:
    """A lightweight directed routing graph."""

    lane_ids: List[str]
    lane_id_to_index: Dict[str, int]
    index_to_lane_id: Dict[int, str]
    adjacency: Dict[int, List[Tuple[int, float, str]]]
    edge_relations: Dict[Tuple[int, int], str]
    csr_graph: csr_matrix


class GraphBuilder:
    """Convert map lanes into a routing graph."""

    def __init__(
        self,
        include_neighbors: bool = True,
        lane_change_penalty: float = 0.0,
        cost_mode: str = "distance",
        cost_fn: Optional[RoutingCostFunction] = None,
        cost_kwargs: Optional[Dict[str, float]] = None,
    ):
        self.include_neighbors = include_neighbors
        self.lane_change_penalty = lane_change_penalty
        self.cost_mode = cost_mode
        self.cost_kwargs = dict(cost_kwargs or {})
        if cost_fn is not None:
            self.cost_fn = cost_fn
        else:
            resolved_cost_kwargs = dict(self.cost_kwargs)
            if cost_mode in {"distance", "time"}:
                resolved_cost_kwargs.setdefault(
                    "lane_change_penalty", lane_change_penalty
                )
            elif cost_mode in {"lanelet2_distance", "lanelet2_time"}:
                resolved_cost_kwargs.setdefault(
                    "lane_change_cost", lane_change_penalty
                )
            elif cost_mode in {"apollo_inspired", "apollo_like"}:
                resolved_cost_kwargs.setdefault("change_penalty", lane_change_penalty)
            self.cost_fn = build_cost_function(
                cost_mode=cost_mode,
                cost_fn=None,
                **resolved_cost_kwargs,
            )

    def build(self, map_: Map) -> RoutingGraph:
        lane_ids = sorted(map_.lanes.keys())
        lane_id_to_index = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
        index_to_lane_id = {idx: lane_id for lane_id, idx in lane_id_to_index.items()}
        adjacency: Dict[int, List[Tuple[int, float, str]]] = {
            lane_id_to_index[lane_id]: [] for lane_id in lane_ids
        }
        edge_relations: Dict[Tuple[int, int], str] = {}

        for lane_id, lane in map_.lanes.items():
            src_idx = lane_id_to_index[lane_id]

            for successor_id in lane.successors:
                if successor_id not in lane_id_to_index:
                    continue
                successor_lane = map_.lanes[successor_id]
                cost = self.cost_fn(map_, lane, successor_lane, "successor")
                dst_idx = lane_id_to_index[successor_id]
                adjacency[src_idx].append((dst_idx, cost, "successor"))
                edge_relations[(src_idx, dst_idx)] = "successor"

            if not self.include_neighbors:
                continue

            for neighbor_id in lane.left_neighbors:
                if neighbor_id not in lane_id_to_index:
                    continue
                if not self._is_lane_change_allowed(map_, lane, "left"):
                    continue
                neighbor_lane = map_.lanes[neighbor_id]
                cost = self.cost_fn(map_, lane, neighbor_lane, "neighbor")
                dst_idx = lane_id_to_index[neighbor_id]
                adjacency[src_idx].append((dst_idx, cost, "neighbor"))
                edge_relations[(src_idx, dst_idx)] = "neighbor"

            for neighbor_id in lane.right_neighbors:
                if neighbor_id not in lane_id_to_index:
                    continue
                if not self._is_lane_change_allowed(map_, lane, "right"):
                    continue
                neighbor_lane = map_.lanes[neighbor_id]
                cost = self.cost_fn(map_, lane, neighbor_lane, "neighbor")
                dst_idx = lane_id_to_index[neighbor_id]
                adjacency[src_idx].append((dst_idx, cost, "neighbor"))
                edge_relations[(src_idx, dst_idx)] = "neighbor"

        csr_graph = self._to_csr_matrix(len(lane_ids), adjacency)

        return RoutingGraph(
            lane_ids=lane_ids,
            lane_id_to_index=lane_id_to_index,
            index_to_lane_id=index_to_lane_id,
            adjacency=adjacency,
            edge_relations=edge_relations,
            csr_graph=csr_graph,
        )

    @staticmethod
    def _to_csr_matrix(
        n_nodes: int, adjacency: Dict[int, List[Tuple[int, float, str]]]
    ) -> csr_matrix:
        matrix = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

        for src_idx, edges in adjacency.items():
            for dst_idx, cost, _ in edges:
                matrix[src_idx, dst_idx] = cost

        return csr_matrix(matrix)

    @staticmethod
    def _iter_boundary_roadlines(
        map_: Map, lane, side: str
    ) -> Optional[Iterable[Tuple[str, Tuple[bool, bool]]]]:
        line_ids = getattr(lane, "line_ids", None)
        if not isinstance(line_ids, dict):
            return None

        boundary_ids = line_ids.get(side, [])
        if not boundary_ids:
            return None

        roadlines = []
        for line_id in boundary_ids:
            roadline = map_.roadlines.get(line_id)
            if roadline is None:
                return None
            roadlines.append((line_id, roadline.lane_change))

        return roadlines

    def _is_lane_change_allowed(self, map_: Map, lane, side: str) -> bool:
        """Return whether a lane's side can be used as a routing lane-change edge.

        The current policy is intentionally conservative: if a boundary is
        represented by multiple roadline segments, all of them must allow the
        relevant crossing direction before the neighbor edge is admitted into the
        routing graph.
        """

        roadlines = self._iter_boundary_roadlines(map_, lane, side)
        if roadlines is None:
            return True

        direction_index = 1 if side == "left" else 0
        return all(
            isinstance(lane_change, tuple)
            and len(lane_change) == 2
            and bool(lane_change[direction_index])
            for _, lane_change in roadlines
        )
