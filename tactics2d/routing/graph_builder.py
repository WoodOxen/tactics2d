# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build lane-level routing graphs from map data."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from tactics2d.map.element import Map

from .utils import get_lane_length


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

    def __init__(self, include_neighbors: bool = True, lane_change_penalty: float = 0.0):
        self.include_neighbors = include_neighbors
        self.lane_change_penalty = lane_change_penalty

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
                cost = get_lane_length(successor_lane)
                dst_idx = lane_id_to_index[successor_id]
                adjacency[src_idx].append((dst_idx, cost, "successor"))
                edge_relations[(src_idx, dst_idx)] = "successor"

            if not self.include_neighbors:
                continue

            for neighbor_id in lane.left_neighbors | lane.right_neighbors:
                if neighbor_id not in lane_id_to_index:
                    continue
                neighbor_lane = map_.lanes[neighbor_id]
                cost = get_lane_length(neighbor_lane) + self.lane_change_penalty
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
