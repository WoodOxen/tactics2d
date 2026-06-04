# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main routing entry point."""

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from tactics2d.map.element import Map

from .algorithm_adapter import AlgorithmAdapter
from .cost_builder import RoutingCostFunction
from .graph_builder import GraphBuilder
from .route import Route, RouteSegment
from .utils import concatenate_centerlines, find_nearest_lane, get_lane_centerline


class Router:
    """A lane-level routing interface.

    Args:
        algorithm: Search backend name. Supported values are ``"dijkstra"``
            and ``"a_star"``.
        include_neighbors: Whether lateral neighbor lanes should be added to
            the routing graph as lane-change candidates.
        lane_change_penalty: Unified lane-change penalty alias for built-in
            presets. It maps to ``lane_change_penalty`` for ``distance`` and
            ``time``, to ``lane_change_cost`` for ``lanelet2_distance`` and
            ``lanelet2_time``, and to ``change_penalty`` for
            ``apollo_inspired`` unless explicitly overridden in
            ``cost_kwargs``.
        cost_mode: Built-in routing cost preset. Supported modes include
            ``distance``, ``time``, ``lanelet2_distance``,
            ``lanelet2_time``, and ``apollo_inspired``. Their implementation
            mechanism is provided by concrete ``CostBuilder`` subclasses.
        cost_fn: Optional custom routing cost callback. When provided, it
            overrides ``cost_mode`` and receives
            ``(map_, from_lane, to_lane, relation)``.
        cost_kwargs: Keyword arguments forwarded to the selected built-in cost
            preset.
        heuristic_builder: Optional A* heuristic builder. When omitted, a
            center-point Euclidean heuristic is used.
    """

    def __init__(
        self,
        algorithm: str = "dijkstra",
        include_neighbors: bool = True,
        lane_change_penalty: float = 0.0,
        cost_mode: str = "distance",
        cost_fn: Optional[RoutingCostFunction] = None,
        cost_kwargs: Optional[Dict[str, float]] = None,
        heuristic_builder: Optional[Callable[[Map, object], Callable[[int, int], float]]] = None,
    ):
        self.algorithm = algorithm
        self.include_neighbors = include_neighbors
        self.lane_change_penalty = lane_change_penalty
        self.cost_mode = cost_mode
        self.cost_fn = cost_fn
        self.cost_kwargs = cost_kwargs or {}
        self.heuristic_builder = heuristic_builder

    def plan(self, map_: Map, start: Sequence[float], goal: Sequence[float]) -> Route:
        graph_builder = GraphBuilder(
            include_neighbors=self.include_neighbors,
            lane_change_penalty=self.lane_change_penalty,
            cost_mode=self.cost_mode,
            cost_fn=self.cost_fn,
            cost_kwargs=self.cost_kwargs,
        )
        routing_graph = graph_builder.build(map_)

        start_lane_id = find_nearest_lane(map_, start)
        goal_lane_id = find_nearest_lane(map_, goal)

        route = Route(tuple(start[:2]), tuple(goal[:2]), start_lane_id, goal_lane_id)
        if start_lane_id is None or goal_lane_id is None:
            return route

        start_idx = routing_graph.lane_id_to_index[start_lane_id]
        goal_idx = routing_graph.lane_id_to_index[goal_lane_id]

        if self.algorithm == "a_star":
            heuristic_fn = self._build_heuristic(map_, routing_graph)
            path_indices, total_cost, edge_relations = AlgorithmAdapter.a_star(
                routing_graph, start_idx, goal_idx, heuristic_fn
            )
        else:
            path_indices, total_cost, edge_relations = AlgorithmAdapter.dijkstra(
                routing_graph, start_idx, goal_idx
            )

        if not path_indices:
            return route

        route.lane_ids = [routing_graph.index_to_lane_id[idx] for idx in path_indices]
        route.total_cost = total_cost

        centerlines = []
        for path_pos, lane_id in enumerate(route.lane_ids):
            centerline = get_lane_centerline(map_.lanes[lane_id])
            centerlines.append(centerline)

            relation_type = "start"
            segment_cost = 0.0
            if path_pos > 0:
                prev_idx = path_indices[path_pos - 1]
                curr_idx = path_indices[path_pos]
                relation_type = edge_relations.get((prev_idx, curr_idx), "successor")
                for neighbor_idx, weight, relation in routing_graph.adjacency[prev_idx]:
                    if neighbor_idx == curr_idx and relation == relation_type:
                        segment_cost = weight
                        break

            route.segments.append(
                RouteSegment(
                    lane_id=lane_id,
                    relation_type=relation_type,
                    cost=segment_cost,
                    centerline=centerline,
                )
            )

        route.path = concatenate_centerlines(centerlines)
        return route

    def _build_heuristic(self, map_: Map, routing_graph) -> Callable[[int, int], float]:
        if self.heuristic_builder is not None:
            return self.heuristic_builder(map_, routing_graph)

        center_cache = {}
        for lane_id, idx in routing_graph.lane_id_to_index.items():
            centerline = get_lane_centerline(map_.lanes[lane_id])
            if centerline is None or len(centerline) == 0:
                center_cache[idx] = None
            else:
                center_cache[idx] = np.mean(centerline, axis=0)

        def heuristic(src_idx: int, dst_idx: int) -> float:
            src_center = center_cache.get(src_idx)
            dst_center = center_cache.get(dst_idx)
            if src_center is None or dst_center is None:
                return 0.0
            return float(np.linalg.norm(src_center - dst_center))

        return heuristic
