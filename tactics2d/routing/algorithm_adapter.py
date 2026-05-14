# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapters for reusing search algorithms on routing graphs."""

from typing import Callable, Dict, List, Tuple

from tactics2d.search.a_star import AStar
from tactics2d.search.dijkstra import Dijkstra

from .graph_builder import RoutingGraph


class AlgorithmAdapter:
    """Adapt a routing graph to shortest-path search."""

    @staticmethod
    def dijkstra(
        graph: RoutingGraph, start_idx: int, goal_idx: int
    ) -> Tuple[List[int], float, Dict[Tuple[int, int], str]]:
        path, total_cost = Dijkstra.plan_graph(start_idx, goal_idx, graph.csr_graph)
        edge_relations = AlgorithmAdapter._collect_edge_relations(graph, path)
        return path, total_cost, edge_relations

    @staticmethod
    def a_star(
        graph: RoutingGraph,
        start_idx: int,
        goal_idx: int,
        heuristic_fn: Callable[[int, int], float],
    ) -> Tuple[List[int], float, Dict[Tuple[int, int], str]]:
        path, total_cost = AStar.plan_graph(
            start_idx=start_idx,
            target_idx=goal_idx,
            graph=graph.csr_graph,
            heuristic_fn=heuristic_fn,
        )
        edge_relations = AlgorithmAdapter._collect_edge_relations(graph, path)
        return path, total_cost, edge_relations

    @staticmethod
    def _collect_edge_relations(graph: RoutingGraph, path: List[int]) -> Dict[Tuple[int, int], str]:
        edge_relations: Dict[Tuple[int, int], str] = {}
        for prev_idx, curr_idx in zip(path[:-1], path[1:]):
            relation = "successor"
            for neighbor_idx, _weight, edge_relation in graph.adjacency.get(prev_idx, []):
                if neighbor_idx == curr_idx:
                    relation = edge_relation
                    break
            edge_relations[(prev_idx, curr_idx)] = relation
        return edge_relations
