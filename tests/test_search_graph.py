# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the generic topology-graph search interfaces."""

import numpy as np
from scipy.sparse import csr_matrix

from tactics2d.search import AStar, Dijkstra


def _build_graph() -> csr_matrix:
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 1] = 1.0
    matrix[1, 3] = 1.0
    matrix[0, 2] = 2.0
    matrix[2, 3] = 1.0
    return csr_matrix(matrix)


def test_dijkstra_plan_graph_returns_shortest_path():
    graph = _build_graph()

    path, total_cost = Dijkstra.plan_graph(0, 3, graph)

    assert path == [0, 1, 3]
    assert total_cost == 2.0


def test_astar_plan_graph_returns_shortest_path():
    graph = _build_graph()

    path, total_cost = AStar.plan_graph(0, 3, graph, heuristic_fn=lambda *_: 0.0)

    assert path == [0, 1, 3]
    assert total_cost == 2.0
