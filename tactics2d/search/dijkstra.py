##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Integration of networkx's implementation of Dijkstra's algorithm.
# @Author: Tactics2D Team
# @Version: 0.1.9

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class Dijkstra:
    @staticmethod
    def plan(
        graph: csr_matrix,
        cell_size: float,
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
    ):
        """"""
        x_min, x_max, y_min, y_max = boundary
        width = int((x_max - x_min) / cell_size)
        height = int((y_max - y_min) / cell_size)

        N = graph.shape[0]
        assert width * height == N, f"width*height={width*height} does not equal to N={N}."

        start_rasterized = [(start[0] - x_min) / cell_size, (start[1] - y_min) / cell_size]
        target_rasterized = [(target[0] - x_min) / cell_size, (target[1] - y_min) / cell_size]

        start_idx = int(round(start_rasterized[1])) * width + int(round(start_rasterized[0]))
        target_idx = int(round(target_rasterized[1])) * width + int(round(target_rasterized[0]))

        _, predecessors = dijkstra(
            csgraph=graph, directed=False, indices=start_idx, return_predecessors=True
        )

        path = []

        i = target_idx
        while i != start_idx and i != -9999:
            y, x = divmod(i, width)
            path.append([x, y])
            i = predecessors[i]

        path.append([int(round(start_rasterized[0])), int(round(start_rasterized[1]))])
        path.reverse()

        path = np.array(path)
        path[:, 0] = path[:, 0] * cell_size + boundary[0]
        path[:, 1] = path[:, 1] * cell_size + boundary[2]

        return path
