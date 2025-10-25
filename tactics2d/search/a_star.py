##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

import heapq
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix


class AStar:
    """This class implements the A* algorithm.

    !!! quote "Reference"
        Hart, Peter E., Nils J. Nilsson, and Bertram Raphael. "A formal basis for the heuristic determination of minimum cost paths." IEEE transactions on Systems Science and Cybernetics 4.2 (1968): 100-107.
    """

    @staticmethod
    def plan(
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
        graph: csr_matrix,
        heuristic_fn: Callable,
        step_size: float,
        max_iter: int = 1e5,
    ):
        """
        Args:
            graph (csr_matrix): _description_
            step_size (float): _description_
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Target point [x, y].
            bounds (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            max_iter (int):

        Returns:
            np.ndarray: The optimal path from the start to target, expressed by global coordinates.
        """
        # rasterize start and target indexes
        x_min, x_max, y_min, y_max = boundary
        width = int((x_max - x_min) / step_size)
        height = int((y_max - y_min) / step_size)

        N = graph.shape[0]
        assert width * height == N, f"width*height={width*height} does not equal to N={N}."

        start_rasterized = [(start[0] - x_min) / step_size, (start[1] - y_min) / step_size]
        target_rasterized = [(target[0] - x_min) / step_size, (target[1] - y_min) / step_size]

        start_idx = int(round(start_rasterized[1])) * width + int(round(start_rasterized[0]))
        target_idx = int(round(target_rasterized[1])) * width + int(round(target_rasterized[0]))

        g_score = np.inf * np.ones(N)
        f_score = np.inf * np.ones(N)
        g_score[start_idx] = 0
        f_score[start_idx] = heuristic_fn(start_rasterized, target_rasterized)

        open_set = []
        heapq.heappush(open_set, (f_score[start_idx], start_idx))

        came_from = {}
        i = 0

        while open_set and i < max_iter:
            _, current_idx = heapq.heappop(open_set)

            if current_idx == target_idx:
                path = []
                while current_idx in came_from:
                    y, x = divmod(current_idx, width)
                    path.append([x, y])
                    current_idx = came_from[current_idx]
                path.append([int(round(start_rasterized[0])), int(round(start_rasterized[1]))])
                path.reverse()

                path = np.array(path)
                path[:, 0] = path[:, 0] * step_size + boundary[0]
                path[:, 1] = path[:, 1] * step_size + boundary[2]
                return path

            neighbors = graph[current_idx].nonzero()[1]

            for neighbor in neighbors:
                tentative_g_score = g_score[current_idx] + graph[current_idx, neighbor]
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_idx
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_fn(
                        divmod(neighbor, width), target_rasterized
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

            i += 1

        return None


# class HybridAStar:
#     @staticmethod
#     def plan(graph, )
