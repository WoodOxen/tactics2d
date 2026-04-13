# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dijkstra algorithm implementation."""

import heapq
import math
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix


class Dijkstra:
    """This class implements the Dijkstra algorithm.

    !!! quote "Reference"
        Dijkstra, Edsger W. "A note on two problems in connexion with graphs."
        Edsger Wybe Dijkstra: his life, work, and legacy. 2022. 287-290.
    """

    @staticmethod
    def plan(
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
        graph: csr_matrix,
        grid_resolution: float,
        max_iter: int = 100000,
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Find the shortest path from start to target using Dijkstra's algorithm.

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Target point [x, y].
            boundary (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            graph (csr_matrix): Sparse adjacency matrix of the search graph.
            grid_resolution (float): Grid resolution for rasterization.
            max_iter (int): Maximum number of iterations before giving up.
            callback (Optional[Callable[[Dict], None]]): Optional callback function called at each iteration.
                Receives a dictionary with algorithm state: iteration count, current node, open set size,
                cost arrays, and parent mapping. Useful for visualization and debugging.

        Returns:
            np.ndarray or None: The optimal path from start to target as a Nx2 array
            of world coordinates. Returns None if no path exists. Returns a single
            point array [[x, y]] if start and target are the same.

        Raises:
            ValueError: If start or target coordinates are outside the boundary.
            ValueError: If grid_resolution is not positive.
            ValueError: If boundary is invalid (x_min >= x_max or y_min >= y_max).
            ValueError: If graph dimensions don't match the calculated grid size.
        """
        # rasterize start and target indexes
        x_min, x_max, y_min, y_max = boundary

        # Validate input parameters
        if grid_resolution <= 0:
            raise ValueError(f"grid_resolution must be positive, got {grid_resolution}")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Invalid boundary: {boundary}. Must satisfy x_min < x_max and y_min < y_max"
            )

        # Check that start and target coordinates are within boundary with epsilon tolerance
        eps = grid_resolution * 1e-9
        if not (x_min - eps <= start[0] < x_max + eps and y_min - eps <= start[1] < y_max + eps):
            raise ValueError(f"start coordinate {start} is outside boundary {boundary}")
        if not (x_min - eps <= target[0] < x_max + eps and y_min - eps <= target[1] < y_max + eps):
            raise ValueError(f"target coordinate {target} is outside boundary {boundary}")

        width = int((x_max - x_min) / grid_resolution)
        height = int((y_max - y_min) / grid_resolution)

        N = graph.shape[0]
        # Allow small floating-point tolerance in grid dimension calculation
        if not math.isclose(width * height, N, rel_tol=1e-9):
            raise ValueError(
                f"width*height={width*height} does not equal to N={N}. "
                f"Graph dimensions don't match the calculated grid size."
            )

        start_rasterized = [
            (start[0] - x_min) / grid_resolution,
            (start[1] - y_min) / grid_resolution,
        ]
        target_rasterized = [
            (target[0] - x_min) / grid_resolution,
            (target[1] - y_min) / grid_resolution,
        ]

        # Compute rasterized indices with clipping to handle floating-point errors at boundaries
        start_x = int(round(start_rasterized[0]))
        start_y = int(round(start_rasterized[1]))
        target_x = int(round(target_rasterized[0]))
        target_y = int(round(target_rasterized[1]))

        # Clip indices to valid range [0, width-1] and [0, height-1]
        start_x = max(0, min(start_x, width - 1))
        start_y = max(0, min(start_y, height - 1))
        target_x = max(0, min(target_x, width - 1))
        target_y = max(0, min(target_y, height - 1))

        start_idx = start_y * width + start_x
        target_idx = target_y * width + target_x

        # Ensure indices are within bounds
        if not (0 <= start_idx < N):
            raise ValueError(f"start_idx {start_idx} out of bounds [0, {N})")
        if not (0 <= target_idx < N):
            raise ValueError(f"target_idx {target_idx} out of bounds [0, {N})")

        # Check if start and target are the same
        if start_idx == target_idx:
            # Optional callback for trivial case
            if callback is not None:
                state = {
                    "iteration": 0,
                    "current_idx": start_idx,
                    "current_coords": (start_idx % width, start_idx // width),
                    "open_set_size": 0,
                    "g_score_current": 0.0,
                    "g_score": np.zeros(N),
                    "closed": np.zeros(N, dtype=bool),
                    "came_from": {},
                    "width": width,
                    "height": height,
                    "target_idx": target_idx,
                    "target_reached": True,
                    "path_found": True,
                    "path": np.array([[start[0], start[1]]]),
                    "open_set_indices": [],
                    "closed_indices": [],
                }
                callback(state)
            return np.array([[start[0], start[1]]])

        g_score = np.inf * np.ones(N)
        closed = np.zeros(N, dtype=bool)
        g_score[start_idx] = 0

        # Priority queue: (g_score, node_index)
        open_set = []
        heapq.heappush(open_set, (g_score[start_idx], start_idx))

        came_from = {}
        i = 0

        while open_set and i < max_iter:
            _, current_idx = heapq.heappop(open_set)

            # Skip if already closed (duplicate entry in open_set)
            if closed[current_idx]:
                i += 1
                continue

            # Mark as closed
            closed[current_idx] = True

            # Call callback if provided
            if callback is not None:
                x_current = current_idx % width
                y_current = current_idx // width
                state = {
                    "iteration": i,
                    "current_idx": current_idx,
                    "current_coords": (x_current, y_current),
                    "open_set_size": len(open_set),
                    "g_score_current": g_score[current_idx],
                    "g_score": g_score.copy(),  # copy of the full array
                    "closed": closed.copy(),  # copy of closed set array
                    "came_from": came_from.copy(),  # copy of the dict
                    "width": width,
                    "height": height,
                    "target_idx": target_idx,
                    "target_reached": False,
                    "open_set_indices": list({idx for _, idx in open_set}),
                    "closed_indices": np.where(closed)[0].tolist(),
                }
                callback(state)

            if current_idx == target_idx:
                path = []
                while current_idx in came_from:
                    y, x = divmod(current_idx, width)
                    path.append([x, y])
                    current_idx = came_from[current_idx]
                # Add start node
                start_y, start_x = divmod(start_idx, width)
                path.append([start_x, start_y])
                path.reverse()

                path = np.array(path)
                path[:, 0] = path[:, 0] * grid_resolution + boundary[0]
                path[:, 1] = path[:, 1] * grid_resolution + boundary[2]

                # Final callback with target reached
                if callback is not None:
                    x_current = target_idx % width
                    y_current = target_idx // width
                    state = {
                        "iteration": i,
                        "current_idx": target_idx,
                        "current_coords": (x_current, y_current),
                        "open_set_size": len(open_set),
                        "g_score_current": g_score[target_idx],
                        "g_score": g_score.copy(),
                        "closed": closed.copy(),
                        "came_from": came_from.copy(),
                        "width": width,
                        "height": height,
                        "target_idx": target_idx,
                        "target_reached": True,
                        "path_found": True,
                        "path": path,  # final path in global coordinates
                        "open_set_indices": list({idx for _, idx in open_set}),
                        "closed_indices": np.where(closed)[0].tolist(),
                    }
                    callback(state)

                return path

            neighbors = graph[current_idx].nonzero()[1]

            for neighbor in neighbors:
                # Skip if neighbor is already closed
                if closed[neighbor]:
                    continue
                tentative_g_score = g_score[current_idx] + graph[current_idx, neighbor]
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_idx
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))

            i += 1

        # Search failed (max iterations reached or open set empty)
        if callback is not None:
            state = {
                "iteration": i,
                "current_idx": None,
                "current_coords": None,
                "open_set_size": len(open_set),
                "g_score_current": None,
                "g_score": g_score.copy(),
                "closed": closed.copy(),
                "came_from": came_from.copy(),
                "width": width,
                "height": height,
                "target_idx": target_idx,
                "target_reached": False,
                "path_found": False,
                "path": None,
                "search_failed": True,
                "max_iterations_reached": i >= max_iter,
                "open_set_empty": len(open_set) == 0,
                "open_set_indices": list({idx for _, idx in open_set}),
                "closed_indices": np.where(closed)[0].tolist(),
            }
            callback(state)

        return None
