# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""D* Litealgorithm implementations."""

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix


class DStar:
    """This class implements the D* Lite algorithm for dynamic path planning.

    !!! quote "Reference"
        Stentz, Anthony. The D* Algorithm for Real-Time Planning of Optimal Traverses. No. CMURITR9437. 1994.
        Koenig, Sven, and Maxim Likhachev. "Improved fast replanning for robot navigation in unknown terrain." Proceedings 2002 IEEE international conference on robotics and automation (Cat. No. 02CH37292). Vol. 1. IEEE, 2002.
    """

    @staticmethod
    def plan(
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
        graph: csr_matrix,
        heuristic_fn: Callable[[ArrayLike, ArrayLike], float],
        grid_resolution: float,
        max_iter: int = 1000000,  # Increased for complex graphs, especially 8-connectivity
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Target point [x, y].
            boundary (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            graph (csr_matrix): Sparse adjacency matrix of the search graph.
            heuristic_fn (Callable[[ArrayLike, ArrayLike], float]): Heuristic function that estimates cost between two points.
            grid_resolution (float): Grid resolution for rasterization.
            max_iter (int): Maximum number of iterations before giving up.
            callback (Optional[Callable[[Dict], None]]): Optional callback function called at each iteration.
                Receives a dictionary with algorithm state: iteration count, current node,
                open set size, g and rhs values, etc. Useful for visualization and debugging.

        Returns:
            np.ndarray or None: The optimal path from start to target, expressed by global
                coordinates, or None if no path exists.

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

        # --- D* Lite Algorithm Implementation ---
        # Based on standard D* Lite pseudocode
        # For static planning, km remains 0

        # Initialize arrays
        g = np.inf * np.ones(N)  # cost from start
        rhs = np.inf * np.ones(N)  # one-step lookahead cost
        km = 0.0  # path cost modifier

        # Helper: index to rasterized coordinates
        def idx_to_coords(idx: int) -> Tuple[float, float]:
            x = idx % width
            y = idx // width
            return (x, y)

        # Helper: heuristic between indices (using rasterized coordinates)
        def heuristic(idx1: int, idx2: int) -> float:
            coords1 = idx_to_coords(idx1)
            coords2 = idx_to_coords(idx2)
            return heuristic_fn(coords1, coords2)

        # Precompute heuristic values from all nodes to start_idx
        h_cache = np.zeros(N)
        for i in range(N):
            h_cache[i] = heuristic(i, start_idx)

        # Precompute neighbor lists for all nodes
        neighbors_cache = [graph[i].nonzero()[1].tolist() for i in range(N)]

        # Priority queue U stores (key1, key2, node_index)
        U = []
        # Track which nodes are in queue with their current keys
        in_queue = np.zeros(N, dtype=bool)
        node_keys = {}  # node_idx -> (key1, key2)

        # Helper: calculate key for node
        def calculate_key(idx: int) -> Tuple[float, float]:
            min_g_rhs = min(g[idx], rhs[idx])
            # D* Lite heuristic: estimate from current node to start
            h_val = h_cache[idx]  # Use precomputed heuristic value
            return (min_g_rhs + h_val + km, min_g_rhs)

        # Helper: get neighbors from graph
        def get_neighbors(idx: int) -> List[int]:
            return neighbors_cache[idx]

        # Helper: get cost between nodes
        def get_cost(u: int, v: int) -> float:
            return graph[u, v]

        # UpdateVertex procedure from D* Lite
        def update_vertex(u: int):
            if u != target_idx:
                # Compute min over successors
                succs = get_neighbors(u)
                min_cost = np.inf
                for succ in succs:
                    cost = get_cost(u, succ) + g[succ]
                    if cost < min_cost - 1e-12:  # Add tolerance for floating-point comparison
                        min_cost = cost
                rhs[u] = (
                    min_cost if not np.isinf(min_cost) else np.inf
                )  # Use np.isinf for numpy floats

            # Update in priority queue
            if in_queue[u]:
                in_queue[u] = False  # Mark old entry as stale

            if g[u] != rhs[u]:
                key = calculate_key(u)
                heapq.heappush(U, (key[0], key[1], u))
                in_queue[u] = True
                node_keys[u] = key

        # Initialize: set rhs of goal to 0 and update
        rhs[target_idx] = 0
        update_vertex(target_idx)

        iteration = 0

        # Main loop: ComputeShortestPath
        while U and iteration < max_iter:
            # Get top node with valid key (skip stale entries)
            while U:
                k1, k2, u = heapq.heappop(U)
                if not in_queue[u]:
                    continue  # Stale entry
                current_key = node_keys.get(u, (float("inf"), float("inf")))
                if (k1, k2) != current_key:
                    continue  # Key changed, skip
                break
            else:
                break  # No valid nodes left

            # Check termination condition
            key_start = calculate_key(start_idx)
            if (k1, k2) > key_start:
                # Reinsert with new key and continue
                update_vertex(u)
                iteration += 1
                continue

            if g[u] > rhs[u]:
                # Overconsistent
                g[u] = rhs[u]
                # Update all predecessors (neighbors in undirected graph)
                preds = get_neighbors(u)
                for pred in preds:
                    update_vertex(pred)
            else:
                # Underconsistent
                g[u] = np.inf
                # Update this node and its neighbors
                nodes_to_update = [u] + get_neighbors(u)
                for node in nodes_to_update:
                    update_vertex(node)

            iteration += 1

            # Call callback if provided
            if callback is not None:
                x_curr, y_curr = idx_to_coords(u)
                state = {
                    "iteration": iteration,
                    "current_idx": u,
                    "current_coords": (x_curr, y_curr),
                    "open_set_size": np.sum(in_queue),
                    "g_current": g[u],
                    "rhs_current": rhs[u],
                    "g": g.copy(),
                    "rhs": rhs.copy(),
                    "in_queue": in_queue.copy(),
                    "width": width,
                    "height": height,
                    "start_idx": start_idx,
                    "target_idx": target_idx,
                    "target_reached": False,
                }
                callback(state)

        # Check if path exists
        if rhs[start_idx] == np.inf:
            # Final callback for failure
            if callback is not None:
                state = {
                    "iteration": iteration,
                    "current_idx": None,
                    "current_coords": None,
                    "open_set_size": np.sum(in_queue),
                    "g_current": None,
                    "rhs_current": None,
                    "g": g.copy(),
                    "rhs": rhs.copy(),
                    "in_queue": in_queue.copy(),
                    "width": width,
                    "height": height,
                    "start_idx": start_idx,
                    "target_idx": target_idx,
                    "target_reached": False,
                    "path_found": False,
                    "search_failed": True,
                }
                callback(state)
            return None

        # Reconstruct path by following minimum cost neighbors
        path_indices = []
        current = start_idx
        path_steps = 0
        max_path_steps = N  # Safety limit: path cannot be longer than total nodes

        while current != target_idx and path_steps < max_path_steps:
            path_indices.append(current)
            # Find neighbor minimizing c(current, neighbor) + g(neighbor)
            neighbors = get_neighbors(current)
            best = current
            best_cost = np.inf
            for nb in neighbors:
                cost = get_cost(current, nb) + g[nb]
                if cost < best_cost - 1e-12:  # Add tolerance for floating-point comparison
                    best_cost = cost
                    best = nb

            if best == current:
                # No progress, check if this node has infinite rhs (dead end)
                if rhs[current] == np.inf:
                    # Dead end, cannot proceed
                    break
                # Otherwise might be floating-point equality, try to continue
                # but we need to ensure we don't get stuck in a loop
                pass

            current = best
            path_steps += 1

        # Check if we reached the target
        if current != target_idx:
            # Failed to reconstruct path even though rhs[start_idx] is finite
            # This indicates a bug in the algorithm or floating-point issues
            if callback is not None:
                state = {
                    "iteration": iteration,
                    "current_idx": current,
                    "current_coords": idx_to_coords(current),
                    "open_set_size": np.sum(in_queue),
                    "g_current": g[current],
                    "rhs_current": rhs[current],
                    "g": g.copy(),
                    "rhs": rhs.copy(),
                    "in_queue": in_queue.copy(),
                    "width": width,
                    "height": height,
                    "start_idx": start_idx,
                    "target_idx": target_idx,
                    "target_reached": False,
                    "path_found": False,
                    "path_reconstruction_failed": True,
                }
                callback(state)
            return None

        path_indices.append(target_idx)

        # Convert indices to global coordinates
        path = []
        for idx in path_indices:
            y, x = divmod(idx, width)
            global_x = x * grid_resolution + x_min
            global_y = y * grid_resolution + y_min
            path.append([global_x, global_y])

        path_array = np.array(path)

        # Final callback for success
        if callback is not None:
            state = {
                "iteration": iteration,
                "current_idx": target_idx,
                "current_coords": (target_x, target_y),
                "open_set_size": np.sum(in_queue),
                "g_current": g[target_idx],
                "rhs_current": rhs[target_idx],
                "g": g.copy(),
                "rhs": rhs.copy(),
                "in_queue": in_queue.copy(),
                "width": width,
                "height": height,
                "start_idx": start_idx,
                "target_idx": target_idx,
                "target_reached": True,
                "path_found": True,
                "path": path_array,
            }
            callback(state)

        return path_array
