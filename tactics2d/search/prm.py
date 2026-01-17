# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""PRM (Probabilistic Roadmap) algorithm implementation."""

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree


def _adjacency_list_to_csr(n_nodes: int, edges: List[Tuple[int, int, float]]) -> csr_matrix:
    """Convert adjacency list to CSR sparse matrix.

    Args:
        n_nodes: Number of nodes in the graph.
        edges: List of (i, j, weight) tuples representing directed edges.

    Returns:
        csr_matrix: Sparse adjacency matrix of shape (n_nodes, n_nodes).
    """
    if not edges:
        return csr_matrix((n_nodes, n_nodes))

    # Create symmetric edges (undirected graph)
    rows = []
    cols = []
    weights = []
    for i, j, w in edges:
        rows.append(i)
        cols.append(j)
        weights.append(w)
        # Add reverse edge
        rows.append(j)
        cols.append(i)
        weights.append(w)

    return csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))


class PRM:
    """This class implements the Probabilistic Roadmap (PRM) algorithm.

    !!! quote "Reference"
        Kavraki, Lydia E., et al. "Probabilistic roadmaps for path planning in high-dimensional configuration spaces." IEEE transactions on Robotics and Automation 12.4 (1996): 566-580.
    """

    @staticmethod
    def plan(
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
        obstacles: Any,
        collide_fn: Callable[[ArrayLike, ArrayLike, Any], bool],
        n_samples: int = 1000,
        k_nearest: int = 10,
        connection_radius: Optional[float] = None,
        max_iter: int = 100000,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> Tuple[List, List]:
        """Find a collision-free path using Probabilistic Roadmap (PRM).

        PRM constructs a graph (roadmap) by sampling random configurations in free space
        and connecting nearby nodes with collision-free edges. The graph is then searched
        to find a path from start to target.

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Goal point [x, y].
            boundary (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            obstacles (Any): Collection of obstacles in the environment.
            collide_fn (Callable[[ArrayLike, ArrayLike, Any], bool]): Collision checking function with signature
                `collide_fn(p1: ArrayLike, p2: ArrayLike, obstacles: Any) -> bool`,
                returning True if the edge (p1 -> p2) is in collision.
            n_samples (int, optional): Number of random samples for roadmap construction.
                Defaults to 1000.
            k_nearest (int, optional): Number of nearest neighbors to connect for each node.
                Defaults to 10. Ignored if connection_radius is specified.
            connection_radius (Optional[float], optional): Radius for connecting nodes.
                If specified, all nodes within this radius are connected (up to max_connections).
                Defaults to None.
            max_iter (int, optional): Maximum number of iterations for graph search.
                Defaults to 100000.
            callback (Optional[Callable[[Dict], None]], optional): Optional callback function called
                at key phases of the algorithm. Receives a dictionary with algorithm state:
                phase, iteration, n_nodes, n_edges, roadmap, path_found, path, etc.
                Useful for visualization and debugging.

        Returns:
            tuple: A tuple containing:
                - path (list): A sequence of waypoints from start to target.
                  Empty list if no path is found.
                - roadmap (tuple): The constructed roadmap as a tuple (nodes, edges), where:
                    nodes: List of [x, y] coordinates of all sampled points including start and target.
                    edges: List of (i, j, cost) tuples representing collision-free edges.
        """
        # Convert inputs to numpy arrays
        start_arr = np.asarray(start)
        target_arr = np.asarray(target)
        boundary_arr = np.asarray(boundary)

        # Validate boundary
        x_min, x_max, y_min, y_max = boundary_arr
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Invalid boundary: {boundary_arr}. "
                f"Must satisfy x_min < x_max and y_min < y_max"
            )

        # Validate start and target are within boundary (with small tolerance)
        eps = 1e-9
        if not (x_min - eps <= start_arr[0] <= x_max + eps):
            raise ValueError(
                f"start x-coordinate {start_arr[0]} is outside boundary x-range [{x_min}, {x_max}]"
            )
        if not (y_min - eps <= start_arr[1] <= y_max + eps):
            raise ValueError(
                f"start y-coordinate {start_arr[1]} is outside boundary y-range [{y_min}, {y_max}]"
            )
        if not (x_min - eps <= target_arr[0] <= x_max + eps):
            raise ValueError(
                f"target x-coordinate {target_arr[0]} is outside boundary x-range [{x_min}, {x_max}]"
            )
        if not (y_min - eps <= target_arr[1] <= y_max + eps):
            raise ValueError(
                f"target y-coordinate {target_arr[1]} is outside boundary y-range [{y_min}, {y_max}]"
            )

        # Validate other parameters
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if k_nearest <= 0:
            raise ValueError(f"k_nearest must be positive, got {k_nearest}")
        if connection_radius is not None and connection_radius <= 0:
            raise ValueError(f"connection_radius must be positive, got {connection_radius}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")

        # Phase 1: Sampling
        nodes = [start_arr.tolist(), target_arr.tolist()]  # Start with start and target
        samples_added = 0
        iteration = 0

        # Generate random samples
        while samples_added < n_samples and iteration < n_samples * 2:  # Avoid infinite loop
            iteration += 1
            # Sample random point within boundary
            rx = np.random.uniform(x_min, x_max)
            ry = np.random.uniform(y_min, y_max)

            # Note: Point collision detection is not performed.
            # The collide_fn only checks edges, not points. Sampled points are assumed
            # to be in free space. If a point is inside an obstacle, edges from it
            # will fail collision checks, making it an isolated node.
            nodes.append([rx, ry])
            samples_added += 1

            # Callback for sampling phase
            if callback is not None:
                state = {
                    "phase": "sampling",
                    "iteration": iteration,
                    "n_nodes": len(nodes),
                    "n_edges": 0,
                    "nodes": nodes.copy(),
                    "edges": [],
                    "current_sample": [rx, ry],
                    "path_found": False,
                    "path": [],
                }
                callback(state)

        # Convert nodes to numpy array for efficient operations
        nodes_array = np.array(nodes)

        # Phase 2: Building roadmap (connecting nodes)
        edges = []
        n_nodes = len(nodes)

        # Build KDTree for efficient nearest neighbor queries
        kdtree = KDTree(nodes_array)

        # Determine connection strategy
        use_radius = connection_radius is not None
        radius = connection_radius if use_radius else None

        # For each node, find neighbors and attempt connections
        for i in range(n_nodes):
            # Find neighbors
            if use_radius:
                # Find all nodes within radius using query_ball_point
                neighbor_indices = kdtree.query_ball_point(nodes_array[i], radius)
                # Convert to numpy array and filter out self
                neighbor_indices = np.array(neighbor_indices)
                neighbor_indices = neighbor_indices[neighbor_indices != i]
            else:
                # Find k_nearest + 1 (to include self), then remove self
                k = min(k_nearest + 1, n_nodes)
                distances, indices = kdtree.query(nodes_array[i], k=k)
                # Remove self
                mask = indices != i
                neighbor_indices = indices[mask]

            # Attempt to connect to each neighbor
            for j in neighbor_indices:
                # Avoid duplicate edges (i < j ensures each edge is considered once)
                if i >= j:
                    continue

                # Check collision between nodes i and j
                if not collide_fn(nodes_array[i], nodes_array[j], obstacles):
                    # Compute Euclidean distance as edge cost
                    dist = np.linalg.norm(nodes_array[i] - nodes_array[j])
                    edges.append((i, j, float(dist)))

            # Callback for connection progress
            if callback is not None and i % 10 == 0:  # Callback every 10 nodes to avoid overhead
                state = {
                    "phase": "connecting",
                    "iteration": i,
                    "n_nodes": n_nodes,
                    "n_edges": len(edges),
                    "nodes": nodes.copy(),
                    "edges": edges.copy(),
                    "current_node": nodes_array[i].tolist(),
                    "path_found": False,
                    "path": [],
                }
                callback(state)

        # Phase 3: Graph search
        # Convert edges to CSR matrix for Dijkstra/A*
        if edges:
            graph = _adjacency_list_to_csr(n_nodes, edges)

            # Use Dijkstra for graph search
            # We need to rasterize start/target indices for Dijkstra
            # But since start and target are already nodes in our graph (indices 0 and 1),
            # we can directly search using a simple Dijkstra implementation

            # Simple Dijkstra on the graph
            start_idx = 0  # start is first node
            target_idx = 1  # target is second node

            # Dijkstra's algorithm
            dist = [float("inf")] * n_nodes
            prev = [-1] * n_nodes
            dist[start_idx] = 0.0

            pq = [(0.0, start_idx)]  # (distance, node)

            search_iteration = 0
            while pq and search_iteration < max_iter:
                search_iteration += 1
                current_dist, current = heapq.heappop(pq)

                # Skip if we found a better path already
                if current_dist > dist[current]:
                    continue

                # Callback for search progress
                if callback is not None and search_iteration % 100 == 0:
                    state = {
                        "phase": "searching",
                        "iteration": search_iteration,
                        "n_nodes": n_nodes,
                        "n_edges": len(edges),
                        "nodes": nodes.copy(),
                        "edges": edges.copy(),
                        "current_node": nodes_array[current].tolist(),
                        "current_distance": current_dist,
                        "path_found": False,
                        "path": [],
                    }
                    callback(state)

                # Check if we reached target
                if current == target_idx:
                    # Reconstruct path
                    path_indices = []
                    node = current
                    while node != -1:
                        path_indices.append(node)
                        node = prev[node]
                    path_indices.reverse()

                    # Convert indices to coordinates
                    path = [nodes[idx] for idx in path_indices]

                    # Final callback with path found
                    if callback is not None:
                        state = {
                            "phase": "complete",
                            "iteration": search_iteration,
                            "n_nodes": n_nodes,
                            "n_edges": len(edges),
                            "nodes": nodes.copy(),
                            "edges": edges.copy(),
                            "current_node": nodes_array[current].tolist(),
                            "current_distance": current_dist,
                            "path_found": True,
                            "path": path.copy(),
                        }
                        callback(state)

                    return path, (nodes, edges)

                # Explore neighbors
                row_start = graph.indptr[current]
                row_end = graph.indptr[current + 1]

                for idx in range(row_start, row_end):
                    neighbor = graph.indices[idx]
                    weight = graph.data[idx]

                    new_dist = current_dist + weight
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        prev[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))

            # Search failed (max iterations reached or no path)
            if callback is not None:
                state = {
                    "phase": "complete",
                    "iteration": search_iteration,
                    "n_nodes": n_nodes,
                    "n_edges": len(edges),
                    "nodes": nodes.copy(),
                    "edges": edges.copy(),
                    "current_node": None,
                    "current_distance": None,
                    "path_found": False,
                    "path": [],
                    "search_failed": True,
                    "max_iterations_reached": search_iteration >= max_iter,
                }
                callback(state)

        # No path found
        return [], (nodes, edges)
