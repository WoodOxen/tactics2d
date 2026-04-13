# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""RRT* algorithm implementation."""

from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from .rrt import RRT


class RRTStar(RRT):
    """This class implements the RRT* algorithm.

    !!! quote "Reference"
        Karaman, Sertac, and Emilio Frazzoli. "Sampling-based algorithms for optimal motion planning." The international journal of robotics research 30.7 (2011): 846-894.
    """

    @staticmethod
    def _choose_parent(tree, new_node, nearest_idx, radius, obstacles, collide_fn):
        new_x, new_y = new_node
        best_parent = nearest_idx
        best_cost = tree[nearest_idx][3] + np.hypot(
            new_x - tree[nearest_idx][0], new_y - tree[nearest_idx][1]
        )

        for i, (nx, ny, _, cost) in enumerate(tree):
            if (nx - new_x) ** 2 + (ny - new_y) ** 2 <= radius**2:
                if not collide_fn([nx, ny], [new_x, new_y], obstacles):
                    new_cost = cost + np.hypot(new_x - nx, new_y - ny)
                    if new_cost < best_cost:
                        best_parent = i
                        best_cost = new_cost
        return best_parent, best_cost

    @staticmethod
    def plan(
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
        obstacles: Any,
        collide_fn: Callable[[ArrayLike, ArrayLike, Any], bool],
        extension_step: float = 1.0,
        max_iter: int = 100000,
        radius: float = 3.0,
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """Find an optimal collision-free path using RRT* (Optimal Rapidly-exploring Random Tree).

        RRT* is an asymptotically optimal variant of RRT that converges to the optimal
        solution as the number of iterations increases. It includes rewiring steps to
        improve path quality.

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Goal point [x, y].
            boundary (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            obstacles (Any): Collection of obstacles in the environment. Can be a list of
                rectangles, polygons, or other shapes depending on `collide_fn`.
            collide_fn (Callable[[ArrayLike, ArrayLike, Any], bool]): Collision checking function with signature
                `collide_fn(p1: ArrayLike, p2: ArrayLike, obstacles: Any) -> bool`,
                returning True if the edge (p1 -> p2) is in collision.
            extension_step (float, optional): Maximum extension distance for tree growth.
                Defaults to 1.0.
            max_iter (int, optional): Maximum number of iterations for tree expansion.
                Defaults to 100000.
            radius (float, optional): Rewiring radius for connecting nearby nodes.
                Defaults to 3.0.
            callback (Optional[Callable[[Dict], None]], optional): Optional callback function called
                at each iteration when a new node is successfully added to the tree. Receives a
                dictionary with algorithm state: iteration count, tree size, new node coordinates,
                random sample point, current tree structure, and node costs. Useful for visualization
                and debugging.

        Returns:
            tuple: A tuple containing:
                - path (list): A sequence of waypoints from start to target.
                  Empty list if no path is found.
                - tree (list): The search tree as a list of nodes (x, y, parent_index, cost).
        """
        # Convert ArrayLike inputs to numpy arrays for type safety
        start_arr = np.asarray(start)
        target_arr = np.asarray(target)
        boundary_arr = np.asarray(boundary)

        tree = [(float(start_arr[0]), float(start_arr[1]), None, 0.0)]

        for iteration in range(max_iter):
            rx = np.random.uniform(boundary_arr[0], boundary_arr[1])
            ry = np.random.uniform(boundary_arr[2], boundary_arr[3])

            nodes = np.array([(x, y) for x, y, _, _ in tree])
            dists = np.sum((nodes - np.array([rx, ry])) ** 2, axis=1)
            nearest_node_idx = int(np.argmin(dists))
            nearest_node = tree[nearest_node_idx]

            # Compute direction vector from nearest node to random point
            dx = rx - nearest_node[0]
            dy = ry - nearest_node[1]
            dist = np.hypot(dx, dy)

            # Determine new node position
            if dist < extension_step:
                # Random point is closer than step_size, use it directly
                new_x, new_y = rx, ry
            else:
                # Move step_size towards random point
                new_x = nearest_node[0] + extension_step * dx / dist
                new_y = nearest_node[1] + extension_step * dy / dist

            # Check if new node is within boundaries
            if not (
                boundary_arr[0] <= new_x <= boundary_arr[1]
                and boundary_arr[2] <= new_y <= boundary_arr[3]
            ):
                continue

            # Check collision from nearest node to new node
            if collide_fn(nearest_node[:2], [new_x, new_y], obstacles):
                continue
            best_parent, best_cost = RRTStar._choose_parent(
                tree, [new_x, new_y], nearest_node_idx, radius, obstacles, collide_fn
            )
            new_node = (new_x, new_y, best_parent, best_cost)

            tree.append(new_node)
            new_idx = len(tree) - 1

            # Call callback if provided
            if callback is not None:
                state = {
                    "iteration": iteration,
                    "tree_size": len(tree),
                    "new_node": (new_x, new_y),
                    "parent_node": tree[best_parent][:2] if best_parent is not None else None,
                    "random_point": (rx, ry),
                    "tree": tree.copy(),  # copy of current tree
                    "path_found": False,
                    "path": [],
                    "node_costs": [node[3] for node in tree],  # cost for each node
                }
                callback(state)

            new_x, new_y, _, new_cost = tree[new_idx]
            for i, (nx, ny, _, cost) in enumerate(tree):
                if i == new_idx:
                    continue
                if (nx - new_x) ** 2 + (ny - new_y) ** 2 <= radius**2:
                    if not collide_fn((nx, ny), (new_x, new_y), obstacles):
                        alt_cost = new_cost + np.hypot(new_x - nx, new_y - ny)
                        if alt_cost < cost:
                            tree[i] = (nx, ny, new_idx, alt_cost)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(target_arr[:2])) < extension_step:
                # Check collision from new_node to target before adding goal
                if not collide_fn(new_node[:2], target_arr[:2], obstacles):
                    # Calculate cost to target: best_cost + distance(new_node, target)
                    target_cost = best_cost + np.hypot(
                        new_node[0] - target_arr[0], new_node[1] - target_arr[1]
                    )
                    tree.append(
                        (float(target_arr[0]), float(target_arr[1]), len(tree) - 1, target_cost)
                    )
                    # Calculate final path
                    path = RRT._reconstruct(tree, len(tree) - 1)
                    # Call callback with path found
                    if callback is not None:
                        state = {
                            "iteration": iteration,
                            "tree_size": len(tree),
                            "new_node": (float(target_arr[0]), float(target_arr[1])),
                            "parent_node": new_node[:2],
                            "random_point": (rx, ry),
                            "tree": tree.copy(),
                            "path_found": True,
                            "path": path,
                            "node_costs": [node[3] for node in tree],
                        }
                        callback(state)
                    break

        # Calculate final path (may be empty if no path found)
        path = RRT._reconstruct(tree, len(tree) - 1)
        # Call callback with final state (path found or not)
        if callback is not None:
            state = {
                "iteration": max_iter,
                "tree_size": len(tree),
                "new_node": None,
                "parent_node": None,
                "random_point": None,
                "tree": tree.copy(),
                "path_found": len(path) > 0,
                "path": path,
                "node_costs": [node[3] for node in tree],
                "search_failed": len(path) == 0,
                "max_iterations_reached": True,
            }
            callback(state)

        return path, tree
