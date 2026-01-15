# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later


from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from tactics2d.participant.trajectory import State


class RRT:
    """This class implements the RRT algorithm.

    !!! quote "Reference"
        LaValle, Steven. "Rapidly-exploring random trees: A new tool for path planning." Research Report 9811 (1998).
    """

    @staticmethod
    def _reconstruct(tree, idx):
        result = []

        while idx is not None:
            node = tree[idx]
            result.append([node[0], node[1]])
            idx = node[2]

        return result[::-1]

    @staticmethod
    def plan(
        start: ArrayLike,
        target: ArrayLike,
        boundary: ArrayLike,
        obstacles: Any,
        collide_fn: Callable[[ArrayLike, ArrayLike, Any], bool],
        extension_step: float = 1.0,
        max_iter: int = 100000,
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """Find a feasible collision-free path using Rapidly-exploring Random Tree (RRT).
        /
                This method finds a feasible collision-free path from the start to the target
                position without considering vehicle size, kinematic constraints, or dynamics.
                The result is a sequence of waypoints in free space.

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
                    callback (Optional[Callable[[Dict], None]], optional): Optional callback function called
                        at each iteration when a new node is successfully added to the tree. Receives a
                        dictionary with algorithm state: iteration count, tree size, new node coordinates,
                        random sample point, and current tree structure. Useful for visualization and debugging.

                Returns:
                    tuple: A tuple containing:
                        - path (list): A sequence of waypoints from start to target.
                          Empty list if no path is found.
                        - tree (list): The search tree as a list of nodes (x, y, parent_index).
        """
        # Convert ArrayLike inputs to numpy arrays for type safety
        start_arr = np.asarray(start)
        target_arr = np.asarray(target)
        boundary_arr = np.asarray(boundary)

        tree = [(float(start_arr[0]), float(start_arr[1]), None)]

        for iteration in range(max_iter):
            rx = np.random.uniform(boundary_arr[0], boundary_arr[1])
            ry = np.random.uniform(boundary_arr[2], boundary_arr[3])

            nodes = np.array([(x, y) for x, y, _ in tree])
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

            new_node = (new_x, new_y, nearest_node_idx)
            tree.append(new_node)

            # Call callback if provided
            if callback is not None:
                state = {
                    "iteration": iteration,
                    "tree_size": len(tree),
                    "new_node": (new_x, new_y),
                    "parent_node": nearest_node[:2],
                    "random_point": (rx, ry),
                    "tree": tree.copy(),  # copy of current tree
                    "path_found": False,
                    "path": [],
                }
                callback(state)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(target_arr[:2])) < extension_step:
                # Check collision from new_node to target before adding goal
                if not collide_fn(new_node[:2], target_arr[:2], obstacles):
                    tree.append((float(target_arr[0]), float(target_arr[1]), len(tree) - 1))
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
                "search_failed": len(path) == 0,
                "max_iterations_reached": True,
            }
            callback(state)

        return path, tree
