# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""RRT-Connect algorithm implementation."""

from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from .rrt import RRT


class RRTConnect(RRT):
    """This class implements the RRT-Connect algorithm.

    RRT-Connect grows two trees simultaneously from start and goal, attempting
    to connect them for faster path finding.

    !!! quote "Reference"
        Kuffner, James J., and Steven M. LaValle. "RRT-connect: An efficient
        approach to single-query path planning." Proceedings 2000 ICRA.
        Millennium Conference. IEEE International Conference on Robotics and
        Automation. Symposia Proceedings (Cat. No. 00CH37065). Vol. 2.
        IEEE, 2000.
    """

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
        """Find a feasible collision-free path using RRT-Connect.

        RRT-Connect grows two trees simultaneously from start and goal,
        attempting to connect them. This often finds paths faster than
        standard RRT.

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
                at each iteration when a new node is successfully added to either tree. Receives a
                dictionary with algorithm state: iteration count, tree sizes, new node coordinates,
                random sample point, current trees structure, and which tree was extended. Useful for
                visualization and debugging.

        Returns:
            tuple: A tuple containing:
                - path (list): A sequence of waypoints from start to target.
                  Empty list if no path is found.
                - trees (tuple): A tuple of two trees (tree_a, tree_b) where each tree
                  is a list of nodes (x, y, parent_index).
        """
        # Convert ArrayLike inputs to numpy arrays for type safety
        start_arr = np.asarray(start)
        target_arr = np.asarray(target)
        boundary_arr = np.asarray(boundary)

        # Initialize two trees: one from start, one from target
        tree_a = [(float(start_arr[0]), float(start_arr[1]), None)]  # Tree from start
        tree_b = [(float(target_arr[0]), float(target_arr[1]), None)]  # Tree from target

        for iteration in range(int(max_iter)):
            # Alternate between extending tree_a and tree_b
            if iteration % 2 == 0:
                from_tree, to_tree = tree_a, tree_b
            else:
                from_tree, to_tree = tree_b, tree_a

            # Sample random point
            rx = np.random.uniform(boundary_arr[0], boundary_arr[1])
            ry = np.random.uniform(boundary_arr[2], boundary_arr[3])

            # Find nearest node in from_tree
            nodes = np.array([(x, y) for x, y, _ in from_tree])
            dists = np.sum((nodes - np.array([rx, ry])) ** 2, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_node = from_tree[nearest_idx]

            # Compute direction vector from nearest node to random point
            dx = rx - nearest_node[0]
            dy = ry - nearest_node[1]
            dist = np.hypot(dx, dy)

            # Determine new node position
            if dist < extension_step:
                new_x, new_y = rx, ry
            else:
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

            # Add new node to from_tree
            new_node = (new_x, new_y, nearest_idx)
            from_tree.append(new_node)

            # Call callback if provided
            if callback is not None:
                extending_tree_a = iteration % 2 == 0  # tree_a is from_tree when iteration is even
                state = {
                    "iteration": iteration,
                    "tree_a_size": len(tree_a),
                    "tree_b_size": len(tree_b),
                    "new_node": (new_x, new_y),
                    "parent_node": nearest_node[:2],
                    "random_point": (rx, ry),
                    "tree_a": tree_a.copy(),  # copy of tree_a
                    "tree_b": tree_b.copy(),  # copy of tree_b
                    "extending_tree": "a" if extending_tree_a else "b",
                    "path_found": False,
                    "path": [],
                }
                callback(state)

            # Try to connect from new node to nearest node in to_tree
            to_nodes = np.array([(x, y) for x, y, _ in to_tree])
            to_dists = np.sum((to_nodes - np.array([new_x, new_y])) ** 2, axis=1)
            to_nearest_idx = int(np.argmin(to_dists))
            to_nearest_node = to_tree[to_nearest_idx]

            # Check if we can connect directly
            if not collide_fn([new_x, new_y], to_nearest_node[:2], obstacles):
                # Connection successful! Reconstruct path
                if from_tree is tree_a:
                    # Path: tree_a from start to new_node, then tree_b from to_nearest_node to target
                    path_a = RRT._reconstruct(tree_a, len(tree_a) - 1)  # start -> new_node
                    path_b = RRT._reconstruct(tree_b, to_nearest_idx)  # target -> to_nearest_node
                    path_b.reverse()  # Reverse to get to_nearest_node -> target
                    # Combine: start -> new_node -> to_nearest_node -> target
                    path = path_a + path_b
                else:
                    # from_tree is tree_b, to_tree is tree_a
                    path_a = RRT._reconstruct(tree_b, len(tree_b) - 1)  # target -> new_node
                    path_b = RRT._reconstruct(tree_a, to_nearest_idx)  # start -> to_nearest_node
                    path_a.reverse()  # Reverse to get new_node -> target
                    # Combine: start -> to_nearest_node -> new_node -> target
                    path = path_b + path_a

                # Call callback with path found
                if callback is not None:
                    extending_tree_a = from_tree is tree_a
                    state = {
                        "iteration": iteration,
                        "tree_a_size": len(tree_a),
                        "tree_b_size": len(tree_b),
                        "new_node": (new_x, new_y),
                        "parent_node": nearest_node[:2],
                        "to_nearest_node": to_nearest_node[:2],
                        "random_point": (rx, ry),
                        "tree_a": tree_a.copy(),
                        "tree_b": tree_b.copy(),
                        "extending_tree": "a" if extending_tree_a else "b",
                        "path_found": True,
                        "path": path,
                    }
                    callback(state)
                return path, (tree_a, tree_b)

        # No path found within max_iter
        # Call callback with final state (path not found)
        if callback is not None:
            state = {
                "iteration": max_iter,
                "tree_a_size": len(tree_a),
                "tree_b_size": len(tree_b),
                "new_node": None,
                "parent_node": None,
                "to_nearest_node": None,
                "random_point": None,
                "tree_a": tree_a.copy(),
                "tree_b": tree_b.copy(),
                "extending_tree": None,
                "path_found": False,
                "path": [],
                "search_failed": True,
                "max_iterations_reached": True,
            }
            callback(state)
        return [], (tree_a, tree_b)
