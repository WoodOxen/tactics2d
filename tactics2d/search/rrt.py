##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

from typing import Any, Callable

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
        collide_fn: Callable,
        step_size: float = 1.0,
        max_iter: int = 1e5,
    ):
        """This method finds a feasible collision-free path from the start to the target position without considering vehicle size, kinematic constraints, or dynamics. The result is a sequence of waypoints in free space.

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Goal point [x, y].
            boundary (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            obstacles (Any): Collection of obstacles in the environment. Can be a list of rectangles, polygons, or other shapes depending on `collide_fn`.
            collide_fn (Callable): Collision checking function with signature `collide_fn(p1: ArrayLike, p2: ArrayLike, obstacles: Any) -> bool`, returning True if the edge (p1 → p2) is in collision.
            max_iter
            step_size

        Returns:
            path (list): A sequence of waypoints from start to target. Empty list if no path is found.
            tree (list):
        """
        tree = [(start[0], start[1], None)]

        for _ in range(max_iter):
            rx = np.random.uniform(boundary[0], boundary[1])
            ry = np.random.uniform(boundary[2], boundary[3])

            nodes = np.array([(x, y) for x, y, _ in tree])
            dists = np.sum((nodes - np.array([rx, ry])) ** 2, axis=1)
            nearest_node_idx = int(np.argmin(dists))
            nearest_node = tree[nearest_node_idx]

            if collide_fn(nearest_node[:2], [rx, ry], obstacles):
                continue

            theta = np.arctan2(ry - nearest_node[1], rx - nearest_node[0])
            new_node = (
                nearest_node[0] + step_size * np.cos(theta),
                nearest_node[1] + step_size * np.sin(theta),
                nearest_node_idx,
            )

            tree.append(new_node)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(target[:2])) < step_size:
                tree.append((target[0], target[1], len(tree) - 1))
                break

        return RRT._reconstruct(tree, len(tree) - 1), tree

    @staticmethod
    def plan_trajectory(self):
        return


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
        collide_fn: Callable,
        step_size: float = 1.0,
        max_iter: int = 1e5,
        radius: float = 3.0,
    ):
        """_summary_

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Target point [x, y].
            boundary (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            obstacles (Any): Collection of obstacles in the environment. Can be a list of rectangles, polygons, or other shapes depending on `collide_fn`.
            collide_fn (Callable): Collision checking function with signature `collide_fn(p1: ArrayLike, p2: ArrayLike, obstacles: Any) -> bool`, returning True if the edge (p1 → p2) is in collision.

        Returns:
            path (list): A sequence of waypoints from start to target. Empty list if no path is found.
            tree (list):
        """
        tree = [(start[0], start[1], None, 0.0)]

        for _ in range(max_iter):
            rx = np.random.uniform(boundary[0], boundary[1])
            ry = np.random.uniform(boundary[2], boundary[3])

            nodes = np.array([(x, y) for x, y, _, _ in tree])
            dists = np.sum((nodes - np.array([rx, ry])) ** 2, axis=1)
            nearest_node_idx = int(np.argmin(dists))
            nearest_node = tree[nearest_node_idx]

            if collide_fn(nearest_node[:2], [rx, ry], obstacles):
                continue

            theta = np.arctan2(ry - nearest_node[1], rx - nearest_node[0])
            new_x = nearest_node[0] + step_size * np.cos(theta)
            new_y = nearest_node[1] + step_size * np.sin(theta)
            best_parent, best_cost = RRTStar._choose_parent(
                tree, [new_x, new_y], nearest_node_idx, radius, obstacles, collide_fn
            )
            new_node = (new_x, new_y, best_parent, best_cost)

            tree.append(new_node)
            new_idx = len(tree) - 1

            new_x, new_y, _, new_cost = tree[new_idx]
            for i, (nx, ny, _, cost) in enumerate(tree):
                if i == new_idx:
                    continue
                if (nx - new_x) ** 2 + (ny - new_y) ** 2 <= radius**2:
                    if not collide_fn((nx, ny), (new_x, new_y), obstacles):
                        alt_cost = new_cost + np.hypot(new_x - nx, new_y - ny)
                        if alt_cost < cost:
                            tree[i] = (nx, ny, new_idx, alt_cost)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(target[:2])) < step_size:
                tree.append((target[0], target[1], len(tree) - 1, best_cost))
                break

        return RRT._reconstruct(tree, len(tree) - 1), tree

    def plan_trajectory(self):
        return
