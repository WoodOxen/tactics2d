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
    def __init__(self, max_iter: int, step_size: float):
        self.max_iter = max_iter
        self.step_size = step_size

    def reconstruct(self, tree, idx):
        result = []

        while idx is not None:
            node = tree[idx]
            result.append([node[0], node[1]])
            idx = node[2]

        return result[::-1]

    def plan(
        self,
        start: ArrayLike,
        target: ArrayLike,
        bounds: ArrayLike,
        obstacles: Any,
        collide_fn: Callable,
    ):
        """This method finds a feasible collision-free path from the start to the target position without considering vehicle size, kinematic constraints, or dynamics. The result is a sequence of waypoints in free space.

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Goal point [x, y].
            bounds (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            obstacles (Any): Collection of obstacles in the environment. Can be a list of rectangles, polygons, or other shapes depending on `collide_fn`.
            collide_fn (Callable): Collision checking function with signature `collide_fn(p1: ArrayLike, p2: ArrayLike, obstacles: Any) -> bool`, returning True if the edge (p1 → p2) is in collision.

        Returns:
            path (list): A sequence of waypoints from start to target. Empty list if no path is found.
            tree (list):
        """
        tree = [(start[0], start[1], None)]

        for _ in range(self.max_iter):
            rx = np.random.uniform(bounds[0], bounds[1])
            ry = np.random.uniform(bounds[2], bounds[3])

            nodes = np.array([(x, y) for x, y, _ in tree])
            dists = np.sum((nodes - np.array([rx, ry])) ** 2, axis=1)
            nearest_node_idx = int(np.argmin(dists))
            nearest_node = tree[nearest_node_idx]

            if collide_fn(nearest_node[:2], [rx, ry], obstacles):
                continue

            theta = np.arctan2(ry - nearest_node[1], rx - nearest_node[0])
            new_node = (
                nearest_node[0] + self.step_size * np.cos(theta),
                nearest_node[1] + self.step_size * np.sin(theta),
                nearest_node_idx,
            )

            tree.append(new_node)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(target[:2])) < self.step_size:
                tree.append((target[0], target[1], len(tree) - 1))
                break

        return self.reconstruct(tree, len(tree) - 1), tree

    def plan_trajectory(self):
        return


class RRTStar(RRT):
    def __init__(self, max_iter, step_size, radius):
        super().__init__(max_iter, step_size)
        self.radius = radius

    def choose_parent(self, tree, new_node, nearest_idx, obstacles, collide_fn):
        new_x, new_y = new_node
        best_parent = nearest_idx
        best_cost = tree[nearest_idx][3] + np.hypot(
            new_x - tree[nearest_idx][0], new_y - tree[nearest_idx][1]
        )

        for i, (nx, ny, _, cost) in enumerate(tree):
            if (nx - new_x) ** 2 + (ny - new_y) ** 2 <= self.radius**2:
                if not collide_fn([nx, ny], [new_x, new_y], obstacles):
                    new_cost = cost + np.hypot(new_x - nx, new_y - ny)
                    if new_cost < best_cost:
                        best_parent = i
                        best_cost = new_cost
        return best_parent, best_cost

    def rewire(self, tree, new_idx, obstacles, collide_fn):
        new_x, new_y, _, new_cost = tree[new_idx]
        for i, (nx, ny, _, cost) in enumerate(tree):
            if i == new_idx:
                continue
            if (nx - new_x) ** 2 + (ny - new_y) ** 2 <= self.radius**2:
                if not collide_fn((nx, ny), (new_x, new_y), obstacles):
                    alt_cost = new_cost + np.hypot(new_x - nx, new_y - ny)
                    if alt_cost < cost:
                        tree[i] = (nx, ny, new_idx, alt_cost)

    def plan(
        self,
        start: ArrayLike,
        target: ArrayLike,
        bounds: ArrayLike,
        obstacles: Any,
        collide_fn: Callable,
    ):
        """_summary_

        Args:
            start (ArrayLike): Starting point [x, y].
            target (ArrayLike): Goal point [x, y].
            bounds (ArrayLike): Search area limits, formatted as [xmin, xmax, ymin, ymax].
            obstacles (Any): Collection of obstacles in the environment. Can be a list of rectangles, polygons, or other shapes depending on `collide_fn`.
            collide_fn (Callable): Collision checking function with signature `collide_fn(p1: ArrayLike, p2: ArrayLike, obstacles: Any) -> bool`, returning True if the edge (p1 → p2) is in collision.

        Returns:
            path (list): A sequence of waypoints from start to target. Empty list if no path is found.
            tree (list):
        """
        tree = [(start[0], start[1], None, 0.0)]

        for _ in range(self.max_iter):
            rx = np.random.uniform(bounds[0], bounds[1])
            ry = np.random.uniform(bounds[2], bounds[3])

            nodes = np.array([(x, y) for x, y, _, _ in tree])
            dists = np.sum((nodes - np.array([rx, ry])) ** 2, axis=1)
            nearest_node_idx = int(np.argmin(dists))
            nearest_node = tree[nearest_node_idx]

            if collide_fn(nearest_node[:2], [rx, ry], obstacles):
                continue

            theta = np.arctan2(ry - nearest_node[1], rx - nearest_node[0])
            new_x = nearest_node[0] + self.step_size * np.cos(theta)
            new_y = nearest_node[1] + self.step_size * np.sin(theta)
            best_parent, best_cost = self.choose_parent(
                tree, [new_x, new_y], nearest_node_idx, obstacles, collide_fn
            )
            new_node = (new_x, new_y, best_parent, best_cost)

            tree.append(new_node)
            new_idx = len(tree) - 1

            self.rewire(tree, new_idx, obstacles, collide_fn)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(target[:2])) < self.step_size:
                tree.append((target[0], target[1], len(tree) - 1, best_cost))
                break

        return self.reconstruct(tree, len(tree) - 1), tree

    def plan_trajectory(self):
        return
