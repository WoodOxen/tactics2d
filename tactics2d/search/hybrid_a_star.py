# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hybrid A* algorithm implementations."""


import heapq
import math
from typing import Any, Callable, List

import numpy as np
from numpy.typing import ArrayLike


class HybridAStar:
    """This class implements the Hybrid A* algorithm for vehicle motion planning.

    Hybrid A* combines discrete graph search with continuous vehicle dynamics.
    It operates in the vehicle's state space (x, y, heading) and considers
    kinematic constraints.

    !!! quote "Reference"
        Dolgov, Dmitri, et al. "Practical search techniques in path planning for autonomous driving." ann arbor 1001.48105 (2008): 18-80.
    """

    @staticmethod
    def plan(
        start: List[float],
        target: List[float],
        boundary: ArrayLike,
        obstacles: Any,
        collide_fn: Callable[[ArrayLike, ArrayLike, Any], bool],
        step_size: float = 1.0,
        max_iter: int = 50000,
        steering_angles: List[float] = [-0.5, 0, 0.5],  # steering angles in radians
        velocity: float = 1.0,
        wheelbase: float = 2.5,
    ) -> List[List[float]]:
        """
        Plan a path considering vehicle kinematics.

        Args:
            start: Starting state [x, y, heading] in meters and radians
            target: Target state [x, y, heading] in meters and radians
            boundary: Search area [xmin, xmax, ymin, ymax]
            obstacles: Collection of obstacles for collision checking
            collide_fn: Function collide_fn(pos1, pos2, obstacles) -> bool where pos1, pos2 are [x, y] positions
            step_size: Distance for each motion primitive (meters)
            max_iter: Maximum number of iterations
            steering_angles: List of steering angles to consider (radians)
            velocity: Constant velocity (m/s)
            wheelbase: Vehicle wheelbase (meters)

        Returns:
            List[List[float]]: Sequence of states [x, y, heading] from start to target, or empty list if not found
        """

        # State node structure: (x, y, heading, parent_idx, g_cost, h_cost)
        start_node = (start[0], start[1], start[2], -1, 0.0, HybridAStar._heuristic(start, target))

        # Grid discretization for state space
        x_min, x_max, y_min, y_max = boundary
        xy_res = step_size * 0.5  # Position resolution
        heading_res = math.pi / 8  # Heading resolution

        # Calculate grid dimensions
        x_cells = int((x_max - x_min) / xy_res) + 1
        y_cells = int((y_max - y_min) / xy_res) + 1
        heading_cells = int(math.ceil(2 * math.pi / heading_res))

        # Data structures
        open_set = []
        heapq.heappush(open_set, (start_node[4] + start_node[5], 0))  # (f_cost, node_idx)

        nodes = [start_node]
        cost_grid = np.full((x_cells, y_cells, heading_cells), np.inf, dtype=float)

        # Convert continuous state to grid indices
        def state_to_grid(x, y, heading):
            xi = int((x - x_min) / xy_res)
            yi = int((y - y_min) / xy_res)
            # Normalize heading to [0, 2π)
            norm_heading = heading % (2 * math.pi)
            # Compute heading index with bounds checking
            hi_float = norm_heading / heading_res
            hi = int(hi_float)
            # Ensure index is within bounds (handles floating-point edge cases)
            if hi >= heading_cells:
                hi = heading_cells - 1
            return (xi, yi, hi)

        start_grid = state_to_grid(*start[:3])
        cost_grid[start_grid] = 0.0

        iteration = 0

        while open_set and iteration < max_iter:
            _, node_idx = heapq.heappop(open_set)
            current_node = nodes[node_idx]
            x, y, heading, parent_idx, g_cost, _ = current_node

            # Check if reached target
            if (
                abs(x - target[0]) < xy_res
                and abs(y - target[1]) < xy_res
                and abs(heading - target[2]) < heading_res
            ):
                # Reconstruct path
                path = []
                idx = node_idx
                while idx != -1:
                    node = nodes[idx]
                    path.append([node[0], node[1], node[2]])
                    idx = node[3]
                path.reverse()
                return path

            # Generate motion primitives
            for steer in steering_angles:
                # Kinematic bicycle model integration
                curvature = math.tan(steer) / wheelbase  # κ = tan(δ) / L
                delta_heading = curvature * step_size  # Δθ = κ * Δs
                new_heading = heading + delta_heading
                # Use average heading for position update (more accurate than using new_heading)
                avg_heading = heading + delta_heading / 2
                new_x = x + math.cos(avg_heading) * step_size
                new_y = y + math.sin(avg_heading) * step_size

                # Check bounds
                if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                    continue

                # Check collision (simplified - check line segment)
                if collide_fn([x, y], [new_x, new_y], obstacles):
                    continue

                # Calculate costs
                new_g = g_cost + step_size
                new_h = HybridAStar._heuristic([new_x, new_y, new_heading], target)
                new_f = new_g + new_h

                # Check if this grid cell has been reached with a lower cost
                new_grid = state_to_grid(new_x, new_y, new_heading)
                if new_g >= cost_grid[new_grid]:
                    continue

                # Update cost and add new node
                cost_grid[new_grid] = new_g
                new_node_idx = len(nodes)
                new_node = (new_x, new_y, new_heading, node_idx, new_g, new_h)
                nodes.append(new_node)

                heapq.heappush(open_set, (new_f, new_node_idx))

            iteration += 1

        return []  # No path found

    @staticmethod
    def _heuristic(state, target):
        """Combined heuristic: Euclidean distance + heading difference."""

        dx = state[0] - target[0]
        dy = state[1] - target[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Add penalty for heading difference
        heading_diff = abs(state[2] - target[2])
        heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
        heading_penalty = heading_diff * 0.1

        return dist + heading_penalty
