# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grid map generator implementation."""


import numpy as np


class GridMapGenerator:
    def __init__(self, size, min_cost=0, max_cost=5, obstacle_proportion=0.1, random_seed=None):
        self.size = size
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.obstacle_proportion = obstacle_proportion

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate(self):
        grid_map = np.random.randint(self.min_cost, self.max_cost + 1, size=self.size).astype(float)

        num_obstacles = int(self.size[0] * self.size[1] * self.obstacle_proportion)
        obstacle_indices = np.random.choice(
            self.size[0] * self.size[1], num_obstacles, replace=False
        )
        for index in obstacle_indices:
            x = index // self.size[1]
            y = index % self.size[1]
            grid_map[x, y] = np.inf  # Represent obstacles with infinite cost

        # randomly select start and goal positions that are not obstacles
        while True:
            start = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            if grid_map[start] != np.inf:
                break
        while True:
            goal = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            if grid_map[goal] != np.inf and goal != start:
                break

        return grid_map, start, goal
