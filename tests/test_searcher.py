##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9


import sys

sys.path.append(".")
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, Point

from tactics2d.map.converter import Rasterization
from tactics2d.map.element import Map
from tactics2d.map.generator import ParkingLotGenerator
from tactics2d.search import MCTS, RRT, AStar, Dijkstra, RRTStar


def generate_scenario():
    # np.random.seed(42)
    np.random.seed(33550336)
    map_generator = ParkingLotGenerator()
    map_ = Map()
    start_state, target_area, target_heading = map_generator.generate(map_)
    areas = map_.areas
    obstacles = [area for area in areas.values() if area.type_ == "obstacle"]
    bounds = map_.boundary
    start = [start_state.x, start_state.y]
    target = shapely.centroid(target_area.geometry)
    target = [target.x, target.y]

    return map_, bounds, obstacles, start, target


def rasterized_map_to_graph(rasterized_map):
    h, w = rasterized_map.shape
    N = h * w
    edges = []
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    for y in range(h):
        for x in range(w):
            if rasterized_map[y, x] == 1:
                continue
            idx = y * w + x
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny) and rasterized_map[ny, nx] == 0:
                    nidx = ny * w + nx
                    edges.append((idx, nidx, 1))

    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    data = [e[2] for e in edges]
    graph = csr_matrix((data, (rows, cols)), shape=(N, N))

    return graph


def heuristic_fn1(pt1, pt2):
    return abs(pt2[0] - pt1[0]) + abs(pt2[1] - pt1[1])


def heuristic_fn2(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def collide_fn(start, target, obstacles):
    line = LineString([start, target])
    for obstacle in obstacles:
        if line.intersects(obstacle.geometry):
            return True

    return False


@pytest.mark.math
def test_dijkstra():
    map_, bounds, obstacles, start, target = generate_scenario()
    cell_size = 0.1
    polygons = [obstacle.geometry for obstacle in obstacles]
    rasterized_map = Rasterization.rasterize_polygons(polygons, bounds, cell_size)
    graph = rasterized_map_to_graph(rasterized_map)
    path = Dijkstra.plan(start, target, bounds, graph, cell_size)

    return path


@pytest.mark.math
def test_a_star():
    map_, bounds, obstacles, start, target = generate_scenario()
    cell_size = 0.1
    polygons = [obstacle.geometry for obstacle in obstacles]
    rasterized_map = Rasterization.rasterize_polygons(polygons, bounds, cell_size)
    graph = rasterized_map_to_graph(rasterized_map)
    path = AStar.plan(start, target, bounds, graph, heuristic_fn1, cell_size)
    path = AStar.plan(start, target, bounds, graph, heuristic_fn2, cell_size)

    return path


@pytest.mark.math
def test_rrt():
    map_, bounds, obstacles, start, target = generate_scenario()
    cell_size = 0.1

    final_path, tree = RRT.plan(
        start, target, bounds, obstacles, collide_fn, max_iter=50000, step_size=cell_size * 5
    )

    return final_path, tree


@pytest.mark.math
def test_rrt_star():
    map_, bounds, obstacles, start, target = generate_scenario()
    cell_size = 0.1

    final_path, tree = RRTStar.plan(
        start,
        target,
        bounds,
        obstacles,
        collide_fn,
        max_iter=10000,
        step_size=cell_size * 5,
        radius=3.0,
    )

    return final_path, tree


def terminal_fn(current_position, target_position):
    return (
        np.sqrt(
            (current_position[0] - target_position[0]) ** 2
            + (current_position[1] - target_position[1]) ** 2
        )
        < 0.1
    )


def expand_fn(current_position, obstacles, step_size=0.1):
    x, y = current_position[:2]
    neighbors = []
    moves = [[step_size, 0], [-step_size, 0], [0, step_size], [0, -step_size]]
    for move in moves:
        position = current_position[0] + move[0], current_position[1] + move[1]
        point = Point(position)
        if not any(point.within(obstacle.geometry) for obstacle in obstacles):
            neighbors.append(position)
    return neighbors


def reward_fn(current_position, target_position):
    dist = np.sqrt(
        (current_position[0] - target_position[0]) ** 2
        + (current_position[1] - target_position[1]) ** 2
    )
    if dist < 0.1:
        return 100
    return -dist


def simulate_fn(current_position, target_position, obstacles, max_iters=1e3):
    x, y = current_position
    i = 0
    while not terminal_fn([x, y], target_position) and i < max_iters:
        neighbors = expand_fn([x, y], obstacles)
        if not neighbors:
            break
        idx = np.random.choice(len(neighbors))
        x, y = neighbors[idx]
        i += 1
    return [x, y]


@pytest.mark.math
def test_mcts():
    map_, bounds, obstacles, start, target = generate_scenario()
    mcts = MCTS(
        terminal_fn=lambda p: terminal_fn(p, target),
        expand_fn=lambda p: expand_fn(p, obstacles, step_size=0.05),
        reward_fn=lambda p: reward_fn(p, target),
        simulate_fn=lambda p: simulate_fn(p, target, obstacles),
    )
    final_path, tree_root = mcts.plan(start, max_try=1e2)

    return final_path, tree_root
