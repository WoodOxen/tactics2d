# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for search algorithms."""

import numpy as np
import pytest

from tactics2d.map.generator.generate_grid_map import GridMapGenerator
from tactics2d.search import (
    RRT,
    AStar,
    Dijkstra,
    DStar,
    HybridAStar,
    RRTConnect,
    RRTStar,
    grid_to_csr,
)


@pytest.fixture
def grid_map_fixture():
    """Generate a grid map as testing data with random parameters."""
    # Generate random size between 3 and 10 for both dimensions (very small for reliable testing)
    width = np.random.randint(3, 11)  # 3 to 10 inclusive, number of columns
    height = np.random.randint(3, 11)  # number of rows
    # GridMapGenerator expects size=(height, width) where height is rows, width is columns
    size = (height, width)

    # Generate random min_cost and max_cost between 1 and 10
    min_cost = np.random.randint(1, 11)  # 1 to 10 inclusive
    max_cost = np.random.randint(1, 11)
    # Ensure min_cost <= max_cost
    if min_cost > max_cost:
        min_cost, max_cost = max_cost, min_cost

    # Fixed obstacle proportion
    obstacle_proportion = 0.1
    # No random_seed to allow different maps each time

    generator = GridMapGenerator(
        size=size,
        min_cost=min_cost,
        max_cost=max_cost,
        obstacle_proportion=obstacle_proportion,
        random_seed=None,
    )
    grid_map, start_idx, goal_idx = generator.generate()
    # start_idx and goal_idx are (row, column) tuples
    # Convert to global coordinates [x, y] where x is column, y is row
    start = [start_idx[1], start_idx[0]]  # x = column, y = row
    goal = [goal_idx[1], goal_idx[0]]  # x = column, y = row
    # Define boundary as [x_min, x_max, y_min, y_max] covering the entire grid
    # Assuming grid cells are unit squares with grid_resolution = 1.0
    # x axis corresponds to columns (0 to width), y axis corresponds to rows (0 to height)
    boundary = [0.0, float(width), 0.0, float(height)]
    grid_resolution = 1.0
    return grid_map, start, goal, boundary, grid_resolution


@pytest.fixture
def grid_map_fixture_with_seed():
    """Generate a grid map with random parameters and fixed seed for reproducibility."""
    # Generate a random seed for this fixture instance
    seed = np.random.randint(0, 2**32 - 1)  # 32-bit random seed

    # Generate random size between 3 and 10 for both dimensions (very small for reliable testing)
    # Use the seed to ensure reproducibility for this fixture instance
    rng = np.random.RandomState(seed)
    width = rng.randint(3, 11)  # 3 to 10 inclusive, number of columns
    height = rng.randint(3, 11)  # number of rows
    size = (height, width)  # GridMapGenerator expects (rows, columns)

    # Generate random min_cost and max_cost between 1 and 10
    min_cost = rng.randint(1, 11)
    max_cost = rng.randint(1, 11)
    if min_cost > max_cost:
        min_cost, max_cost = max_cost, min_cost

    # Fixed obstacle proportion
    obstacle_proportion = 0.1
    # Use the generated seed for GridMapGenerator
    generator = GridMapGenerator(
        size=size,
        min_cost=min_cost,
        max_cost=max_cost,
        obstacle_proportion=obstacle_proportion,
        random_seed=seed,
    )
    grid_map, start_idx, goal_idx = generator.generate()
    # Convert to global coordinates
    start = [start_idx[1], start_idx[0]]  # x = column, y = row
    goal = [goal_idx[1], goal_idx[0]]
    boundary = [0.0, float(width), 0.0, float(height)]
    grid_resolution = 1.0
    return grid_map, start, goal, boundary, grid_resolution, seed


def euclidean_heuristic(a, b):
    """Euclidean distance heuristic for A* and D*."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def extract_obstacles_from_grid(grid_map):
    """Extract obstacle coordinates from grid map.

    Args:
        grid_map: 2D numpy array with obstacles marked as np.inf

    Returns:
        list: List of (x, y) obstacle cell centers where x=col, y=row
    """
    height, width = grid_map.shape
    obstacles = []
    for row in range(height):
        for col in range(width):
            if np.isinf(grid_map[row, col]):
                obstacles.append((col, row))
    return obstacles


def create_grid_collision_checker(obstacles):
    """Create a collision checking function for grid obstacles.

    Args:
        obstacles: List of (x, y) obstacle cell centers

    Returns:
        function: collide_fn(p1, p2, obstacles) that returns True if line segment
                  from p1 to p2 intersects any obstacle unit square.
    """

    def collide_fn(p1, p2, obstacles):
        # p1, p2: (x, y) coordinates
        # obstacles: list of (x, y) obstacle cell centers
        # Each obstacle is a unit square centered at (ox, oy) with side length 1
        # Use axis-aligned bounding box intersection test
        # Parameterize segment: p1 + t*(p2-p1), t in [0,1]
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1

        for ox, oy in obstacles:
            # Obstacle square bounds
            ox_min = ox - 0.5
            ox_max = ox + 0.5
            oy_min = oy - 0.5
            oy_max = oy + 0.5

            # Check if segment intersects AABB using slab method
            # Compute t for x and y slabs
            if abs(dx) < 1e-9:
                # Segment vertical, check if x within obstacle x range
                if not (ox_min <= x1 <= ox_max):
                    continue
                tx_min = -np.inf
                tx_max = np.inf
            else:
                tx1 = (ox_min - x1) / dx
                tx2 = (ox_max - x1) / dx
                tx_min = min(tx1, tx2)
                tx_max = max(tx1, tx2)

            if abs(dy) < 1e-9:
                # Segment horizontal, check if y within obstacle y range
                if not (oy_min <= y1 <= oy_max):
                    continue
                ty_min = -np.inf
                ty_max = np.inf
            else:
                ty1 = (oy_min - y1) / dy
                ty2 = (oy_max - y1) / dy
                ty_min = min(ty1, ty2)
                ty_max = max(ty1, ty2)

            # Intersection of intervals
            t_min = max(tx_min, ty_min, 0.0)
            t_max = min(tx_max, ty_max, 1.0)

            if t_min <= t_max:
                return True  # collision
        return False  # no collision

    return collide_fn


@pytest.mark.search
@pytest.mark.parametrize("connectivity", [4, 8])
def test_dijkstra_search(grid_map_fixture, connectivity):
    """Test Dijkstra algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture
    graph = grid_to_csr(grid_map, obstacle_value=np.inf, connectivity=connectivity)

    path = Dijkstra.plan(
        start=start,
        target=goal,
        boundary=boundary,
        graph=graph,
        grid_resolution=grid_resolution,
        max_iter=100000,
        callback=None,
    )

    # Assert that a path is found (could be None if no path exists due to obstacles)
    # For a grid with 10% obstacles, there should be a path most of the time
    if path is not None:
        assert path.shape[1] == 2  # path should be Nx2
        assert np.allclose(path[0], start)  # start matches
        assert np.allclose(path[-1], goal)  # goal matches
        # Path should be within boundary
        assert np.all(path[:, 0] >= boundary[0] - 1e-9)
        assert np.all(path[:, 0] < boundary[1] + 1e-9)
        assert np.all(path[:, 1] >= boundary[2] - 1e-9)
        assert np.all(path[:, 1] < boundary[3] + 1e-9)
    # If path is None, it's acceptable (no path exists)


@pytest.mark.search
@pytest.mark.parametrize("connectivity", [4, 8])
def test_a_star_search(grid_map_fixture, connectivity):
    """Test A* algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture
    graph = grid_to_csr(grid_map, obstacle_value=np.inf, connectivity=connectivity)

    path = AStar.plan(
        start=start,
        target=goal,
        boundary=boundary,
        graph=graph,
        heuristic_fn=euclidean_heuristic,
        grid_resolution=grid_resolution,
        max_iter=100000,
        callback=None,
    )

    if path is not None:
        assert path.shape[1] == 2
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)
        assert np.all(path[:, 0] >= boundary[0] - 1e-9)
        assert np.all(path[:, 0] < boundary[1] + 1e-9)
        assert np.all(path[:, 1] >= boundary[2] - 1e-9)
        assert np.all(path[:, 1] < boundary[3] + 1e-9)


@pytest.mark.search
@pytest.mark.parametrize("connectivity", [4, 8])
def test_d_star_search(grid_map_fixture, connectivity):
    """Test D* algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture
    graph = grid_to_csr(grid_map, obstacle_value=np.inf, connectivity=connectivity)

    path = DStar.plan(
        start=start,
        target=goal,
        boundary=boundary,
        graph=graph,
        heuristic_fn=euclidean_heuristic,
        grid_resolution=grid_resolution,
        max_iter=1000000,  # Increased for D* which may need more iterations for 8-connectivity
        callback=None,
    )

    if path is not None:
        assert path.shape[1] == 2
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)
        assert np.all(path[:, 0] >= boundary[0] - 1e-9)
        assert np.all(path[:, 0] < boundary[1] + 1e-9)
        assert np.all(path[:, 1] >= boundary[2] - 1e-9)
        assert np.all(path[:, 1] < boundary[3] + 1e-9)


@pytest.mark.search
@pytest.mark.parametrize("connectivity", [4, 8])
def test_search_algorithms_consistency(grid_map_fixture_with_seed, connectivity):
    """Test that Dijkstra, A* and D* produce consistent results on the same static map.

    For a static grid map, all algorithms should find the same optimal path
    (or agree that no path exists). D* may have performance issues but should
    produce correct results for static environments.
    """
    grid_map, start, goal, boundary, grid_resolution, seed = grid_map_fixture_with_seed
    graph = grid_to_csr(grid_map, obstacle_value=np.inf, connectivity=connectivity)

    # Run all three algorithms with the same input
    dijkstra_path = Dijkstra.plan(
        start=start,
        target=goal,
        boundary=boundary,
        graph=graph,
        grid_resolution=grid_resolution,
        max_iter=100000,
        callback=None,
    )

    a_star_path = AStar.plan(
        start=start,
        target=goal,
        boundary=boundary,
        graph=graph,
        heuristic_fn=euclidean_heuristic,
        grid_resolution=grid_resolution,
        max_iter=100000,
        callback=None,
    )

    d_star_path = DStar.plan(
        start=start,
        target=goal,
        boundary=boundary,
        graph=graph,
        heuristic_fn=euclidean_heuristic,
        grid_resolution=grid_resolution,
        max_iter=1000000,  # Increased for D* due to performance issues
        callback=None,
    )

    # Check consistency: all algorithms should agree on path existence
    dijkstra_has_path = dijkstra_path is not None
    a_star_has_path = a_star_path is not None
    d_star_has_path = d_star_path is not None

    # All algorithms must agree (fundamental correctness)
    if not (dijkstra_has_path == a_star_has_path == d_star_has_path):
        raise AssertionError(
            f"Algorithms disagree on path existence (seed={seed}): "
            f"Dijkstra={dijkstra_has_path}, A*={a_star_has_path}, D*={d_star_has_path}. "
            f"Connectivity={connectivity}. This is a serious algorithm error."
        )

    # If no path exists, both algorithms agree and test passes
    if not dijkstra_has_path:
        return  # No path exists, both algorithms agree

    # Both algorithms found paths - check they are valid and equivalent
    # We allow different but equivalent paths (same start, goal, and approximate cost)

    # First, validate each path individually
    for path, name in zip([dijkstra_path, a_star_path, d_star_path], ["Dijkstra", "A*", "D*"]):
        assert path.shape[1] == 2, f"{name} path shape invalid: {path.shape}"
        assert np.allclose(path[0], start), f"{name} start mismatch"
        assert np.allclose(path[-1], goal), f"{name} goal mismatch"
        # Check path is within boundary
        assert np.all(path[:, 0] >= boundary[0] - 1e-9), f"{name} path outside x_min"
        assert np.all(path[:, 0] < boundary[1] + 1e-9), f"{name} path outside x_max"
        assert np.all(path[:, 1] >= boundary[2] - 1e-9), f"{name} path outside y_min"
        assert np.all(path[:, 1] < boundary[3] + 1e-9), f"{name} path outside y_max"

    # Calculate path costs
    dijkstra_cost = np.sum(np.linalg.norm(np.diff(dijkstra_path, axis=0), axis=1))
    a_star_cost = np.sum(np.linalg.norm(np.diff(a_star_path, axis=0), axis=1))
    d_star_cost = np.sum(np.linalg.norm(np.diff(d_star_path, axis=0), axis=1))

    # Check that all paths have approximately the same total cost
    # Dijkstra provides the optimal cost, A* and D* should match (all are optimal algorithms)
    tolerance = 1e-9
    assert np.isclose(
        dijkstra_cost, a_star_cost, rtol=tolerance, atol=tolerance
    ), f"Path costs differ: Dijkstra={dijkstra_cost:.6f}, A*={a_star_cost:.6f}"
    assert np.isclose(
        dijkstra_cost, d_star_cost, rtol=tolerance, atol=tolerance
    ), f"Path costs differ: Dijkstra={dijkstra_cost:.6f}, D*={d_star_cost:.6f}"

    # Note: We don't require paths to have the same number of waypoints or identical shape
    # Different algorithms can produce different but equivalent paths
    # (e.g., different sequences of diagonal/straight moves with same total cost)
    # This satisfies the requirement to "allow different but equivalent paths"


@pytest.mark.search
def test_rrt_search(grid_map_fixture):
    """Test RRT algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture

    # Extract obstacles from grid map (inf indicates obstacle)
    obstacles = extract_obstacles_from_grid(grid_map)
    collide_fn = create_grid_collision_checker(obstacles)

    # Run RRT planning
    path, tree = RRT.plan(
        start=start,
        target=goal,
        boundary=boundary,
        obstacles=obstacles,
        collide_fn=collide_fn,
        extension_step=1.0,
        max_iter=10000,
        callback=None,
    )

    # Basic validation: if path is found, check its properties
    if path:
        assert len(path) > 0
        # Path should be a list of [x, y] points
        assert isinstance(path, list)
        # Start and goal should match (within tolerance)
        assert np.allclose(path[0], start, atol=1e-9)
        assert np.allclose(path[-1], goal, atol=1e-9)
        # Path should be within boundary
        for point in path:
            x, y = point
            assert boundary[0] - 1e-9 <= x <= boundary[1] + 1e-9
            assert boundary[2] - 1e-9 <= y <= boundary[3] + 1e-9
    # If no path found, that's acceptable (obstacles may block)


@pytest.mark.search
def test_rrt_star_search(grid_map_fixture):
    """Test RRT* algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture

    obstacles = extract_obstacles_from_grid(grid_map)
    collide_fn = create_grid_collision_checker(obstacles)

    # Run RRT* planning with default radius
    path, tree = RRTStar.plan(
        start=start,
        target=goal,
        boundary=boundary,
        obstacles=obstacles,
        collide_fn=collide_fn,
        extension_step=1.0,
        max_iter=10000,
        radius=3.0,
        callback=None,
    )

    if path:
        assert len(path) > 0
        assert isinstance(path, list)
        assert np.allclose(path[0], start, atol=1e-9)
        assert np.allclose(path[-1], goal, atol=1e-9)
        for point in path:
            x, y = point
            assert boundary[0] - 1e-9 <= x <= boundary[1] + 1e-9
            assert boundary[2] - 1e-9 <= y <= boundary[3] + 1e-9


@pytest.mark.search
def test_rrt_connect_search(grid_map_fixture):
    """Test RRT-Connect algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture

    obstacles = extract_obstacles_from_grid(grid_map)
    collide_fn = create_grid_collision_checker(obstacles)

    # Run RRT-Connect planning
    path, trees = RRTConnect.plan(
        start=start,
        target=goal,
        boundary=boundary,
        obstacles=obstacles,
        collide_fn=collide_fn,
        extension_step=1.0,
        max_iter=10000,
        callback=None,
    )

    if path:
        assert len(path) > 0
        assert isinstance(path, list)
        assert np.allclose(path[0], start, atol=1e-9)
        assert np.allclose(path[-1], goal, atol=1e-9)
        for point in path:
            x, y = point
            assert boundary[0] - 1e-9 <= x <= boundary[1] + 1e-9
            assert boundary[2] - 1e-9 <= y <= boundary[3] + 1e-9
        # trees should be a tuple of two trees
        assert isinstance(trees, tuple)
        assert len(trees) == 2


@pytest.mark.search
def test_hybrid_a_star(grid_map_fixture):
    """Test Hybrid A* algorithm on a grid map."""
    grid_map, start, goal, boundary, grid_resolution = grid_map_fixture

    # Convert 2D start and goal to 3D states with heading = 0
    start_3d = [start[0], start[1], 0.0]  # [x, y, heading]
    goal_3d = [goal[0], goal[1], 0.0]  # [x, y, heading]

    # Extract obstacles from grid map (inf indicates obstacle)
    obstacles = extract_obstacles_from_grid(grid_map)
    collide_fn = create_grid_collision_checker(obstacles)

    # Run Hybrid A* planning
    path = HybridAStar.plan(
        start=start_3d,
        target=goal_3d,
        boundary=boundary,
        obstacles=obstacles,
        collide_fn=collide_fn,
        step_size=1.0,
        max_iter=50000,
        steering_angles=[-0.5, 0, 0.5],  # radians
        velocity=1.0,
        wheelbase=2.5,
    )

    # Basic validation: if path is found, check its properties
    if path:
        assert len(path) > 0
        assert isinstance(path, list)
        # Path should be a list of [x, y, heading] states
        # Check that first state matches start (position only, heading may differ)
        assert np.allclose(path[0][:2], start_3d[:2], atol=0.5)
        # Check that last state matches goal (position only)
        assert np.allclose(path[-1][:2], goal_3d[:2], atol=0.5)
        # Path should be within boundary
        for state in path:
            x, y, heading = state
            assert boundary[0] - 1e-9 <= x <= boundary[1] + 1e-9
            assert boundary[2] - 1e-9 <= y <= boundary[3] + 1e-9
            # Heading should be in valid range [0, 2π)
            assert 0.0 <= heading < 2 * np.pi or np.isclose(heading, 2 * np.pi)
    # If no path found, that's acceptable (obstacles may block)
