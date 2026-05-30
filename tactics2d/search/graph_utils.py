# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph conversion utilities for search algorithms."""

import numpy as np
from scipy.sparse import csr_matrix


def grid_to_csr(
    weight_grid: np.ndarray,
    obstacle_value: float = None,
    connectivity: int = 4,
    diagonal_cost_multiplier: float = 1.4142135623730951,
) -> csr_matrix:
    """
    Convert a 2D weight grid to CSR sparse graph for Dijkstra/A* algorithms.

    The weight grid represents traversal costs for each cell. Edge weights between
    adjacent cells are computed as the average of the two cell weights, multiplied
    by the distance factor (1.0 for orthogonal moves, diagonal_cost_multiplier for
    diagonal moves).

    Args:
        weight_grid: 2D numpy array of shape (height, width) containing traversal
            costs. Use obstacle_value to mark impassable cells.
        obstacle_value: Value representing obstacles (default None). Cells with this
            value are treated as impassable (infinite cost).
        connectivity: 4 or 8, defines neighbor connections. 4 considers only
            orthogonal moves (up, down, left, right). 8 includes diagonal moves.
        diagonal_cost_multiplier: Multiplier for diagonal edge weights, typically
            sqrt(2) ≈ 1.414 for Euclidean distance.

    Returns:
        csr_matrix: Sparse adjacency matrix of shape (N, N) where N = height * width.
            Edge weights represent traversal costs between adjacent cells.

    Raises:
        ValueError: If connectivity is not 4 or 8, or grid is not 2D.
        TypeError: If weight_grid is not a numpy array.

    Example:
        >>> import numpy as np
        >>> grid = np.array([[1.0, 2.0, 3.0],
        ...                  [4.0, None, 5.0],
        ...                  [6.0, 7.0, 8.0]])
        >>> graph = grid_to_csr(grid, obstacle_value=None, connectivity=4)
    """
    if not isinstance(weight_grid, np.ndarray):
        raise TypeError(f"weight_grid must be numpy array, got {type(weight_grid)}")

    if weight_grid.ndim != 2:
        raise ValueError(f"weight_grid must be 2D array, got shape {weight_grid.shape}")

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    height, width = weight_grid.shape
    N = height * width

    # Create a copy with obstacles replaced by np.inf
    grid = weight_grid.copy()
    if obstacle_value is not None:
        # Replace obstacle_value with np.inf
        grid[grid == obstacle_value] = np.inf
    else:
        # Handle None values (common case)
        mask = np.vectorize(lambda x: x is None)(grid)
        grid[mask] = np.inf

    # Ensure numeric dtype for calculations
    grid = grid.astype(float)
    # Convert any NaN values to inf (from None values that didn't get converted)
    grid[np.isnan(grid)] = np.inf

    # Precompute linear indices for all cells
    linear_indices = np.arange(N).reshape(height, width)

    # Directions for orthogonal moves
    orthogonal_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Directions for diagonal moves (only used if connectivity == 8)
    diagonal_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Collect edges: row indices, column indices, and weights
    rows = []
    cols = []
    weights = []

    for i in range(height):
        for j in range(width):
            idx = linear_indices[i, j]
            current_weight = grid[i, j]

            # Skip if current cell is obstacle (infinite cost)
            if np.isinf(current_weight):
                continue

            # Check orthogonal neighbors
            for di, dj in orthogonal_dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_idx = linear_indices[ni, nj]
                    # Only add edge once (from lower index to higher index)
                    if neighbor_idx <= idx:
                        continue
                    neighbor_weight = grid[ni, nj]

                    # Skip if neighbor is obstacle
                    if np.isinf(neighbor_weight):
                        continue

                    # Edge weight = average of cell weights * distance factor (1.0 for orthogonal)
                    edge_weight = (current_weight + neighbor_weight) / 2.0

                    rows.append(idx)
                    cols.append(neighbor_idx)
                    weights.append(edge_weight)

            # Check diagonal neighbors if connectivity == 8
            if connectivity == 8:
                for di, dj in diagonal_dirs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = linear_indices[ni, nj]
                        # Only add edge once (from lower index to higher index)
                        if neighbor_idx <= idx:
                            continue
                        neighbor_weight = grid[ni, nj]

                        if np.isinf(neighbor_weight):
                            continue

                        # Edge weight = average * diagonal distance multiplier
                        edge_weight = (
                            (current_weight + neighbor_weight) / 2.0 * diagonal_cost_multiplier
                        )

                        rows.append(idx)
                        cols.append(neighbor_idx)
                        weights.append(edge_weight)

    # Build CSR matrix (symmetric edges added only once for undirected graph)
    # Note: This creates a directed graph; for undirected, add reverse edges too
    # But search algorithms typically work with directed graphs where edge (i,j) implies (j,i)
    # We'll create symmetric matrix for compatibility with current Dijkstra/AStar
    symmetric_rows = rows + cols
    symmetric_cols = cols + rows
    symmetric_weights = weights + weights

    return csr_matrix((symmetric_weights, (symmetric_rows, symmetric_cols)), shape=(N, N))
