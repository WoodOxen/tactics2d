# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Tensor-level graph primitives for radius queries and sparse ops."""

import warnings
from typing import Optional

import torch

# Chunk size for the pairwise distance sweep, in rows of the query set, and the
# cap on the ``chunk x N x F`` product it is allowed to allocate.
_DEFAULT_CHUNK = 512
_DISTANCE_BUDGET = 8_000_000


def _squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the squared Euclidean distance between every pair of rows.

    Args:
        a (torch.Tensor): Rows of shape ``(M, F)``.
        b (torch.Tensor): Rows of shape ``(N, F)``.

    Returns:
        A ``(M, N)`` tensor.
    """

    return (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)


def _chunk_rows(num_queries: int, num_candidates: int, feature_dim: int, cap: int) -> int:
    """Return how many queries one distance sweep may hold.

    Args:
        num_queries (int): Number of query points.
        num_candidates (int): Number of candidate points.
        feature_dim (int): Number of columns per point.
        cap (int): Largest row count to use.

    Returns:
        A row count of at least one.
    """

    per_row = max(1, num_candidates * feature_dim)
    return max(1, min(num_queries, cap, _DISTANCE_BUDGET // per_row))


def _segment_offsets(batch: torch.Tensor) -> torch.Tensor:
    """Return cumulative segment boundaries of a sorted batch vector.

    Empty segments are included.

    Args:
        batch (torch.Tensor): Sorted non-negative assignment vector.

    Returns:
        A ``(B + 1,)`` tensor of segment start offsets.
    """

    batch_size = int(batch.max()) + 1
    boundaries = torch.arange(batch_size + 1, device=batch.device)
    return torch.bucketize(boundaries, batch)


def _radius_single(
    x: torch.Tensor, y: torch.Tensor, r: float, max_num_neighbors: Optional[int], chunk_size: int
) -> torch.Tensor:
    """Solve one radius query for a single, unbatched pair of point sets.

    Keeper of the nearest neighbours, ties broken by the lower index; the
    predicate is strict (``distance < r``) over all feature columns.

    Args:
        x (torch.Tensor): Candidate neighbours of shape ``(N, F)``.
        y (torch.Tensor): Query points of shape ``(M, F)``.
        r (float): Radius in the units of ``x`` and ``y``.
        max_num_neighbors (Optional[int]): Per-query cap. None keeps every
            neighbour.
        chunk_size (int): Number of queries per distance sweep.

    Returns:
        An ``(2, E)`` index tensor where row 0 selects ``y`` and row 1
        selects ``x``, grouped by query index and, inside a group, by
        ascending distance.
    """

    empty = torch.zeros((2, 0), dtype=torch.long, device=x.device)
    if x.numel() == 0 or y.numel() == 0:
        return empty

    chunk = _chunk_rows(y.shape[0], x.shape[0], x.shape[1], chunk_size)
    threshold = r * r
    rows = []
    cols = []
    for start in range(0, y.shape[0], chunk):
        stop = min(start + chunk, y.shape[0])
        query = y[start:stop]
        # Filter before ranking, so the sorts see O(hits) elements.
        local_rows, local_cols = (_squared_distances(query, x) < threshold).nonzero(as_tuple=True)
        if local_rows.numel() == 0:
            continue
        # ``nonzero`` yields (row, col) ascending; a stable sort by distance
        # then ranks each query's hits nearest-first, ties by lower index.
        distance = (query[local_rows] - x[local_cols]).pow(2).sum(-1)
        order = torch.sort(distance, stable=True).indices
        local_rows, local_cols = local_rows[order], local_cols[order]
        # Stable sort by query index keeps the distance order inside a group.
        order = torch.sort(local_rows, stable=True).indices
        local_rows, local_cols = local_rows[order], local_cols[order]
        if max_num_neighbors is not None:
            counts = torch.bincount(local_rows, minlength=query.shape[0])
            offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
            rank = torch.arange(local_rows.numel(), device=local_rows.device) - offsets[local_rows]
            keep = rank < max_num_neighbors
            local_rows, local_cols = local_rows[keep], local_cols[keep]
        rows.append(local_rows + start)
        cols.append(local_cols)
    if not rows:
        return empty
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: Optional[int] = 32,
    chunk_size: int = _DEFAULT_CHUNK,
) -> torch.Tensor:
    """Find, for each point in ``y``, every point in ``x`` within ``r``.

    Row 0 of the result indexes ``y`` and row 1 indexes ``x``.

    Args:
        x (torch.Tensor): Candidate neighbours of shape ``(N, F)``.
        y (torch.Tensor): Query points of shape ``(M, F)``.
        r (float): Radius.
        batch_x (Optional[torch.Tensor], optional): Assignment vector for ``x``. Defaults to None.
        batch_y (Optional[torch.Tensor], optional): Assignment vector for ``y``. Defaults to None.
        max_num_neighbors (Optional[int], optional): Per-query cap on kept
            neighbours. Defaults to 32.
        chunk_size (int, optional): Maximum queries per distance sweep. Defaults to 512.

    Returns:
        An ``(2, E)`` index tensor.

    Raises:
        AssertionError: If a batch vector does not match its point set.
    """

    if batch_x is None and batch_y is None:
        return _radius_single(x, y, r, max_num_neighbors, chunk_size)

    batch_size = 1
    if batch_x is not None:
        assert x.shape[0] == batch_x.numel()
        batch_size = int(batch_x.max()) + 1
    if batch_y is not None:
        assert y.shape[0] == batch_y.numel()
        batch_size = max(batch_size, int(batch_y.max()) + 1)

    ptr_x = _segment_offsets(batch_x) if batch_x is not None else None
    ptr_y = _segment_offsets(batch_y) if batch_y is not None else None

    rows = []
    cols = []
    for index in range(batch_size):
        x_start = int(ptr_x[index]) if ptr_x is not None else 0
        x_stop = int(ptr_x[index + 1]) if ptr_x is not None else x.shape[0]
        y_start = int(ptr_y[index]) if ptr_y is not None else 0
        y_stop = int(ptr_y[index + 1]) if ptr_y is not None else y.shape[0]
        if x_start == x_stop or y_start == y_stop:
            continue
        local = _radius_single(
            x[x_start:x_stop], y[y_start:y_stop], r, max_num_neighbors, chunk_size
        )
        rows.append(local[0] + y_start)
        cols.append(local[1] + x_start)
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: Optional[int] = 32,
    chunk_size: int = _DEFAULT_CHUNK,
) -> torch.Tensor:
    """Connect every point to the points of ``x`` within ``r`` of it.

    The underlying query is capped at ``max_num_neighbors + 1`` when
    ``loop=False``, so a node keeps at most ``max_num_neighbors`` others.

    Args:
        x (torch.Tensor): Points of shape ``(N, F)``.
        r (float): Radius.
        batch (Optional[torch.Tensor], optional): Assignment vector. Defaults to None.
        loop (bool, optional): Whether to keep self-loops. Defaults to False.
        max_num_neighbors (Optional[int], optional): Per-node cap on kept
            neighbours. Defaults to 32.
        chunk_size (int, optional): Maximum queries per distance sweep. Defaults to 512.

    Returns:
        An ``(2, E)`` index tensor where row 0 is the neighbour and row 1 is
        the centre node.
    """

    inflated = None if max_num_neighbors is None else max_num_neighbors + (0 if loop else 1)
    edge_index = radius(x, x, r, batch, batch, inflated, chunk_size)
    row, col = edge_index[1], edge_index[0]
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def dense_to_sparse(mask: torch.Tensor) -> torch.Tensor:
    """Return the edge index of a dense adjacency, with 3-D batching.

    A ``(A, S, S)`` mask is flattened to ``(a * S + i, a * S + j)`` pairs.

    Args:
        mask (torch.Tensor): Boolean adjacency of shape ``(N, N)`` or ``(A, S, S)``.

    Returns:
        A ``(2, E)`` index tensor.

    Raises:
        ValueError: If the mask is not 2-D or 3-D.
    """

    if mask.dim() == 3:
        batch_src, index, neighbours = mask.nonzero(as_tuple=True)
        source = batch_src * mask.shape[1] + index
        target = batch_src * mask.shape[2] + neighbours
        return torch.stack([source, target], dim=0)
    if mask.dim() == 2:
        return mask.nonzero().t()
    raise ValueError("dense_to_sparse expects a 2-D or 3-D mask.")


def subgraph(
    subset: torch.Tensor, edge_index: torch.Tensor, relabel_nodes: bool = False
) -> torch.Tensor:
    """Keep only the edges whose two endpoints are both in ``subset``.

    Args:
        subset (torch.Tensor): Boolean node mask of shape ``(N,)``.
        edge_index (torch.Tensor): ``(2, E)`` index tensor.
        relabel_nodes (bool, optional): Whether to compress the surviving
            node ids to ``[0, K)``. Defaults to False.

    Returns:
        The filtered ``(2, E')`` index tensor.

    Raises:
        ValueError: If ``subset`` is not a boolean mask.
    """

    if subset.dtype != torch.bool:
        raise ValueError("subgraph expects a boolean node mask for 'subset'.")
    edge_mask = subset[edge_index[0]] & subset[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    if relabel_nodes:
        index = torch.full_like(subset, -1, dtype=torch.long)
        index[subset] = torch.arange(int(subset.sum()), device=subset.device)
        edge_index = index[edge_index]
    return edge_index


def segment_softmax(
    src: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None
) -> torch.Tensor:
    """Normalize ``src`` independently inside each group of ``index``.

    ``1e-16`` is added to the group sum, so a group with no members yields all
    zeros.

    Args:
        src (torch.Tensor): Values of shape ``(E, ...)``.
        index (torch.Tensor): Group assignment of shape ``(E,)``.
        num_nodes (Optional[int], optional): Number of groups; defaults to max index + 1.

    Returns:
        Normalized values with the same shape as ``src``.
    """

    if num_nodes is None:
        num_nodes = int(index.max()) + 1 if index.numel() else 0
    expanded = index
    for _ in range(src.dim() - 1):
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(src)
    shape = (num_nodes,) + src.shape[1:]
    # ``scatter_reduce_`` is the only core-torch per-group maximum; it warns as
    # beta on torch 1.12, which the caller does not need to hear.
    group_max = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group_max.scatter_reduce_(0, expanded, src.detach(), reduce="amax", include_self=True)
    # A group with no members shifts by zero, not by -inf.
    group_max = torch.where(torch.isinf(group_max), torch.zeros_like(group_max), group_max)
    weights = (src - group_max.gather(0, expanded)).exp()
    total = torch.zeros(shape, dtype=src.dtype, device=src.device)
    total = total.scatter_add_(0, expanded, weights) + 1e-16
    return weights / total.gather(0, expanded)
