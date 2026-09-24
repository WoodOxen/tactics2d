# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tensor transforms shared by BITS network components."""

import torch


def add_batch_dim(tensor, min_ndim: int):
    """Add leading singleton dimensions until ``min_ndim`` is reached."""

    while tensor.ndim < min_ndim:
        tensor = tensor.unsqueeze(0)
    return tensor


def homogeneous_transform(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 homogeneous transform to batched two-dimensional points."""

    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    transform = matrix.to(device=points.device, dtype=points.dtype)
    return torch.matmul(homogeneous, transform.transpose(1, 2))[..., :2]
