# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Supervision utilities for BITS-style imitation."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .schema import BitsBatch


@dataclass(frozen=True)
class BitsGoalSupervision:
    """Spatial goal supervision for the BITS high-level planner."""

    goal_position: np.ndarray
    goal_yaw: np.ndarray
    goal_index: int
    goal_position_pixel: Optional[np.ndarray] = None
    goal_position_pixel_flat: Optional[int] = None
    goal_position_residual: Optional[np.ndarray] = None
    goal_spatial_map: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "goal_position": self.goal_position,
            "goal_yaw": self.goal_yaw,
            "goal_index": self.goal_index,
            "goal_position_pixel": self.goal_position_pixel,
            "goal_position_pixel_flat": self.goal_position_pixel_flat,
            "goal_position_residual": self.goal_position_residual,
            "goal_spatial_map": self.goal_spatial_map,
        }


def build_goal_supervision(batch: BitsBatch) -> BitsGoalSupervision:
    """Build goal supervision from the last available future state."""

    available = np.flatnonzero(batch.target_availabilities)
    if available.size == 0:
        raise ValueError("Cannot build goal supervision without available future targets.")

    goal_index = int(available[-1])
    goal_position = np.asarray(batch.target_positions[goal_index], dtype=float).copy()
    goal_yaw = np.asarray(batch.target_yaws[goal_index], dtype=float).copy()

    if batch.raster_from_agent is None or batch.image is None:
        return BitsGoalSupervision(
            goal_position=goal_position,
            goal_yaw=goal_yaw,
            goal_index=goal_index,
        )

    height, width = batch.image.shape[-2:]
    raster_position = _transform_point(goal_position, batch.raster_from_agent)
    clipped = np.asarray(
        [
            np.clip(raster_position[0], 0.0, width - 1e-5),
            np.clip(raster_position[1], 0.0, height - 1e-5),
        ],
        dtype=float,
    )
    goal_pixel = np.floor(clipped).astype(int)
    residual = clipped - goal_pixel.astype(float)
    flat = int(goal_pixel[1] * width + goal_pixel[0])
    spatial_map = np.zeros((height, width), dtype=np.float32)
    spatial_map[goal_pixel[1], goal_pixel[0]] = 1.0

    # BITS spatial planner supervision decomposes a continuous goal into
    # pixel classification plus within-pixel residual regression.
    return BitsGoalSupervision(
        goal_position=goal_position,
        goal_yaw=goal_yaw,
        goal_index=goal_index,
        goal_position_pixel=goal_pixel,
        goal_position_pixel_flat=flat,
        goal_position_residual=residual,
        goal_spatial_map=spatial_map,
    )


def _transform_point(point, transform: np.ndarray) -> np.ndarray:
    transformed = transform @ np.asarray([point[0], point[1], 1.0], dtype=float)
    return transformed[:2]
