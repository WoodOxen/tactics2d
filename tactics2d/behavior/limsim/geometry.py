# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry helpers for LimSim-style behavior planning."""

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Polygon

from .schema import AgentDecisionState


def footprint(state: AgentDecisionState) -> Polygon:
    """Build an oriented rectangular footprint for a predicted agent state."""

    length = max(float(state.length), 0.1)
    width = max(float(state.width), 0.1)
    bbox = LinearRing(
        [
            [0.5 * length, -0.5 * width],
            [0.5 * length, 0.5 * width],
            [-0.5 * length, 0.5 * width],
            [-0.5 * length, -0.5 * width],
        ]
    )
    transform = [
        np.cos(state.heading),
        -np.sin(state.heading),
        np.sin(state.heading),
        np.cos(state.heading),
        state.x,
        state.y,
    ]
    return Polygon(affine_transform(bbox, transform))

