# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Route result data structures."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RouteSegment:
    """A routing segment represented by one routed lane.

    Attributes:
        lane_id: The unique lane identifier in the source map.
        relation_type: The transition type used to enter this lane in the route,
            such as ``start``, ``successor``, or ``neighbor``.
        cost: The incremental routing cost associated with this segment.
        centerline: The geometric centerline of this routed lane, if available.
    """

    lane_id: str
    relation_type: str = "successor"
    cost: float = 0.0
    centerline: Optional[np.ndarray] = None


@dataclass
class Route:
    """A lane-level routing result.

    Attributes:
        start: Input start position in world coordinates.
        goal: Input goal position in world coordinates.
        start_lane_id: The lane matched from ``start``.
        goal_lane_id: The lane matched from ``goal``.
        lane_ids: Ordered lane identifiers of the planned route. Their order is
            the route sequence shown in the notebook visualization.
        segments: Structured per-lane route segments aligned with ``lane_ids``.
        path: Concatenated geometric path built from routed lane centerlines.
            This is the visualized route polyline, not a list of topology nodes.
        total_cost: Total routing cost returned by the search algorithm.
    """

    start: Tuple[float, float]
    goal: Tuple[float, float]
    start_lane_id: Optional[str] = None
    goal_lane_id: Optional[str] = None
    lane_ids: List[str] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)
    path: Optional[np.ndarray] = None
    total_cost: float = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.lane_ids) == 0
