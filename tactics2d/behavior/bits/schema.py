# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data schemas for BITS-style imitation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BitsRaster:
    """Rasterized map context for a BITS sample."""

    image: np.ndarray
    drivable_map: np.ndarray
    raster_from_agent: np.ndarray
    agent_from_raster: np.ndarray
    static_image: Optional[np.ndarray] = None
    dynamic_image: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        """Return raster fields using the BITS/TBSIM batch key style."""

        return {
            "image": self.image,
            "drivable_map": self.drivable_map,
            "raster_from_agent": self.raster_from_agent,
            "agent_from_raster": self.agent_from_raster,
            "static_image": self.static_image,
            "dynamic_image": self.dynamic_image,
        }


@dataclass(frozen=True)
class BitsBatch:
    """A single agent-centric BITS training or inference sample.

    Positions are expressed in the current ego frame. The ego current pose is
    at the origin, with the x-axis aligned to the ego heading and y-axis to the
    ego's left. Yaws are relative to the ego heading.
    """

    ego_id: object
    frame: int
    history_positions: np.ndarray
    history_yaws: np.ndarray
    history_availabilities: np.ndarray
    target_positions: np.ndarray
    target_yaws: np.ndarray
    target_availabilities: np.ndarray
    curr_speed: float
    centroid: np.ndarray
    yaw: float
    extent: np.ndarray
    type: int
    agent_from_world: np.ndarray
    world_from_agent: np.ndarray
    all_other_agents_history_positions: np.ndarray
    all_other_agents_history_yaws: np.ndarray
    all_other_agents_history_availability: np.ndarray
    all_other_agents_future_positions: np.ndarray
    all_other_agents_future_yaws: np.ndarray
    all_other_agents_future_availability: np.ndarray
    all_other_agents_curr_speed: np.ndarray
    all_other_agents_types: np.ndarray
    all_other_agents_extents: np.ndarray
    all_other_agents_history_extents: np.ndarray
    agent_ids: List[object] = field(default_factory=list)
    lane_id: Optional[str] = None
    image: Optional[np.ndarray] = None
    drivable_map: Optional[np.ndarray] = None
    raster_from_agent: Optional[np.ndarray] = None
    agent_from_raster: Optional[np.ndarray] = None
    static_image: Optional[np.ndarray] = None
    dynamic_image: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        """Return a dictionary following the BITS/TBSIM batch key style."""

        return {
            "ego_id": self.ego_id,
            "frame": self.frame,
            "history_positions": self.history_positions,
            "history_yaws": self.history_yaws,
            "history_availabilities": self.history_availabilities,
            "target_positions": self.target_positions,
            "target_yaws": self.target_yaws,
            "target_availabilities": self.target_availabilities,
            "curr_speed": self.curr_speed,
            "centroid": self.centroid,
            "yaw": self.yaw,
            "extent": self.extent,
            "type": self.type,
            "agent_from_world": self.agent_from_world,
            "world_from_agent": self.world_from_agent,
            "all_other_agents_history_positions": self.all_other_agents_history_positions,
            "all_other_agents_history_yaws": self.all_other_agents_history_yaws,
            "all_other_agents_history_availability": self.all_other_agents_history_availability,
            "all_other_agents_future_positions": self.all_other_agents_future_positions,
            "all_other_agents_future_yaws": self.all_other_agents_future_yaws,
            "all_other_agents_future_availability": self.all_other_agents_future_availability,
            "all_other_agents_curr_speed": self.all_other_agents_curr_speed,
            "all_other_agents_types": self.all_other_agents_types,
            "all_other_agents_extents": self.all_other_agents_extents,
            "all_other_agents_history_extents": self.all_other_agents_history_extents,
            "agent_ids": self.agent_ids,
            "lane_id": self.lane_id,
            "image": self.image,
            "drivable_map": self.drivable_map,
            "raster_from_agent": self.raster_from_agent,
            "agent_from_raster": self.agent_from_raster,
            "static_image": self.static_image,
            "dynamic_image": self.dynamic_image,
        }

    @property
    def history_shape(self) -> Tuple[int, ...]:
        """Shape of ego history positions, useful in tests and diagnostics."""

        return self.history_positions.shape

    @property
    def future_shape(self) -> Tuple[int, ...]:
        """Shape of ego future positions, useful in tests and diagnostics."""

        return self.target_positions.shape
