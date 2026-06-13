# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""TypedDict definitions for geometry_data shared between sensor and display."""

from __future__ import annotations

from typing import TypedDict


class RoadElement(TypedDict, total=False):
    """A single road element in the map_data."""

    id: str
    shape: str  # "polygon" | "line"
    type: str
    geometry: list[list[float]]
    color: str | None
    line_width: float
    line_style: str | None


class MapData(TypedDict, total=False):
    """Map-related data in geometry_data."""

    road_id_to_remove: list[str]
    road_elements: list[RoadElement]


class ParticipantElement(TypedDict, total=False):
    """A single participant in the participant_data."""

    id: str
    shape: str  # "polygon" | "circle"
    type: str
    geometry: list[list[float]]
    position: list[float]
    rotation: float
    color: str | None
    line_width: float
    velocity: list[float] | None


class PointCloudElement(TypedDict, total=False):
    """A point cloud element in participant_data."""

    id: str
    points: list[list[float]]
    color: str
    point_size: float
    alpha: float
    type: str


class ParticipantData(TypedDict, total=False):
    """Participant-related data in geometry_data."""

    participant_id_to_create: list[str]
    participant_id_to_remove: list[str]
    participants: list[ParticipantElement]
    point_clouds: list[PointCloudElement]


class Metadata(TypedDict, total=False):
    """Camera/sensor metadata in geometry_data."""

    sensor_position: list[float]
    sensor_yaw: float
    perception_range: float | None


class GeometryData(TypedDict, total=False):
    """Top-level geometry_data structure shared between sensor and display.

    This dict is produced by sensor ``update()`` and consumed by display
    renderers (MatplotlibRenderer, browser frontend, etc.).
    """

    metadata: Metadata
    map_data: MapData
    participant_data: ParticipantData
