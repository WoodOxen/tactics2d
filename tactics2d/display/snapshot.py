# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene snapshot implementation for unified display backend."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


@dataclasses.dataclass
class RoadElement:
    """Static road network element (lane, area, road line, junction, etc.).

    Attributes:
        id_: Unique identifier for this road element.
        shape: Geometry shape type ("polygon" | "line").
        geometry: Vertex coordinates in world space.
        type_: Semantic type ("lane" | "road" | "crosswalk" | "roadline" | "curbstone" | ...).
        subtype: Optional subtype (e.g. "driving", "parking", "sidewalk").
        color: Optional color hint or semantic color key.
        line_width: Stroke width for line elements.
        line_style: Line style ("solid" | "dashed" | "solid_dashed" | ...).
        z_order: Rendering z-order (higher = on top).
    """

    id_: str | int
    shape: str  # "polygon" | "line"
    geometry: list[tuple[float, float]]
    type_: str
    subtype: str | None = None
    color: str | None = None
    line_width: float = 1.0
    line_style: str | None = None
    z_order: int = 1


@dataclasses.dataclass
class ParticipantElement:
    """Dynamic traffic participant (vehicle, cyclist, pedestrian).

    Attributes:
        id_: Unique identifier.
        shape: Geometry shape type ("polygon" | "circle").
        geometry: Footprint vertices in local/world coordinates.
        position: World position (x, y).
        rotation: Heading angle in radians.
        type_: Entity type ("vehicle" | "cyclist" | "pedestrian").
        color: Optional color hint or semantic color key.
        velocity: Optional velocity vector (vx, vy) in m/s.
        length: Vehicle length in meters.
        width: Vehicle width in meters.
    """

    id_: str | int
    shape: str  # "polygon" | "circle"
    geometry: list[tuple[float, float]]
    position: tuple[float, float]
    rotation: float
    type_: str
    color: str | None = None
    velocity: tuple[float, float] | None = None
    length: float | None = None
    width: float | None = None


@dataclasses.dataclass
class PointCloudElement:
    """Point cloud data from LiDAR or depth sensor.

    Attributes:
        id_: Unique identifier.
        points: Point coordinates (Nx2), in world or sensor-local space.
        color: Point color.
        point_size: Point rendering size.
        alpha: Point opacity.
        coordinate_frame: "world" or "sensor".
    """

    id_: str | int
    points: list[tuple[float, float]]
    color: str = "red"
    point_size: float = 2.0
    alpha: float = 0.8
    coordinate_frame: str = "world"


@dataclasses.dataclass
class TrafficLightState:
    """Traffic light signal state.

    Attributes:
        id_: Unique identifier.
        position: World position.
        state: Signal state ("red" | "yellow" | "green" | "red_yellow" | "off").
        heading: Optional heading of the signal head.
    """

    id_: str | int
    position: tuple[float, float]
    state: str  # "red" | "yellow" | "green" | "red_yellow" | "off"
    heading: float | None = None


@dataclasses.dataclass
class CameraMetadata:
    """Sensor / camera metadata that controls the viewport.

    Attributes:
        id_: Unique identifier.
        position: Camera world position (x, y).
        yaw: Camera yaw angle in radians.
        perception_range: Maximum detection distance in meters.
        viewport_aspect: Viewport aspect ratio (width / height).
    """

    id_: str
    position: tuple[float, float]
    yaw: float
    perception_range: float
    viewport_aspect: float = 16.0 / 9.0


@dataclasses.dataclass
class SceneSnapshot:
    """Unified scene snapshot consumed by all display backends.

    This dataclass holds a complete description of a single simulation frame.
    It does **not** reference any env / simulator internal objects (Map,
    Participant, Trajectory, State, shapely geometries).  All geometry data
    is stored as plain Python lists and tuples, making it JSON-serializable.

    The snapshot supports both **full-state** (all elements present) and
    **incremental** (only changed elements with removal lists) modes.
    """

    version: str = "1.0"
    """Schema version string for frontend/backend compatibility."""

    frame: int = 0
    """Frame index (simulation step count)."""

    timestamp_ms: int | None = None
    """Millisecond timestamp of this frame (dataset time)."""

    scene_name: str | None = None
    """Human-readable scene / recording name."""

    road_elements: dict[str | int, RoadElement] = dataclasses.field(default_factory=dict)
    """All visible road network elements, keyed by id."""

    road_ids_to_remove: list[str | int] = dataclasses.field(default_factory=list)
    """Road element ids that should be removed (incremental update)."""

    participants: dict[str | int, ParticipantElement] = dataclasses.field(default_factory=dict)
    """All visible participants, keyed by id."""

    participant_ids_to_create: list[str | int] = dataclasses.field(default_factory=list)
    """Participant ids that are new in this frame (incremental create)."""

    participant_ids_to_remove: list[str | int] = dataclasses.field(default_factory=list)
    """Participant ids that should be removed (incremental update)."""

    point_clouds: list[PointCloudElement] = dataclasses.field(default_factory=list)
    """Point clouds (replaced fully each frame)."""

    traffic_lights: dict[str | int, TrafficLightState] = dataclasses.field(default_factory=dict)
    """Traffic light states, keyed by id."""

    cameras: list[CameraMetadata] = dataclasses.field(default_factory=list)
    """Camera / sensor metadata controlling viewports."""

    ego_participant_id: str | int | None = None
    """Ego participant id (used for highlighting)."""

    debug_overlays: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Optional debug visualisation data."""

    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Arbitrary extension metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary (for browser frontend HTTP transport)."""
        return json.loads(self.to_json())

    def to_json(self) -> str:
        """Serialize to a JSON string (for WebSocket transport)."""
        return json.dumps(dataclasses.asdict(self), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSnapshot:
        """Deserialize from a dictionary (for testing / replay)."""

        def _dict_to_typed(obj, cls_type):
            if isinstance(obj, dict) and not isinstance(obj, cls_type):
                field_set = {f.name for f in dataclasses.fields(cls_type)}
                filtered = {k: v for k, v in obj.items() if k in field_set}
                return cls_type(**filtered)
            return obj

        road_elements = {
            k: _dict_to_typed(v, RoadElement) for k, v in data.get("road_elements", {}).items()
        }
        participants = {
            k: _dict_to_typed(v, ParticipantElement)
            for k, v in data.get("participants", {}).items()
        }
        point_clouds = [
            _dict_to_typed(pc, PointCloudElement) for pc in data.get("point_clouds", [])
        ]
        traffic_lights = {
            k: _dict_to_typed(v, TrafficLightState)
            for k, v in data.get("traffic_lights", {}).items()
        }
        cameras = [_dict_to_typed(c, CameraMetadata) for c in data.get("cameras", [])]

        return cls(
            version=data.get("version", "1.0"),
            frame=data.get("frame", 0),
            timestamp_ms=data.get("timestamp_ms"),
            scene_name=data.get("scene_name"),
            road_elements=road_elements,
            road_ids_to_remove=data.get("road_ids_to_remove", []),
            participants=participants,
            participant_ids_to_create=data.get("participant_ids_to_create", []),
            participant_ids_to_remove=data.get("participant_ids_to_remove", []),
            point_clouds=point_clouds,
            traffic_lights=traffic_lights,
            cameras=cameras,
            ego_participant_id=data.get("ego_participant_id"),
            debug_overlays=data.get("debug_overlays", {}),
            extra=data.get("extra", {}),
        )
