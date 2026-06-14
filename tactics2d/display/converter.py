# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene snapshot converter — translates simulator state to SceneSnapshot."""

from __future__ import annotations

from typing import Any

from shapely.geometry import Point

from tactics2d.display.sensor import BEVCamera
from tactics2d.map.element import Map
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle

from .snapshot import (
    CameraMetadata,
    ParticipantElement,
    PointCloudElement,
    RoadElement,
    SceneSnapshot,
)


class SceneSnapshotConverter:
    """Convert simulator / env internal state to :class:`SceneSnapshot`.

    This converter maintains incremental state across frames so that
    ``participant_ids_to_create`` and ``participant_ids_to_remove`` are
    correctly populated.

    Usage::

        converter = SceneSnapshotConverter()
        snapshot = converter.convert_from_simulator(participants, ids, map_, frame)
        snapshot = converter.convert_from_camera(camera, ...)
    """

    def __init__(self) -> None:
        self._prev_participant_ids: set[int] = set()

    def reset(self) -> None:
        """Reset incremental tracking (call on env reset)."""
        self._prev_participant_ids.clear()

    # ------------------------------------------------------------------
    # Public high-level converters
    # ------------------------------------------------------------------

    def convert_from_simulator(
        self,
        participants: dict[int, Any],
        participant_ids: list[int],
        map_: Map,
        frame: int,
        position: Point | None = None,
        heading: float = 0.0,
    ) -> SceneSnapshot:
        """Build a snapshot from raw simulator objects.

        Args:
            participants: All participants dict (id → participant).
            participant_ids: Active participant ids for this frame.
            map_: The current map.
            frame: Current frame number.
            position: Optional camera position.
            heading: Optional camera heading.

        Returns:
            A :class:`SceneSnapshot` for this frame.
        """
        road_elements = _build_road_elements(map_)
        participant_elements, created_ids, removed_ids = self._build_participant_elements(
            participants, participant_ids
        )
        cameras = _build_camera_metadata(position, heading) if position is not None else []

        return SceneSnapshot(
            frame=frame,
            road_elements=road_elements,
            participants=participant_elements,
            participant_ids_to_create=created_ids,
            participant_ids_to_remove=removed_ids,
            cameras=cameras,
        )

    def convert_from_camera(
        self,
        camera: BEVCamera,
        participants: dict[int, Any],
        participant_ids: list[int],
        frame: int,
        position: Point,
        heading: float,
    ) -> SceneSnapshot:
        """Build a snapshot using a :class:`BEVCamera` for geometry extraction.

        This mirrors the pattern used by ``preview.py`` and the scenario
        managers: delegate to ``BEVCamera.update()`` then convert the result.

        Args:
            camera: A configured BEVCamera instance.
            participants: All participants dict.
            participant_ids: Active participant ids for this frame.
            frame: Current frame number.
            position: Camera position.
            heading: Camera heading in radians.

        Returns:
            A :class:`SceneSnapshot`.
        """
        geometry_data, prev_road_ids, prev_participant_ids = camera.update(
            frame,
            participants,
            participant_ids,
            set(),  # prev_road_id_set (rebuilt each time)
            set(),  # prev_participant_id_set (rebuilt each time)
            position,
            heading,
        )

        snapshot = self._from_geometry_data(geometry_data, position, heading, frame)
        self._prev_participant_ids = prev_participant_ids
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_geometry_data(
        self, geometry_data: dict[str, Any], position: Point, heading: float, frame: int
    ) -> SceneSnapshot:
        """Convert a ``geometry_data`` dict (from BEVCamera) to SceneSnapshot."""
        map_data = geometry_data.get("map_data", {})
        participant_data = geometry_data.get("participant_data", {})

        road_elements = {}
        for elem in map_data.get("road_elements", []):
            road_elements[elem["id"]] = RoadElement(
                id_=elem["id"],
                shape=elem.get("shape", "polygon"),
                geometry=[tuple(pt) for pt in elem.get("geometry", [])],
                type_=elem.get("type", ""),
                color=elem.get("color"),
                line_width=elem.get("line_width", 1.0),
            )

        participants = {}
        for part in participant_data.get("participants", []):
            participants[part["id"]] = ParticipantElement(
                id_=part["id"],
                shape=part.get("shape", "polygon"),
                geometry=[tuple(pt) for pt in part.get("geometry", [])],
                position=tuple(part.get("position", (0, 0))),
                rotation=part.get("rotation", 0.0),
                type_=part.get("type", "vehicle"),
                color=part.get("color"),
                line_width=part.get("line_width", 1.0),
            )

        metadata = geometry_data.get("metadata", {})
        sensor_position = metadata.get("sensor_position", position)
        sensor_yaw = metadata.get("sensor_yaw", heading)
        perception_range = metadata.get("perception_range", None)

        cameras = [
            CameraMetadata(
                id_="camera-0",
                position=(
                    (
                        sensor_position.x
                        if hasattr(sensor_position, "x")
                        else float(sensor_position[0])
                    ),
                    (
                        sensor_position.y
                        if hasattr(sensor_position, "y")
                        else float(sensor_position[1])
                    ),
                ),
                yaw=float(sensor_yaw),
                perception_range=float(perception_range) if perception_range else 80.0,
            )
        ]

        point_clouds = []
        for pc in participant_data.get("point_clouds", []):
            point_clouds.append(
                PointCloudElement(
                    id_=pc.get("id", f"pc_{len(point_clouds)}"),
                    points=[tuple(pt) for pt in pc.get("points", [])],
                    color=pc.get("color", "red"),
                    point_size=pc.get("point_size", 2.0),
                    alpha=pc.get("alpha", 0.8),
                )
            )

        participant_id_to_remove = participant_data.get("participant_id_to_remove", [])
        participant_id_to_create = participant_data.get("participant_id_to_create", [])

        return SceneSnapshot(
            version="1.0",
            frame=frame,
            road_elements=road_elements,
            road_ids_to_remove=map_data.get("road_id_to_remove", []),
            participants=participants,
            participant_ids_to_create=participant_id_to_create,
            participant_ids_to_remove=participant_id_to_remove,
            point_clouds=point_clouds,
            cameras=cameras,
        )

    def _build_participant_elements(
        self, participants: dict[int, Any], participant_ids: list[int]
    ) -> tuple[dict[int, ParticipantElement], list[int], list[int]]:
        """Convert participant objects to ParticipantElements with delta tracking."""
        current_ids = set(participant_ids)
        created_ids = sorted(current_ids - self._prev_participant_ids)
        removed_ids = sorted(self._prev_participant_ids - current_ids)
        self._prev_participant_ids = current_ids

        elements: dict[int, ParticipantElement] = {}
        for pid in participant_ids:
            participant = participants[pid]
            elements[pid] = _participant_to_element(participant)

        return elements, created_ids, removed_ids


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _build_road_elements(map_: Map) -> dict[int, RoadElement]:
    """Convert a tactics2d Map to RoadElement dict."""
    elements: dict[int, RoadElement] = {}
    for lane_id, lane in map_.lanes.items():
        coords = list(lane.geometry.exterior.coords) if hasattr(lane.geometry, "exterior") else []
        elements[lane_id] = RoadElement(
            id_=lane_id,
            shape="polygon",
            geometry=coords,
            type_="lane",
            subtype=lane.subtype if hasattr(lane, "subtype") else None,
        )
    for area_id, area in map_.areas.items():
        coords = list(area.geometry.exterior.coords) if hasattr(area.geometry, "exterior") else []
        elements[area_id] = RoadElement(
            id_=area_id,
            shape="polygon",
            geometry=coords,
            type_="area",
            subtype=area.subtype if hasattr(area, "subtype") else None,
        )
    for roadline_id, roadline in map_.roadlines.items():
        coords = list(roadline.geometry.coords) if hasattr(roadline.geometry, "coords") else []
        elements[roadline_id] = RoadElement(
            id_=roadline_id,
            shape="line",
            geometry=coords,
            type_="roadline",
            line_style=roadline.type_ if hasattr(roadline, "type_") else "solid",
        )
    return elements


def _participant_to_element(participant: Any) -> ParticipantElement:
    """Convert a Participant (Vehicle/Cyclist/Pedestrian) to ParticipantElement."""
    from shapely.geometry import Polygon

    if hasattr(participant, "get_pose"):
        pose_poly = participant.get_pose()
        if isinstance(pose_poly, Polygon):
            geometry = list(pose_poly.exterior.coords)
        else:
            geometry = list(pose_poly.coords)
    else:
        geometry = []

    state = participant.trajectory.last_state if hasattr(participant, "trajectory") else None

    if isinstance(participant, Vehicle):
        type_ = "vehicle"
    elif isinstance(participant, Cyclist):
        type_ = "cyclist"
    elif isinstance(participant, Pedestrian):
        type_ = "pedestrian"
    else:
        type_ = "vehicle"

    return ParticipantElement(
        id_=participant.id_,
        shape="polygon",
        geometry=geometry,
        position=(state.x, state.y) if state else (0, 0),
        rotation=state.heading if state else 0.0,
        type_=type_,
        velocity=(state.vx, state.vy) if state else None,
        length=participant.length if hasattr(participant, "length") else None,
        width=participant.width if hasattr(participant, "width") else None,
    )


def _build_camera_metadata(position: Point, heading: float) -> list[CameraMetadata]:
    """Build a single CameraMetadata from position and heading."""
    return [
        CameraMetadata(
            id_="ego-camera",
            position=(position.x, position.y) if hasattr(position, "x") else tuple(position),
            yaw=heading,
            perception_range=80.0,
            viewport_aspect=16.0 / 9.0,
        )
    ]
