# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Planner-facing semantic queries derived from Tactics2D maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon, box

from tactics2d.geometry import frenet
from tactics2d.map.element import Map, Regulatory


@dataclass(frozen=True)
class StopTarget:
    """A planner-facing stop target projected from map regulatory elements."""

    lane_id: str
    point: Point
    reason: str
    source_id: str
    state: str | None = None


@dataclass(frozen=True)
class StopLine:
    """A planner-facing stop line, usually inferred from a stop target."""

    lane_id: str
    geometry: LineString
    source_id: str | None = None
    reason: str | None = None
    virtual: bool = True


@dataclass(frozen=True)
class LaneConflict:
    """A conflict relation between two lanes inferred for planners."""

    lane_id: str
    conflict_lane_id: str
    points: tuple[Point, ...]
    inferred: bool = True
    source: str = "geometry"


class SemanticMapQuery:
    """High-level map semantics inferred from lanes, junctions, and geometry."""

    def __init__(self, map_: Map):
        self.map = map_

    def get_reference_path(
        self, lane_id: str, route_lane_ids: Sequence[str] | None = None, lookahead_lanes: int = 3
    ) -> ReferencePath | None:
        """Build a reference path by concatenating lane centerlines."""

        if lane_id not in self.map.lanes:
            return None

        if route_lane_ids is None:
            lane_ids = [lane_id]
            current_lane_id = lane_id
            while len(lane_ids) < max(1, lookahead_lanes):
                current_lane = self.map.lanes.get(current_lane_id)
                if current_lane is None or not current_lane.successors:
                    break
                next_lane_id = sorted(current_lane.successors)[0]
                if next_lane_id in lane_ids or next_lane_id not in self.map.lanes:
                    break
                lane_ids.append(next_lane_id)
                current_lane_id = next_lane_id
        else:
            lane_ids = [candidate for candidate in route_lane_ids if candidate in self.map.lanes]

        centerlines = [self.map.lanes[candidate].centerline() for candidate in lane_ids]
        path = self._concatenate_lines(centerlines)
        if path is None:
            return None
        width = self.map.lanes[lane_id].get_width()
        return frenet.ReferencePath(path=path, lane_ids=tuple(lane_ids), lane_width=width)

    def get_lanes_in_region(self, region: tuple[float, float, float, float] | Polygon):
        """Return lane ids whose geometry intersects a region."""

        geometry = (
            box(region[0], region[2], region[1], region[3]) if isinstance(region, tuple) else region
        )
        lane_ids = []
        for lane_id, lane in self.map.lanes.items():
            if lane.geometry is not None and lane.geometry.intersects(geometry):
                lane_ids.append(lane_id)
        return lane_ids

    def get_traffic_light_for_lane(self, lane_id: str) -> Regulatory | None:
        """Return the traffic-light regulatory element bound to a lane."""

        for regulation in self.map.regulations.values():
            if regulation.is_traffic_light() and regulation.applies_to_lane(lane_id):
                return regulation
        return None

    def get_traffic_light_state(self, lane_id: str, time_ms: int | None = None):
        """Return the traffic-light state record nearest to ``time_ms``."""

        regulation = self.get_traffic_light_for_lane(lane_id)
        if regulation is None:
            return None
        return regulation.state_at(time_ms)

    def get_stop_signs(self, lane_id: str | None = None) -> list[Regulatory]:
        """Return stop signs, optionally filtered by lane id."""

        stop_signs = []
        for regulation in self.map.regulations.values():
            if regulation.is_stop_sign() and (
                lane_id is None or regulation.applies_to_lane(lane_id)
            ):
                stop_signs.append(regulation)
        return stop_signs

    def get_stop_targets(
        self,
        lane_id: str,
        time_ms: int | None = None,
        include_stop_signs: bool = True,
        include_traffic_lights: bool = True,
    ) -> list[StopTarget]:
        """Return stop targets from stop signs and traffic lights."""

        targets = []
        if include_stop_signs:
            for stop_sign in self.get_stop_signs(lane_id):
                if stop_sign.position is not None:
                    targets.append(
                        StopTarget(
                            lane_id=lane_id,
                            point=stop_sign.position,
                            reason="stop_sign",
                            source_id=stop_sign.id_,
                        )
                    )

        if include_traffic_lights:
            regulation = self.get_traffic_light_for_lane(lane_id)
            state_record = self.get_traffic_light_state(lane_id, time_ms)
            point = regulation.stop_point_at(time_ms) if regulation is not None else None
            state = state_record.get("state") if state_record is not None else None
            if regulation is not None and point is not None:
                targets.append(
                    StopTarget(
                        lane_id=lane_id,
                        point=point,
                        reason="traffic_light",
                        source_id=regulation.id_,
                        state=state,
                    )
                )
        return targets

    def get_lane_change_permission(
        self, lane_id: str, direction: str, s: float | None = None
    ) -> bool:
        """Return whether a lane change is allowed at a lane position."""

        lane = self.map.lanes.get(lane_id)
        if lane is None:
            return False
        direction = direction.lower()
        if direction in {"left", "lcl"}:
            if not lane.left_neighbors:
                return False
            side = "left"
            lane_change_index = 1
        elif direction in {"right", "lcr"}:
            if not lane.right_neighbors:
                return False
            side = "right"
            lane_change_index = 0
        else:
            raise ValueError("direction must be 'left'/'lcl' or 'right'/'lcr'.")

        roadline_ids = self._boundary_roadline_ids(lane, side, s)
        if not roadline_ids:
            return True
        permissions = []
        for roadline_id in roadline_ids:
            roadline = self.map.roadlines.get(roadline_id)
            if roadline is None or roadline.lane_change is None:
                continue
            permissions.append(bool(roadline.lane_change[lane_change_index]))
        return all(permissions) if permissions else True

    def is_lane_in_junction(self, lane_id: str) -> bool:
        """Return whether a lane is marked or inferred as junction-related."""

        lane = self.map.lanes.get(lane_id)
        if lane is None:
            return False

        tags = lane.custom_tags or {}
        if bool(tags.get("is_intersection")) or bool(tags.get("junction")):
            return True
        subtype = (lane.subtype or "").lower()
        if "junction" in subtype or "intersection" in subtype:
            return True
        return self.get_junction_by_lane(lane_id) is not None

    def get_junction_area(self, junction_id: str | None = None, lane_id: str | None = None):
        """Return an available junction/intersection polygon."""

        junction = None
        if junction_id is not None:
            junction = self.map.junctions.get(junction_id)
        elif lane_id is not None:
            junction = self.get_junction_by_lane(lane_id)

        if junction is not None:
            shape = (junction.custom_tags or {}).get("shape")
            if shape:
                return Polygon(shape)

        for area in self.map.areas.values():
            subtype = (area.subtype or "").lower()
            if "junction" in subtype or "intersection" in subtype:
                lane = self.map.lanes.get(lane_id) if lane_id is not None else None
                if lane_id is None or (
                    lane is not None
                    and lane.geometry is not None
                    and area.geometry.intersects(lane.geometry)
                ):
                    return area.geometry
        return None

    def get_junction_by_lane(self, lane_id: str):
        """Return the first junction containing a lane, if one is available."""

        for junction in self.map.junctions.values():
            if lane_id in self._junction_lane_ids(junction):
                return junction
        return None

    def get_conflict_lanes(
        self,
        lane_id: str,
        candidate_lane_ids: Iterable[str] | None = None,
        angle_threshold: float = np.deg2rad(20.0),
    ) -> list[str]:
        """Infer lanes whose centerlines geometrically conflict with a lane."""

        return [
            conflict.conflict_lane_id
            for conflict in self.get_junction_conflicts(
                lane_id, candidate_lane_ids=candidate_lane_ids, angle_threshold=angle_threshold
            )
        ]

    def get_junction_conflicts(
        self,
        lane_id: str,
        candidate_lane_ids: Iterable[str] | None = None,
        angle_threshold: float = np.deg2rad(20.0),
    ) -> list[LaneConflict]:
        """Infer planner-facing lane conflicts from geometry and topology."""

        conflicts = []
        candidates = candidate_lane_ids if candidate_lane_ids is not None else self.map.lanes.keys()
        for other_lane_id in candidates:
            if other_lane_id == lane_id:
                continue
            if self.has_conflict(lane_id, other_lane_id, angle_threshold=angle_threshold):
                conflicts.append(
                    LaneConflict(
                        lane_id=lane_id,
                        conflict_lane_id=other_lane_id,
                        points=tuple(self.get_conflict_points(lane_id, other_lane_id)),
                    )
                )
        return conflicts

    def has_conflict(
        self, lane_id_a: str, lane_id_b: str, angle_threshold: float = np.deg2rad(20.0)
    ) -> bool:
        """Return whether two lanes have crossing or overlapping drive paths."""

        if self._are_directly_connected(lane_id_a, lane_id_b):
            return False

        line_a = self._lane_centerline(lane_id_a)
        line_b = self._lane_centerline(lane_id_b)
        if line_a is None or line_b is None:
            return False

        intersection = line_a.intersection(line_b)
        if intersection.is_empty:
            lane_a = self.map.lanes.get(lane_id_a)
            lane_b = self.map.lanes.get(lane_id_b)
            if (
                lane_a is None
                or lane_b is None
                or lane_a.geometry is None
                or lane_b.geometry is None
            ):
                return False
            intersection = lane_a.geometry.intersection(lane_b.geometry)
            if intersection.is_empty:
                return False

        conflict_points = self.get_conflict_points(lane_id_a, lane_id_b)
        if not conflict_points:
            return not intersection.is_empty

        for point in conflict_points:
            heading_a = self._heading_at(line_a, point)
            heading_b = self._heading_at(line_b, point)
            angle = abs(np.arctan2(np.sin(heading_a - heading_b), np.cos(heading_a - heading_b)))
            if angle_threshold <= angle <= np.pi - angle_threshold:
                return True
        return False

    def get_conflict_points(self, lane_id_a: str, lane_id_b: str) -> list[Point]:
        """Return geometric conflict points between two lane centerlines."""

        line_a = self._lane_centerline(lane_id_a)
        line_b = self._lane_centerline(lane_id_b)
        if line_a is None or line_b is None:
            return []

        intersection = line_a.intersection(line_b)
        return self._points_from_geometry(intersection)

    def get_stop_line(
        self,
        lane_id: str,
        stop_target: StopTarget | None = None,
        point: Point | tuple[float, float] | None = None,
        width: float | None = None,
    ) -> StopLine | None:
        """Build a virtual stop line perpendicular to a lane centerline."""

        lane = self.map.lanes.get(lane_id)
        if lane is None:
            return None

        target_point = point
        if target_point is None and stop_target is not None:
            target_point = stop_target.point
        if target_point is None:
            targets = self.get_stop_targets(lane_id)
            if not targets:
                return None
            target_point = targets[0].point

        projection = lane.project_point(target_point)
        if projection is None:
            return None
        line_width = float(width if width is not None else lane.get_width(default=3.6))
        half_width = line_width / 2.0
        normal = np.array([-np.sin(projection.heading), np.cos(projection.heading)])
        center = np.array([projection.point.x, projection.point.y])
        start = center - normal * half_width
        end = center + normal * half_width
        return StopLine(
            lane_id=lane_id,
            geometry=LineString([tuple(start), tuple(end)]),
            source_id=stop_target.source_id if stop_target is not None else None,
            reason=stop_target.reason if stop_target is not None else None,
            virtual=True,
        )

    def get_stop_line_geometry(
        self,
        lane_id: str,
        stop_target: StopTarget | None = None,
        point: Point | tuple[float, float] | None = None,
        width: float | None = None,
    ) -> LineString | None:
        """Build a virtual stop-line geometry perpendicular to a lane centerline."""

        stop_line = self.get_stop_line(
            lane_id=lane_id, stop_target=stop_target, point=point, width=width
        )
        return stop_line.geometry if stop_line is not None else None

    def _lane_centerline(self, lane_id: str) -> LineString | None:
        lane = self.map.lanes.get(lane_id)
        return lane.centerline() if lane is not None else None

    def _concatenate_lines(self, lines: Sequence[LineString | None]) -> LineString | None:
        coords = []
        for line in lines:
            if line is None or line.length <= 0.0:
                continue
            line_coords = list(line.coords)
            if not coords:
                coords.extend(line_coords)
            elif np.allclose(coords[-1], line_coords[0]):
                coords.extend(line_coords[1:])
            else:
                coords.extend(line_coords)
        return LineString(coords) if len(coords) >= 2 else None

    def _boundary_roadline_ids(self, lane, side: str, s: float | None) -> list[str]:
        tags = lane.custom_tags or {}
        segments = tags.get("boundary_segments", {}).get(side, [])
        if s is not None and segments:
            matched = [
                segment["roadline_id"]
                for segment in segments
                if float(segment.get("start_s", -np.inf))
                <= s
                <= float(segment.get("end_s", np.inf))
            ]
            if matched:
                return matched
        return list((lane.line_ids or {}).get(side, []))

    def _are_directly_connected(self, lane_id_a: str, lane_id_b: str) -> bool:
        lane_a = self.map.lanes.get(lane_id_a)
        lane_b = self.map.lanes.get(lane_id_b)
        if lane_a is None or lane_b is None:
            return False
        return (
            lane_id_b in lane_a.successors
            or lane_id_b in lane_a.predecessors
            or lane_id_b in lane_a.left_neighbors
            or lane_id_b in lane_a.right_neighbors
            or lane_id_a in lane_b.successors
            or lane_id_a in lane_b.predecessors
            or lane_id_a in lane_b.left_neighbors
            or lane_id_a in lane_b.right_neighbors
        )

    def _junction_lane_ids(self, junction) -> set:
        lane_ids = set()
        tags = junction.custom_tags or {}
        for key in ("lane_ids", "incoming_lanes", "inside_lanes", "outgoing_lanes"):
            lane_ids.update(tags.get(key, []) or [])
        for from_lane_id, to_lane_id in junction.lane_links:
            lane_ids.add(from_lane_id)
            lane_ids.add(to_lane_id)
        for connection in junction.connections.values():
            lane_ids.update(self._junction_lane_ids(connection))
        return lane_ids

    def _heading_at(self, line: LineString, point: Point) -> float:
        progress = float(line.project(point))
        ahead = line.interpolate(min(progress + 0.5, line.length))
        behind = line.interpolate(max(progress - 0.5, 0.0))
        return float(np.arctan2(ahead.y - behind.y, ahead.x - behind.x))

    def _points_from_geometry(self, geometry) -> list[Point]:
        if geometry.is_empty:
            return []
        if isinstance(geometry, Point):
            return [geometry]
        if isinstance(geometry, MultiPoint):
            return list(geometry.geoms)
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            if not coords:
                return []
            return [Point(coords[len(coords) // 2])]
        if isinstance(geometry, GeometryCollection):
            points = []
            for sub_geometry in geometry.geoms:
                points.extend(self._points_from_geometry(sub_geometry))
            return points
        return []
