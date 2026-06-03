# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene extraction from Tactics2D participants and maps."""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import normalize_angle
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State
from tactics2d.routing.utils import find_nearest_lane, get_lane_centerline

from .config import LimSimConfig
from .schema import AgentDecisionState


class SceneBuilder:
    """Build planner states from Tactics2D data structures."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    def build(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, AgentDecisionState]:
        """Extract active participants at a frame as decision states."""

        selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
        states = {}
        for agent_id in selected_ids:
            participant = participants.get(agent_id)
            if participant is None or not participant.trajectory.has_state(frame):
                continue

            raw_state = participant.get_state(frame)
            if not isinstance(raw_state, State):
                continue

            lane_id = self._match_lane(map_, raw_state)
            route_progress, lateral_offset = self._project_on_lane(
                map_, lane_id, raw_state.location
            )
            route_lane_ids = tuple([lane_id]) if lane_id is not None else tuple()
            states[agent_id] = AgentDecisionState(
                agent_id=agent_id,
                x=raw_state.x,
                y=raw_state.y,
                heading=normalize_angle(raw_state.heading),
                speed=max(raw_state.speed or 0.0, 0.0),
                lane_id=lane_id,
                lateral_offset=lateral_offset,
                route_lane_ids=route_lane_ids,
                route_progress=route_progress,
                length=participant.length or self.config.default_vehicle_length,
                width=participant.width or self.config.default_vehicle_width,
            )
        return states

    def _match_lane(self, map_: Optional[Map], state: State) -> Optional[str]:
        if map_ is None or len(map_.lanes) == 0:
            return None

        lane_id = self._find_heading_consistent_lane(map_, state)
        if lane_id is None:
            return None

        lane = map_.lanes[lane_id]
        projection = lane.project_point(state.location)
        if projection is None:
            distance = lane.geometry.distance(Point(state.location))
        else:
            distance = projection.distance

        if distance > self.config.lane_match_radius:
            return None
        return lane_id

    def _find_heading_consistent_lane(self, map_: Map, state: State) -> Optional[str]:
        nearby_lane_id = find_nearest_lane(map_, state.location)
        if nearby_lane_id is None:
            return None

        point = Point(state.location)
        best_lane_id = nearby_lane_id
        best_score = np.inf
        lane_ids = self._candidate_lane_ids(map_, nearby_lane_id, state.location)
        for lane_id in lane_ids:
            lane = map_.lanes.get(lane_id)
            if lane is None:
                continue
            projection = lane.project_point(state.location)
            distance = lane.geometry.distance(point) if projection is None else projection.distance
            if distance > self.config.lane_match_radius:
                continue
            lane_heading = self._lane_heading_at(
                lane, projection.s if projection is not None else None
            )
            heading_error = 0.0
            if lane_heading is not None:
                heading_error = abs(normalize_angle(state.heading - lane_heading))
                heading_error = min(heading_error, abs(np.pi - heading_error))
            score = distance + self.config.lane_heading_match_weight * heading_error
            if score < best_score:
                best_score = score
                best_lane_id = lane_id
        return best_lane_id

    def _candidate_lane_ids(self, map_: Map, lane_id: str, point_xy):
        lane = map_.lanes.get(lane_id)
        point = Point(point_xy)
        lane_ids = set()
        for candidate_id, candidate_lane in map_.lanes.items():
            if candidate_lane.geometry is None:
                continue
            centerline = get_lane_centerline(candidate_lane)
            if centerline is not None:
                distance = LineString(centerline).distance(point)
            else:
                distance = candidate_lane.geometry.distance(point)
            if distance <= self.config.lane_match_radius:
                lane_ids.add(candidate_id)
        if lane is None:
            return list(lane_ids or {lane_id})
        lane_ids.update(
            {
                lane_id,
                *lane.left_neighbors,
                *lane.right_neighbors,
                *lane.predecessors,
                *lane.successors,
            }
        )
        return list(lane_ids)

    def _lane_heading_at(self, lane, s: Optional[float]) -> Optional[float]:
        centerline = get_lane_centerline(lane)
        if centerline is None or len(centerline) < 2:
            return None
        line = LineString(centerline)
        if line.length <= 1e-6:
            return None
        route_s = 0.0 if s is None else float(np.clip(s, 0.0, line.length))
        before = line.interpolate(max(0.0, route_s - 1.0))
        after = line.interpolate(min(line.length, route_s + 1.0))
        return float(np.arctan2(after.y - before.y, after.x - before.x))

    def _project_on_lane(
        self, map_: Optional[Map], lane_id: Optional[str], point
    ) -> Tuple[float, float]:
        if map_ is None or lane_id is None or lane_id not in map_.lanes:
            return 0.0, 0.0
        projection = map_.lanes[lane_id].project_point(point)
        if projection is None:
            return 0.0, 0.0
        return projection.s, projection.d
