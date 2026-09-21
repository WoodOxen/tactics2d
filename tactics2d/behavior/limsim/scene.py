# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene extraction from Tactics2D participants and maps."""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State
from tactics2d.routing.utils import get_lane_centerline

from .config import LimSimConfig
from .schema import AgentDecisionState

# Score added to a lane that is not on the vehicle's route, in metres. Large
# enough that the nearest on-route lane wins a junction fork, small enough that
# an actual lane change still matches the lane the vehicle is moving onto.
_OFF_ROUTE_PENALTY = 4.0


class SceneBuilder:
    """Build planner states from Tactics2D data structures."""

    def __init__(self, config: LimSimConfig):
        self.config = config
        self._lane_index_cache = {}  # {id(map_): (strtree, lane_ids, centerlines)}

    def build(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        route_map: Optional[Dict[object, Tuple[str, ...]]] = None,
    ) -> Dict[object, AgentDecisionState]:
        """Extract active participants at a frame as decision states.

        Args:
            route_map: Optional ``{agent_id: (lane_id_0, lane_id_1, ...)}``
                providing ground-truth lane sequences.  When set, the agent's
                ``route_lane_ids`` is populated from this map instead of being
                inferred from the current lane alone.
        """

        selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
        states = {}
        for agent_id in selected_ids:
            participant = participants.get(agent_id)
            if participant is None or not participant.trajectory.has_state(frame):
                continue

            raw_state = participant.get_state(frame)
            if not isinstance(raw_state, State):
                continue

            preferred = route_map.get(agent_id, ()) if route_map else ()
            lane_id = self._match_lane(map_, raw_state, preferred)
            route_progress, lateral_offset = self._project_on_lane(
                map_, lane_id, raw_state.location
            )
            if route_map and agent_id in route_map:
                route_lane_ids = route_map[agent_id]
            else:
                route_lane_ids = tuple([lane_id]) if lane_id is not None else tuple()
            states[agent_id] = AgentDecisionState(
                agent_id=agent_id,
                x=raw_state.x,
                y=raw_state.y,
                heading=spatial.normalize_angle(raw_state.heading),
                speed=max(raw_state.speed or 0.0, 0.0),
                lane_id=lane_id,
                lateral_offset=lateral_offset,
                route_lane_ids=route_lane_ids,
                route_progress=route_progress,
                length=participant.length or self.config.default_vehicle_length,
                width=participant.width or self.config.default_vehicle_width,
            )
        return states

    def _match_lane(
        self, map_: Optional[Map], state: State, route_lane_ids: Iterable[str] = ()
    ) -> Optional[str]:
        if map_ is None or len(map_.lanes) == 0:
            return None

        lane_id = self._find_heading_consistent_lane(map_, state, route_lane_ids)
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

    def _find_heading_consistent_lane(
        self, map_: Map, state: State, route_lane_ids: Iterable[str] = ()
    ) -> Optional[str]:
        point = Point(state.location)

        # broad phase: the cached STRtree of lane centerlines, no brute-force scan
        strtree, indexed_lane_ids, indexed_centerlines = self._get_lane_index(map_)
        if strtree is None:
            return None

        search_region = point.buffer(self.config.lane_match_radius)
        hit_indices = strtree.query(search_region)
        if len(hit_indices) == 0:
            return None

        # narrow phase: score the nearest hit together with its topology neighbours
        nearest_idx = min(hit_indices, key=lambda idx: indexed_centerlines[idx].distance(point))
        best_lane_id = indexed_lane_ids[nearest_idx]
        best_score = np.inf
        for lane_id in self._candidate_lane_ids(map_, best_lane_id, point):
            lane = map_.lanes.get(lane_id)
            if lane is None:
                continue
            centerline = get_lane_centerline(lane)
            if centerline is None:
                continue
            distance = LineString(centerline).distance(point)
            if distance > self.config.lane_match_radius:
                continue
            projection = lane.project_point(state.location)
            lane_heading = self._lane_heading_at(
                lane, projection.s if projection is not None else None
            )
            heading_error = 0.0
            if lane_heading is not None:
                heading_error = abs(spatial.normalize_angle(state.heading - lane_heading))
            score = distance + self.config.lane_heading_match_weight * heading_error
            if route_lane_ids and lane_id not in route_lane_ids:
                score += _OFF_ROUTE_PENALTY
            if score < best_score:
                best_score = score
                best_lane_id = lane_id

        return best_lane_id

    def _candidate_lane_ids(self, map_: Map, lane_id: str, point_xy) -> List[str]:
        """Nearby lane ids: STRtree hits within the match radius plus topology neighbours.

        Args:
            map_ (Map): The map holding the lanes.
            lane_id (str): The lane whose neighbours join the candidate set.
            point_xy: The query point as an ``(x, y)`` pair.

        Returns:
            The candidate lane ids, always including ``lane_id``.
        """
        point = Point(point_xy)
        lane = map_.lanes.get(lane_id)
        strtree, indexed_lane_ids, indexed_centerlines = self._get_lane_index(map_)

        lane_ids = set()
        if strtree is not None:
            for idx in strtree.query(point.buffer(self.config.lane_match_radius)):
                if indexed_centerlines[idx].distance(point) <= self.config.lane_match_radius:
                    lane_ids.add(indexed_lane_ids[idx])

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

    def _get_lane_index(self, map_: Map):
        """Build or retrieve a cached STRtree of lane centerlines.

        Returns:
            ``(strtree, lane_ids, centerlines)`` where all three lists are
            aligned by index.  Returns ``(None, [], [])`` when no indexable
            lanes exist.
        """
        map_key = id(map_)
        if map_key in self._lane_index_cache:
            return self._lane_index_cache[map_key]

        indexed_lane_ids: List[str] = []
        indexed_centerlines: List[LineString] = []
        for candidate_id, candidate_lane in map_.lanes.items():
            centerline = candidate_lane.centerline()
            centerline = (
                np.asarray(centerline.coords, dtype=float) if centerline is not None else None
            )
            if centerline is not None and len(centerline) >= 2:
                indexed_lane_ids.append(candidate_id)
                indexed_centerlines.append(LineString(centerline))

        if indexed_centerlines:
            strtree = STRtree(indexed_centerlines)
        else:
            strtree = None

        result = (strtree, indexed_lane_ids, indexed_centerlines)
        self._lane_index_cache[map_key] = result
        return result

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

        # ``Lane.project_point`` measures the distance to the projected point,
        # which past the end of the lane is the longitudinal overshoot rather
        # than a lateral offset. Project onto the lane's direction at the
        # projection instead, so a vehicle that has driven past a short lane
        # keeps the offset it had while driving in it.
        offset = np.array([point[0] - projection.point.x, point[1] - projection.point.y])
        normal = np.array([-np.sin(projection.heading), np.cos(projection.heading)])
        return projection.s, float(np.dot(normal, offset))
