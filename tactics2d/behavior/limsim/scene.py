# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene extraction from Tactics2D participants and maps."""

from typing import Dict, Iterable, Optional, Tuple

from shapely.geometry import Point

from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State
from tactics2d.routing.utils import find_nearest_lane

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
                heading=raw_state.heading,
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

        lane_id = find_nearest_lane(map_, state.location)
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

    def _project_on_lane(
        self, map_: Optional[Map], lane_id: Optional[str], point
    ) -> Tuple[float, float]:
        if map_ is None or lane_id is None or lane_id not in map_.lanes:
            return 0.0, 0.0
        projection = map_.lanes[lane_id].project_point(point)
        if projection is None:
            return 0.0, 0.0
        return projection.s, projection.d
