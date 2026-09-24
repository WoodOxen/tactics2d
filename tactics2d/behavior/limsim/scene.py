# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene extraction from Tactics2D participants and maps."""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State

from .config import LimSimConfig
from .decision_state import AgentDecisionState

# Score added to a lane that is not on the vehicle's route, in metres. Large
# enough that the nearest on-route lane wins a junction fork, small enough that
# an actual lane change still matches the lane the vehicle is moving onto.
_OFF_ROUTE_PENALTY = 4.0


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
        if map_ is None:
            return None
        return map_.match_lane(
            state.x,
            state.y,
            state.heading,
            radius=self.config.lane_match_radius,
            heading_weight=self.config.lane_heading_match_weight,
            preferred_lane_ids=route_lane_ids,
            off_route_penalty=_OFF_ROUTE_PENALTY,
        )

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
