# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build BITS-style agent-centric samples from Tactics2D data."""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State

from .batch import BitsBatch
from .config import BitsConfig
from .rasterizer import BitsRasterizer


class BitsBatchBuilder:
    """Convert Tactics2D participants into a BITS-compatible single sample."""

    VEHICLE_TYPE = 3
    OTHER_TYPE = 0

    def __init__(
        self, config: Optional[BitsConfig] = None, rasterizer: Optional[BitsRasterizer] = None
    ):
        self.config = config or BitsConfig()
        self.rasterizer = rasterizer or BitsRasterizer(self.config)

    def build(
        self,
        participants: Dict[object, object],
        frame: int,
        ego_id: object,
        map_: Optional[Map] = None,
        agent_ids: Optional[Iterable[object]] = None,
        include_raster: bool = False,
    ) -> BitsBatch:
        """Build one agent-centric sample around ``ego_id`` at ``frame``."""

        if ego_id not in participants:
            raise KeyError(f"Ego participant {ego_id!r} is not available.")
        ego = participants[ego_id]
        if not ego.trajectory.has_state(frame):
            raise KeyError(f"Ego participant {ego_id!r} has no state at frame {frame}.")

        ego_state = ego.trajectory.get_state(frame)
        agent_from_world = self._agent_from_world(ego_state)
        world_from_agent = np.linalg.inv(agent_from_world)
        candidate_ids = self._select_neighbor_ids(participants, frame, ego_id, agent_ids)

        history_frames = self._history_frames(frame)
        future_frames = self._future_frames(frame)
        neighbor_count = self.config.max_agents

        history_positions, history_yaws, history_availabilities = self._extract_sequence(
            ego, history_frames, agent_from_world, ego_state.heading
        )
        target_positions, target_yaws, target_availabilities = self._extract_sequence(
            ego, future_frames, agent_from_world, ego_state.heading
        )

        other_history_positions = np.zeros((neighbor_count, len(history_frames), 2), dtype=float)
        other_history_yaws = np.zeros((neighbor_count, len(history_frames), 1), dtype=float)
        other_history_availability = np.zeros((neighbor_count, len(history_frames)), dtype=bool)
        other_future_positions = np.zeros((neighbor_count, len(future_frames), 2), dtype=float)
        other_future_yaws = np.zeros((neighbor_count, len(future_frames), 1), dtype=float)
        other_future_availability = np.zeros((neighbor_count, len(future_frames)), dtype=bool)
        other_curr_speed = np.zeros(neighbor_count, dtype=float)
        other_types = np.zeros(neighbor_count, dtype=int)
        other_extents = np.zeros((neighbor_count, 2), dtype=float)
        other_history_extents = np.zeros((neighbor_count, len(history_frames), 2), dtype=float)

        kept_agent_ids = []
        for index, agent_id in enumerate(candidate_ids[:neighbor_count]):
            participant = participants[agent_id]
            kept_agent_ids.append(agent_id)
            hist_pos, hist_yaw, hist_avail = self._extract_sequence(
                participant, history_frames, agent_from_world, ego_state.heading
            )
            fut_pos, fut_yaw, fut_avail = self._extract_sequence(
                participant, future_frames, agent_from_world, ego_state.heading
            )
            extent = self._participant_extent(participant)

            other_history_positions[index] = hist_pos
            other_history_yaws[index] = hist_yaw
            other_history_availability[index] = hist_avail
            other_future_positions[index] = fut_pos
            other_future_yaws[index] = fut_yaw
            other_future_availability[index] = fut_avail
            other_curr_speed[index] = self._state_speed(participant.trajectory.get_state(frame))
            other_types[index] = self._participant_type(participant)
            other_extents[index] = extent
            other_history_extents[index, hist_avail] = extent

        raster = None
        if include_raster:
            if map_ is None:
                raise ValueError("map_ is required when include_raster=True.")
            raster = self.rasterizer.rasterize(map_, agent_from_world)
            raster = self.rasterizer.attach_agent_history(
                raster=raster,
                ego_history_positions=history_positions,
                ego_history_yaws=history_yaws,
                ego_history_availabilities=history_availabilities,
                ego_extent=self._participant_extent(ego),
                other_history_positions=other_history_positions,
                other_history_yaws=other_history_yaws,
                other_history_availabilities=other_history_availability,
                other_extents=other_extents,
            )

        return BitsBatch(
            ego_id=ego_id,
            frame=frame,
            history_positions=history_positions,
            history_yaws=history_yaws,
            history_availabilities=history_availabilities,
            target_positions=target_positions,
            target_yaws=target_yaws,
            target_availabilities=target_availabilities,
            curr_speed=self._state_speed(ego_state),
            centroid=np.asarray(ego_state.location, dtype=float),
            yaw=float(ego_state.heading),
            extent=self._participant_extent(ego),
            type=self._participant_type(ego),
            agent_from_world=agent_from_world,
            world_from_agent=world_from_agent,
            all_other_agents_history_positions=other_history_positions,
            all_other_agents_history_yaws=other_history_yaws,
            all_other_agents_history_availability=other_history_availability,
            all_other_agents_future_positions=other_future_positions,
            all_other_agents_future_yaws=other_future_yaws,
            all_other_agents_future_availability=other_future_availability,
            all_other_agents_curr_speed=other_curr_speed,
            all_other_agents_types=other_types,
            all_other_agents_extents=other_extents,
            all_other_agents_history_extents=other_history_extents,
            agent_ids=kept_agent_ids,
            lane_id=self._match_lane(map_, ego_state),
            image=None if raster is None else raster.image,
            drivable_map=None if raster is None else raster.drivable_map,
            raster_from_agent=None if raster is None else raster.raster_from_agent,
            agent_from_raster=None if raster is None else raster.agent_from_raster,
            static_image=None if raster is None else raster.static_image,
            dynamic_image=None if raster is None else raster.dynamic_image,
        )

    def _history_frames(self, frame: int) -> Sequence[int]:
        step_ms = self.config.step_ms
        return [frame - step_ms * offset for offset in range(self.config.history_steps, -1, -1)]

    def _future_frames(self, frame: int) -> Sequence[int]:
        step_ms = self.config.step_ms
        return [frame + step_ms * offset for offset in range(1, self.config.future_steps + 1)]

    def _select_neighbor_ids(
        self,
        participants: Dict[object, object],
        frame: int,
        ego_id: object,
        agent_ids: Optional[Iterable[object]],
    ) -> list:
        ego_state = participants[ego_id].trajectory.get_state(frame)
        selected = []
        candidate_ids = participants.keys() if agent_ids is None else agent_ids
        for agent_id in candidate_ids:
            if agent_id == ego_id:
                continue
            participant = participants.get(agent_id)
            if participant is None or not participant.trajectory.has_state(frame):
                continue
            if not self.config.include_non_vehicle_neighbors and not isinstance(
                participant, Vehicle
            ):
                continue
            distance = spatial.euclidean_distance(
                ego_state.location, participant.trajectory.get_state(frame).location
            )
            if distance <= self.config.max_agents_distance:
                selected.append((distance, agent_id))
        selected.sort(key=lambda item: (item[0], str(item[1])))
        return [agent_id for _, agent_id in selected]

    def _extract_sequence(
        self, participant, frames: Sequence[int], agent_from_world: np.ndarray, ego_heading: float
    ):
        positions = np.zeros((len(frames), 2), dtype=float)
        yaws = np.zeros((len(frames), 1), dtype=float)
        availabilities = np.zeros(len(frames), dtype=bool)
        for index, frame in enumerate(frames):
            if not participant.trajectory.has_state(frame):
                continue
            state = participant.trajectory.get_state(frame)
            positions[index] = spatial.transform_point(state.location, agent_from_world)
            yaws[index, 0] = spatial.normalize_angle(state.heading - ego_heading)
            availabilities[index] = True
        return positions, yaws, availabilities

    @staticmethod
    def _agent_from_world(state: State) -> np.ndarray:
        x, y = state.location
        c = float(np.cos(state.heading))
        s = float(np.sin(state.heading))
        return np.asarray(
            [[c, s, -(c * x + s * y)], [-s, c, s * x - c * y], [0.0, 0.0, 1.0]], dtype=float
        )

    def _participant_extent(self, participant) -> np.ndarray:
        length = getattr(participant, "length", None) or self.config.default_vehicle_length
        width = getattr(participant, "width", None) or self.config.default_vehicle_width
        return np.asarray([float(length), float(width)], dtype=float)

    def _participant_type(self, participant) -> int:
        return self.VEHICLE_TYPE if isinstance(participant, Vehicle) else self.OTHER_TYPE

    @staticmethod
    def _state_speed(state: State) -> float:
        return float(state.speed or 0.0)

    @staticmethod
    def _match_lane(map_: Optional[Map], state: State) -> Optional[str]:
        if map_ is None or not map_.lanes:
            return None
        return map_.find_nearest_lane(state.location)
