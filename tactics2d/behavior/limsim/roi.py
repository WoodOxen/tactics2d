# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Region-of-interest selection for LimSim-style local scenes."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tactics2d.geometry import euclidean_distance


@dataclass(frozen=True)
class RoISelection:
    """Vehicles selected for local interaction and background prediction."""

    agent_ids: List[object] = field(default_factory=list)
    background_agent_ids: List[object] = field(default_factory=list)
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    outer_radius: Optional[float] = None


class RoISelector:
    """Select local Region of Interest agents from participants.

    Tactics2D uses the general Region of Interest naming for this local
    control area: vehicles inside the radius are controlled, and vehicles
    between the radius and roughly twice the radius are kept as background
    obstacles.
    """

    @staticmethod
    def select_by_radius(
        participants: Dict[object, object],
        frame: int,
        center: Sequence[float],
        radius: float,
        outer_radius: Optional[float] = None,
        candidate_ids: Optional[Iterable[object]] = None,
    ) -> RoISelection:
        """Select agents by distance to a fixed physical center."""

        if radius < 0:
            raise ValueError("radius must be non-negative.")
        outer = 2.0 * radius if outer_radius is None else outer_radius
        if outer < radius:
            raise ValueError("outer_radius must be greater than or equal to radius.")

        center_xy = (float(center[0]), float(center[1]))
        agent_ids = []
        background_agent_ids = []
        selected_ids = list(participants.keys()) if candidate_ids is None else list(candidate_ids)
        for agent_id in selected_ids:
            participant = participants.get(agent_id)
            if participant is None or not participant.trajectory.has_state(frame):
                continue
            state = participant.trajectory.get_state(frame)
            distance = euclidean_distance(state.location, center_xy)
            if distance <= radius:
                agent_ids.append(agent_id)
            elif distance <= outer:
                background_agent_ids.append(agent_id)

        return RoISelection(
            agent_ids=agent_ids,
            background_agent_ids=background_agent_ids,
            center=center_xy,
            radius=float(radius),
            outer_radius=float(outer),
        )

    @staticmethod
    def select_around_agent(
        participants: Dict[object, object],
        frame: int,
        ego_id: object,
        radius: float,
        outer_radius: Optional[float] = None,
        include_ego: bool = True,
        candidate_ids: Optional[Iterable[object]] = None,
    ) -> RoISelection:
        """Select agents around an ego participant."""

        ego = participants.get(ego_id)
        if ego is None or not ego.trajectory.has_state(frame):
            raise KeyError(f"ego_id {ego_id!r} is not active at frame {frame}.")

        ego_state = ego.trajectory.get_state(frame)
        selection = RoISelector.select_by_radius(
            participants=participants,
            frame=frame,
            center=ego_state.location,
            radius=radius,
            outer_radius=outer_radius,
            candidate_ids=candidate_ids,
        )
        if not include_ego and ego_id in selection.agent_ids:
            agent_ids = [agent_id for agent_id in selection.agent_ids if agent_id != ego_id]
            return RoISelection(
                agent_ids=agent_ids,
                background_agent_ids=selection.background_agent_ids,
                center=selection.center,
                radius=selection.radius,
                outer_radius=selection.outer_radius,
            )
        return selection

    @staticmethod
    def select_dense_region(
        participants: Dict[object, object],
        frame: int,
        max_agents: int,
        neighbor_count: int = 6,
        candidate_ids: Optional[Iterable[object]] = None,
    ) -> RoISelection:
        """Select a compact local region around the densest active participant."""

        if max_agents <= 0:
            return RoISelection()

        active = []
        selected_ids = list(participants.keys()) if candidate_ids is None else list(candidate_ids)
        for agent_id in selected_ids:
            participant = participants.get(agent_id)
            if participant is None or not participant.trajectory.has_state(frame):
                continue
            active.append((agent_id, participant.trajectory.get_state(frame)))

        if len(active) <= max_agents:
            return RoISelection(
                agent_ids=[agent_id for agent_id, _ in active],
                center=active[0][1].location if active else None,
            )

        scored = []
        for agent_id, state in active:
            distances = sorted(
                euclidean_distance(state.location, other_state.location)
                for other_id, other_state in active
                if other_id != agent_id
            )
            k = min(neighbor_count, len(distances))
            mean_distance = sum(distances[:k]) / max(k, 1)
            scored.append((mean_distance, agent_id, state))

        _, _, anchor_state = min(scored, key=lambda item: item[0])
        ordered = sorted(
            active,
            key=lambda item: euclidean_distance(item[1].location, anchor_state.location),
        )
        agent_ids = [agent_id for agent_id, _ in ordered[:max_agents]]
        return RoISelection(agent_ids=agent_ids, center=anchor_state.location)
