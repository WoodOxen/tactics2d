# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lightweight rolling PDP runner for LimSim-style behavior planning."""

from copy import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from tactics2d.geometry import normalize_angle
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

from .action import LimSimAction
from .config import LimSimConfig
from .model import LimSimBehaviorModel
from .planner import LaneFollower
from .roi import RoISelector
from .scene import SceneBuilder
from .schema import PlanningResult


@dataclass
class RollingSimulationResult:
    """Output of a lightweight rolling LimSim-style simulation."""

    participants: Dict[object, object]
    results: List[PlanningResult] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    predicted_trajectories: List[Dict[object, Trajectory]] = field(default_factory=list)


class LimSimRollingRunner:
    """Run repeated single-step PDP updates over a short simulation horizon.

    This runner keeps the behavior model itself single-frame and wraps it in a
    receding-horizon loop. Controlled RoI vehicles execute the first state of
    their planned trajectory; other vehicles advance with a KS lane-following
    rule, matching LimSim's lightweight background handling.
    """

    def __init__(self, config: Optional[LimSimConfig] = None):
        self.config = config or LimSimConfig()
        self.behavior_model = LimSimBehaviorModel(self.config)
        self.scene_builder = SceneBuilder(self.config)
        self.follower = LaneFollower(self.config)

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        start_frame: int,
        simulation_steps: int,
        agent_ids: Optional[Iterable[object]] = None,
        roi_center: Optional[Sequence[float]] = None,
        roi_radius: Optional[float] = None,
        roi_outer_radius: Optional[float] = None,
        ego_id: Optional[object] = None,
    ) -> RollingSimulationResult:
        """Run rolling PDP for ``simulation_steps`` control updates."""

        frame = int(start_frame)
        simulation_agent_ids = self._simulation_agent_ids(
            participants,
            frame,
            agent_ids=agent_ids,
            roi_center=roi_center,
            roi_radius=roi_radius,
            roi_outer_radius=roi_outer_radius,
            ego_id=ego_id,
        )
        simulated_participants = self._clone_participants_at_frame(
            participants,
            frame,
            simulation_agent_ids,
        )
        frames = [frame]
        results = []
        predictions = []
        last_planned_trajectories = {}
        committed_trajectories = {}
        committed_actions = {}
        fixed_roi_center = None if ego_id is not None else roi_center

        for _ in range(simulation_steps):
            result = self.behavior_model.plan(
                simulated_participants,
                map_,
                frame=frame,
                agent_ids=agent_ids,
                roi_center=fixed_roi_center,
                roi_radius=roi_radius,
                roi_outer_radius=roi_outer_radius,
                ego_id=ego_id,
                last_planned_trajectories=last_planned_trajectories,
            )
            results.append(result)
            self._update_committed_trajectories(
                result,
                frame,
                committed_trajectories,
                committed_actions,
            )
            predictions.append(
                self.behavior_model.predictor.predict(
                    simulated_participants,
                    map_,
                    frame,
                    agent_ids=result.roi_agent_ids,
                    last_planned_trajectories=last_planned_trajectories,
                )
            )
            next_frame = self._next_frame(frame)
            self._advance_participants(
                simulated_participants,
                map_,
                frame,
                next_frame,
                result,
                committed_trajectories,
            )
            last_planned_trajectories = self._prediction_memory(
                result,
                committed_trajectories,
            )
            frame = next_frame
            frames.append(frame)

        return RollingSimulationResult(
            participants=simulated_participants,
            results=results,
            frames=frames,
            predicted_trajectories=predictions,
        )

    def _simulation_agent_ids(
        self,
        participants: Dict[object, object],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        roi_center: Optional[Sequence[float]] = None,
        roi_radius: Optional[float] = None,
        roi_outer_radius: Optional[float] = None,
        ego_id: Optional[object] = None,
    ) -> Optional[Iterable[object]]:
        if agent_ids is not None:
            return agent_ids
        if roi_radius is None:
            return None
        if ego_id is not None:
            selection = RoISelector.select_around_agent(
                participants,
                frame,
                ego_id=ego_id,
                radius=roi_radius,
                outer_radius=roi_outer_radius,
            )
        elif roi_center is not None:
            selection = RoISelector.select_by_radius(
                participants,
                frame,
                center=roi_center,
                radius=roi_radius,
                outer_radius=roi_outer_radius,
            )
        else:
            return None
        return [*selection.agent_ids, *selection.background_agent_ids]

    def _clone_participants_at_frame(
        self,
        participants: Dict[object, object],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ):
        clones = {}
        selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
        for agent_id in selected_ids:
            participant = participants.get(agent_id)
            if not isinstance(participant, Vehicle) or not participant.trajectory.has_state(frame):
                continue
            initial_state = participant.trajectory.get_state(frame)
            clone = copy(participant)
            clone.trajectory = Trajectory(
                id_=participant.trajectory.id_,
                fps=round(1.0 / self.config.dt, 3),
                stable_freq=True,
            )
            clone.trajectory.add_state(
                State(
                    frame=frame,
                    x=initial_state.x,
                    y=initial_state.y,
                    heading=normalize_angle(initial_state.heading),
                    vx=initial_state.vx,
                    vy=initial_state.vy,
                )
            )
            clones[agent_id] = clone
        return clones

    def _advance_participants(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        next_frame: int,
        result: PlanningResult,
        committed_trajectories: Optional[Dict[object, Trajectory]] = None,
    ) -> None:
        controlled_ids = set(result.trajectories)
        background_states = self.scene_builder.build(
            participants,
            map_,
            frame,
            agent_ids=[agent_id for agent_id in participants if agent_id not in controlled_ids],
        )
        background_next_states = {
            agent_id: rollout[0]
            for agent_id, state in background_states.items()
            for rollout in [self.follower.rollout(state, LimSimAction.KS, map_, steps=1)]
            if rollout
        }

        for agent_id, participant in participants.items():
            next_state = None
            planned_trajectory = (committed_trajectories or {}).get(agent_id)
            if planned_trajectory is None:
                planned_trajectory = result.trajectories.get(agent_id)
            if planned_trajectory is not None and planned_trajectory.has_state(next_frame):
                planned_state = planned_trajectory.get_state(next_frame)
                next_state = State(
                    frame=next_frame,
                    x=planned_state.x,
                    y=planned_state.y,
                    heading=normalize_angle(planned_state.heading),
                    vx=planned_state.vx,
                    vy=planned_state.vy,
                )
            elif agent_id in background_next_states:
                next_state = self._state_from_decision_state(background_next_states[agent_id], next_frame)

            if next_state is not None:
                participant.trajectory.add_state(next_state)

    def _update_committed_trajectories(
        self,
        result: PlanningResult,
        frame: int,
        committed_trajectories: Dict[object, Trajectory],
        committed_actions: Dict[object, LimSimAction],
    ) -> None:
        next_frame = self._next_frame(frame)
        for agent_id in list(committed_trajectories):
            if agent_id not in result.actions or not committed_trajectories[agent_id].has_state(next_frame):
                committed_trajectories.pop(agent_id, None)
                committed_actions.pop(agent_id, None)
                continue
            result.actions[agent_id] = committed_actions[agent_id]

        for agent_id, action in result.actions.items():
            if not action.is_lane_change:
                continue
            if (
                agent_id not in committed_trajectories
                or committed_actions.get(agent_id) != action
            ):
                trajectory = result.trajectories.get(agent_id)
                if trajectory is not None:
                    committed_trajectories[agent_id] = trajectory
                    committed_actions[agent_id] = action

    def _prediction_memory(
        self,
        result: PlanningResult,
        committed_trajectories: Dict[object, Trajectory],
    ) -> Dict[object, Trajectory]:
        memory = dict(result.trajectories)
        for agent_id, trajectory in committed_trajectories.items():
            if agent_id in memory:
                memory[agent_id] = trajectory
        return memory

    def _state_from_decision_state(self, state, frame: int) -> State:
        return State(
            frame=frame,
            x=state.x,
            y=state.y,
            heading=normalize_angle(state.heading),
            vx=state.speed * np.cos(state.heading),
            vy=state.speed * np.sin(state.heading),
        )

    def _next_frame(self, frame: int) -> int:
        return int(round(frame + self.config.dt * 1000))
