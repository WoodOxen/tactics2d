# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Short-horizon closed-loop runner for BITS-style behavior models."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from tactics2d.geometry import normalize_angle
from tactics2d.behavior._rolling import clone_vehicle_participants_at_frames, copy_state
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State, Trajectory

from .config import BitsConfig
from .model import BitsBehaviorModel


@dataclass
class BitsRollingResult:
    """Output of a rolling BITS simulation."""

    participants: Dict[object, object]
    frames: List[int] = field(default_factory=list)
    predicted_trajectories: List[Dict[object, Trajectory]] = field(default_factory=list)


class BitsRollingRunner:
    """Wrap a single-frame BITS model in a short receding-horizon loop.

    The runner intentionally stays narrow: BITS remains responsible for
    predicting trajectories, while this class only commits the first predicted
    state and advances background vehicles by log replay or one-step constant
    velocity extrapolation.
    """

    def __init__(
        self,
        config: Optional[BitsConfig] = None,
        behavior_model: Optional[BitsBehaviorModel] = None,
    ):
        self.config = config or BitsConfig()
        if behavior_model is None:
            raise ValueError(
                "behavior_model is required; pass BitsBehaviorModel() or a custom "
                "BitsBehaviorModel with an explicit policy."
            )
        self.behavior_model = behavior_model

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        start_frame: int,
        simulation_steps: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> BitsRollingResult:
        """Run repeated BITS updates for ``simulation_steps`` control steps."""

        if simulation_steps < 0:
            raise ValueError("simulation_steps must be non-negative.")

        frame = int(start_frame)
        simulated_participants = clone_vehicle_participants_at_frames(
            participants,
            frames=self._history_frames(frame),
            required_frame=frame,
            fps=round(1.0 / self.config.dt, 3),
        )
        controlled_ids = set(simulated_participants) if agent_ids is None else {
            agent_id for agent_id in agent_ids if agent_id in simulated_participants
        }
        frames = [frame]
        predicted_trajectories = []

        for _ in range(simulation_steps):
            predictions = self.behavior_model.predict(
                simulated_participants,
                map_,
                frame=frame,
                agent_ids=controlled_ids,
            )
            predicted_trajectories.append(predictions)
            next_frame = self._next_frame(frame)
            self._advance_participants(
                simulated_participants=simulated_participants,
                source_participants=participants,
                predictions=predictions,
                controlled_ids=controlled_ids,
                frame=frame,
                next_frame=next_frame,
            )
            frame = next_frame
            frames.append(frame)

        return BitsRollingResult(
            participants=simulated_participants,
            frames=frames,
            predicted_trajectories=predicted_trajectories,
        )

    def _advance_participants(
        self,
        simulated_participants: Dict[object, object],
        source_participants: Dict[object, object],
        predictions: Dict[object, Trajectory],
        controlled_ids: Iterable[object],
        frame: int,
        next_frame: int,
    ) -> None:
        controlled_ids = set(controlled_ids)
        for agent_id, participant in simulated_participants.items():
            next_state = None
            planned = predictions.get(agent_id)
            if planned is not None and planned.has_state(next_frame):
                next_state = copy_state(planned.get_state(next_frame), next_frame)
            elif agent_id not in controlled_ids:
                next_state = self._background_next_state(
                    source_participants.get(agent_id),
                    participant,
                    frame,
                    next_frame,
                )

            if next_state is not None:
                participant.trajectory.add_state(next_state)

    def _background_next_state(
        self,
        source_participant,
        simulated_participant,
        frame: int,
        next_frame: int,
    ) -> Optional[State]:
        if (
            source_participant is not None
            and source_participant.trajectory.has_state(next_frame)
        ):
            return copy_state(source_participant.trajectory.get_state(next_frame), next_frame)

        if not simulated_participant.trajectory.has_state(frame):
            return None
        state = simulated_participant.trajectory.get_state(frame)
        dt = max(0.0, (next_frame - frame) / 1000.0)
        vx, vy = state.velocity or (
            (state.speed or 0.0) * np.cos(state.heading),
            (state.speed or 0.0) * np.sin(state.heading),
        )
        return State(
            frame=next_frame,
            x=float(state.x + vx * dt),
            y=float(state.y + vy * dt),
            heading=normalize_angle(state.heading),
            vx=float(vx),
            vy=float(vy),
        )

    def _next_frame(self, frame: int) -> int:
        return int(round(frame + self.config.step_ms))

    def _history_frames(self, frame: int) -> List[int]:
        return [
            int(frame - self.config.step_ms * offset)
            for offset in range(self.config.history_steps, -1, -1)
        ]
