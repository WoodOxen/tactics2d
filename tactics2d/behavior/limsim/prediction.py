# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rule-based prediction used by the LimSim-style behavior stack."""

from typing import Dict, Iterable, Optional

from tactics2d.map.element import Map
from tactics2d.participant.trajectory import Trajectory

from .config import LimSimConfig
from .planner import LaneFollower
from .scene import SceneBuilder
from .schema import states_to_trajectory


class LimSimPredictor:
    """Predict short horizon trajectories with LimSim's default assumptions.

    The original LimSim predictor reuses remaining planned trajectories for
    controlled vehicles and uses constant-speed lane following for background
    vehicles. This lightweight version follows the same rule-based spirit.
    """

    def __init__(self, config: LimSimConfig):
        self.config = config
        self.scene_builder = SceneBuilder(config)
        self.follower = LaneFollower(config)

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        last_planned_trajectories: Optional[Dict[object, Trajectory]] = None,
    ) -> Dict[object, Trajectory]:
        """Predict future trajectories for active participants."""

        prediction = {}
        selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
        for agent_id in selected_ids:
            remaining = self._remaining_trajectory(agent_id, frame, last_planned_trajectories or {})
            if remaining is not None:
                prediction[agent_id] = remaining
                continue

            states = self.scene_builder.build(participants, map_, frame, [agent_id])
            if agent_id not in states:
                continue
            future_states = self.follower.rollout(states[agent_id], states[agent_id].action, map_)
            prediction[agent_id] = states_to_trajectory(
                agent_id, future_states, frame, self.config.dt
            )
        return prediction

    def _remaining_trajectory(
        self, agent_id: object, frame: int, last_planned_trajectories: Dict[object, Trajectory]
    ) -> Optional[Trajectory]:
        trajectory = last_planned_trajectories.get(agent_id)
        if trajectory is None:
            return None

        remaining_frames = [state_frame for state_frame in trajectory.frames if state_frame > frame]
        if not remaining_frames:
            return None

        remaining = Trajectory(id_=agent_id, fps=trajectory.fps, stable_freq=trajectory.stable_freq)
        for state_frame in remaining_frames:
            remaining.add_state(trajectory.get_state(state_frame))
        return remaining
