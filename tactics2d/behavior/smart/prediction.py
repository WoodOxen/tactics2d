# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""World-frame predictions returned by SMART."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tactics2d.behavior.trajectory_processing import trajectory_from_poses
from tactics2d.participant.trajectory import Trajectory


@dataclass(frozen=True)
class SmartPrediction:
    """A joint SMART rollout for every modelled agent."""

    agent_ids: List[object]
    positions: np.ndarray
    headings: np.ndarray
    availabilities: np.ndarray
    frame_ms0: int
    step_ms: int
    frames: Optional[List[int]] = None

    def trajectory(self, agent_id: object) -> Trajectory:
        """Return one agent's rollout as a native trajectory."""

        try:
            row = self.agent_ids.index(agent_id)
        except ValueError as error:
            raise KeyError(agent_id) from error
        frames = self.frames or [
            self.frame_ms0 + self.step_ms * (step + 1) for step in range(self.positions.shape[1])
        ]
        return trajectory_from_poses(
            agent_id,
            self.positions[row],
            self.headings[row],
            frames,
            availabilities=self.availabilities[row],
            step_ms=self.step_ms,
        )
