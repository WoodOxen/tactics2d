# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Receding-horizon closed-loop replay for BITS."""

from typing import Dict, Iterable, Optional

from tactics2d.map.element import Map

from ..results import TrajectoryRolloutResult
from ..trajectory_rollout import TrajectoryReplayRunner
from .config import BitsConfig

DEFAULT_REPLAN_INTERVAL = 20


class BitsRollingRunner:
    """Configure the shared trajectory replay for BITS predictions."""

    def __init__(
        self,
        model,
        config: Optional[BitsConfig] = None,
        horizon_ms: Optional[int] = None,
        replan_interval: int = DEFAULT_REPLAN_INTERVAL,
    ):
        self.model = model
        self.config = config if config is not None else model.config
        self.horizon_ms = (
            self.config.planning_steps * self.config.step_ms
            if horizon_ms is None
            else int(horizon_ms)
        )
        self.replan_interval = int(replan_interval)

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms: Optional[int] = None,
        controlled_ids: Optional[Iterable[object]] = None,
    ) -> TrajectoryRolloutResult:
        runner = TrajectoryReplayRunner(
            self.config.step_ms, self.horizon_ms, commit_steps=self.replan_interval
        )

        def predict(current_participants, current, controlled, _context):
            return self.model.predict(current_participants, map_, current, agent_ids=controlled)

        return runner.run(
            participants, ego_id, predict, frame_ms=frame_ms, controlled_ids=controlled_ids
        )
