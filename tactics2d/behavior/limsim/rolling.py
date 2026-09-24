# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Receding-horizon closed-loop replay for LimSim."""

from typing import Dict, Iterable, Optional, Tuple

from tactics2d.map.element import Map

from ..results import TrajectoryRolloutResult
from ..trajectory_rollout import TrajectoryReplayRunner
from .config import LimSimConfig


class LimSimRollingRunner:
    """Configure the shared trajectory replay for LimSim predictions."""

    def __init__(
        self, model, config: Optional[LimSimConfig] = None, horizon_ms: Optional[int] = None
    ):
        self.model = model
        self.config = config if config is not None else model.config
        self.horizon_ms = (
            self.config.planning_steps * self.config.step_ms
            if horizon_ms is None
            else int(horizon_ms)
        )

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms: Optional[int] = None,
        route_map: Optional[Dict[object, Tuple[str, ...]]] = None,
        controlled_ids: Optional[Iterable[object]] = None,
    ) -> TrajectoryRolloutResult:
        runner = TrajectoryReplayRunner(
            self.config.step_ms, self.horizon_ms, commit_steps=1, suppress_prediction_errors=True
        )

        def prepare(current_participants, take_over, controlled):
            if route_map is not None:
                return route_map
            from tactics2d.dataset_parser.route_extractor import extract_all_lane_sequences

            return extract_all_lane_sequences(
                current_participants, map_, take_over, agent_ids=controlled
            )

        def predict(current_participants, current, controlled, routes):
            return self.model.predict(
                current_participants, map_, current, agent_ids=controlled, route_map=routes
            )

        return runner.run(
            participants,
            ego_id,
            predict,
            frame_ms=frame_ms,
            controlled_ids=controlled_ids,
            prepare=prepare,
        )
