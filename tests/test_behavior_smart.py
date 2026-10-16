# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the SMART behavior model (public API only)."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

pytest.importorskip("torch", reason="SMART tests require torch.")

from tactics2d.behavior import SmartBehaviorModel, SmartConfig
from tactics2d.behavior.smart.prediction import SmartPrediction
from tactics2d.behavior.smart.rollout import SmartRolloutRunner
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

# SMART assets are downloaded separately; ``asset_root`` points at them.
ASSET_ROOT = Path(__file__).resolve().parent.parent / "tactics2d/data/checkpoints/smart"


def _state(frame, x, y, heading=0.0, speed=5.0):
    return State(
        frame=frame,
        x=x,
        y=y,
        heading=heading,
        vx=speed * np.cos(heading),
        vy=speed * np.sin(heading),
    )


def _vehicle(agent_id, frames, x=1.0, y=0.0, length=4.5, width=1.8):
    """Build a straight-moving vehicle over a frame grid."""

    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    for step, frame in enumerate(frames):
        trajectory.add_state(_state(frame, x + 0.5 * step, y))
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=length, width=width)


def _lane(id_, centerline, subtype="road"):
    """Build a lane whose centerline is the given polyline."""

    points = np.asarray(centerline, dtype=float)
    left = LineString([(x, y + 1.75) for x, y in points])
    right = LineString([(x, y - 1.75) for x, y in points])
    return Lane(
        id_=id_,
        left_side=left,
        right_side=right,
        subtype=subtype,
        custom_tags={"centerline": points},
    )


def _straight_map():
    """Build a map with one long straight lane and nothing else."""

    map_ = Map(name="smart_test_map")
    map_.add_lane(_lane("A", [[0.0, 0.0], [9.0, 0.0], [18.0, 0.0], [27.0, 0.0], [36.0, 0.0]]))
    return map_


def _history_participants(count=3, spacing=8.0, frames=None):
    """Build a column of vehicles with a full history ending at 1000 ms."""

    if frames is None:
        frames = range(0, 1100, 100)
    return {index: _vehicle(index, frames, x=0.0, y=spacing * index) for index in range(count)}


class _ShortHorizonSmartModel:
    def __init__(self, config):
        self.config = config
        self.requested_steps = []

    def predict_scene(self, participants, map_, frame, agent_ids=None, future_steps=None):
        del map_
        self.requested_steps.append(future_steps)
        agent_ids = list(agent_ids)
        positions = np.zeros((len(agent_ids), future_steps, 2), dtype=float)
        headings = np.zeros((len(agent_ids), future_steps), dtype=float)
        for row, agent_id in enumerate(agent_ids):
            state = participants[agent_id].trajectory.get_state(frame)
            positions[row, :, 0] = state.x + np.arange(1, future_steps + 1) * 0.5
            positions[row, :, 1] = state.y
            headings[row] = state.heading
        return SmartPrediction(
            agent_ids=agent_ids,
            positions=positions,
            headings=headings,
            availabilities=np.ones(positions.shape[:2], dtype=bool),
            frame_ms0=frame,
            step_ms=self.config.step_ms,
        )


def test_smart_closed_loop_requests_only_committed_future():
    config = SmartConfig()
    model = _ShortHorizonSmartModel(config)
    participants = _history_participants(count=1, frames=range(0, 4100, 100))
    runner = SmartRolloutRunner(
        model, config=config, warmup_steps=11, planning_interval=10, scenario_steps=31
    )

    runner.run(participants, _straight_map(), ego_id=0, start_frame=0, controlled_ids=[0])

    assert model.requested_steps == [10, 8]


@pytest.mark.integration
@pytest.mark.slow
def test_smart_rollout_runs(womd_behavior_scene):
    """The released SMART integration completes a CPU rollout on WOMD data."""
    required = ["smart_waymo_ep0.pt", "motion_codebook.pkl", "map_codebook.pkl"]
    if not all((ASSET_ROOT / asset).exists() for asset in required):
        pytest.skip(f"Released SMART assets are not present under {ASSET_ROOT}.")
    model = SmartBehaviorModel.from_checkpoint(
        str(ASSET_ROOT / "smart_waymo_ep0.pt"),
        config=SmartConfig(asset_root=str(ASSET_ROOT)),
        device="cpu",
    )
    participants, map_, controlled_ids = womd_behavior_scene
    ego_id = controlled_ids[0]

    result = model.rollout(
        participants,
        map_,
        ego_id=ego_id,
        start_frame=0,
        warmup_steps=11,
        planning_interval=10,
        scenario_steps=31,
        controlled_ids=controlled_ids,
    )

    assert result.ego_id == ego_id
    assert set(result.poses) == set(controlled_ids)
    assert result.frames
    assert result.cycles > 0
    assert result.controlled_ids == result.modeled_ids
    assert set(result.trajectories) == set(result.modeled_ids)
