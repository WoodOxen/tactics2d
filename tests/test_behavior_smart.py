# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the SMART behavior model (public API only)."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

pytest.importorskip("torch", reason="SMART tests require torch.")

from tactics2d.behavior import BehaviorModelBase, SmartBehaviorModel, SmartConfig
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

HISTORY_END_MS = 1000

# SMART assets are downloaded separately; ``asset_root`` points at them.
ASSET_ROOT = Path(__file__).resolve().parent.parent / "tactics2d/data/checkpoints/smart"


def _config(**kwargs):
    """Build a config aimed at the downloaded SMART assets.

    Returns:
        A config with ``asset_root`` set.

    Raises:
        Skipped: If the codebooks are not present on this machine.
    """

    for asset in ("motion_codebook.pkl", "map_codebook.pkl"):
        if not (ASSET_ROOT / asset).exists():
            pytest.skip(f"SMART assets are not present under {ASSET_ROOT}.")
    return SmartConfig(asset_root=str(ASSET_ROOT), **kwargs)


@pytest.fixture(scope="module")
def smart_model():
    """Load the self-trained SMART checkpoint once for the whole module."""
    checkpoint = ASSET_ROOT / "smart_waymo_ep0.pt"
    if not checkpoint.exists():
        pytest.skip("no self-trained SMART checkpoint on this machine.")

    return SmartBehaviorModel.from_checkpoint(str(checkpoint), config=_config(), device="cpu")


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


@pytest.mark.integration
@pytest.mark.slow
def test_smart_loads_released_checkpoint(smart_model):
    """The self-trained checkpoint loads into a usable behavior model."""
    assert isinstance(smart_model, BehaviorModelBase)
    assert smart_model.policy is not None


@pytest.mark.integration
@pytest.mark.slow
def test_smart_predict_runs(smart_model):
    """The public prediction entry point runs."""
    participants = _history_participants(count=3)

    predicted = smart_model.predict(
        participants, _straight_map(), frame=HISTORY_END_MS, agent_ids=[0]
    )

    assert set(predicted) == {0}
    assert isinstance(predicted[0], Trajectory)


@pytest.mark.integration
@pytest.mark.slow
def test_smart_closed_loop_runs(smart_model):
    """The closed loop replans over a 41-step scenario and commits every agent."""
    # A closed loop needs ground truth for the whole span, not just the warmup:
    # the model overwrites the future, but it can only start from frames the
    # scenario actually occupies.
    participants = _history_participants(count=3, frames=range(0, 4100, 100))

    result = smart_model.rollout(
        participants,
        _straight_map(),
        ego_id=0,
        frame_ms0=0,
        warmup_steps=11,
        planning_interval=10,
        scenario_steps=41,
    )

    assert result.ego_id == 0
    assert set(result.poses) == {0, 1, 2}
