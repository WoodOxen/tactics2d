# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the BITS behavior model (public API only)."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

pytest.importorskip("torch", reason="BITS torch tests require the tactics2d[behavior] extra.")
pytest.importorskip("torchvision", reason="BITS torch tests require the tactics2d[behavior] extra.")

from tactics2d.behavior import BehaviorModelBase, BitsBehaviorModel
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

# Released BITS checkpoints; downloaded separately from the Hugging Face repo.
ASSET_ROOT = Path(__file__).resolve().parent.parent / "tactics2d/data/checkpoints/bits"
PLANNER_CHECKPOINT = ASSET_ROOT / "bits_planner_nusc_resnet50.ckpt"
PREDICTOR_CHECKPOINT = ASSET_ROOT / "bits_predictor_nusc_resnet18.ckpt"


@pytest.fixture(scope="module")
def bits_model():
    """Load the released TBSIM planner and predictor once for the whole module."""
    if not PLANNER_CHECKPOINT.exists() or not PREDICTOR_CHECKPOINT.exists():
        pytest.skip(f"Released BITS checkpoints are not present under {ASSET_ROOT}.")

    # The checkpoint is self-describing; the two paths are the whole input.
    return BitsBehaviorModel.from_trained_planner(
        planner_checkpoint=PLANNER_CHECKPOINT,
        predictor_checkpoint=PREDICTOR_CHECKPOINT,
        map_location="cpu",
        device="cpu",
    )


def _state(frame, x, y, heading=0.0, speed=5.0):
    return State(
        frame=frame,
        x=x,
        y=y,
        heading=heading,
        vx=speed * np.cos(heading),
        vy=speed * np.sin(heading),
    )


def _straight_map():
    """Build two parallel lanes; the raster reads both the sides and the centerline."""
    map_ = Map(name="bits_test_map")
    for lane_id, y in (("A", 0.0), ("B", 5.0)):
        map_.add_lane(
            Lane(
                id_=lane_id,
                left_side=LineString([(0.0, y - 2.0), (200.0, y - 2.0)]),
                right_side=LineString([(0.0, y + 2.0), (200.0, y + 2.0)]),
                custom_tags={"centerline": np.array([[0.0, y], [200.0, y]])},
            )
        )
    return map_


def _straight_vehicle(agent_id, frames, x, y, speed):
    """Build a vehicle cruising along +x at ``speed`` over the given frame grid."""
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    for index, frame in enumerate(frames):
        trajectory.add_state(_state(frame, x + speed * 0.1 * index, y, speed=speed))
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


@pytest.mark.integration
@pytest.mark.slow
def test_bits_loads_released_checkpoints(bits_model):
    """The released planner and predictor load into a usable behavior model."""
    assert isinstance(bits_model, BehaviorModelBase)
    assert bits_model.policy is not None


@pytest.mark.integration
@pytest.mark.slow
def test_bits_predict_runs(bits_model):
    """The public prediction entry point runs."""
    participants = {
        0: _straight_vehicle(0, range(0, 3100, 100), 0.0, 0.0, speed=8.0),
        1: _straight_vehicle(1, range(0, 3100, 100), 20.0, 5.0, speed=3.0),
    }

    predicted = bits_model.predict(participants, _straight_map(), frame=1000, agent_ids=[0])

    assert set(predicted) == {0}
    assert isinstance(predicted[0], Trajectory)


@pytest.mark.integration
@pytest.mark.slow
def test_bits_rollout_runs(bits_model):
    """The public closed-loop entry point runs on recorded participants."""
    map_ = _straight_map()
    participants = {
        0: _straight_vehicle(0, range(0, 3100, 100), 0.0, 0.0, speed=8.0),
        1: _straight_vehicle(1, range(0, 3100, 100), 20.0, 5.0, speed=3.0),
    }

    result = bits_model.rollout(participants, map_, ego_id=0, frame_ms=1000, horizon_ms=2000)

    assert result.cycles > 0
    assert isinstance(result.trajectory, Trajectory)
