# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the BITS behavior model (public API only)."""

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

pytest.importorskip("torch", reason="BITS torch tests require the tactics2d[bits] extra.")
pytest.importorskip("torchvision", reason="BITS torch tests require the tactics2d[bits] extra.")

from tactics2d.behavior import BehaviorModelBase
from tactics2d.behavior.bits import BitsBehaviorModel
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


def _poses(trajectory):
    """Return the frames and ``[x, y, heading, speed]`` poses of a trajectory."""
    frames = sorted(trajectory.frames)
    poses = np.zeros((len(frames), 4), dtype=float)
    for index, frame in enumerate(frames):
        state = trajectory.get_state(frame)
        poses[index] = [state.x, state.y, state.heading, state.speed]
    return np.asarray(frames, dtype=float), poses


def _trajectory_arrays(trajectories):
    """Flatten trajectories into a ``frames_<agent>`` / ``poses_<agent>`` mapping."""
    arrays = {}
    for agent_id, trajectory in trajectories.items():
        frames, poses = _poses(trajectory)
        arrays[f"frames_{agent_id}"] = frames
        arrays[f"poses_{agent_id}"] = poses
    return arrays


def _is_finite(poses):
    """Return whether the pose rows hold finite numbers."""
    return bool(np.isfinite(np.asarray(poses, dtype=float)).all())


def _dump(runtime_dir, name, arrays, meta=None):
    """Write the arrays (and optional summary) under the test's runtime directory."""
    if arrays:
        np.savez_compressed(str(runtime_dir / f"{name}.npz"), **arrays)
    if meta is not None:
        (runtime_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    return runtime_dir / f"{name}.json" if meta is not None else runtime_dir / f"{name}.npz"


@pytest.mark.integration
@pytest.mark.slow
def test_bits_loads_released_checkpoints(bits_model, runtime_dir):
    """The released planner and predictor load into a usable behavior model."""
    assert isinstance(bits_model, BehaviorModelBase)
    assert bits_model.policy is not None

    path = _dump(
        runtime_dir,
        "bits_load",
        {},
        {
            "future_steps": bits_model.config.future_steps,
            "history_steps": bits_model.config.history_steps,
            "dt": bits_model.config.dt,
        },
    )
    assert path.exists()


@pytest.mark.integration
@pytest.mark.slow
def test_bits_predict_returns_finite_trajectories(bits_model, runtime_dir):
    """predict() returns a finite world-frame trajectory per requested agent."""
    participants = {
        0: _straight_vehicle(0, range(0, 3100, 100), 0.0, 0.0, speed=8.0),
        1: _straight_vehicle(1, range(0, 3100, 100), 20.0, 5.0, speed=3.0),
    }

    predicted = bits_model.predict(participants, _straight_map(), frame=1000, agent_ids=[0])

    assert set(predicted) == {0}
    trajectory = predicted[0]
    assert isinstance(trajectory, Trajectory)
    assert 1 <= len(trajectory.frames) <= bits_model.config.future_steps
    assert _is_finite(_poses(trajectory)[1])

    assert _dump(runtime_dir, "bits_predict", _trajectory_arrays(predicted)).exists()


@pytest.mark.integration
@pytest.mark.slow
def test_bits_mpc_closed_loop_runs(bits_model, runtime_dir):
    """A receding-horizon loop replans every 500 ms and commits the ego plan."""
    map_ = _straight_map()
    participants = {
        0: _straight_vehicle(0, range(0, 1100, 100), 0.0, 0.0, speed=8.0),
        1: _straight_vehicle(1, range(0, 3100, 100), 20.0, 5.0, speed=3.0),
    }

    committed = Trajectory(id_=0, fps=10, stable_freq=False)
    frame = 1000
    for _ in range(4):
        predicted = bits_model.predict(participants, map_, frame=frame, agent_ids=[0])
        future = [f for f in sorted(predicted[0].frames) if f > frame]
        assert future

        for next_frame in future[:5]:
            state = predicted[0].get_state(next_frame)
            participants[0].trajectory.add_state(state)
            committed.add_state(state)
        frame = future[4]

    assert len(committed.frames) == 20
    assert sorted(committed.frames) == list(committed.frames)
    assert _is_finite(_poses(committed)[1])

    path = _dump(
        runtime_dir,
        "bits_mpc",
        _trajectory_arrays({0: committed}),
        {"committed_rows": len(committed.frames)},
    )
    assert path.exists()
