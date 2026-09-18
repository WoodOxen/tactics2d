# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the LimSim behavior model (public API only)."""

import json

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.behavior.limsim import LimSimBehaviorModel, LimSimConfig
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _lane(lane_id, x_left, x_right, y_start, y_end):
    left_side = LineString([(x_left, y_start), (x_left, y_end)])
    right_side = LineString([(x_right, y_start), (x_right, y_end)])
    return Lane(
        id_=lane_id,
        left_side=left_side,
        right_side=right_side,
        custom_tags={
            "centerline": np.array(
                [[(x_left + x_right) / 2.0, y_start], [(x_left + x_right) / 2.0, y_end]]
            )
        },
    )


def _parallel_map():
    map_ = Map(name="parallel")
    map_.add_lane(_lane("A", 0.0, 2.0, 0.0, 80.0))
    map_.add_lane(_lane("B", 2.0, 4.0, 0.0, 80.0))
    return map_


def _vehicle(agent_id, frames, x, y, heading=np.pi / 2, speed=5.0):
    """Build a vehicle moving at ``speed`` along ``heading`` over ``frames``."""
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for index, frame in enumerate(frames):
        travelled = speed * 0.1 * index
        trajectory.add_state(
            State(
                frame=frame,
                x=x + travelled * np.cos(heading),
                y=y + travelled * np.sin(heading),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
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
def test_limsim_plan_contract(runtime_dir):
    """plan() returns actions and finite trajectories for every controlled agent."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    map_ = _parallel_map()
    participants = {
        1: _vehicle(1, [0], 1.0, 5.0, speed=5.0),
        2: _vehicle(2, [0], 3.0, 10.0, speed=2.0),
    }

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, route_map={1: ("A",), 2: ("B",)}
    )

    assert set(result.actions) == {1, 2}
    assert set(result.trajectories) == {1, 2}
    for trajectory in result.trajectories.values():
        assert isinstance(trajectory, Trajectory)
        assert 1 <= len(trajectory.frames) <= config.horizon_steps
        assert _is_finite(_poses(trajectory)[1])

    path = _dump(
        runtime_dir,
        "limsim_plan",
        _trajectory_arrays(result.trajectories),
        {"actions": {str(key): str(action) for key, action in result.actions.items()}},
    )
    assert path.exists()


@pytest.mark.integration
def test_limsim_predict_is_behavior_interface(runtime_dir):
    """predict() honours the shared behavior-model interface."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    participants = {1: _vehicle(1, [0], 3.0, 5.0, speed=5.0)}
    model = LimSimBehaviorModel(config)

    assert isinstance(model, BehaviorModelBase)
    predicted = model.predict(
        participants, _parallel_map(), frame=0, agent_ids=[1], route_map={1: ("B",)}
    )

    assert set(predicted) == {1}
    assert isinstance(predicted[1], Trajectory)
    assert 1 <= len(predicted[1].frames) <= config.horizon_steps
    assert _is_finite(_poses(predicted[1])[1])

    assert _dump(runtime_dir, "limsim_predict", _trajectory_arrays(predicted)).exists()


@pytest.mark.integration
def test_limsim_mpc_closed_loop_runs(runtime_dir):
    """A receding-horizon loop replans and commits the ego one step at a time."""
    config = LimSimConfig(horizon_steps=10, dt=0.1, mcts_iterations=50, interaction_distance=20.0)
    map_ = _parallel_map()
    participants = {
        1: _vehicle(1, range(0, 1100, 100), 1.0, 0.0, speed=5.0),
        2: _vehicle(2, range(0, 2100, 100), 3.0, 20.0, speed=3.0),
    }
    model = LimSimBehaviorModel(config)
    route_map = {1: ("A",), 2: ("B",)}

    committed = Trajectory(id_=1, fps=10, stable_freq=True)
    frame = 1000
    for _ in range(10):
        result = model.plan(participants, map_, frame, route_map=route_map, agent_ids=[1])
        future = [f for f in sorted(result.trajectories[1].frames) if f > frame]
        assert future
        state = result.trajectories[1].get_state(future[0])
        participants[1].trajectory.add_state(state)
        committed.add_state(state)
        frame = future[0]

    assert len(committed.frames) == 10
    assert sorted(committed.frames) == list(committed.frames)
    assert _is_finite(_poses(committed)[1])

    path = _dump(
        runtime_dir,
        "limsim_mpc",
        _trajectory_arrays({1: committed}),
        {"committed_rows": len(committed.frames)},
    )
    assert path.exists()
