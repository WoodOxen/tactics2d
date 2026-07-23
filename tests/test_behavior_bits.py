# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the BITS behavior model (public API only)."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

pytest.importorskip("torch", reason="BITS torch tests require the tactics2d[bits] extra.")
pytest.importorskip("torchvision", reason="BITS torch tests require the tactics2d[bits] extra.")

from tactics2d.behavior import BehaviorModelBase
from tactics2d.behavior.bits import BitsBehaviorModel, BitsConfig
from tactics2d.behavior.bits.predictor import BitsPrediction
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _vehicle(agent_id, states, length=4.5, width=1.8):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    for state in states:
        trajectory.add_state(state)
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=length, width=width)


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
    map_ = Map(name="bits_test_map")
    lane = Lane(
        id_="A",
        left_side=LineString([(0.0, -2.0), (60.0, -2.0)]),
        right_side=LineString([(0.0, 2.0), (60.0, 2.0)]),
        custom_tags={"centerline": np.array([[0.0, 0.0], [60.0, 0.0]])},
    )
    map_.add_lane(lane)
    return map_


class _FixedPolicy:
    """Minimal policy mock that returns pre-set predictions."""

    def __init__(self, positions, yaws=None, scores=None):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 2:
            positions = positions[None]
        self.positions = positions
        self.yaws = (
            np.zeros((*positions.shape[:2], 1), dtype=float) if yaws is None else np.asarray(yaws)
        )
        self.scores = (
            np.ones(positions.shape[0], dtype=float) if scores is None else np.asarray(scores)
        )

    def predict_batch(self, batch):
        return BitsPrediction(
            positions=self.positions.copy(),
            yaws=self.yaws.copy(),
            availabilities=np.ones(self.positions.shape[:2], dtype=bool),
            scores=self.scores.copy(),
        )


def test_bits_is_behavior_model():
    """BitsBehaviorModel implements the shared BehaviorModelBase interface."""
    model = BitsBehaviorModel(
        BitsConfig(history_steps=0, future_steps=1),
        policy=_FixedPolicy([[1.0, 0.0]]),
        include_raster=False,
    )
    assert isinstance(model, BehaviorModelBase)


def test_bits_requires_policy():
    """Construction without a policy raises ValueError."""
    with pytest.raises(ValueError, match="A BitsPolicy is required"):
        BitsBehaviorModel(BitsConfig(history_steps=0, future_steps=1), include_raster=False)


def test_bits_predict():
    """predict() returns world-frame trajectories for each requested agent."""
    config = BitsConfig(history_steps=1, future_steps=2, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(
            1,
            [_state(-100, 10.0, 8.0, heading=np.pi / 2), _state(0, 10.0, 10.0, heading=np.pi / 2)],
        )
    }
    model = BitsBehaviorModel(
        config, policy=_FixedPolicy([[2.0, 0.0], [4.0, 0.0]]), include_raster=False
    )

    trajectories = model.predict(participants, _straight_map(), frame=0, agent_ids=[1])
    trajectory = trajectories[1]

    assert trajectory.frames == [100, 200]
    np.testing.assert_allclose(trajectory.get_state(100).location, (10.0, 12.0), atol=1e-9)
    np.testing.assert_allclose(trajectory.get_state(200).location, (10.0, 14.0), atol=1e-9)


def test_bits_predict_batch():
    """predict_batch() returns a BitsPrediction with the expected shape."""
    config = BitsConfig(history_steps=0, future_steps=1)
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0, speed=3.0)])}
    model = BitsBehaviorModel(config, policy=_FixedPolicy([[0.3, 0.0]]), include_raster=False)

    batch = model.builder.build(participants, frame=0, ego_id=1)
    prediction = model.predict_batch(batch)

    assert isinstance(prediction, BitsPrediction)
    assert prediction.positions.shape == (1, 1, 2)
    np.testing.assert_allclose(prediction.positions[0, 0], [0.3, 0.0])


@pytest.mark.slow
def test_bits_from_checkpoint(tmp_path):
    """from_checkpoint() loads weights and produces structurally valid predictions."""
    import torch

    from tactics2d.behavior.bits.model import BitsBiLevelTorchModel

    # --- create a synthetic checkpoint ---
    ckpt_config = BitsConfig(history_steps=0, future_steps=2, dt=0.1, raster_size=64)
    # image_channels matches raster output: static(3) + other-agent blob(1) = 4
    model = BitsBiLevelTorchModel(
        image_channels=4, future_steps=2, hidden_dim=8, config=ckpt_config
    )
    model.eval()

    checkpoint = {
        "metadata": {
            "image_channels": 4,
            "future_steps": 2,
            "hidden_dim": 8,
            "model_arch": "resnet18",
            "config": {"future_steps": 2, "dt": 0.1, "history_steps": 0, "raster_size": 64},
        },
        "model_state_dict": model.state_dict(),
    }
    ckpt_path = tmp_path / "model.ckpt"
    torch.save(checkpoint, ckpt_path)

    # --- load and run inference ---
    loaded = BitsBehaviorModel.from_checkpoint(ckpt_path)
    participants = {
        1: _vehicle(
            1,
            [
                _state(0, 10.0, 10.0, heading=np.pi / 2),
                _state(100, 10.0, 12.0, heading=np.pi / 2),
                _state(200, 10.0, 14.0, heading=np.pi / 2),
            ],
        )
    }
    trajectories = loaded.predict(participants, _straight_map(), frame=0, agent_ids=[1])
    trajectory = trajectories[1]

    assert trajectory.frames == [100, 200]
    assert len(trajectory.frames) == 2


@pytest.mark.slow
@pytest.mark.dataset_parser
def test_bits_from_nuplan_checkpoint_with_history_conditioning(tmp_path):
    """from_checkpoint() with history_conditioning=True exercises the RNN encoder.

    Loads a real NuPlan scenario, resamples its trajectory to 50 ms intervals
    (matching NuPlan's native FPS=20), then runs inference through a model
    with ``history_conditioning=True``.
    """
    import torch

    from tactics2d.behavior.bits.model import BitsBiLevelTorchModel
    from tactics2d.dataset_parser import NuPlanParser

    # --- parse a real NuPlan scenario ---
    folder_path = "./tactics2d/data/trajectory_sample/NuPlan/data/cache"
    file_name = "train_vegas_1/2021.05.18.21.31.22_veh-30_00062_00160.db"
    if not Path(folder_path, file_name).exists():
        pytest.skip("NuPlan data not found — skipping NuPlan-based test.")

    participants, _ = NuPlanParser().parse_trajectory(file_name, folder_path)
    vehicles = [(pid, p) for pid, p in participants.items() if p.type_ == "vehicle"]
    vehicles.sort(key=lambda x: len(x[1].trajectory.history_states), reverse=True)
    nuplan_pid, nuplan_p = vehicles[0]

    # resample with exact 50 ms spacing (NuPlan FPS=20, but raw timestamps
    # have occasional 49 ms jitter)
    raw_states = list(nuplan_p.trajectory.history_states.values())
    base_frame = raw_states[len(raw_states) // 2 - 5].frame
    trajectory = Trajectory(id_=nuplan_pid, fps=20, stable_freq=True)
    for i, raw in enumerate(raw_states):
        trajectory.add_state(
            State(
                frame=base_frame + i * 50,
                x=raw.x,
                y=raw.y,
                heading=raw.heading,
                vx=raw.vx,
                vy=raw.vy,
            )
        )
    ego = Vehicle(
        nuplan_pid, "vehicle", trajectory=trajectory, length=nuplan_p.length, width=nuplan_p.width
    )

    # --- config matching NuPlan's FPS=20 (dt=50 ms) ---
    dt = 0.05
    hist_steps, fut_steps = 2, 5
    ckpt_config = BitsConfig(
        history_steps=hist_steps,
        future_steps=fut_steps,
        dt=dt,
        raster_size=64,
        max_agents_distance=5.0,
    )
    # image_channels: static(3) + time_steps(hist_steps+1)
    img_channels = 4 + hist_steps

    model = BitsBiLevelTorchModel(
        image_channels=img_channels,
        future_steps=fut_steps,
        hidden_dim=8,
        history_conditioning=True,
        config=ckpt_config,
    )
    model.eval()

    checkpoint = {
        "metadata": {
            "image_channels": img_channels,
            "future_steps": fut_steps,
            "hidden_dim": 8,
            "model_arch": "resnet18",
            "history_conditioning": True,
            "config": {
                "history_steps": hist_steps,
                "future_steps": fut_steps,
                "dt": dt,
                "raster_size": 64,
                "max_agents_distance": 5.0,
            },
        },
        "model_state_dict": model.state_dict(),
    }
    ckpt_path = tmp_path / "nuplan_model.ckpt"
    torch.save(checkpoint, ckpt_path)

    # --- pick a reference frame with enough history/future coverage ---
    mid = len(raw_states) // 2
    ref_frame = base_frame + mid * 50

    loaded = BitsBehaviorModel.from_checkpoint(ckpt_path)

    trajectories = loaded.predict({nuplan_pid: ego}, _straight_map(), frame=ref_frame)
    trajectory = trajectories[nuplan_pid]

    expected_frames = [ref_frame + (t + 1) * int(round(dt * 1000)) for t in range(fut_steps)]
    assert trajectory.frames == expected_frames
    assert len(trajectory.frames) == fut_steps
