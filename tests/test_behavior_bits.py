# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the BITS behavior model (public API only)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="BITS torch tests require the tactics2d[behavior] extra.")
pytest.importorskip("torchvision", reason="BITS torch tests require the tactics2d[behavior] extra.")

from tactics2d.behavior import BitsBehaviorModel, BitsConfig
from tactics2d.behavior.bits.policy import BitsPolicy
from tactics2d.behavior.bits.prediction import BitsPrediction
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

# Released BITS checkpoints; downloaded separately from the Hugging Face repo.
ASSET_ROOT = Path(__file__).resolve().parent.parent / "tactics2d/data/checkpoints/bits"
PLANNER_CHECKPOINT = ASSET_ROOT / "bits_planner_nusc_resnet50.ckpt"
PREDICTOR_CHECKPOINT = ASSET_ROOT / "bits_predictor_nusc_resnet18.ckpt"


def _state(frame, x, y, heading=0.0, speed=5.0):
    return State(
        frame=frame,
        x=x,
        y=y,
        heading=heading,
        vx=speed * np.cos(heading),
        vy=speed * np.sin(heading),
    )


def _straight_vehicle(agent_id, frames, x, y, speed):
    """Build a vehicle cruising along +x at ``speed`` over the given frame grid."""
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    for index, frame in enumerate(frames):
        trajectory.add_state(_state(frame, x + speed * 0.1 * index, y, speed=speed))
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


class _BatchPolicy(BitsPolicy):
    def __init__(self, config):
        self.config = config
        self.batch_sizes = []

    def _prediction(self):
        positions = np.zeros((1, self.config.future_steps, 2), dtype=float)
        positions[0, :, 0] = np.arange(1, self.config.future_steps + 1)
        return BitsPrediction(
            positions=positions,
            yaws=np.zeros((1, self.config.future_steps, 1), dtype=float),
            availabilities=np.ones((1, self.config.future_steps), dtype=bool),
            scores=np.ones(1, dtype=float),
        )

    def predict_batch(self, batch):
        raise AssertionError("multi-agent prediction should use predict_batches")

    def predict_batches(self, batches):
        self.batch_sizes.append(len(batches))
        return [self._prediction() for _ in batches]


def test_bits_predict_micro_batches_controlled_agents():
    config = BitsConfig(future_steps=4)
    policy = _BatchPolicy(config)
    model = BitsBehaviorModel(config=config, policy=policy, include_raster=False)
    participants = {
        index: _straight_vehicle(index, range(0, 2100, 100), 0.0, 4.0 * index, speed=5.0)
        for index in range(6)
    }

    trajectories = model.predict(participants, None, frame=1000, agent_ids=participants)

    assert policy.batch_sizes == [4, 2]
    assert set(trajectories) == set(participants)


@pytest.mark.integration
@pytest.mark.slow
def test_bits_rollout_runs(womd_behavior_scene):
    """The released BITS integration completes a CPU rollout on WOMD data."""
    if not PLANNER_CHECKPOINT.exists() or not PREDICTOR_CHECKPOINT.exists():
        pytest.skip(f"Released BITS checkpoints are not present under {ASSET_ROOT}.")
    model = BitsBehaviorModel.from_trained_planner(
        planner_checkpoint=PLANNER_CHECKPOINT,
        predictor_checkpoint=PREDICTOR_CHECKPOINT,
        map_location="cpu",
        device="cpu",
    )
    participants, map_, controlled_ids = womd_behavior_scene
    ego_id = controlled_ids[0]

    result = model.rollout(
        participants,
        map_,
        ego_id=ego_id,
        start_frame=1000,
        horizon_ms=2000,
        controlled_ids=controlled_ids[:2],
    )

    assert result.cycles > 0
    assert isinstance(result.trajectory, Trajectory)
    assert result.controlled_ids == controlled_ids[:2]
    assert set(result.trajectories) == set(controlled_ids[:2])
