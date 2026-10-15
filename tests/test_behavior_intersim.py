# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the InterSim-style relation-driven behavior model."""

from pathlib import Path

import numpy as np
import pytest
import torch
from shapely.geometry import LineString

from tactics2d.behavior import InterSimBehaviorModel, InterSimConfig
from tactics2d.behavior.intersim import m2i_features, relation_decider
from tactics2d.behavior.intersim.relation_model import RelationVectorNet
from tactics2d.behavior.intersim.scene import current_state
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

RELATION_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "tactics2d/data/checkpoints/intersim/relation_predictor_waymo.pt"
)


def _lane(lane_id, start, end, width=1.9):
    """Build a straight lane from ``start`` to ``end`` along an axis."""
    (x0, y0), (x1, y1) = start, end
    if x0 == x1:
        left = LineString([(x0 - width, y0), (x0 - width, y1)])
        right = LineString([(x0 + width, y0), (x0 + width, y1)])
    else:
        left = LineString([(x0, y0 - width), (x1, y1 - width)])
        right = LineString([(x0, y0 + width), (x1, y1 + width)])
    samples = np.linspace(0.0, 1.0, 41)
    centerline = np.asarray(start) + np.outer(samples, np.asarray(end) - np.asarray(start))
    return Lane(
        id_=lane_id, left_side=left, right_side=right, custom_tags={"centerline": centerline}
    )


def _vertical_lane(lane_id, x=0.0, y_start=-80.0, y_end=80.0):
    return _lane(lane_id, (x, y_start), (x, y_end))


def _moving_participant(agent_id, x0, y0, heading, speed, steps):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for index in range(steps):
        distance = speed * index * 0.1
        trajectory.add_state(
            State(
                frame=index * 100,
                x=x0 + distance * np.cos(heading),
                y=y0 + distance * np.sin(heading),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.8, width=1.9)


@pytest.mark.integration
@pytest.mark.slow
def test_intersim_rollout_runs(womd_behavior_scene):
    """The checkpoint-backed integrated closed loop runs on WOMD data."""
    if not RELATION_CHECKPOINT.exists():
        pytest.skip(f"Released InterSim checkpoint is not present at {RELATION_CHECKPOINT}.")
    config = InterSimConfig(relation_mode="nn", relation_model_path=str(RELATION_CHECKPOINT))
    participants, map_, controlled_ids = womd_behavior_scene
    ego_id = controlled_ids[0]

    result = InterSimBehaviorModel(config).rollout(
        participants, map_, ego_id=ego_id, controlled_ids=controlled_ids
    )

    assert result.ego_id == ego_id
    assert set(result.poses) == set(controlled_ids)
    assert result.frames
    assert result.cycles > 0
    assert result.controlled_ids == result.relevant_ids
    assert set(result.trajectories) == set(result.relevant_ids)


def test_relation_forward_many_matches_single_scores():
    torch.manual_seed(7)
    model = RelationVectorNet().eval()
    mappings = []
    for seed, road_rows in [(3, 12), (5, 31)]:
        rng = np.random.default_rng(seed)
        mappings.append(
            m2i_features.build_relation_model_input(
                agent_trajectories=rng.normal(size=(2, 11, 7)).astype(np.float32),
                tracks_type=np.ones(2, dtype=np.int32),
                road_points=rng.normal(size=(road_rows, 3)).astype(np.float32),
                road_types=np.ones(road_rows, dtype=np.int32),
                road_ids=np.arange(road_rows, dtype=np.int32) // 4,
                normalizer=(1.0, -2.0, 0.3),
                all_agent_ids=["reactor", "influencer"],
                scenario_id=str(seed),
            )
        )

    expected = np.concatenate(
        [
            model.forward(
                mapping["matrix"],
                mapping["polyline_spans"],
                mapping["map_start_polyline_idx"],
                torch.device("cpu"),
            )
            for mapping in mappings
        ]
    )
    actual = model.forward_many(mappings, torch.device("cpu"))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(expected, axis=1))


def test_road_feature_grid_matches_full_filter():
    map_ = Map(name="road_grid")
    for index, x in enumerate((-300.0, -80.0, 0.0, 90.0, 310.0)):
        map_.add_lane(_vertical_lane(f"lane-{index}", x=x, y_start=-250.0, y_end=250.0))

    cached = relation_decider._road_samples(map_)
    for cx, cy in ((0.0, 0.0), (140.0, 100.0), (-200.0, -100.0)):
        keep = (
            (np.abs(cached.points[:, 0] - cx) <= 150.0)
            & (cached.points[:, 1] - cy >= -60.0)
            & (cached.points[:, 1] - cy <= 170.0)
        )
        expected = (cached.points[keep], cached.types[keep], cached.ids[keep])
        actual = relation_decider.sample_road_features(map_, cx, cy)
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_array_equal(actual_array, expected_array)


def test_current_state_uses_latest_frame_not_after_request():
    participant = _moving_participant("car", 0.0, 0.0, 0.0, 2.0, 5)

    assert current_state(participant, -1) is None
    assert current_state(participant, 0).frame == 0
    assert current_state(participant, 250).frame == 200
    assert current_state(participant, 999).frame == 400
