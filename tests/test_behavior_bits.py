# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for BITS-style data preparation (inference-only subset)."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

pytest.importorskip("torch", reason="BITS torch tests require the tactics2d[bits] extra.")
pytest.importorskip("torchvision", reason="BITS torch tests require the tactics2d[bits] extra.")

from tactics2d.behavior import BehaviorModelBase
from tactics2d.behavior.bits import BitsBehaviorModel, BitsConfig
from tactics2d.behavior.bits.dataset import BitsBatchBuilder, BitsSampleDataset
from tactics2d.behavior.bits.model import (
    BitsAgentPrediction,
    BitsPlan,
    BitsPlanScorer,
    BitsPolicy,
    BitsPrediction,
)
from tactics2d.behavior.bits.rasterizer import BitsRasterizer
from tactics2d.behavior.bits.torch_base import (
    BitsRasterBackbone,
    BitsRasterizedMapUNet,
    _BitsPositionalEncodingNd,
    integrate_unicycle_controls,
)
from tactics2d.behavior.bits.torch_model import (
    BitsAgentAwareTrajectoryModule,
    BitsBiLevelTorchModel,
    BitsRasterizeROIEncoder,
    TorchBitsPolicy,
    bits_batch_to_torch,
    bits_prediction_from_torch,
    collate_bits_batches_to_torch,
    decode_bits_spatial_prediction,
)
from tactics2d.map.element import Area, Lane, LaneRelationship, Map
from tactics2d.participant.element import Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _vehicle(agent_id, states, length=4.5, width=1.8):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    for state in states:
        trajectory.add_state(state)
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=length, width=width)


def _pedestrian(agent_id, frame, x, y):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    trajectory.add_state(State(frame=frame, x=x, y=y, heading=0.0, vx=0.0, vy=0.0))
    return Pedestrian(agent_id, "pedestrian", trajectory=trajectory, width=0.5)


def _state(frame, x, y, heading=0.0, speed=5.0):
    return State(
        frame=frame,
        x=x,
        y=y,
        heading=heading,
        vx=speed * np.cos(heading),
        vy=speed * np.sin(heading),
    )


class _FixedBitsPolicy(BitsPolicy):
    def __init__(self, positions, yaws=None, scores=None):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 2:
            positions = positions[None]
        self.positions = positions

        if yaws is None:
            yaws = np.zeros((*positions.shape[:2], 1), dtype=float)
        else:
            yaws = np.asarray(yaws, dtype=float)
            if yaws.ndim == 2:
                yaws = yaws[None, :, None]
            elif yaws.ndim == 3 and yaws.shape[-1] != 1:
                yaws = yaws[..., None]
        self.yaws = yaws

        if scores is None:
            scores = np.ones(positions.shape[0], dtype=float)
        self.scores = np.asarray(scores, dtype=float)

    def predict_batch(self, batch):
        return BitsPrediction(
            positions=self.positions.copy(),
            yaws=self.yaws.copy(),
            availabilities=np.ones(self.positions.shape[:2], dtype=bool),
            scores=self.scores.copy(),
        )


def _forward_policy(config, step_distance=1.0):
    return _FixedBitsPolicy([[(step + 1) * step_distance for step in range(config.future_steps)]])


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


def _successor_map():
    map_ = Map(name="bits_route_test_map")
    lane_a = Lane(
        id_="A",
        left_side=LineString([(0.0, -2.0), (10.0, -2.0)]),
        right_side=LineString([(0.0, 2.0), (10.0, 2.0)]),
        custom_tags={"centerline": np.array([[0.0, 0.0], [10.0, 0.0]])},
    )
    lane_b = Lane(
        id_="B",
        left_side=LineString([(10.0, -2.0), (25.0, -2.0)]),
        right_side=LineString([(10.0, 2.0), (25.0, 2.0)]),
        custom_tags={"centerline": np.array([[10.0, 0.0], [25.0, 0.0]])},
    )
    lane_a.add_related_lane("B", LaneRelationship.SUCCESSOR)
    map_.add_lane(lane_a)
    map_.add_lane(lane_b)
    return map_


def _map_with_crosswalk():
    map_ = _straight_map()
    map_.add_area(
        Area(
            id_="crosswalk",
            geometry=Polygon([(7.0, -3.0), (9.0, -3.0), (9.0, 3.0), (7.0, 3.0)]),
            subtype="crosswalk",
        )
    )
    return map_


def test_bits_config_exposes_tbsim_style_cost_weights():
    config = BitsConfig(
        future_steps=7,
        dt=0.2,
        likelihood_weight=0.5,
        progress_weight=1.5,
        lane_weight=2.5,
        collision_weight=3.5,
    )

    assert config.planning_steps == 7
    assert config.step_ms == 200
    assert config.cost_weights == {
        "likelihood_weight": 0.5,
        "progress_weight": 1.5,
        "lane_weight": 2.5,
        "collision_weight": 3.5,
    }


def test_bits_batch_builder_outputs_agent_centric_ego_sequences():
    config = BitsConfig(history_steps=2, future_steps=3, max_agents=2, max_agents_distance=20.0)
    participants = {
        1: _vehicle(
            1,
            [
                _state(-200, 8.0, 5.0),
                _state(-100, 9.0, 5.0),
                _state(0, 10.0, 5.0),
                _state(100, 11.0, 5.0),
                _state(200, 12.0, 5.0),
                _state(300, 13.0, 5.0),
            ],
        ),
        2: _vehicle(2, [_state(0, 14.0, 6.0), _state(100, 15.0, 6.0)]),
    }

    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1, map_=_straight_map())

    assert batch.history_positions.shape == (3, 2)
    assert batch.target_positions.shape == (3, 2)
    assert batch.all_other_agents_history_positions.shape == (2, 3, 2)
    assert batch.all_other_agents_future_positions.shape == (2, 3, 2)
    assert batch.lane_id == "A"
    np.testing.assert_allclose(batch.history_positions, [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(batch.target_positions, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(
        batch.agent_from_world @ batch.world_from_agent, np.eye(3), atol=1e-9
    )
    assert batch.history_availabilities.tolist() == [True, True, True]
    assert batch.target_availabilities.tolist() == [True, True, True]
    assert batch.agent_ids == [2]
    np.testing.assert_allclose(batch.all_other_agents_history_positions[0, -1], [4.0, 1.0])


def test_bits_batch_builder_rotates_world_into_ego_heading_frame():
    config = BitsConfig(history_steps=0, future_steps=1, max_agents=1)
    participants = {
        1: _vehicle(
            1,
            [_state(0, 10.0, 10.0, heading=np.pi / 2), _state(100, 10.0, 12.0, heading=np.pi / 2)],
        ),
        2: _vehicle(2, [_state(0, 8.0, 10.0, heading=np.pi / 2)]),
    }

    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)

    np.testing.assert_allclose(batch.history_positions[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(batch.target_positions[0], [2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(
        batch.all_other_agents_history_positions[0, 0], [0.0, 2.0], atol=1e-9
    )


def test_bits_batch_builder_masks_missing_frames_and_filters_neighbors():
    config = BitsConfig(history_steps=2, future_steps=2, max_agents=1, max_agents_distance=12.0)
    participants = {
        1: _vehicle(1, [_state(-100, -1.0, 0.0), _state(0, 0.0, 0.0), _state(200, 2.0, 0.0)]),
        2: _vehicle(2, [_state(0, 5.0, 0.0), _state(100, 6.0, 0.0)]),
        3: _vehicle(3, [_state(0, 40.0, 0.0)]),
        4: _pedestrian(4, 0, 4.0, 0.0),
    }

    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)

    assert batch.history_availabilities.tolist() == [False, True, True]
    assert batch.target_availabilities.tolist() == [False, True]
    assert batch.agent_ids == [2]
    assert batch.all_other_agents_future_availability[0].tolist() == [True, False]
    assert not batch.all_other_agents_history_availability[0, 0]
    np.testing.assert_allclose(batch.all_other_agents_extents[0], [4.5, 1.8])


def test_bits_batch_builder_can_include_non_vehicle_neighbors():
    config = BitsConfig(
        history_steps=0,
        future_steps=1,
        max_agents=2,
        max_agents_distance=12.0,
        include_non_vehicle_neighbors=True,
    )
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)]),
        2: _pedestrian(2, 0, 3.0, 0.0),
    }

    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)

    assert batch.agent_ids == [2]
    assert batch.all_other_agents_types[0] == BitsBatchBuilder.OTHER_TYPE


def test_bits_batch_builder_requires_available_ego_state():
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0)])}

    with pytest.raises(KeyError):
        BitsBatchBuilder(BitsConfig(history_steps=0, future_steps=1)).build(
            participants, frame=100, ego_id=1
        )


def test_bits_rasterizer_rasterizes_static_map_layers():
    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    rasterizer = BitsRasterizer(config)
    agent_from_world = BitsBatchBuilder(config)._agent_from_world(_state(0, 10.0, 0.0))

    raster = rasterizer.rasterize(_map_with_crosswalk(), agent_from_world)

    assert raster.image.shape == (3, 64, 64)
    assert raster.drivable_map.shape == (64, 64)
    assert raster.drivable_map.dtype == bool
    assert raster.image[0].sum() > 0.0
    assert raster.image[1].sum() > 0.0
    assert raster.image[2].sum() > 0.0
    assert raster.drivable_map[32, 16]
    np.testing.assert_allclose(raster.raster_from_agent @ raster.agent_from_raster, np.eye(3))


def test_bits_rasterizer_can_prepend_agent_history_channels():
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    rasterizer = BitsRasterizer(config)
    agent_from_world = BitsBatchBuilder(config)._agent_from_world(_state(0, 10.0, 0.0))
    static_raster = rasterizer.rasterize(_map_with_crosswalk(), agent_from_world)

    raster = rasterizer.attach_agent_history(
        raster=static_raster,
        ego_history_positions=np.asarray([[-1.0, 0.0], [0.0, 0.0]], dtype=float),
        ego_history_yaws=np.asarray([[[0.0]], [[0.0]]], dtype=float),
        ego_history_availabilities=np.asarray([False, True]),
        ego_extent=np.asarray([4.5, 1.8], dtype=float),
        other_history_positions=np.asarray([[[3.0, 1.0], [4.0, 1.0]]], dtype=float),
        other_history_yaws=np.asarray([[[0.0], [0.0]]], dtype=float),
        other_history_availabilities=np.asarray([[True, True]]),
        other_extents=np.asarray([[4.5, 1.8]], dtype=float),
    )

    assert raster.dynamic_image.shape == (2, 64, 64)
    assert raster.static_image.shape == (3, 64, 64)
    assert raster.image.shape == (5, 64, 64)
    assert raster.dynamic_image[0].min() < 0.0
    assert raster.dynamic_image[0].max() == 0.0
    assert raster.dynamic_image[1].min() < 0.0
    assert raster.dynamic_image[1].max() > 0.0
    np.testing.assert_allclose(raster.image[:2], raster.dynamic_image)
    np.testing.assert_allclose(raster.image[-3:], raster.static_image)


def test_bits_rasterizer_uses_agent_forward_left_coordinate_convention():
    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    rasterizer = BitsRasterizer(config)
    raster_from_agent = rasterizer.raster_from_agent()

    ego_pixel = raster_from_agent @ np.array([0.0, 0.0, 1.0])
    forward_pixel = raster_from_agent @ np.array([1.0, 0.0, 1.0])
    left_pixel = raster_from_agent @ np.array([0.0, 1.0, 1.0])

    np.testing.assert_allclose(ego_pixel[:2], [16.0, 32.0])
    np.testing.assert_allclose(forward_pixel[:2], [18.0, 32.0])
    np.testing.assert_allclose(left_pixel[:2], [16.0, 30.0])


def test_bits_batch_builder_can_attach_optional_raster_context():
    config = BitsConfig(
        history_steps=0, future_steps=1, max_agents=1, raster_size=64, pixel_size=0.5
    )
    participants = {1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0)])}

    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_map_with_crosswalk(), include_raster=True
    )
    batch_dict = batch.as_dict()

    assert batch.dynamic_image.shape == (1, 64, 64)
    assert batch.static_image.shape == (3, 64, 64)
    assert batch.image.shape == (4, 64, 64)
    assert batch.drivable_map.shape == (64, 64)
    assert batch_dict["image"] is batch.image
    assert batch_dict["dynamic_image"] is batch.dynamic_image
    assert batch_dict["static_image"] is batch.static_image
    assert batch_dict["drivable_map"] is batch.drivable_map
    assert batch_dict["raster_from_agent"] is batch.raster_from_agent
    assert batch_dict["agent_from_raster"] is batch.agent_from_raster


def test_bits_batch_builder_requires_map_when_attaching_raster_context():
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)])}

    with pytest.raises(ValueError):
        BitsBatchBuilder(BitsConfig(history_steps=0, future_steps=1)).build(
            participants, frame=0, ego_id=1, include_raster=True
        )


def test_bits_sample_dataset_indexes_valid_training_samples():
    config = BitsConfig(history_steps=1, future_steps=1, max_agents=1)
    participants = {
        1: _vehicle(1, [_state(-100, -1.0, 0.0), _state(0, 0.0, 0.0), _state(100, 1.0, 0.0)]),
        2: _vehicle(2, [_state(0, 5.0, 0.0), _state(100, 6.0, 0.0), _state(200, 7.0, 0.0)]),
    }

    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=False,
        require_full_history=True,
        require_full_future=True,
    )

    assert len(dataset) == 2
    assert [(index.frame, index.ego_id) for index in dataset.indices] == [(0, 1), (100, 2)]
    assert dataset.frames() == [0, 100]
    assert dataset.ego_ids() == [1, 2]
    assert dataset[0].ego_id == 1
    assert [batch.ego_id for batch in dataset] == [1, 2]


def test_bits_sample_dataset_can_relax_future_requirement_for_inference():
    config = BitsConfig(history_steps=1, future_steps=1)
    participants = {1: _vehicle(1, [_state(-100, -1.0, 0.0), _state(0, 0.0, 0.0)])}

    dataset = BitsSampleDataset(
        participants=participants,
        map_=None,
        config=config,
        require_full_history=True,
        require_full_future=False,
    )

    assert len(dataset) == 1
    batch = dataset[0]
    assert batch.ego_id == 1
    assert batch.target_availabilities.tolist() == [False]


def test_bits_behavior_model_predicts_world_frame_trajectories():
    config = BitsConfig(history_steps=1, future_steps=2, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(
            1,
            [_state(-100, 10.0, 8.0, heading=np.pi / 2), _state(0, 10.0, 10.0, heading=np.pi / 2)],
        )
    }
    model = BitsBehaviorModel(
        config, policy=_FixedBitsPolicy([[2.0, 0.0], [4.0, 0.0]]), include_raster=False
    )

    trajectories = model.predict(participants, _straight_map(), frame=0, agent_ids=[1])
    trajectory = trajectories[1]

    assert trajectory.frames == [100, 200]
    np.testing.assert_allclose(trajectory.get_state(100).location, (10.0, 12.0), atol=1e-9)
    np.testing.assert_allclose(trajectory.get_state(200).location, (10.0, 14.0), atol=1e-9)
    assert trajectory.get_state(100).heading == pytest.approx(np.pi / 2)
    assert trajectory.get_state(100).vx == pytest.approx(0.0)
    assert trajectory.get_state(100).vy == pytest.approx(20.0)
    assert trajectory.get_state(100).speed == pytest.approx(20.0)


def test_bits_behavior_model_implements_shared_behavior_interface():
    model = BitsBehaviorModel(
        BitsConfig(history_steps=0, future_steps=1),
        policy=_FixedBitsPolicy([[1.0, 0.0]]),
        include_raster=False,
    )

    assert isinstance(model, BehaviorModelBase)


def test_bits_behavior_model_exposes_batch_prediction_api():
    config = BitsConfig(history_steps=0, future_steps=1)
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0, speed=3.0)])}
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)

    prediction = BitsBehaviorModel(
        config, policy=_FixedBitsPolicy([[0.3, 0.0]]), include_raster=False
    ).predict_batch(batch)

    assert prediction.positions.shape == (1, 1, 2)
    np.testing.assert_allclose(prediction.positions[0, 0], [0.3, 0.0])


def test_bits_behavior_model_loads_default_policy_when_policy_is_omitted(monkeypatch):
    import tactics2d.behavior.bits.defaults as defaults_module

    default_config = BitsConfig(history_steps=0, future_steps=1)

    def fake_load_default_bits_policy(**kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] is None
        return default_config, _FixedBitsPolicy([[1.0, 0.0]])

    monkeypatch.setattr(defaults_module, "load_default_bits_policy", fake_load_default_bits_policy)

    model = BitsBehaviorModel(include_raster=False, device="cpu")

    assert model.config == default_config
    assert isinstance(model.policy, _FixedBitsPolicy)


def test_bits_behavior_model_rejects_config_without_custom_policy():
    with pytest.raises(ValueError, match="config is loaded from the default BITS policy"):
        BitsBehaviorModel(BitsConfig(history_steps=0, future_steps=1), include_raster=False)


def test_bits_plan_scorer_prefers_safe_drivable_candidate():
    config = BitsConfig(
        history_steps=0,
        future_steps=2,
        max_agents=1,
        raster_size=64,
        pixel_size=0.5,
        progress_weight=1.0,
        lane_weight=100.0,
        collision_weight=100.0,
    )
    participants = {
        1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0), _state(200, 12.0, 0.0)]),
        2: _vehicle(2, [_state(0, 13.0, 0.0), _state(100, 15.0, 0.0), _state(200, 17.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    plan = BitsPlan(
        positions=np.asarray([[[5.0, 0.0], [7.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]], dtype=float),
        yaws=np.zeros((2, 2, 1), dtype=float),
        availabilities=np.ones((2, 2), dtype=bool),
        scores=np.asarray([0.0, 0.0], dtype=float),
    )

    scores = BitsPlanScorer(config).score_batch(batch, plan)
    selected = BitsPlanScorer.select_plan(plan, scores)

    assert scores.collision[0] < 0.0
    np.testing.assert_allclose(scores.lane, [0.0, 0.0])
    assert scores.total[1] > scores.total[0]
    np.testing.assert_allclose(selected.positions[0], plan.positions[1])


def test_bits_plan_scorer_uses_total_distance_progress():
    config = BitsConfig(
        history_steps=0,
        future_steps=3,
        max_agents=0,
        likelihood_weight=0.0,
        progress_weight=1.0,
        lane_weight=0.0,
        collision_weight=0.0,
    )
    participants = {1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0)])}
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)
    plan = BitsPlan(
        positions=np.asarray(
            [[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 3.0], [0.0, 6.0]]],
            dtype=float,
        ),
        yaws=np.zeros((2, 3, 1), dtype=float),
        availabilities=np.ones((2, 3), dtype=bool),
        scores=np.zeros(2, dtype=float),
    )

    scores = BitsPlanScorer(config).score_batch(batch, plan)

    np.testing.assert_allclose(scores.progress, [4.0, 6.0])
    assert scores.total[1] > scores.total[0]


def test_bits_plan_scorer_penalizes_distance_from_drivable_area():
    config = BitsConfig(
        history_steps=0,
        future_steps=2,
        max_agents=0,
        raster_size=64,
        pixel_size=0.5,
        likelihood_weight=0.0,
        progress_weight=0.0,
        lane_weight=1.0,
        collision_weight=0.0,
    )
    participants = {1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0)])}
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    plan = BitsPlan(
        positions=np.asarray([[[1.0, 0.0], [2.0, 0.0]], [[1.0, 5.0], [2.0, 5.0]]], dtype=float),
        yaws=np.zeros((2, 2, 1), dtype=float),
        availabilities=np.ones((2, 2), dtype=bool),
        scores=np.zeros(2, dtype=float),
    )

    scores = BitsPlanScorer(config).score_batch(batch, plan)

    np.testing.assert_allclose(scores.lane[0], 0.0)
    assert scores.lane[1] < scores.lane[0]
    assert scores.total[0] > scores.total[1]


def test_bits_plan_scorer_can_use_predicted_agent_trajectories_for_collision_cost():
    config = BitsConfig(
        history_steps=0,
        future_steps=1,
        max_agents=1,
        likelihood_weight=0.0,
        progress_weight=0.0,
        lane_weight=0.0,
        collision_weight=1.0,
    )
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)]),
        2: _vehicle(2, [_state(0, 20.0, 0.0), _state(100, 20.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)
    plan = BitsPlan(
        positions=np.asarray([[[10.0, 0.0]], [[1.0, 0.0]]], dtype=float),
        yaws=np.zeros((2, 1, 1), dtype=float),
        availabilities=np.ones((2, 1), dtype=bool),
        scores=np.zeros(2, dtype=float),
    )
    predicted_agents = BitsAgentPrediction(
        positions=np.asarray([[[[10.0, 0.0]]], [[[20.0, 0.0]]]], dtype=float),
        yaws=np.zeros((2, 1, 1, 1), dtype=float),
        availabilities=np.ones((2, 1, 1), dtype=bool),
    )

    scores = BitsPlanScorer(config).score_batch(batch, plan, agent_prediction=predicted_agents)

    assert scores.collision[0] < scores.collision[1]
    assert scores.total[1] > scores.total[0]


def test_bits_batch_to_torch_converts_required_and_optional_fields():
    import torch

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)])
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )

    torch_batch = bits_batch_to_torch(batch)

    assert torch_batch.tensors["history_positions"].shape == (2, 2)
    assert torch_batch.tensors["history_positions"].dtype == torch.float32
    assert torch_batch.tensors["history_availabilities"].dtype == torch.bool
    assert torch_batch.tensors["type"].dtype == torch.long
    assert torch_batch.tensors["image"].shape == (5, 64, 64)
    assert torch_batch.metadata["ego_id"] == 1
    assert torch_batch.as_dict()["metadata"]["frame"] == 0


def test_collate_bits_batches_to_torch_stacks_same_shaped_samples():
    import torch

    config = BitsConfig(history_steps=0, future_steps=1)
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)]),
        2: _vehicle(2, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants, map_=_straight_map(), config=config, include_raster=False
    )

    torch_batch = collate_bits_batches_to_torch(dataset, include_optional=False)

    assert torch_batch.tensors["history_positions"].shape == (2, 1, 2)
    assert torch_batch.tensors["target_positions"].shape == (2, 1, 2)
    assert torch_batch.tensors["target_availabilities"].dtype == torch.bool
    assert torch_batch.metadata["ego_id"] == [1, 2]


def test_torch_bits_policy_wraps_module_outputs():
    import torch

    class TinyPolicy(torch.nn.Module):
        def forward(self, tensors):
            position = tensors["target_positions"][None]
            yaw = tensors["target_yaws"][None]
            availability = tensors["target_availabilities"][None]
            score = torch.ones(1)
            return {
                "positions": position,
                "yaws": yaw,
                "availabilities": availability,
                "scores": score,
            }

    config = BitsConfig(history_steps=0, future_steps=1)
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)])}
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)
    policy = TorchBitsPolicy(TinyPolicy())

    prediction = policy.predict_batch(batch)

    assert prediction.positions.shape == (1, 1, 2)
    np.testing.assert_allclose(prediction.positions[0, 0], [1.0, 0.0])
    assert prediction.availabilities.tolist() == [[True]]
    np.testing.assert_allclose(prediction.scores, [1.0])


def test_torch_bits_policy_scores_bilevel_outputs_like_bits_closed_loop():
    import torch

    class TinyBiLevelPolicy(torch.nn.Module):
        def forward(self, tensors):
            positions = torch.tensor(
                [[[[5.0, 0.0], [7.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]]],
                dtype=tensors["target_positions"].dtype,
                device=tensors["target_positions"].device,
            )
            yaws = torch.zeros(
                (1, 2, 2, 1),
                dtype=tensors["target_yaws"].dtype,
                device=tensors["target_yaws"].device,
            )
            availabilities = torch.ones((1, 2, 2), dtype=torch.bool, device=positions.device)
            log_likelihood = torch.tensor(
                [[5.0, 0.0]], dtype=positions.dtype, device=positions.device
            )
            return {
                "plan": {
                    "positions": positions[:, :, -1],
                    "yaws": yaws[:, :, -1],
                    "log_likelihood": log_likelihood,
                },
                "predictions": {
                    "positions": positions,
                    "yaws": yaws,
                    "availabilities": availabilities,
                    "scores": log_likelihood,
                },
            }

    config = BitsConfig(
        history_steps=0,
        future_steps=2,
        max_agents=1,
        raster_size=64,
        pixel_size=0.5,
        likelihood_weight=1.0,
        progress_weight=1.0,
        lane_weight=100.0,
        collision_weight=100.0,
    )
    participants = {
        1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0), _state(200, 12.0, 0.0)]),
        2: _vehicle(2, [_state(0, 13.0, 0.0), _state(100, 15.0, 0.0), _state(200, 17.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    policy = TorchBitsPolicy(TinyBiLevelPolicy(), plan_scorer=BitsPlanScorer(config))

    prediction = policy.predict_batch(batch)

    assert prediction.positions.shape == (1, 2, 2)
    np.testing.assert_allclose(prediction.positions[0], [[1.0, 0.0], [2.0, 0.0]])
    assert policy.last_plan_scores is not None
    assert policy.last_plan_scores.likelihood[0] > policy.last_plan_scores.likelihood[1]
    assert policy.last_plan_scores.collision[0] < policy.last_plan_scores.collision[1]
    assert policy.last_selected_plan is not None
    np.testing.assert_allclose(policy.last_selected_plan.positions[0], [[1.0, 0.0], [2.0, 0.0]])


def test_torch_bits_policy_scores_collision_against_predicted_agents():
    import torch

    class TinyBiLevelPolicy(torch.nn.Module):
        def forward(self, tensors):
            positions = torch.tensor(
                [[[[10.0, 0.0]], [[1.0, 0.0]]]],
                dtype=tensors["target_positions"].dtype,
                device=tensors["target_positions"].device,
            )
            yaws = torch.zeros(
                (1, 2, 1, 1), dtype=tensors["target_yaws"].dtype, device=positions.device
            )
            agent_positions = torch.tensor(
                [[[[10.0, 0.0]], [[20.0, 0.0]]]], dtype=positions.dtype, device=positions.device
            ).unsqueeze(2)
            agent_yaws = torch.zeros(
                (1, 2, 1, 1, 1), dtype=positions.dtype, device=positions.device
            )
            scene_availabilities = torch.ones(
                (1, 2, 2, 1), dtype=torch.bool, device=positions.device
            )
            return {
                "plan": {
                    "positions": positions[:, :, -1],
                    "yaws": yaws[:, :, -1],
                    "log_likelihood": torch.zeros(
                        (1, 2), dtype=positions.dtype, device=positions.device
                    ),
                },
                "predictions": {
                    "positions": positions,
                    "yaws": yaws,
                    "agent_positions": agent_positions,
                    "agent_yaws": agent_yaws,
                    "scene_availabilities": scene_availabilities,
                },
            }

    config = BitsConfig(
        history_steps=0,
        future_steps=1,
        max_agents=1,
        likelihood_weight=0.0,
        progress_weight=0.0,
        lane_weight=0.0,
        collision_weight=10.0,
    )
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)]),
        2: _vehicle(2, [_state(0, 20.0, 0.0), _state(100, 20.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)
    policy = TorchBitsPolicy(TinyBiLevelPolicy(), plan_scorer=BitsPlanScorer(config))

    prediction = policy.predict_batch(batch)

    assert policy.last_plan_scores is not None
    assert policy.last_plan_scores.collision[0] < policy.last_plan_scores.collision[1]
    np.testing.assert_allclose(prediction.positions[0], [[1.0, 0.0]])


def test_torch_bits_policy_can_return_all_scored_bilevel_modes():
    import torch

    class TinyBiLevelPolicy(torch.nn.Module):
        def forward(self, tensors):
            positions = torch.tensor(
                [[[[1.0, 0.0]], [[2.0, 0.0]]]],
                dtype=tensors["target_positions"].dtype,
                device=tensors["target_positions"].device,
            )
            yaws = torch.zeros(
                (1, 2, 1, 1), dtype=tensors["target_yaws"].dtype, device=positions.device
            )
            return {
                "plan": {
                    "positions": positions[:, :, -1],
                    "yaws": yaws[:, :, -1],
                    "log_likelihood": torch.tensor(
                        [[0.0, 1.0]], dtype=positions.dtype, device=positions.device
                    ),
                },
                "predictions": {"positions": positions, "yaws": yaws},
            }

    config = BitsConfig(
        history_steps=0, future_steps=1, progress_weight=0.0, lane_weight=0.0, collision_weight=0.0
    )
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)])}
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)
    policy = TorchBitsPolicy(
        TinyBiLevelPolicy(), plan_scorer=BitsPlanScorer(config), select_best_plan=False
    )

    prediction = policy.predict_batch(batch)

    assert prediction.positions.shape == (2, 1, 2)
    np.testing.assert_allclose(prediction.scores, [0.0, 1.0])
    assert policy.last_selected_plan is not None
    np.testing.assert_allclose(policy.last_selected_plan.scores, [0.0, 1.0])


def test_decode_bits_spatial_prediction_uses_top_pixel_and_residual():
    import torch

    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    raster_from_agent = BitsRasterizer(config).raster_from_agent()
    agent_from_raster = np.linalg.inv(raster_from_agent)
    spatial_prediction = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    spatial_prediction[0, 0, 32, 22] = 10.0
    spatial_prediction[0, 1:3, 32, 22] = torch.logit(torch.tensor([0.25, 0.75]))

    decoded = decode_bits_spatial_prediction(
        spatial_prediction, torch.as_tensor(agent_from_raster[None], dtype=torch.float32)
    )

    assert decoded["positions"].shape == (1, 1, 2)
    np.testing.assert_allclose(decoded["pixel_positions"].detach().numpy()[0, 0], [22.25, 32.75])
    np.testing.assert_allclose(decoded["positions"].detach().numpy()[0, 0], [3.125, -0.375])
    assert decoded["scores"][0, 0] > 0.1


def test_decode_bits_spatial_prediction_samples_from_probability_map():
    import torch

    torch.manual_seed(7)
    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    agent_from_raster = np.linalg.inv(BitsRasterizer(config).raster_from_agent())
    spatial_prediction = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    spatial_prediction[0, 0, 32, 22] = 10.0
    spatial_prediction[0, 0, 32, 24] = 8.0

    decoded = decode_bits_spatial_prediction(
        spatial_prediction,
        torch.as_tensor(agent_from_raster[None], dtype=torch.float32),
        num_samples=4,
    )

    assert decoded["positions"].shape == (1, 4, 2)
    assert decoded["scores"].shape == (1, 4)
    assert torch.all(decoded["scores"] > 0.0)


def test_bits_bilevel_torch_model_preserves_sampled_goal_modes():
    import torch

    torch.manual_seed(11)
    config = BitsConfig(history_steps=1, future_steps=2, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(
            1,
            [
                _state(-100, 9.0, 0.0),
                _state(0, 10.0, 0.0),
                _state(100, 11.0, 0.0),
                _state(200, 12.0, 0.0),
            ],
        )
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    torch_batch = bits_batch_to_torch(batch)
    model = BitsBiLevelTorchModel(
        image_channels=batch.image.shape[0], future_steps=config.future_steps, hidden_dim=32
    )

    output = model(torch_batch.tensors, num_samples=3)

    assert output["plan"]["positions"].shape == (1, 3, 2)
    assert output["predictions"]["positions"].shape == (1, 3, 2, 2)
    assert output["predictions"]["scores"].shape == (1, 3)


def test_bits_bilevel_torch_model_exposes_planner_and_predictor_checkpoint_boundaries():
    model = BitsBiLevelTorchModel(image_channels=5, future_steps=1, hidden_dim=16)
    state_keys = set(model.state_dict())

    assert hasattr(model, "planner")
    assert hasattr(model, "predictor")
    assert any(key.startswith("shared_encoder.encoder_heads.map_model") for key in state_keys)
    assert any(key.startswith("planner.spatial_goal_decoder.decoder") for key in state_keys)
    assert any(key.startswith("predictor.roi_head.agent_net") for key in state_keys)
    assert any(key.startswith("predictor.policy_head.goal_encoder") for key in state_keys)
    assert any(key.startswith("predictor.policy_head.ego_decoder.mlp._model") for key in state_keys)
    assert any(
        key.startswith("predictor.future_state_head.agents_decoder.mlp._model")
        for key in state_keys
    )


def test_integrate_unicycle_controls_keeps_zero_control_constant_velocity():
    import torch

    config = BitsConfig(dt=0.5, future_steps=3)
    current_states = torch.tensor([[0.0, 0.0, 2.0, 0.0]], dtype=torch.float32)
    controls = torch.zeros((1, 1, 3, 2), dtype=torch.float32)

    positions, yaws = integrate_unicycle_controls(controls, current_states, config)

    np.testing.assert_allclose(
        positions.detach().numpy()[0, 0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], atol=1e-6
    )
    np.testing.assert_allclose(yaws.detach().numpy()[0, 0], [[0.0], [0.0], [0.0]], atol=1e-6)


def test_integrate_unicycle_controls_applies_acceleration_and_yaw_rate():
    import torch

    config = BitsConfig(
        dt=1.0,
        future_steps=2,
        dynamics_max_steer=10.0,
        dynamics_max_yawvel=10.0,
        dynamics_acceleration_min=-10.0,
        dynamics_acceleration_max=10.0,
        dynamics_speed_min=-10.0,
        dynamics_speed_max=10.0,
    )
    current_states = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    controls = torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]], dtype=torch.float32)

    positions, yaws = integrate_unicycle_controls(controls, current_states, config)

    assert positions.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(positions.detach().numpy()[0, 0, 0], [1.5, 0.0], atol=1e-6)
    assert positions.detach().numpy()[0, 0, 1, 0] > 1.5
    assert positions.detach().numpy()[0, 0, 1, 1] > 0.0
    np.testing.assert_allclose(yaws.detach().numpy()[0, 0], [[0.5], [0.5]], atol=1e-6)


def test_bits_xy_positional_encoding_keeps_axes_in_separate_sin_cos_channels():
    import torch

    encoder = _BitsPositionalEncodingNd(dim=8, dropout=0.0, step_size=(1.0, 1.0))
    inputs = torch.zeros((1, 3, 8), dtype=torch.float32)
    positions = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)

    encoded = encoder(inputs, positions)

    assert encoded.shape == (1, 3, 8)
    assert encoded[0, 0, 1].item() == pytest.approx(1.0)
    assert encoded[0, 0, 5].item() == pytest.approx(1.0)
    assert torch.allclose(encoded[0, 0, 4:], encoded[0, 1, 4:])
    assert not torch.allclose(encoded[0, 0, :4], encoded[0, 1, :4])
    assert torch.allclose(encoded[0, 0, :4], encoded[0, 2, :4])
    assert not torch.allclose(encoded[0, 0, 4:], encoded[0, 2, 4:])


def test_bits_raster_backbone_exposes_official_style_feature_maps():
    import torch

    backbone = BitsRasterBackbone(image_channels=5, model_arch="resnet18", feature_dim=8)
    features = backbone.extract_features(torch.zeros((1, 5, 64, 64), dtype=torch.float32))

    assert hasattr(backbone, "map_model")
    assert hasattr(backbone.map_model, "conv1")
    assert hasattr(backbone.map_model, "layer1")
    assert set(features) == {"layer1", "layer2", "layer3", "layer4", "final"}
    assert features["layer1"].shape == (1, 64, 16, 16)
    assert features["layer4"].shape == (1, 512, 2, 2)


def test_bits_rasterized_map_unet_outputs_official_goal_channels():
    import torch

    planner_net = BitsRasterizedMapUNet(image_channels=5, model_arch="resnet18")

    output = planner_net(torch.zeros((1, 5, 64, 64), dtype=torch.float32))

    assert output.shape == (1, 4, 64, 64)


def test_bits_rasterize_roi_encoder_feeds_low_level_traffic_head():
    import torch

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=32, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 0.0, 0.0), _state(0, 1.0, 0.0), _state(100, 2.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 5.0, 0.0), _state(0, 6.0, 0.0), _state(100, 7.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    torch_batch = bits_batch_to_torch(batch)
    encoder = BitsRasterizeROIEncoder(
        image_channels=batch.image.shape[0],
        global_feature_dim=8,
        agent_feature_dim=8,
        context_size=8,
        roi_feature_size=4,
        model_arch="resnet18",
    )
    encoder.eval()
    agent_positions = torch.tensor([[[0.0, 0.0], [5.0, 0.0]]], dtype=torch.float32)

    agent_features, global_features, encoder_features = encoder(
        torch_batch.tensors, agent_positions
    )
    loss = agent_features.sum() + global_features.sum()
    loss.backward()

    assert agent_features.shape == (1, 2, 8)
    assert global_features.shape == (1, 8)
    assert "layer2" in encoder_features
    assert any(param.grad is not None for param in encoder.parameters())


def test_bits_rasterize_roi_encoder_uses_official_context_and_feature_size():
    from torchvision.ops import RoIAlign

    encoder = BitsRasterizeROIEncoder(
        image_channels=5, context_size=30, roi_feature_size=7, model_arch="resnet18"
    )

    assert encoder.context_size == 30
    assert encoder.roi_feature_size == 7
    assert encoder.roi_layer_key == "layer2"
    assert isinstance(encoder.roi_align, RoIAlign)
    assert encoder.agent_net[2].in_features == 128


def test_bits_agent_aware_module_uses_official_decoder_names():
    config = BitsConfig(
        history_steps=1, future_steps=1, max_agents=1, raster_size=32, pixel_size=0.5
    )
    participants = {
        1: _vehicle(1, [_state(-100, 0.0, 0.0), _state(0, 1.0, 0.0), _state(100, 2.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 4.0, 0.0), _state(0, 5.0, 0.0), _state(100, 6.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    module = BitsAgentAwareTrajectoryModule(
        future_steps=config.future_steps,
        image_channels=batch.image.shape[0],
        global_feature_dim=8,
        agent_feature_dim=8,
        goal_feature_dim=4,
        decoder_layer_dims=(8,),
        model_arch="resnet18",
    )

    state_keys = set(module.state_dict())

    assert any(
        key.startswith("shared_encoder.encoder_heads.map_model.layer1") for key in state_keys
    )
    assert any(key.startswith("roi_head.agent_net") for key in state_keys)
    assert any(key.startswith("policy_head.goal_encoder._model") for key in state_keys)
    assert any(key.startswith("policy_head.ego_decoder.mlp._model") for key in state_keys)
    assert any(key.startswith("future_state_head.agents_decoder.mlp._model") for key in state_keys)


def test_decode_bits_spatial_prediction_masks_non_drivable_pixels():
    import torch

    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    agent_from_raster = np.linalg.inv(BitsRasterizer(config).raster_from_agent())
    spatial_prediction = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    spatial_prediction[0, 0, 32, 22] = 10.0
    spatial_prediction[0, 0, 34, 24] = 8.0
    drivable_map = torch.zeros((1, 64, 64), dtype=torch.bool)
    drivable_map[0, 34, 24] = True

    decoded = decode_bits_spatial_prediction(
        spatial_prediction,
        torch.as_tensor(agent_from_raster[None], dtype=torch.float32),
        drivable_map=drivable_map,
        mask_drivable=True,
    )

    np.testing.assert_allclose(decoded["pixel_positions"].detach().numpy()[0, 0], [24.5, 34.5])
    np.testing.assert_allclose(decoded["location_prob_map"][0, 34, 24].item(), 1.0)


def test_decode_bits_spatial_prediction_falls_back_when_drivable_mask_is_empty():
    import torch

    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    agent_from_raster = np.linalg.inv(BitsRasterizer(config).raster_from_agent())
    spatial_prediction = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    spatial_prediction[0, 0, 32, 22] = 10.0
    drivable_map = torch.zeros((1, 64, 64), dtype=torch.bool)

    decoded = decode_bits_spatial_prediction(
        spatial_prediction,
        torch.as_tensor(agent_from_raster[None], dtype=torch.float32),
        drivable_map=drivable_map,
        mask_drivable=True,
    )

    np.testing.assert_allclose(decoded["pixel_positions"].detach().numpy()[0, 0], [22.5, 32.5])
    assert decoded["location_prob_map"].sum().item() == pytest.approx(1.0, abs=1e-6)
