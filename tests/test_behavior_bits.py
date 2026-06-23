# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for BITS-style data preparation."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

pytest.importorskip("torch", reason="BITS torch tests require the tactics2d[bits] extra.")
pytest.importorskip("torchvision", reason="BITS torch tests require the tactics2d[bits] extra.")

from tactics2d.behavior import BehaviorModelBase
from tactics2d.behavior.bits import BitsBehaviorModel, BitsConfig
from tactics2d.behavior.bits.cache import (
    build_bits_batch_cache,
    load_bits_batch_cache,
    load_bits_batch_cache_manifest,
    rebuild_bits_batch_cache_manifest,
)
from tactics2d.behavior.bits.dataset import (
    BitsBatchBuilder,
    BitsSampleDataset,
    NuPlanBitsDataset,
)
from tactics2d.behavior.bits.evaluation import evaluate_bits_rolling_result
from tactics2d.behavior.bits.model import (
    BitsAgentPrediction,
    BitsPlan,
    BitsPlanScorer,
    BitsPolicy,
    BitsPrediction,
)
from tactics2d.behavior.bits.rasterizer import BitsRasterizer
from tactics2d.behavior.bits.rolling import BitsRollingRunner
from tactics2d.behavior.bits.supervision import build_goal_supervision
from tactics2d.behavior.bits.training import (
    BitsRunConfig,
    BitsTrainingSchedule,
    NuPlanBitsSplit,
    NuPlanLogSpec,
    bits_run_config_from_dict,
    bits_run_config_to_dict,
    evaluate_nuplan_bits_planner_split,
    load_bits_checkpoint,
    load_bits_run_config,
    load_bits_inference_model,
    load_tbsim_bits_inference_weights,
    map_tbsim_bits_planner_state_dict,
    map_tbsim_bits_predictor_state_dict,
    merge_tbsim_bits_state_dicts,
    run_nuplan_bits_open_loop_protocol,
    run_nuplan_bits_planner_protocol,
    run_nuplan_bits_rolling_protocol,
    run_nuplan_bits_torch_validation,
    save_bits_checkpoint,
    save_bits_run_config,
    train_nuplan_bits_model,
    train_nuplan_bits_planner,
)
from tactics2d.behavior.bits.torch_model import (
    BitsAgentAwareTrajectoryModule,
    BitsBiLevelTorchModel,
    BitsRasterBackbone,
    BitsRasterizeROIEncoder,
    BitsRasterizedMapUNet,
    TorchBitsPolicy,
    bits_batch_to_torch,
    bits_goal_supervision_to_torch,
    bits_prediction_from_torch,
    collate_bits_batches_to_torch,
    collate_bits_goal_supervisions_to_torch,
    compute_bits_torch_losses,
    decode_bits_spatial_prediction,
    run_bits_planner_torch_epoch,
    integrate_unicycle_controls,
    run_bits_torch_epoch,
    _BitsPositionalEncodingNd,
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
    return _FixedBitsPolicy(
        [[(step + 1) * step_distance, 0.0] for step in range(config.future_steps)]
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
            [
                _state(0, 10.0, 10.0, heading=np.pi / 2),
                _state(100, 10.0, 12.0, heading=np.pi / 2),
            ],
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
    participants = {
        1: _vehicle(1, [_state(-100, -1.0, 0.0), _state(0, 0.0, 0.0)]),
    }

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


def test_nuplan_bits_dataset_uses_parser_and_builds_batches():
    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            assert file == "log.db"
            assert folder == "logs"
            assert time_range == (-100, 100)
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            assert file == "map.gpkg"
            assert folder == "maps"
            return _successor_map()

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    dataset = NuPlanBitsDataset(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
        time_range=(-100, 100),
        parser=FakeNuPlanParser(),
        config=config,
        include_raster=True,
    )

    assert dataset.actual_time_range == (-100, 100)
    assert len(dataset) == 1
    batch = dataset[0]
    assert batch.ego_id == 1
    assert batch.image.shape == (5, 64, 64)


def test_nuplan_bits_dataset_runs_torch_epoch_pipeline_sanity():
    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                ),
                2: _vehicle(
                    2,
                    [
                        _state(-100, 14.0, 0.0),
                        _state(0, 15.0, 0.0),
                        _state(100, 16.0, 0.0),
                    ],
                ),
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _successor_map()

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    dataset = NuPlanBitsDataset(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
        time_range=(-100, 100),
        parser=FakeNuPlanParser(),
        config=config,
        include_raster=True,
    )
    model = BitsBiLevelTorchModel(
        image_channels=dataset[0].image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=32,
    )

    result = run_bits_torch_epoch(model, dataset, batch_size=2)

    assert dataset.actual_time_range == (-100, 100)
    assert result.sample_count == 2
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0


def test_bits_batch_cache_builds_and_reuses_nuplan_samples(tmp_path):
    class FakeNuPlanParser:
        calls = 0

        def parse_trajectory(self, file, folder, time_range=None):
            self.calls += 1
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                ),
                2: _vehicle(
                    2,
                    [
                        _state(-100, 14.0, 1.0),
                        _state(0, 15.0, 1.0),
                        _state(100, 16.0, 1.0),
                    ],
                ),
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    parser = FakeNuPlanParser()
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    log_spec = NuPlanLogSpec(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
    )
    run_config = BitsRunConfig(
        config=config,
        split=NuPlanBitsSplit(train=(log_spec,), val=(log_spec,)),
        schedule=BitsTrainingSchedule(
            batch_size=1,
            max_train_samples_per_log=1,
            max_val_samples_per_log=1,
        ),
    )

    manifest = build_bits_batch_cache(
        run_config,
        tmp_path,
        splits=("train", "val"),
        parser=parser,
        progress_interval=0,
    )
    train_batches = load_bits_batch_cache(tmp_path, "train")
    loaded_manifest = load_bits_batch_cache_manifest(tmp_path)
    reused = build_bits_batch_cache(
        run_config,
        tmp_path,
        splits=("train",),
        parser=parser,
        progress_interval=0,
    )

    assert manifest["splits"]["train"]["sample_count"] == 1
    assert manifest["splits"]["val"]["sample_count"] == 1
    assert loaded_manifest["run_config"]["config"]["raster_size"] == 64
    assert len(train_batches) == 1
    assert train_batches[0].image.shape[0] == 5
    assert train_batches[0].static_image is None
    assert train_batches[0].dynamic_image is None
    assert reused["splits"]["train"]["sample_count"] == 1
    assert parser.calls == 2


def test_bits_batch_cache_reuses_parsed_maps_across_logs(tmp_path):
    class FakeNuPlanParser:
        trajectory_calls = 0
        map_calls = 0

        def parse_trajectory(self, file, folder, time_range=None):
            self.trajectory_calls += 1
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            self.map_calls += 1
            return _straight_map()

    parser = FakeNuPlanParser()
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    split = NuPlanBitsSplit(
        train=(
            NuPlanLogSpec(data_file="a.db", data_folder="logs", map_file="same.gpkg", map_folder="maps"),
            NuPlanLogSpec(data_file="b.db", data_folder="logs", map_file="same.gpkg", map_folder="maps"),
        )
    )
    run_config = BitsRunConfig(
        config=config,
        split=split,
        schedule=BitsTrainingSchedule(max_train_samples_per_log=1),
    )

    manifest = build_bits_batch_cache(
        run_config,
        tmp_path,
        splits=("train",),
        parser=parser,
        progress_interval=0,
    )

    assert manifest["splits"]["train"]["sample_count"] == 2
    assert parser.trajectory_calls == 2
    assert parser.map_calls == 1


def test_bits_batch_cache_manifest_can_be_rebuilt_from_sample_files(tmp_path):
    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    run_config = BitsRunConfig(
        config=config,
        split=NuPlanBitsSplit(
            train=(
                NuPlanLogSpec(
                    data_file="log.db",
                    data_folder="logs",
                    map_file="map.gpkg",
                    map_folder="maps",
                ),
            )
        ),
        schedule=BitsTrainingSchedule(max_train_samples_per_log=1),
    )
    build_bits_batch_cache(
        run_config,
        tmp_path,
        splits=("train",),
        parser=FakeNuPlanParser(),
        progress_interval=0,
    )
    (tmp_path / "manifest.json").unlink()

    manifest = rebuild_bits_batch_cache_manifest(run_config, tmp_path, splits=("train",))
    batches = load_bits_batch_cache(tmp_path, "train")

    assert manifest["splits"]["train"]["sample_count"] == 1
    assert manifest["splits"]["train"]["rebuilt_from_files"] is True
    assert len(batches) == 1
    assert batches[0].image.shape[0] == 5


def test_nuplan_bits_dataset_runs_agent_aware_torch_epoch_pipeline_sanity():
    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                        _state(200, 12.0, 0.0),
                    ],
                ),
                2: _vehicle(
                    2,
                    [
                        _state(-100, 14.0, 1.0),
                        _state(0, 15.0, 1.0),
                        _state(100, 16.0, 1.0),
                        _state(200, 17.0, 1.0),
                    ],
                ),
            }
            return participants, (-100, 200)

        def parse_map(self, file, folder=None):
            return _successor_map()

    config = BitsConfig(
        history_steps=1,
        future_steps=2,
        max_agents=1,
        raster_size=64,
        pixel_size=0.5,
    )
    dataset = NuPlanBitsDataset(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
        time_range=(-100, 200),
        parser=FakeNuPlanParser(),
        config=config,
        include_raster=True,
    )
    model = BitsBiLevelTorchModel(
        image_channels=dataset[0].image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=32,
    )

    result = run_bits_torch_epoch(model, dataset, batch_size=2)

    assert result.sample_count == 2
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0
    assert result.mean_losses["prediction_loss"] > 0.0


@pytest.mark.bits_workflow
def test_train_nuplan_bits_planner_saves_planner_checkpoint(tmp_path):
    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    split = NuPlanBitsSplit(
        train=(
            NuPlanLogSpec(
                data_file="log.db",
                data_folder="logs",
                map_file="map.gpkg",
                map_folder="maps",
            ),
        ),
    )
    schedule = BitsTrainingSchedule(
        epochs=1,
        batch_size=1,
        hidden_dim=16,
        max_train_samples_per_log=1,
        checkpoint_every_epochs=1,
        seed=3,
    )

    model, history = train_nuplan_bits_planner(
        split,
        tmp_path,
        config=config,
        schedule=schedule,
        parser=FakeNuPlanParser(),
    )
    loaded_model, metadata, _payload = load_bits_checkpoint(history.checkpoints[0])
    val_result = evaluate_nuplan_bits_planner_split(
        model,
        split.train,
        config=config,
        schedule=schedule,
        parser=FakeNuPlanParser(),
    )

    assert len(history.train) == 1
    assert history.train[0].sample_count == 1
    assert len(history.checkpoints) == 1
    assert "spatial planner only" in metadata["official_checkpoint_note"]
    assert isinstance(loaded_model, BitsBiLevelTorchModel)
    assert val_result.sample_count == 1


@pytest.mark.bits_workflow
def test_run_nuplan_bits_planner_protocol_saves_repeatable_results(tmp_path):
    import json
    import torch

    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    train_log = NuPlanLogSpec(
        data_file="train.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
    )
    run_config = BitsRunConfig(
        config=config,
        split=NuPlanBitsSplit(train=(train_log,), val=(train_log,), test=(train_log,)),
        schedule=BitsTrainingSchedule(
            epochs=1,
            batch_size=1,
            hidden_dim=16,
            max_train_samples_per_log=1,
            max_val_samples_per_log=1,
            checkpoint_every_epochs=1,
            seed=13,
        ),
    )

    model, result = run_nuplan_bits_planner_protocol(
        run_config,
        tmp_path,
        parser=FakeNuPlanParser(),
    )
    with open(result.result_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    assert isinstance(model, BitsBiLevelTorchModel)
    assert result.protocol == "nuplan_bits_planner_v0"
    assert result.train[0]["sample_count"] == 1
    assert result.val[0]["sample_count"] == 1
    assert result.test["sample_count"] == 1
    assert len(result.checkpoints) == 1
    assert (tmp_path / "bits_run_config.json").exists()
    assert (tmp_path / "bits_planner_protocol_result.json").exists()
    assert payload["protocol"] == "nuplan_bits_planner_v0"
    assert payload["run_config"]["schedule"]["seed"] == 13
    assert payload["train"][0]["mean_total_loss"] > 0.0

    init_model = BitsBiLevelTorchModel(
        image_channels=5,
        future_steps=config.future_steps,
        hidden_dim=16,
        config=config,
    )
    shared_key = "shared_encoder.encoder_heads.map_model.conv1.weight"
    predictor_payload = {
        "state_dict": {
            "model.map_encoder.encoder_heads.map_model.conv1.weight": torch.full_like(
                init_model.state_dict()[shared_key],
                0.125,
            ),
        },
    }
    frozen_model, frozen_result = run_nuplan_bits_planner_protocol(
        run_config,
        tmp_path / "frozen",
        predictor_checkpoint=predictor_payload,
        parser=FakeNuPlanParser(),
        freeze_shared_encoder=True,
    )

    assert frozen_result.inference["uses_tbsim_predictor_checkpoint"] is True
    assert frozen_result.inference["freeze_shared_encoder"] is True
    assert torch.allclose(
        frozen_model.state_dict()[shared_key],
        torch.full_like(frozen_model.state_dict()[shared_key], 0.125),
    )


@pytest.mark.bits_workflow
def test_run_nuplan_bits_open_loop_protocol_loads_checkpoint_and_scores_splits(tmp_path):
    import json

    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                ),
                2: _vehicle(
                    2,
                    [
                        _state(-100, 13.0, 1.0),
                        _state(0, 14.0, 1.0),
                        _state(100, 15.0, 1.0),
                    ],
                ),
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    config = BitsConfig(
        history_steps=1,
        future_steps=1,
        max_agents=1,
        raster_size=64,
        pixel_size=0.5,
    )
    log_spec = NuPlanLogSpec(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
    )
    run_config = BitsRunConfig(
        config=config,
        split=NuPlanBitsSplit(train=(log_spec,), val=(log_spec,), test=(log_spec,)),
        schedule=BitsTrainingSchedule(
            epochs=1,
            batch_size=1,
            hidden_dim=16,
            max_train_samples_per_log=1,
            max_val_samples_per_log=1,
            checkpoint_every_epochs=1,
            seed=23,
        ),
    )
    _model, history = train_nuplan_bits_model(
        run_config.split,
        tmp_path / "train",
        config=config,
        schedule=run_config.schedule,
        parser=FakeNuPlanParser(),
    )

    loaded_model, result = run_nuplan_bits_open_loop_protocol(
        run_config,
        tmp_path / "eval",
        checkpoint_path=history.checkpoints[0],
        parser=FakeNuPlanParser(),
    )
    with open(result.result_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    assert isinstance(loaded_model, BitsBiLevelTorchModel)
    assert result.protocol == "nuplan_bits_open_loop_v0"
    assert result.inference["source"] == "tactics2d"
    assert result.train[0]["sample_count"] == 1
    assert result.val[0]["sample_count"] == 1
    assert result.test["sample_count"] == 1
    assert result.checkpoints == (history.checkpoints[0],)
    assert (tmp_path / "eval" / "bits_open_loop_run_config.json").exists()
    assert payload["inference"]["source"] == "tactics2d"
    assert payload["test"]["mean_total_loss"] > 0.0


@pytest.mark.bits_workflow
def test_run_nuplan_bits_open_loop_protocol_accepts_mixed_weight_sources(tmp_path):
    import torch

    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    model = BitsBiLevelTorchModel(image_channels=5, future_steps=1, hidden_dim=16, config=config)
    model_state = model.state_dict()
    planner_key = "planner.spatial_goal_decoder.decoder.conv2.0.bias"
    predictor_key = "predictor.policy_head.ego_decoder.mlp._model.0.weight"
    planner_payload = {
        "model_state_dict": {
            planner_key: model_state[planner_key].clone(),
        },
    }
    predictor_payload = {
        "state_dict": {
            "model.ego_decoder.mlp._model.0.weight": model_state[predictor_key].clone(),
        },
    }
    log_spec = NuPlanLogSpec(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
    )
    run_config = BitsRunConfig(
        config=config,
        split=NuPlanBitsSplit(test=(log_spec,)),
        schedule=BitsTrainingSchedule(
            batch_size=1,
            hidden_dim=16,
            max_val_samples_per_log=1,
        ),
    )

    _model, result = run_nuplan_bits_open_loop_protocol(
        run_config,
        tmp_path,
        tactics2d_planner_checkpoint=planner_payload,
        predictor_checkpoint=predictor_payload,
        image_channels=5,
        future_steps=1,
        hidden_dim=16,
        parser=FakeNuPlanParser(),
        strict=False,
    )

    assert result.inference["source"] == "mixed"
    assert result.test["sample_count"] == 1
    assert planner_key in result.inference["compatibility"]["matched_keys"]
    assert predictor_key in result.inference["compatibility"]["matched_keys"]


@pytest.mark.bits_workflow
def test_run_nuplan_bits_rolling_protocol_scores_short_closed_loop(tmp_path):
    import json

    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                    ],
                ),
                2: _vehicle(
                    2,
                    [
                        _state(-100, 20.0, 0.0),
                        _state(0, 20.5, 0.0),
                        _state(100, 21.0, 0.0),
                    ],
                ),
            }
            return participants, (-100, 100)

        def parse_map(self, file, folder=None):
            return _straight_map()

    config = BitsConfig(
        history_steps=1,
        future_steps=1,
        max_agents=1,
        raster_size=64,
        pixel_size=0.5,
    )
    log_spec = NuPlanLogSpec(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
    )
    run_config = BitsRunConfig(
        config=config,
        split=NuPlanBitsSplit(test=(log_spec,)),
        schedule=BitsTrainingSchedule(
            epochs=1,
            batch_size=1,
            hidden_dim=16,
            max_train_samples_per_log=1,
            max_val_samples_per_log=1,
            checkpoint_every_epochs=1,
            seed=29,
        ),
    )
    _model, history = train_nuplan_bits_model(
        NuPlanBitsSplit(train=(log_spec,)),
        tmp_path / "train",
        config=config,
        schedule=run_config.schedule,
        parser=FakeNuPlanParser(),
    )

    loaded_model, result = run_nuplan_bits_rolling_protocol(
        run_config,
        tmp_path / "rolling",
        checkpoint_path=history.checkpoints[0],
        simulation_steps=1,
        parser=FakeNuPlanParser(),
        num_samples=2,
        mask_drivable=True,
    )
    with open(result.result_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    assert isinstance(loaded_model, BitsBiLevelTorchModel)
    assert result.protocol == "nuplan_bits_rolling_v0"
    assert result.inference["source"] == "tactics2d"
    assert result.test["split"] == "test"
    assert result.test["simulation_steps"] == 1
    assert result.test["summary"]["evaluated_log_count"] == 1
    assert result.test["log_results"][0]["start_frame"] == 0
    assert result.test["log_results"][0]["ego_id"] == 1
    assert result.test["log_results"][0]["metrics"]["prediction_round_count"] == 1
    assert payload["test"]["summary"]["evaluated_log_count"] == 1


@pytest.mark.bits_workflow
def test_run_nuplan_bits_torch_validation_builds_dataset_and_model():
    import torch

    torch.manual_seed(19)

    class FakeNuPlanParser:
        def parse_trajectory(self, file, folder, time_range=None):
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                        _state(200, 12.0, 0.0),
                    ],
                ),
                2: _vehicle(
                    2,
                    [
                        _state(-100, 14.0, 1.0),
                        _state(0, 15.0, 1.0),
                        _state(100, 16.0, 1.0),
                        _state(200, 17.0, 1.0),
                    ],
                ),
            }
            return participants, time_range or (-100, 200)

        def parse_map(self, file, folder=None):
            return _successor_map()

    config = BitsConfig(
        history_steps=1,
        future_steps=2,
        max_agents=1,
        raster_size=64,
        pixel_size=0.5,
    )

    result = run_nuplan_bits_torch_validation(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        map_folder="maps",
        time_range=(-100, 200),
        frame_range=(0, 0),
        parser=FakeNuPlanParser(),
        config=config,
        batch_size=2,
        max_samples=1,
        hidden_dim=32,
        model_arch="resnet18",
        use_ground_truth_goal=False,
        num_samples=2,
        mask_drivable=True,
    )

    assert result.sample_count == 1
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0


@pytest.mark.bits_workflow
def test_run_nuplan_bits_torch_validation_filters_frame_range_after_parse():
    class FakeNuPlanParser:
        seen_time_range = None

        def parse_trajectory(self, file, folder, time_range=None):
            self.seen_time_range = time_range
            participants = {
                1: _vehicle(
                    1,
                    [
                        _state(-100, 9.0, 0.0),
                        _state(0, 10.0, 0.0),
                        _state(100, 11.0, 0.0),
                        _state(200, 12.0, 0.0),
                        _state(300, 13.0, 0.0),
                    ],
                )
            }
            return participants, (-100, 300)

        def parse_map(self, file, folder=None):
            return _successor_map()

    parser = FakeNuPlanParser()
    config = BitsConfig(history_steps=1, future_steps=1, max_agents=0)

    result = run_nuplan_bits_torch_validation(
        data_file="log.db",
        data_folder="logs",
        map_file="map.gpkg",
        time_range=(-100, 300),
        frame_range=(200, 200),
        parser=parser,
        config=config,
        max_samples=4,
    )

    assert parser.seen_time_range == (-100, 300)
    assert result.sample_count == 1
    assert result.step_count == 1


@pytest.mark.bits_workflow
def test_run_nuplan_bits_torch_validation_rejects_negative_max_samples():
    with pytest.raises(ValueError, match="max_samples"):
        run_nuplan_bits_torch_validation(
            data_file="log.db",
            data_folder="logs",
            map_file="map.gpkg",
            parser=object(),
            config=BitsConfig(history_steps=0, future_steps=1),
            max_samples=-1,
        )


def test_bits_run_config_round_trip_preserves_split_and_schedule(tmp_path):
    run_config = BitsRunConfig(
        config=BitsConfig(history_steps=2, future_steps=3, max_agents=4),
        split=NuPlanBitsSplit(
            train=(
                NuPlanLogSpec(
                    data_file="train.db",
                    data_folder="logs",
                    map_file="train_map.json",
                    map_folder="maps",
                    frame_range=(10, 20),
                    ego_ids=("ego",),
                ),
            ),
            val=(
                NuPlanLogSpec(
                    data_file="val.db",
                    map_file="val_map.json",
                    time_range=(100, 200),
                ),
            ),
        ),
        schedule=BitsTrainingSchedule(
            epochs=2,
            batch_size=3,
            model_arch="resnet50",
            max_train_samples_per_log=5,
            seed=7,
        ),
    )
    config_path = tmp_path / "bits_run.json"

    save_bits_run_config(config_path, run_config)
    loaded = load_bits_run_config(config_path)
    payload = bits_run_config_to_dict(loaded)
    rebuilt = bits_run_config_from_dict(payload)

    assert loaded == run_config
    assert rebuilt == run_config
    assert payload["split"]["train"][0]["frame_range"] == (10, 20)
    assert payload["schedule"]["model_arch"] == "resnet50"


def test_bits_run_config_requires_split_object():
    with pytest.raises(ValueError, match="split"):
        bits_run_config_from_dict({"config": {}, "schedule": {}})


def test_bits_behavior_model_predicts_world_frame_trajectories():
    config = BitsConfig(history_steps=1, future_steps=2, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(
            1,
            [
                _state(-100, 10.0, 8.0, heading=np.pi / 2),
                _state(0, 10.0, 10.0, heading=np.pi / 2),
            ],
        ),
    }
    model = BitsBehaviorModel(
        config,
        policy=_FixedBitsPolicy([[2.0, 0.0], [4.0, 0.0]]),
        include_raster=False,
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
        config,
        policy=_FixedBitsPolicy([[0.3, 0.0]]),
        include_raster=False,
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

    model = BitsBehaviorModel(
        include_raster=False,
        device="cpu",
    )

    assert model.config == default_config
    assert isinstance(model.policy, _FixedBitsPolicy)


def test_bits_behavior_model_rejects_config_without_custom_policy():
    with pytest.raises(ValueError, match="config is loaded from the default BITS policy"):
        BitsBehaviorModel(BitsConfig(history_steps=0, future_steps=1), include_raster=False)


def test_bits_rolling_runner_requires_explicit_behavior_model():
    with pytest.raises(ValueError, match="behavior_model"):
        BitsRollingRunner(BitsConfig(history_steps=0, future_steps=1))


def test_bits_rolling_runner_advances_controlled_agents_with_bits_prediction():
    config = BitsConfig(history_steps=1, future_steps=2, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(1, [_state(-100, -1.0, 0.0), _state(0, 0.0, 0.0)]),
    }

    result = BitsRollingRunner(
        config,
        behavior_model=BitsBehaviorModel(
            config,
            policy=_forward_policy(config),
            include_raster=False,
        ),
    ).run(
        participants,
        _straight_map(),
        start_frame=0,
        simulation_steps=2,
        agent_ids=[1],
    )

    assert result.frames == [0, 100, 200]
    assert len(result.predicted_trajectories) == 2
    assert result.participants[1].trajectory.has_state(200)
    np.testing.assert_allclose(result.participants[1].trajectory.get_state(100).location, (1.0, 0.0))
    np.testing.assert_allclose(result.participants[1].trajectory.get_state(200).location, (2.0, 0.0))
    assert not participants[1].trajectory.has_state(100)


def test_bits_rolling_runner_replays_background_log_states():
    config = BitsConfig(history_steps=0, future_steps=1, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0, speed=1.0)]),
        2: _vehicle(
            2,
            [
                _state(0, 10.0, 0.0, speed=2.0),
                _state(100, 10.0, 2.0, speed=2.0),
            ],
        ),
    }

    result = BitsRollingRunner(
        config,
        behavior_model=BitsBehaviorModel(
            config,
            policy=_forward_policy(config, step_distance=0.1),
            include_raster=False,
        ),
    ).run(
        participants,
        _straight_map(),
        start_frame=0,
        simulation_steps=1,
        agent_ids=[1],
    )

    assert set(result.participants) == {1, 2}
    np.testing.assert_allclose(result.participants[2].trajectory.get_state(100).location, (10.0, 2.0))
    assert 2 not in result.predicted_trajectories[0]


def test_bits_rolling_runner_extrapolates_background_when_log_state_is_missing():
    config = BitsConfig(history_steps=0, future_steps=1, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0, speed=1.0)]),
        2: _vehicle(2, [_state(0, 10.0, 0.0, heading=np.pi / 2, speed=3.0)]),
    }

    result = BitsRollingRunner(
        config,
        behavior_model=BitsBehaviorModel(
            config,
            policy=_forward_policy(config, step_distance=0.1),
            include_raster=False,
        ),
    ).run(
        participants,
        _straight_map(),
        start_frame=0,
        simulation_steps=1,
        agent_ids=[1],
    )

    np.testing.assert_allclose(
        result.participants[2].trajectory.get_state(100).location,
        (10.0, 0.3),
        atol=1e-9,
    )


def test_bits_rolling_runner_rejects_negative_steps():
    with pytest.raises(ValueError, match="simulation_steps"):
        config = BitsConfig()
        BitsRollingRunner(
            config,
            behavior_model=BitsBehaviorModel(
                config,
                policy=_forward_policy(config),
                include_raster=False,
            ),
        ).run(
            participants={},
            map_=None,
            start_frame=0,
            simulation_steps=-1,
        )


def test_evaluate_bits_rolling_result_reports_safety_and_log_error():
    config = BitsConfig(history_steps=1, future_steps=1, dt=0.1, raster_size=64)
    participants = {
        1: _vehicle(
            1,
            [
                _state(-100, -1.0, 0.0),
                _state(0, 0.0, 0.0),
                _state(100, 1.0, 0.0),
            ],
        ),
        2: _vehicle(
            2,
            [
                _state(-100, 8.0, 0.0, speed=1.0),
                _state(0, 8.0, 0.0, speed=1.0),
                _state(100, 8.0, 0.1, speed=1.0),
            ],
        ),
    }
    result = BitsRollingRunner(
        config,
        behavior_model=BitsBehaviorModel(
            config,
            policy=_forward_policy(config),
            include_raster=False,
        ),
    ).run(
        participants,
        _straight_map(),
        start_frame=0,
        simulation_steps=1,
        agent_ids=[1],
    )

    evaluation = evaluate_bits_rolling_result(
        result,
        reference_participants=participants,
        map_=_straight_map(),
    )

    assert evaluation.frame_count == 2
    assert evaluation.prediction_round_count == 1
    assert evaluation.min_distance > 0.0
    assert evaluation.collision_count == 0
    assert evaluation.first_collision is None
    assert evaluation.off_drivable_count == 0
    assert evaluation.off_drivable_rate == 0.0
    assert evaluation.first_off_drivable is None
    assert evaluation.mean_ade == pytest.approx(0.0)
    assert evaluation.mean_fde == pytest.approx(0.0)
    assert set(evaluation.trajectory_errors) == {1, 2}
    assert evaluation.as_dict()["frame_count"] == 2


def test_evaluate_bits_rolling_result_detects_collision():
    config = BitsConfig(history_steps=0, future_steps=1, dt=0.1)
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 0.0, speed=0.0)]),
        2: _vehicle(2, [_state(0, 0.0, 0.0, speed=0.0)]),
    }
    result = BitsRollingRunner(
        config,
        behavior_model=BitsBehaviorModel(
            config,
            policy=_forward_policy(config),
            include_raster=False,
        ),
    ).run(
        participants,
        _straight_map(),
        start_frame=0,
        simulation_steps=0,
    )

    evaluation = evaluate_bits_rolling_result(result)

    assert evaluation.collision_count == 1
    assert evaluation.first_collision == (0, 1, 2)


def test_evaluate_bits_rolling_result_reports_off_drivable_states():
    config = BitsConfig(history_steps=0, future_steps=1, dt=0.1)
    participants = {
        1: _vehicle(1, [_state(0, 0.0, 5.0, speed=0.0)]),
    }
    result = BitsRollingRunner(
        config,
        behavior_model=BitsBehaviorModel(
            config,
            policy=_forward_policy(config),
            include_raster=False,
        ),
    ).run(
        participants,
        _straight_map(),
        start_frame=0,
        simulation_steps=0,
    )

    evaluation = evaluate_bits_rolling_result(result, map_=_straight_map())

    assert evaluation.off_drivable_count == 1
    assert evaluation.off_drivable_rate == 1.0
    assert evaluation.first_off_drivable == (0, 1)
    assert evaluation.as_dict()["first_off_drivable"] == (0, 1)


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
            [
                [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
                [[0.0, 0.0], [0.0, 3.0], [0.0, 6.0]],
            ],
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
        positions=np.asarray(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[1.0, 5.0], [2.0, 5.0]],
            ],
            dtype=float,
        ),
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




def test_bits_goal_supervision_uses_last_available_future_target():
    config = BitsConfig(
        history_steps=0, future_steps=3, max_agents=1, raster_size=64, pixel_size=0.5
    )
    participants = {
        1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0), _state(300, 13.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )

    supervision = build_goal_supervision(batch)

    assert supervision.goal_index == 2
    np.testing.assert_allclose(supervision.goal_position, [3.0, 0.0])
    np.testing.assert_allclose(supervision.goal_position_pixel, [22, 32])
    np.testing.assert_allclose(supervision.goal_position_residual, [0.0, 0.0])
    assert supervision.goal_position_pixel_flat == 32 * 64 + 22
    assert supervision.goal_spatial_map.shape == (64, 64)
    assert supervision.goal_spatial_map[32, 22] == 1.0
    assert supervision.goal_spatial_map.sum() == 1.0


def test_bits_batch_to_torch_converts_required_and_optional_fields():
    import torch

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
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
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=False,
    )

    torch_batch = collate_bits_batches_to_torch(dataset, include_optional=False)

    assert torch_batch.tensors["history_positions"].shape == (2, 1, 2)
    assert torch_batch.tensors["target_positions"].shape == (2, 1, 2)
    assert torch_batch.tensors["target_availabilities"].dtype == torch.bool
    assert torch_batch.metadata["ego_id"] == [1, 2]


def test_bits_goal_supervision_to_torch_and_prediction_round_trip():
    import torch

    config = BitsConfig(history_steps=0, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {1: _vehicle(1, [_state(0, 10.0, 0.0), _state(100, 11.0, 0.0)])}
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    goal = build_goal_supervision(batch)

    goal_tensors = bits_goal_supervision_to_torch(goal)
    prediction = bits_prediction_from_torch(
        torch.zeros((1, 1, 2), dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
    )

    assert goal_tensors["goal_position"].dtype == torch.float32
    assert goal_tensors["goal_position_pixel_flat"].dtype == torch.long
    assert prediction.positions.shape == (1, 1, 2)
    assert prediction.availabilities.tolist() == [[True]]
    np.testing.assert_allclose(prediction.scores, [1.0])


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
            log_likelihood = torch.tensor([[5.0, 0.0]], dtype=positions.dtype, device=positions.device)
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
            yaws = torch.zeros((1, 2, 1, 1), dtype=tensors["target_yaws"].dtype, device=positions.device)
            agent_positions = torch.tensor(
                [[[[10.0, 0.0]], [[20.0, 0.0]]]],
                dtype=positions.dtype,
                device=positions.device,
            ).unsqueeze(2)
            agent_yaws = torch.zeros((1, 2, 1, 1, 1), dtype=positions.dtype, device=positions.device)
            scene_availabilities = torch.ones((1, 2, 2, 1), dtype=torch.bool, device=positions.device)
            return {
                "plan": {
                    "positions": positions[:, :, -1],
                    "yaws": yaws[:, :, -1],
                    "log_likelihood": torch.zeros((1, 2), dtype=positions.dtype, device=positions.device),
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
            yaws = torch.zeros((1, 2, 1, 1), dtype=tensors["target_yaws"].dtype, device=positions.device)
            return {
                "plan": {
                    "positions": positions[:, :, -1],
                    "yaws": yaws[:, :, -1],
                    "log_likelihood": torch.tensor([[0.0, 1.0]], dtype=positions.dtype, device=positions.device),
                },
                "predictions": {"positions": positions, "yaws": yaws},
            }

    config = BitsConfig(
        history_steps=0,
        future_steps=1,
        progress_weight=0.0,
        lane_weight=0.0,
        collision_weight=0.0,
    )
    participants = {1: _vehicle(1, [_state(0, 0.0, 0.0), _state(100, 1.0, 0.0)])}
    batch = BitsBatchBuilder(config).build(participants, frame=0, ego_id=1)
    policy = TorchBitsPolicy(
        TinyBiLevelPolicy(),
        plan_scorer=BitsPlanScorer(config),
        select_best_plan=False,
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
        spatial_prediction,
        torch.as_tensor(agent_from_raster[None], dtype=torch.float32),
    )

    assert decoded["positions"].shape == (1, 1, 2)
    np.testing.assert_allclose(decoded["pixel_positions"].detach().numpy()[0, 0], [22.25, 32.75])
    np.testing.assert_allclose(decoded["positions"].detach().numpy()[0, 0], [3.125, -0.375])
    assert decoded["scores"][0, 0] > 0.1


def test_bits_spatial_planner_residual_loss_uses_sigmoid_like_tbsim():
    import torch

    output = {
        "plan": {
            "spatial_prediction": torch.zeros((1, 4, 2, 2), dtype=torch.float32),
        },
        "predictions": {
            "positions": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            "yaws": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "availabilities": torch.ones((1, 1, 1), dtype=torch.bool),
        },
    }
    tensors = {
        "target_positions": torch.zeros((1, 1, 2), dtype=torch.float32),
        "target_yaws": torch.zeros((1, 1, 1), dtype=torch.float32),
        "target_availabilities": torch.ones((1, 1), dtype=torch.bool),
    }
    goal_tensors = {
        "goal_position_pixel_flat": torch.tensor([0], dtype=torch.long),
        "goal_spatial_map": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        "goal_position_residual": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        "goal_yaw": torch.zeros((1, 1), dtype=torch.float32),
    }

    losses = compute_bits_torch_losses(
        output,
        tensors,
        goal_tensors,
        loss_weights={
            "prediction_loss": 0.0,
            "goal_loss": 0.0,
            "pixel_bce_loss": 0.0,
            "pixel_ce_loss": 0.0,
            "pixel_res_loss": 1.0,
            "pixel_yaw_loss": 0.0,
        },
    )

    assert torch.isclose(losses["pixel_res_loss"], torch.tensor(0.0))


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
        ),
    }
    batch = BitsBatchBuilder(config).build(
        participants, frame=0, ego_id=1, map_=_straight_map(), include_raster=True
    )
    torch_batch = bits_batch_to_torch(batch)
    model = BitsBiLevelTorchModel(
        image_channels=batch.image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=32,
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
    assert any(key.startswith("predictor.future_state_head.agents_decoder.mlp._model") for key in state_keys)


def test_integrate_unicycle_controls_keeps_zero_control_constant_velocity():
    import torch

    config = BitsConfig(dt=0.5, future_steps=3)
    current_states = torch.tensor([[0.0, 0.0, 2.0, 0.0]], dtype=torch.float32)
    controls = torch.zeros((1, 1, 3, 2), dtype=torch.float32)

    positions, yaws = integrate_unicycle_controls(controls, current_states, config)

    np.testing.assert_allclose(
        positions.detach().numpy()[0, 0],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        atol=1e-6,
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


def test_bits_agent_aware_trajectory_module_predicts_ego_and_agents():
    config = BitsConfig(history_steps=1, future_steps=2, max_agents=2, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(
            1,
            [
                _state(-100, 9.0, 0.0),
                _state(0, 10.0, 0.0),
                _state(100, 11.0, 0.0),
                _state(200, 12.0, 0.0),
            ],
        ),
        2: _vehicle(
            2,
            [
                _state(-100, 12.0, 1.0),
                _state(0, 13.0, 1.0),
                _state(100, 14.0, 1.0),
                _state(200, 15.0, 1.0),
            ],
        ),
    }
    batch = BitsBatchBuilder(config).build(
        participants,
        frame=0,
        ego_id=1,
        map_=_straight_map(),
        include_raster=True,
    )
    torch_batch = bits_batch_to_torch(batch)
    goals = bits_goal_supervision_to_torch(build_goal_supervision(batch))
    module = BitsAgentAwareTrajectoryModule(
        future_steps=config.future_steps,
        image_channels=batch.image.shape[0],
        global_feature_dim=32,
        agent_feature_dim=32,
        goal_feature_dim=8,
        decoder_layer_dims=(32,),
        context_size=6,
        roi_feature_size=4,
        model_arch="resnet18",
    )

    output = module(
        torch_batch.tensors,
        goals["goal_position"],
        goals["goal_yaw"],
    )

    assert output["positions"].shape == (1, 1, 2, 2)
    assert output["controls"].shape == (1, 1, 2, 2)
    assert output["agent_positions"].shape == (1, 1, 2, 2, 2)
    assert output["agent_controls"].shape == (1, 1, 2, 2, 2)
    assert output["scene_positions"].shape == (1, 1, 3, 2, 2)
    assert output["scene_controls"].shape == (1, 1, 3, 2, 2)
    assert output["trajectories"].shape == (1, 1, 3, 2, 3)
    assert output["scene_availabilities"][0, 0, 0].all()
    assert output["scene_availabilities"][0, 0, 1].all()
    assert not output["scene_availabilities"][0, 0, 2].any()


def test_bits_agent_aware_module_can_enable_official_history_transformer_branch():
    import torch

    config = BitsConfig(
        history_steps=2,
        future_steps=1,
        max_agents=1,
        raster_size=32,
        pixel_size=0.5,
    )
    participants = {
        1: _vehicle(
            1,
            [_state(-200, 0.0, 0.0), _state(-100, 0.5, 0.0), _state(0, 1.0, 0.0), _state(100, 2.0, 0.0)],
        ),
        2: _vehicle(
            2,
            [_state(-200, 4.0, 0.0), _state(-100, 4.5, 0.0), _state(0, 5.0, 0.0), _state(100, 6.0, 0.0)],
        ),
    }
    batch = BitsBatchBuilder(config).build(
        participants,
        frame=0,
        ego_id=1,
        map_=_straight_map(),
        include_raster=True,
    )
    torch_batch = bits_batch_to_torch(batch)
    goals = bits_goal_supervision_to_torch(build_goal_supervision(batch))
    module = BitsAgentAwareTrajectoryModule(
        future_steps=config.future_steps,
        image_channels=batch.image.shape[0],
        global_feature_dim=16,
        agent_feature_dim=16,
        goal_feature_dim=8,
        decoder_layer_dims=(16,),
        context_size=6,
        roi_feature_size=4,
        history_conditioning=True,
        use_transformer=True,
        config=config,
    )
    module.eval()

    output = module(torch_batch.tensors, goals["goal_position"], goals["goal_yaw"])
    loss = output["trajectories"].sum()
    loss.backward()

    assert output["trajectories"].shape == (1, 1, 2, 1, 3)
    assert module.history_encoder is not None
    assert module.transformer is not None
    assert module.roi_size.shape == (4,)
    assert module.weights_scaling.shape == (3,)
    assert any(param.grad is not None for param in module.transformer.parameters())


def test_bits_xy_positional_encoding_keeps_axes_in_separate_sin_cos_channels():
    import torch

    encoder = _BitsPositionalEncodingNd(dim=8, dropout=0.0, step_size=(1.0, 1.0))
    inputs = torch.zeros((1, 3, 8), dtype=torch.float32)
    positions = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]],
        dtype=torch.float32,
    )

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
        participants,
        frame=0,
        ego_id=1,
        map_=_straight_map(),
        include_raster=True,
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

    agent_features, global_features, encoder_features = encoder(torch_batch.tensors, agent_positions)
    loss = agent_features.sum() + global_features.sum()
    loss.backward()

    assert agent_features.shape == (1, 2, 8)
    assert global_features.shape == (1, 8)
    assert "layer2" in encoder_features
    assert any(param.grad is not None for param in encoder.parameters())


def test_bits_rasterize_roi_encoder_uses_official_context_and_feature_size():
    from torchvision.ops import RoIAlign

    encoder = BitsRasterizeROIEncoder(
        image_channels=5,
        context_size=30,
        roi_feature_size=7,
        model_arch="resnet18",
    )

    assert encoder.context_size == 30
    assert encoder.roi_feature_size == 7
    assert encoder.roi_layer_key == "layer2"
    assert isinstance(encoder.roi_align, RoIAlign)
    assert encoder.agent_net[2].in_features == 128


def test_bits_agent_aware_module_uses_official_decoder_names():
    config = BitsConfig(history_steps=1, future_steps=1, max_agents=1, raster_size=32, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 0.0, 0.0), _state(0, 1.0, 0.0), _state(100, 2.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 4.0, 0.0), _state(0, 5.0, 0.0), _state(100, 6.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants,
        frame=0,
        ego_id=1,
        map_=_straight_map(),
        include_raster=True,
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

    assert any(key.startswith("shared_encoder.encoder_heads.map_model.layer1") for key in state_keys)
    assert any(key.startswith("roi_head.agent_net") for key in state_keys)
    assert any(key.startswith("policy_head.goal_encoder._model") for key in state_keys)
    assert any(key.startswith("policy_head.ego_decoder.mlp._model") for key in state_keys)
    assert any(key.startswith("future_state_head.agents_decoder.mlp._model") for key in state_keys)


def test_bits_agent_aware_prediction_loss_uses_other_agent_targets():
    import torch

    tensors = {
        "target_positions": torch.zeros((1, 1, 2), dtype=torch.float32),
        "target_yaws": torch.zeros((1, 1, 1), dtype=torch.float32),
        "target_availabilities": torch.ones((1, 1), dtype=torch.bool),
        "all_other_agents_future_positions": torch.tensor(
            [[[[10.0, 0.0]]]],
            dtype=torch.float32,
        ),
        "all_other_agents_future_yaws": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        "all_other_agents_future_availability": torch.ones((1, 1, 1), dtype=torch.bool),
    }
    output = {
        "predictions": {
            "scene_positions": torch.zeros((1, 1, 2, 1, 2), dtype=torch.float32),
            "scene_yaws": torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32),
        }
    }

    losses = compute_bits_torch_losses(output, tensors, loss_weights={"prediction_loss": 1.0})

    assert torch.isclose(losses["prediction_loss"], torch.tensor(100.0 / 6.0))
    assert torch.isclose(losses["total"], losses["prediction_loss"])


def test_bits_torch_losses_compute_tbsim_style_goal_loss():
    import torch

    tensors = {
        "target_positions": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]], dtype=torch.float32),
        "target_yaws": torch.zeros((1, 2, 1), dtype=torch.float32),
        "target_availabilities": torch.tensor([[True, True]]),
    }
    output = {
        "predictions": {
            "positions": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "yaws": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
        }
    }

    losses = compute_bits_torch_losses(
        output,
        tensors,
        loss_weights={"prediction_loss": 0.0, "goal_loss": 1.0},
    )

    assert torch.isclose(losses["prediction_loss"], torch.tensor(5.0 / 6.0))
    assert torch.isclose(losses["goal_loss"], torch.tensor(4.0 / 6.0))
    assert torch.isclose(losses["total"], losses["goal_loss"])


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


def test_bits_bilevel_torch_model_forward_and_losses_are_differentiable():
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 14.0, 0.0), _state(0, 15.0, 0.0), _state(100, 16.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=True,
    )
    batches = list(dataset)
    torch_batch = collate_bits_batches_to_torch(batches)
    goals = collate_bits_goal_supervisions_to_torch(
        [build_goal_supervision(batch) for batch in batches]
    )
    model = BitsBiLevelTorchModel(
        image_channels=torch_batch.tensors["image"].shape[1],
        future_steps=config.future_steps,
        hidden_dim=32,
    )

    output = model(
        torch_batch.tensors,
        goal_tensors=goals,
        use_ground_truth_goal=True,
    )
    losses = compute_bits_torch_losses(output, torch_batch.tensors, goals)

    assert output["plan"]["positions"].shape == (2, 1, 2)
    assert output["predictions"]["positions"].shape == (2, 1, 1, 2)
    assert set(losses) >= {
        "prediction_loss",
        "goal_loss",
        "pixel_bce_loss",
        "pixel_ce_loss",
        "pixel_res_loss",
        "pixel_yaw_loss",
        "total",
    }
    losses["total"].backward()
    assert any(param.grad is not None for param in model.parameters())


def test_bits_bilevel_torch_model_can_use_agent_aware_traffic_head():
    config = BitsConfig(history_steps=1, future_steps=1, max_agents=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 14.0, 0.0), _state(0, 15.0, 0.0), _state(100, 16.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants,
        frame=0,
        ego_id=1,
        map_=_straight_map(),
        include_raster=True,
    )
    torch_batch = bits_batch_to_torch(batch)
    goals = collate_bits_goal_supervisions_to_torch([build_goal_supervision(batch)])
    model = BitsBiLevelTorchModel(
        image_channels=batch.image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=32,
    )

    output = model(
        torch_batch.tensors,
        goal_tensors=goals,
        use_ground_truth_goal=True,
    )
    losses = compute_bits_torch_losses(output, torch_batch.tensors, goals)

    assert output["predictions"]["scene_positions"].shape == (1, 1, 2, 1, 2)
    assert output["predictions"]["agent_positions"].shape == (1, 1, 1, 1, 2)
    assert losses["prediction_loss"] > 0.0
    losses["total"].backward()
    assert any(param.grad is not None for param in model.parameters())


def test_compute_bits_torch_losses_uses_tbsim_style_weights():
    import torch

    config = BitsConfig(
        history_steps=1,
        future_steps=1,
        raster_size=64,
        pixel_size=0.5,
        prediction_loss_weight=2.0,
        goal_loss_weight=5.0,
        pixel_bce_loss_weight=0.5,
        pixel_ce_loss_weight=3.0,
        pixel_res_loss_weight=0.0,
        pixel_yaw_loss_weight=4.0,
    )
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
    }
    batch = BitsBatchBuilder(config).build(
        participants,
        frame=0,
        ego_id=1,
        map_=_straight_map(),
        include_raster=True,
    )
    torch_batch = bits_batch_to_torch(batch)
    goals = collate_bits_goal_supervisions_to_torch([build_goal_supervision(batch)])
    model = BitsBiLevelTorchModel(
        image_channels=batch.image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=32,
    )

    output = model(
        torch_batch.tensors,
        goal_tensors=goals,
        use_ground_truth_goal=True,
    )
    losses = compute_bits_torch_losses(
        output,
        torch_batch.tensors,
        goals,
        config=config,
    )

    expected_total = (
        losses["prediction_loss"] * config.prediction_loss_weight
        + losses["goal_loss"] * config.goal_loss_weight
        + losses["pixel_bce_loss"] * config.pixel_bce_loss_weight
        + losses["pixel_ce_loss"] * config.pixel_ce_loss_weight
        + losses["pixel_res_loss"] * config.pixel_res_loss_weight
        + losses["pixel_yaw_loss"] * config.pixel_yaw_loss_weight
    )
    assert torch.allclose(losses["total"], expected_total)


def test_run_bits_torch_epoch_can_validate_without_optimizer():
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=True,
    )
    sample = dataset[0]
    image_channels = sample.image.shape[0]
    model = BitsBiLevelTorchModel(
        image_channels=image_channels,
        future_steps=config.future_steps,
        hidden_dim=32,
    )

    result = run_bits_torch_epoch(model, dataset, batch_size=1)

    assert result.sample_count == 1
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0
    assert "pixel_ce_loss" in result.mean_losses
    assert result.as_dict()["sample_count"] == 1


def test_run_bits_torch_epoch_can_use_sampled_plans_and_drivable_mask():
    import torch

    torch.manual_seed(17)
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 14.0, 0.0), _state(0, 15.0, 0.0), _state(100, 16.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=True,
    )
    model = BitsBiLevelTorchModel(
        image_channels=dataset[0].image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=32,
        config=config,
    )

    result = run_bits_torch_epoch(
        model,
        dataset,
        batch_size=2,
        use_ground_truth_goal=False,
        num_samples=3,
        mask_drivable=True,
        config=config,
    )

    assert result.sample_count == 2
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0
    assert "pixel_ce_loss" in result.mean_losses


def test_run_bits_planner_torch_epoch_updates_only_spatial_planner():
    import torch

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=True,
    )
    model = BitsBiLevelTorchModel(
        image_channels=dataset[0].image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=16,
        config=config,
    )
    optimizer = torch.optim.Adam(
        list(model.shared_encoder.parameters())
        + list(model.planner.spatial_goal_decoder.parameters()),
        lr=1e-3,
    )

    result = run_bits_planner_torch_epoch(
        model,
        dataset,
        optimizer=optimizer,
        batch_size=1,
        config=config,
    )

    assert result.sample_count == 1
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0
    assert "pixel_ce_loss" in result.mean_losses
    assert any(param.grad is not None for param in model.planner.spatial_goal_decoder.parameters())
    assert any(param.grad is not None for param in model.shared_encoder.parameters())
    assert all(param.grad is None for param in model.predictor.parameters())


def test_run_bits_planner_torch_epoch_can_freeze_shared_encoder():
    import torch

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=True,
    )
    model = BitsBiLevelTorchModel(
        image_channels=dataset[0].image.shape[0],
        future_steps=config.future_steps,
        hidden_dim=16,
        config=config,
    )
    shared_encoder_before = {
        name: tensor.detach().clone()
        for name, tensor in model.shared_encoder.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.planner.spatial_goal_decoder.parameters(), lr=1e-3)

    result = run_bits_planner_torch_epoch(
        model,
        dataset,
        optimizer=optimizer,
        batch_size=1,
        config=config,
        freeze_shared_encoder=True,
    )

    assert result.sample_count == 1
    assert result.mean_total_loss > 0.0
    assert any(param.grad is not None for param in model.planner.spatial_goal_decoder.parameters())
    assert all(param.grad is None for param in model.shared_encoder.parameters())
    assert all(param.grad is None for param in model.predictor.parameters())
    assert all(
        torch.equal(model.shared_encoder.state_dict()[name], before)
        for name, before in shared_encoder_before.items()
    )


def test_run_bits_torch_epoch_handles_empty_dataset():
    import torch

    model = BitsBiLevelTorchModel(image_channels=4, future_steps=1, hidden_dim=16)

    result = run_bits_torch_epoch(model, [], batch_size=1)

    assert result.sample_count == 0
    assert result.step_count == 0
    assert result.mean_total_loss == 0.0
    assert result.mean_losses == {}
    assert all(param.grad is None for param in model.parameters())


def test_bits_checkpoint_round_trip_preserves_model_structure(tmp_path):
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    model = BitsBiLevelTorchModel(
        image_channels=5,
        future_steps=config.future_steps,
        hidden_dim=16,
        model_arch="resnet18",
        context_size=12,
        roi_feature_size=5,
        roi_layer_key="layer1",
        config=config,
    )
    schedule = BitsTrainingSchedule(
        hidden_dim=16,
        model_arch="resnet18",
        context_size=12,
        roi_feature_size=5,
        roi_layer_key="layer1",
    )
    checkpoint_path = tmp_path / "bits.pt"

    save_bits_checkpoint(
        checkpoint_path,
        model=model,
        epoch=3,
        image_channels=5,
        config=config,
        schedule=schedule,
    )
    loaded_model, metadata, payload = load_bits_checkpoint(checkpoint_path)

    assert metadata["epoch"] == 3
    assert metadata["model_arch"] == "resnet18"
    assert metadata["context_size"] == 12
    assert metadata["roi_feature_size"] == 5
    assert metadata["roi_layer_key"] == "layer1"
    assert loaded_model.context_size == 12
    assert loaded_model.roi_feature_size == 5
    assert loaded_model.roi_layer_key == "layer1"
    assert set(payload["model_state_dict"]) == set(model.state_dict())


def test_bits_inference_loader_can_use_tactics2d_checkpoint(tmp_path):
    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    model = BitsBiLevelTorchModel(
        image_channels=5,
        future_steps=config.future_steps,
        hidden_dim=16,
        config=config,
    )
    checkpoint_path = tmp_path / "bits.pt"
    save_bits_checkpoint(
        checkpoint_path,
        model=model,
        epoch=2,
        image_channels=5,
        config=config,
    )

    result = load_bits_inference_model(checkpoint_path=checkpoint_path)

    assert result.source == "tactics2d"
    assert result.compatibility is None
    assert result.metadata["epoch"] == 2
    assert isinstance(result.model, BitsBiLevelTorchModel)


def test_tbsim_bits_checkpoint_key_mapping_shares_encoder_and_keeps_heads_separate():
    import torch

    planner_weight = torch.ones(1)
    predictor_weight = torch.zeros(1)
    planner_mapped = map_tbsim_bits_planner_state_dict(
        {
            "state_dict.nets.policy.encoder_heads.map_model.conv1.weight": planner_weight,
            "nets.policy.decoder.conv2.0.bias": planner_weight,
        }
    )
    predictor_mapped = map_tbsim_bits_predictor_state_dict(
        {
            "state_dict.model.map_encoder.encoder_heads.map_model.conv1.weight": predictor_weight,
            "model.ego_decoder.mlp._model.0.weight": predictor_weight,
        }
    )
    merged = merge_tbsim_bits_state_dicts(
        {
            "nets.policy.encoder_heads.map_model.conv1.weight": planner_weight,
        },
        {
            "model.map_encoder.encoder_heads.map_model.conv1.weight": predictor_weight,
        },
    )

    assert planner_mapped["shared_encoder.encoder_heads.map_model.conv1.weight"] is planner_weight
    assert planner_mapped["planner.spatial_goal_decoder.decoder.conv2.0.bias"] is planner_weight
    assert predictor_mapped["shared_encoder.encoder_heads.map_model.conv1.weight"] is predictor_weight
    assert predictor_mapped["predictor.policy_head.ego_decoder.mlp._model.0.weight"] is predictor_weight
    assert merged["shared_encoder.encoder_heads.map_model.conv1.weight"] is predictor_weight
    assert set(merged) == {
        "shared_encoder.encoder_heads.map_model.conv1.weight",
    }


def test_tbsim_bits_inference_weight_loader_can_load_partial_weights_when_not_strict():
    import torch

    model = BitsBiLevelTorchModel(image_channels=5, future_steps=1, hidden_dim=16)
    target_key = "planner.spatial_goal_decoder.decoder.conv2.0.bias"
    new_value = torch.full_like(model.state_dict()[target_key], 3.0)
    planner_payload = {
        "state_dict": {
            "nets.policy.decoder.conv2.0.bias": new_value,
        }
    }

    report = load_tbsim_bits_inference_weights(
        model,
        planner_checkpoint=planner_payload,
        strict=False,
    )

    assert target_key in report.matched_keys
    assert not report.is_compatible
    assert torch.allclose(model.state_dict()[target_key], new_value)


def test_tbsim_bits_inference_weight_loader_rejects_incompatible_strict_load():
    import torch

    model = BitsBiLevelTorchModel(image_channels=5, future_steps=1, hidden_dim=16)
    planner_payload = {
        "state_dict": {
            "nets.policy.decoder.conv2.0.bias": torch.zeros(1),
        }
    }

    with pytest.raises(ValueError, match="not compatible"):
        load_tbsim_bits_inference_weights(
            model,
            planner_checkpoint=planner_payload,
            strict=True,
        )


def test_bits_inference_loader_can_use_official_planner_predictor_checkpoints():
    model = BitsBiLevelTorchModel(image_channels=5, future_steps=1, hidden_dim=16)
    model_state = model.state_dict()
    planner_payload = {
        "state_dict": {
            "nets.policy.decoder.conv2.0.bias": model_state[
                "planner.spatial_goal_decoder.decoder.conv2.0.bias"
            ].clone(),
        }
    }
    predictor_payload = {
        "state_dict": {
            "model.ego_decoder.mlp._model.0.weight": model_state[
                "predictor.policy_head.ego_decoder.mlp._model.0.weight"
            ].clone(),
        }
    }

    result = load_bits_inference_model(
        planner_checkpoint=planner_payload,
        predictor_checkpoint=predictor_payload,
        image_channels=5,
        future_steps=1,
        hidden_dim=16,
        strict=False,
    )

    assert result.source == "tbsim"
    assert result.compatibility is not None
    assert "planner.spatial_goal_decoder.decoder.conv2.0.bias" in result.compatibility.matched_keys
    assert "predictor.policy_head.ego_decoder.mlp._model.0.weight" in result.compatibility.matched_keys
    assert result.metadata["image_channels"] == 5
    assert isinstance(result.model, BitsBiLevelTorchModel)


def test_bits_inference_loader_can_mix_tactics2d_planner_with_official_predictor():
    import torch

    model = BitsBiLevelTorchModel(image_channels=5, future_steps=1, hidden_dim=16)
    model_state = model.state_dict()
    planner_key = "planner.spatial_goal_decoder.decoder.conv2.0.bias"
    predictor_key = "predictor.policy_head.ego_decoder.mlp._model.0.weight"
    planner_value = torch.full_like(model_state[planner_key], 4.0)
    planner_payload = {
        "metadata": {},
        "model_state_dict": {
            planner_key: planner_value,
            "predictor.policy_head.ego_decoder.mlp._model.0.bias": model_state[
                "predictor.policy_head.ego_decoder.mlp._model.0.bias"
            ].clone(),
        },
    }
    predictor_payload = {
        "state_dict": {
            "model.ego_decoder.mlp._model.0.weight": model_state[predictor_key].clone(),
        }
    }

    result = load_bits_inference_model(
        tactics2d_planner_checkpoint=planner_payload,
        predictor_checkpoint=predictor_payload,
        image_channels=5,
        future_steps=1,
        hidden_dim=16,
        strict=False,
    )

    assert result.source == "mixed"
    assert result.metadata["uses_tactics2d_planner_checkpoint"] is True
    assert result.metadata["uses_tbsim_predictor_checkpoint"] is True
    assert planner_key in result.compatibility.matched_keys
    assert predictor_key in result.compatibility.matched_keys
    assert "predictor.policy_head.ego_decoder.mlp._model.0.bias" in result.compatibility.missing_keys
    assert torch.allclose(result.model.state_dict()[planner_key], planner_value)


def test_bits_inference_loader_can_enable_official_predictor_history_transformer():
    import torch

    model = BitsBiLevelTorchModel(
        image_channels=5,
        future_steps=1,
        hidden_dim=16,
        history_conditioning=True,
        use_transformer=True,
    )
    model_state = model.state_dict()
    predictor_payload = {
        "state_dict": {
            "model.history_encoder.lstm.weight_ih_l0": model_state[
                "predictor.history_encoder.lstm.weight_ih_l0"
            ].clone(),
            "model.transformer.pre_emb.weight": model_state[
                "predictor.transformer.pre_emb.weight"
            ].clone(),
            "model.weights_scaling": model_state["predictor.weights_scaling"].clone(),
        }
    }

    result = load_bits_inference_model(
        predictor_checkpoint=predictor_payload,
        image_channels=5,
        future_steps=1,
        hidden_dim=16,
        history_conditioning=True,
        use_transformer=True,
        strict=False,
    )

    assert result.model.history_conditioning is True
    assert result.model.use_transformer is True
    assert "predictor.history_encoder.lstm.weight_ih_l0" in result.compatibility.matched_keys
    assert "predictor.transformer.pre_emb.weight" in result.compatibility.matched_keys
    assert "predictor.weights_scaling" in result.compatibility.matched_keys


def test_bits_inference_loader_rejects_ambiguous_weight_sources(tmp_path):
    with pytest.raises(ValueError, match="either checkpoint_path"):
        load_bits_inference_model(
            checkpoint_path=tmp_path / "bits.pt",
            planner_checkpoint={},
            image_channels=5,
            future_steps=1,
        )

    with pytest.raises(ValueError, match="only one planner source"):
        load_bits_inference_model(
            tactics2d_planner_checkpoint={"model_state_dict": {}},
            planner_checkpoint={},
            image_channels=5,
            future_steps=1,
        )


def test_run_bits_torch_epoch_updates_model_with_optimizer():
    import torch

    config = BitsConfig(history_steps=1, future_steps=1, raster_size=64, pixel_size=0.5)
    participants = {
        1: _vehicle(1, [_state(-100, 9.0, 0.0), _state(0, 10.0, 0.0), _state(100, 11.0, 0.0)]),
        2: _vehicle(2, [_state(-100, 14.0, 0.0), _state(0, 15.0, 0.0), _state(100, 16.0, 0.0)]),
    }
    dataset = BitsSampleDataset(
        participants=participants,
        map_=_straight_map(),
        config=config,
        include_raster=True,
    )
    image_channels = dataset[0].image.shape[0]
    model = BitsBiLevelTorchModel(
        image_channels=image_channels,
        future_steps=config.future_steps,
        hidden_dim=32,
    )
    run_bits_torch_epoch(model, dataset, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [param.detach().clone() for param in model.parameters()]

    result = run_bits_torch_epoch(model, dataset, optimizer=optimizer, batch_size=2)

    after = [param.detach().clone() for param in model.parameters()]
    assert result.sample_count == 2
    assert result.step_count == 1
    assert result.mean_total_loss > 0.0
    assert any(not torch.allclose(prev, curr) for prev, curr in zip(before, after))
