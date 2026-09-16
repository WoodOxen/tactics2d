# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the SMART behavior model (public API only)."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

pytest.importorskip("torch", reason="SMART tests require torch.")

from tactics2d.behavior import BehaviorModelBase
from tactics2d.behavior.smart import SmartBehaviorModel, SmartConfig
from tactics2d.behavior.smart.dataset import SmartBatchBuilder
from tactics2d.behavior.smart.map_tokenizer import (
    _LIGHT_TYPE_BY_STATE,
    MapTokenizer,
    _resample_polyline,
)
from tactics2d.behavior.smart.motion_tokenizer import MotionTokenizer
from tactics2d.behavior.smart.schema import (
    AgentType,
    LightType,
    PointType,
    PolygonType,
    SmartPrediction,
)
from tactics2d.geometry import polyline, spatial
from tactics2d.map.element import Area, Lane, Map, Regulatory, RoadLine
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

HISTORY_END_MS = 1000


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _state(frame, x, y, heading=0.0, speed=5.0):
    return State(
        frame=frame,
        x=x,
        y=y,
        heading=heading,
        vx=speed * np.cos(heading),
        vy=speed * np.sin(heading),
    )


def _vehicle(agent_id, x=1.0, y=0.0, frames=None, length=4.5, width=1.8):
    """Build a straight-moving vehicle over a frame grid."""

    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=False)
    for step, frame in enumerate(HISTORY_END_MS if frames is None else frames):
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
    return {
        index: _vehicle(index, x=0.0, y=spacing * index, frames=frames) for index in range(count)
    }


class _EchoPolicy:
    """Policy mock that keeps the batch's agents but predicts a fixed line.

    Each tensor row keeps a lane of its own, so a joint rollout does not drive
    the agents into each other and the closed-loop collision counter stays zero.
    """

    def __init__(self, row_spacing=8.0):
        self.row_spacing = row_spacing
        self.decoded = []

    def predict_batch(self, batch):
        self.decoded.append(len(batch.agents.agent_ids))
        count = batch.agents.num_agents
        positions = np.zeros((count, 80, 2), dtype=float)
        positions[:, :, 0] = np.arange(1, 81, dtype=float) * 0.5
        positions[:, :, 1] = self.row_spacing * np.arange(count, dtype=float)[:, None]
        return SmartPrediction(
            agent_ids=list(batch.agents.agent_ids),
            positions=positions,
            headings=np.zeros((count, 80), dtype=float),
            availabilities=np.ones((count, 80), dtype=bool),
            frame_ms0=int(batch.frame_ms),
            step_ms=100,
        )


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@pytest.mark.integration
def test_config_defaults_match_upstream():
    """The defaults carry the architecture the released checkpoint was trained with."""
    config = SmartConfig()
    assert (config.history_steps, config.future_steps) == (11, 80)
    assert (config.token_size, config.map_token_size) == (2048, 1024)
    assert (config.hidden_dim, config.num_heads, config.head_dim) == (128, 8, 16)
    assert (config.num_map_layers, config.num_agent_layers) == (3, 6)
    assert (config.pl2pl_radius, config.pl2a_radius, config.a2a_radius) == (10.0, 30.0, 60.0)
    assert config.time_span == 30


@pytest.mark.integration
def test_config_token_arithmetic():
    """The token time base is the one the checkpoint's slot layout assumes."""
    config = SmartConfig()
    assert config.step_ms == 100
    assert config.planning_steps == 80
    assert config.history_token_slots == 2
    assert config.future_token_slots == 16


@pytest.mark.integration
def test_config_rejects_a_future_that_is_not_whole_tokens():
    """A future span indivisible by the token shift is refused."""
    with pytest.raises(ValueError, match="divisible by shift"):
        SmartConfig(future_steps=81)


@pytest.mark.integration
def test_config_rejects_a_history_shorter_than_two_tokens():
    """A history that cannot fill two token slots is refused."""
    with pytest.raises(ValueError, match="at least two motion tokens"):
        SmartConfig(history_steps=5)


# ------------------------------------------------------------------
# Native geometry the map tokenizer leans on
# ------------------------------------------------------------------


@pytest.mark.integration
def test_native_geometry_contract_holds_for_a_straight_line():
    """The native sampler returns the requested spacing, tangents and normals."""
    points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    positions, headings, normals = polyline.sample_uniformly(points, 0.0, 20.0, 21)
    assert positions.shape == (21, 2)
    gaps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    np.testing.assert_allclose(gaps, 1.0, atol=1e-9)
    np.testing.assert_allclose(headings, 0.0, atol=1e-9)
    tangents = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0, atol=1e-9)


@pytest.mark.integration
def test_native_geometry_contract_holds_for_a_circle():
    """Sampling a circle at unit spacing tracks its analytic tangent."""
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    points = np.stack([10.0 * np.cos(angles), 10.0 * np.sin(angles)], axis=1)
    positions, headings, _ = polyline.sample_uniformly(points, 0.0, polyline.arc_length(points), 63)
    gaps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # Equal arc-length sampling reproduces equal spacing to within the float
    # precision of the interpolation, not exactly.
    assert np.ptp(gaps) < 1e-4
    # A heading is the direction of the *input* segment the sample falls on, not
    # the direction from one sample to the next: the input vertices are 1 degree
    # apart, so a heading may sit up to half that (pi/360) away from the analytic
    # tangent at its own sample point. Bounding by pi/360 + slack pins that down
    # -- a sampler that interpolated headings would come in far tighter, and one
    # that mis-indexed the segment would blow past it.
    analytic = np.arctan2(positions[:, 1], positions[:, 0]) + np.pi / 2.0
    offset = np.abs(np.arctan2(np.sin(analytic - headings), np.cos(analytic - headings)))
    assert np.max(offset) < np.pi / 360.0 + 1e-3
    # Normals stay unit-length and orthogonal to their heading.
    _, _, normals = polyline.sample_uniformly(points, 0.0, polyline.arc_length(points), 63)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
    tangents = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0, atol=1e-9)


# ------------------------------------------------------------------
# Map tokenizer
# ------------------------------------------------------------------


@pytest.mark.integration
def test_map_tokenizer_cuts_a_five_metre_window():
    """A five-metre run becomes exactly one token sampled at 0, 2.5 and 5 m."""
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    headings = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tokens = _resample_polyline(positions, headings)
    assert tokens.shape == (1, 3, 3)
    np.testing.assert_allclose(tokens[0, :, 0], [0.0, 2.5, 5.0], atol=1e-6)
    np.testing.assert_allclose(tokens[0, :, 2], 0.0, atol=1e-6)


@pytest.mark.integration
def test_map_tokenizer_windows_a_nine_metre_run():
    """A nine-metre run yields one full window plus a three-point tail."""
    positions = np.array([[0.0, 0.0], [9.0, 0.0]], dtype=np.float32)
    headings = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tokens = _resample_polyline(positions, headings)
    # 9 m resamples to 19 points: one full 11-point window (arc 0..5 m) and eight
    # leftover points, which is at least three, so the remainder contributes one
    # extra tail token sampled at its own quarter points (arc 5, 7 and 9 m).
    assert tokens.shape == (2, 3, 3)
    np.testing.assert_allclose(tokens[0, :, 0], [0.0, 2.5, 5.0], atol=1e-6)
    np.testing.assert_allclose(tokens[1, :, 0], [5.0, 7.0, 9.0], atol=1e-5)


@pytest.mark.integration
def test_map_tokenizer_builds_tokens_from_a_native_map():
    """A native lane yields centerline tokens with the codebook's type codes."""
    tokens = MapTokenizer(SmartConfig()).build(_straight_map(), 1100)
    assert tokens.num_tokens > 0
    assert tokens.traj_pos.shape[1:] == (3, 2)
    assert set(tokens.pt_type.tolist()) == {PointType.CENTERLINE}
    assert set(tokens.pl_type.tolist()) == {PolygonType.VEHICLE}
    assert set(tokens.pt_side.tolist()) == {0}
    assert set(tokens.light_type.tolist()) == {LightType.UNKNOWN}


@pytest.mark.integration
def test_map_tokenizer_covers_the_lane_with_overlapping_windows():
    """Consecutive tokens step five metres along the lane and overlap by a metre."""
    tokens = MapTokenizer(SmartConfig()).build(_straight_map(), 1100)
    starts = tokens.traj_pos[:, 0, 0].numpy()
    # The lane's kept vertices span 36 m and its windows step 5 m, plus a tail.
    assert starts[0] == pytest.approx(0.0, abs=1e-6)
    steps = np.diff(starts[:8])
    np.testing.assert_allclose(steps, 5.0, atol=1e-5)
    assert np.all(np.diff(tokens.traj_pos[:, 2, 0].numpy()) >= 0.0)


@pytest.mark.integration
def test_map_tokenizer_uses_the_tagged_centerline():
    """The tokenizer reads ``custom_tags['centerline']``, not the boundary midpoint."""
    map_ = Map(name="tagged_centerline")
    # The boundaries sit at y = 50, so their midpoint is nowhere near the tagged
    # centerline at y = 0. If the tokenizer fell back to the midpoint path the
    # token coordinates would move by fifty metres.
    lane = Lane(
        id_="A",
        left_side=LineString([(0.0, 51.75), (9.0, 51.75), (18.0, 51.75), (27.0, 51.75)]),
        right_side=LineString([(0.0, 48.25), (9.0, 48.25), (18.0, 48.25), (27.0, 48.25)]),
        subtype="road",
        custom_tags={"centerline": np.array([[0.0, 0.0], [9.0, 0.0], [18.0, 0.0], [27.0, 0.0]])},
    )
    map_.add_lane(lane)
    tokens = MapTokenizer(SmartConfig()).build(map_, 1100)
    assert tokens.num_tokens > 0
    np.testing.assert_allclose(tokens.traj_pos[:, :, 1].numpy(), 0.0, atol=1e-3)


@pytest.mark.integration
def test_map_tokenizer_type_codes_follow_the_source_element():
    """Lanes, road edges, markings and crosswalks each carry their own codes."""
    map_ = Map(name="typed_map")
    map_.add_lane(_lane("A", [[0.0, 0.0], [9.0, 0.0], [18.0, 0.0], [27.0, 0.0]]))
    map_.add_roadline(
        RoadLine(
            "edge",
            LineString([(0.0, 5.0), (9.0, 5.0), (18.0, 5.0), (27.0, 5.0)]),
            type_="road_border",
        )
    )
    map_.add_roadline(
        RoadLine(
            "marking",
            LineString([(0.0, 10.0), (9.0, 10.0), (18.0, 10.0), (27.0, 10.0)]),
            type_="line_thin",
            subtype="solid",
            color="white",
        )
    )
    map_.add_area(
        Area(
            "crosswalk",
            Polygon([(0.0, 20.0), (4.0, 20.0), (4.0, 24.0), (0.0, 24.0)]),
            subtype="crosswalk",
        )
    )
    tokens = MapTokenizer(SmartConfig()).build(map_, 1100)
    groups: dict = {}
    for point_type, polygon_type, polyline in zip(
        tokens.pt_type.tolist(), tokens.pl_type.tolist(), tokens.token2pl[1].tolist()
    ):
        groups.setdefault(polyline, (point_type, polygon_type))
    # Sources are walked lanes first, then road edges, then markings, then
    # crosswalks, so the polyline indices come out in that order.
    assert groups == {
        0: (PointType.CENTERLINE, PolygonType.VEHICLE),
        1: (PointType.EDGE, PolygonType.VEHICLE),
        2: (PointType.SOLID_WHITE, PolygonType.VEHICLE),
        3: (PointType.CROSSWALK, PolygonType.PEDESTRIAN),
    }


@pytest.mark.integration
def test_map_tokenizer_skips_segments_it_cannot_tokenize():
    """A two-vertex lane and an unknown road line type produce no tokens."""
    map_ = Map(name="too_short")
    map_.add_lane(_lane("A", [[0.0, 0.0], [30.0, 0.0]]))
    map_.add_roadline(RoadLine("virtual", LineString([(0.0, 9.0), (30.0, 9.0)]), type_="virtual"))
    tokens = MapTokenizer(SmartConfig()).build(map_, 1100)
    assert tokens.num_tokens == 0
    assert tokens.traj_pos.shape == (0, 3, 2)


@pytest.mark.integration
def test_map_tokenizer_reads_the_traffic_light_state_from_the_map():
    """The light state comes through the native query, matched on the exact frame."""
    map_ = Map(name="signalled_map")
    map_.add_lane(_lane("A", [[0.0, 0.0], [9.0, 0.0], [18.0, 0.0], [27.0, 0.0]]))
    map_.add_regulatory(
        Regulatory(
            "light",
            subtype="traffic_light",
            custom_tags={"lane_id": "A", "states": [{"time_ms": 1100, "state": "go"}]},
        )
    )
    tokens = MapTokenizer(SmartConfig()).build(map_, 1100)
    assert set(tokens.light_type.tolist()) == {LightType.GO}
    # One hundred milliseconds on, the record is one frame away but not at the
    # frame, and upstream reads a lamp as unknown unless its timestamp matches.
    later = MapTokenizer(SmartConfig()).build(map_, 1200)
    assert set(later.light_type.tolist()) == {LightType.UNKNOWN}


@pytest.mark.integration
def test_light_type_table_maps_every_parser_state():
    """The nine parser states map onto SMART's four light indices."""
    assert _LIGHT_TYPE_BY_STATE["stop"] == LightType.STOP
    assert _LIGHT_TYPE_BY_STATE["arrow_stop"] == LightType.STOP
    assert _LIGHT_TYPE_BY_STATE["flashing_stop"] == LightType.STOP
    assert _LIGHT_TYPE_BY_STATE["go"] == LightType.GO
    assert _LIGHT_TYPE_BY_STATE["arrow_go"] == LightType.GO
    assert _LIGHT_TYPE_BY_STATE["caution"] == 2
    assert _LIGHT_TYPE_BY_STATE["arrow_caution"] == 2
    assert _LIGHT_TYPE_BY_STATE["flashing_caution"] == 2
    assert _LIGHT_TYPE_BY_STATE["unknown"] == LightType.UNKNOWN
    assert _LIGHT_TYPE_BY_STATE.get("not_a_state", LightType.UNKNOWN) == LightType.UNKNOWN


# ------------------------------------------------------------------
# Motion tokenizer
# ------------------------------------------------------------------


def _history_arrays(frames, base_x=1.0):
    positions = np.zeros((1, frames, 2), dtype=float)
    positions[0, :, 0] = base_x + np.arange(frames) * 0.5
    return positions


@pytest.mark.integration
def test_motion_tokenizer_slot_arithmetic():
    """A history becomes two slots and a full scenario eighteen."""
    tokenizer = MotionTokenizer(SmartConfig())
    for frames, slots in ((11, 2), (91, 18)):
        positions = _history_arrays(frames)
        headings = np.zeros((1, frames), dtype=float)
        valid = np.ones((1, frames), dtype=bool)
        velocity = np.zeros((1, frames, 2), dtype=float)
        velocity[:, :, 0] = 5.0
        tokens = tokenizer.tokenize(positions, headings, valid, velocity, np.array([0]))
        assert tokens.token_idx.shape == (1, slots)
        assert tokens.agent_valid_mask.shape == (1, slots)
        assert bool(tokens.agent_valid_mask[:, :2].all())


@pytest.mark.integration
def test_motion_tokenizer_needs_a_position_to_keep_the_first_slot():
    """The first slot survives frame 0 only when the agent has a nonzero x there.

    Upstream restores frame 0 for an agent it thinks is present, and it decides
    presence by ``position[:, 0, 0] != 0`` -- so a track whose first x is exactly
    zero loses its oldest slot. The sentinel is a quirk of the checkpoint's input
    distribution, not a reading of the data, and the port reproduces it.
    """
    tokenizer = MotionTokenizer(SmartConfig())
    headings = np.zeros((1, 11), dtype=float)
    valid = np.ones((1, 11), dtype=bool)
    velocity = np.zeros((1, 11, 2), dtype=float)
    velocity[:, :, 0] = 5.0

    present = tokenizer.tokenize(
        _history_arrays(11, base_x=1.0), headings, valid, velocity, np.array([0])
    )
    assert present.agent_valid_mask[0].tolist() == [True, True]

    at_origin = tokenizer.tokenize(
        _history_arrays(11, base_x=0.0), headings, valid, velocity, np.array([0])
    )
    assert at_origin.agent_valid_mask[0].tolist() == [False, True]


@pytest.mark.integration
def test_motion_tokenizer_keeps_the_newest_slot_across_a_gap():
    """Only a slot's first and last frame decide it, so a middle gap survives.

    Upstream reduces a slot to the conjunction of its two ends rather than all
    of its frames, which is what lets a track with a dropout in the middle of
    its history still drive the rollout.
    """
    tokenizer = MotionTokenizer(SmartConfig())
    positions = _history_arrays(11)
    headings = np.zeros((1, 11), dtype=float)
    velocity = np.zeros((1, 11, 2), dtype=float)
    velocity[:, :, 0] = 5.0

    gap = np.ones((1, 11), dtype=bool)
    gap[0, 7] = False
    tokens = tokenizer.tokenize(positions, headings, gap, velocity, np.array([0]))
    assert tokens.agent_valid_mask[0].tolist() == [True, True]

    # Frame 5 is the last frame of the first slot and the first of the second,
    # so losing it drops the older slot; the newer one is restored because the
    # agent is present at the newest frame and absent five frames before it.
    shared = np.ones((1, 11), dtype=bool)
    shared[0, 5] = False
    tokens = tokenizer.tokenize(positions, headings, shared, velocity, np.array([0]))
    assert tokens.agent_valid_mask[0].tolist() == [False, True]


# ------------------------------------------------------------------
# Batch builder
# ------------------------------------------------------------------


@pytest.mark.integration
def test_batch_builder_selects_a_deterministic_agent_set():
    """The modelled set is the nearest agents inside the radius, twice over."""
    config = SmartConfig(predicted_radius=40.0, max_predicted_agents=3)
    builder = SmartBatchBuilder(config)
    participants = _history_participants(count=8, spacing=8.0)
    frame = HISTORY_END_MS

    first = builder.select_agent_ids(participants, frame, center_id=0)
    second = builder.select_agent_ids(participants, frame, center_id=0)
    assert first == second
    # Vehicles sit at y = 0, 8, 16, ...; only those within 40 m are candidates,
    # and the cap keeps the three nearest.
    assert first == [0, 1, 2]


@pytest.mark.integration
def test_batch_builder_requires_an_active_centre():
    """A centre that is not active at the frame is refused by name."""
    builder = SmartBatchBuilder(SmartConfig())
    participants = _history_participants(count=2)
    with pytest.raises(ValueError, match="not a modelled participant active at frame"):
        builder.select_agent_ids(participants, HISTORY_END_MS, center_id=99)


@pytest.mark.integration
def test_batch_builder_anchors_on_the_newest_observed_frame():
    """A frame between samples snaps back to a frame the scenario was seen at."""
    builder = SmartBatchBuilder(SmartConfig())
    participants = _history_participants(count=3)
    batch = builder.build(participants, _straight_map(), HISTORY_END_MS + 50, center_id=0)
    assert batch.frame_ms == HISTORY_END_MS
    assert len(batch.agents.agent_ids) == 3
    assert batch.agents.num_slots == 2
    assert bool(batch.agents.eval_mask.all())


@pytest.mark.integration
def test_batch_builder_uses_the_scenario_frame_grid_for_the_history():
    """A drifting timestamp does not empty the history window."""
    builder = SmartBatchBuilder(SmartConfig())
    # Real WOMD logs are only nominally 10 Hz: index 11 can land at 1152 ms.
    frames = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1152]
    participants = _history_participants(count=2, frames=frames)
    scene_frames = builder.history_frames(1152, frames)
    # The window is the last eleven frames the scenario was seen at, so it ends
    # at the anchor rather than on a multiple of the step.
    assert scene_frames == frames[1:]
    lattice = [1152 - 100 * (10 - index) for index in range(11)]
    assert scene_frames != lattice
    batch = builder.build(participants, _straight_map(), 1152, center_id=0)
    assert batch.frame_ms == 1152
    assert bool(batch.agents.agent_valid_mask.all())


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


def _smart_model(**kwargs):
    config = SmartConfig(**kwargs)
    policy = _EchoPolicy()
    return SmartBehaviorModel(config=config, policy=policy), policy


@pytest.mark.integration
def test_smart_model_is_a_behavior_model():
    """SmartBehaviorModel implements the shared BehaviorModelBase interface."""
    model, _ = _smart_model()
    assert isinstance(model, BehaviorModelBase)


@pytest.mark.integration
def test_smart_model_requires_a_policy():
    """Construction without a policy raises ValueError."""
    with pytest.raises(ValueError, match="A policy is required"):
        SmartBehaviorModel(SmartConfig())


@pytest.mark.integration
def test_predict_returns_a_world_frame_trajectory():
    """The prediction is a 10 Hz trajectory on the lattice after the anchor."""
    model, _ = _smart_model(vehicle_only=False)
    participants = _history_participants(count=3)
    result = model.predict(participants, _straight_map(), HISTORY_END_MS, agent_ids=[0])
    assert set(result) == {0}
    trajectory = result[0]
    assert trajectory.fps == 10.0
    frames = list(trajectory.frames)
    assert frames == [HISTORY_END_MS + 100 * (step + 1) for step in range(80)]
    assert frames == sorted(frames)
    states = [trajectory.get_state(frame) for frame in frames]
    assert all(np.isfinite([state.x, state.y, state.heading]).all() for state in states)


@pytest.mark.integration
def test_predict_filters_to_the_requested_agents_but_decodes_jointly():
    """Asking for one agent still decodes the whole modelled scene once."""
    model, policy = _smart_model(vehicle_only=False)
    participants = _history_participants(count=4)
    result = model.predict(participants, _straight_map(), HISTORY_END_MS, agent_ids=[0])
    assert set(result) == {0}
    assert policy.decoded == [4]


@pytest.mark.integration
def test_predict_returns_every_modelled_agent_when_asked_for_none():
    """A None agent list returns the whole modelled set."""
    model, _ = _smart_model(vehicle_only=False)
    participants = _history_participants(count=4)
    scene = model.predict_scene(participants, _straight_map(), HISTORY_END_MS)
    result = model.predict(participants, _straight_map(), HISTORY_END_MS)
    assert set(result) == set(scene.agent_ids)
    assert len(result) == 4


@pytest.mark.integration
def test_predict_scene_carries_the_scenario_frame_grid():
    """Predicted frames follow the scenario's own timestamps when it has them."""
    model, _ = _smart_model(vehicle_only=False)
    frames = list(range(0, 1100, 100)) + [1100 + 100 * step for step in range(1, 81)]
    participants = _history_participants(count=2, frames=frames)
    scene = model.predict_scene(participants, _straight_map(), HISTORY_END_MS)
    assert scene.frames == frames[11:91]


@pytest.mark.integration
def test_predict_scene_needs_a_map():
    """A missing map is refused rather than silently tokenized as empty."""
    model, _ = _smart_model(vehicle_only=False)
    with pytest.raises(ValueError, match="needs a map"):
        model.predict_scene(_history_participants(count=2), None, HISTORY_END_MS)


@pytest.mark.integration
def test_predict_is_reproducible_with_a_seed():
    """The same model and the same scene give bit-identical rollouts."""
    from tactics2d.behavior.smart.model import SmartTorchModel, TorchSmartPolicy

    config = SmartConfig(seed=0, vehicle_only=False)
    module = SmartTorchModel(config)
    module.eval()
    policy = TorchSmartPolicy(module, device="cpu", seed=config.seed)
    model = SmartBehaviorModel(config=config, policy=policy)
    participants = _history_participants(count=3)
    first = model.predict_scene(participants, _straight_map(), HISTORY_END_MS)
    second = model.predict_scene(participants, _straight_map(), HISTORY_END_MS)
    assert np.array_equal(first.positions, second.positions)
    assert list(first.agent_ids) == list(second.agent_ids)


@pytest.mark.integration
def test_state_dict_from_checkpoint_unwraps_the_lightning_envelope():
    """The Lightning envelope is unwrapped; a payload with no weights is refused."""
    from tactics2d.behavior.smart.model import _state_dict_from_checkpoint

    payload = {"state_dict": {"encoder.map_encoder.token_emb.weight": None}}
    assert _state_dict_from_checkpoint(payload) == payload["state_dict"]
    with pytest.raises(ValueError, match="no state dict"):
        _state_dict_from_checkpoint({"epoch": 0})


@pytest.mark.integration
def test_encoder_state_dict_requires_the_lightning_prefix():
    """A key outside ``encoder.`` is refused rather than silently dropped."""
    from tactics2d.behavior.smart.model import SmartTorchModel

    with pytest.raises(ValueError, match="does not start with"):
        SmartTorchModel.encoder_state_dict({"map_encoder.token_emb.weight": None})


@pytest.mark.integration
def test_from_checkpoint_loads_onto_the_device_it_was_asked_for(monkeypatch):
    """A checkpoint trained on a GPU is still loaded onto the requested device.

    A checkpoint stores its tensors' device. Without a ``map_location``,
    ``torch.load`` puts them back on that GPU, so a CPU replay takes VRAM it
    never asked for -- which on a machine with a training job running takes it
    from the trainer.
    """
    import torch

    seen = {}

    def fake_load(path, map_location=None):
        seen["map_location"] = map_location
        raise FileNotFoundError(path)

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(FileNotFoundError):
        SmartBehaviorModel.from_checkpoint("missing.ckpt", device="cpu")
    assert seen["map_location"] == "cpu"
    with pytest.raises(FileNotFoundError):
        SmartBehaviorModel.from_checkpoint("missing.ckpt")
    assert seen["map_location"] == "cpu"
    with pytest.raises(FileNotFoundError):
        SmartBehaviorModel.from_checkpoint("missing.ckpt", device="cuda:0")
    assert seen["map_location"] == "cuda:0"


# ------------------------------------------------------------------
# Closed loop
# ------------------------------------------------------------------


@pytest.mark.slow
def test_from_checkpoint_loads_a_self_trained_checkpoint():
    """A self-trained SMART checkpoint loads strictly, or the test is skipped."""
    import os

    # Upstream releases no Waymo-trained weights (WOMD terms), so the only
    # checkpoint available here is one trained from the released code. This is
    # the end of epoch 0 of the full-data run -- the same file upstream scored
    # 0.7268 realism_meta on and the one the closed-loop eval used.
    path = "/home/cyberc3/smart_work/ckpt_full1000/epoch=00.ckpt"
    if not os.path.exists(path):
        pytest.skip("no self-trained SMART checkpoint on this machine.")
    model = SmartBehaviorModel.from_checkpoint(path, device="cpu")
    assert isinstance(model, BehaviorModelBase)
    assert model.policy.seed == model.config.seed


@pytest.mark.integration
def test_run_closed_loop_commits_every_modelled_agent():
    """One joint pass per planning step commits the whole modelled set."""
    model, _ = _smart_model(vehicle_only=False)
    # A closed loop needs ground truth for the whole span, not just the warmup:
    # the model overwrites the future, but it can only start from frames the
    # scenario actually occupies.
    participants = _history_participants(count=3, frames=range(0, 4100, 100))
    result = model.run_closed_loop(
        participants,
        _straight_map(),
        ego_id=0,
        frame_ms0=0,
        warmup_steps=11,
        planning_interval=10,
        scenario_steps=41,
    )
    assert result.ego_id == 0
    assert result.total_agents_controlled == len(result.modelled_ids)
    assert len(result.modelled_ids) == 3
    for agent_id, poses in result.poses.items():
        assert poses.shape == (41, 4)
    assert not result.collided
    assert result.progress > 0.0


@pytest.mark.integration
def test_run_closed_loop_refuses_an_unknown_ego():
    """An ego outside the scenario is reported by name."""
    model, _ = _smart_model(vehicle_only=False)
    with pytest.raises(KeyError, match="not a participant"):
        model.run_closed_loop(
            _history_participants(count=2), _straight_map(), ego_id=99, scenario_steps=41
        )


@pytest.mark.integration
def test_collision_classification_reads_the_relative_heading():
    """Overlapping bodies are classified front, side or rear by heading."""
    from tactics2d.behavior.smart.rolling import collision_kind

    dims = {0: (4.8, 1.9), 1: (4.8, 1.9)}
    poses = {0: np.array([[0.0, 0.0, 0.0, 0.0]], dtype=float), 1: None}
    poses[1] = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=float)
    assert collision_kind(poses, dims, 0, 0) == 2
    poses[1] = np.array([[0.0, 0.0, 0.0, np.pi]], dtype=float)
    assert collision_kind(poses, dims, 0, 0) == 1
    poses[1] = np.array([[0.0, 0.0, 0.0, np.pi / 2.0]], dtype=float)
    assert collision_kind(poses, dims, 0, 0) == 0
    poses[1] = np.array([[100.0, 100.0, 0.0, 0.0]], dtype=float)
    assert collision_kind(poses, dims, 0, 0) is None


@pytest.mark.slow
def test_primitives_radius_graph_matches_torch_cluster():
    """The pure-torch radius graph agrees with torch_cluster when it is present."""
    torch_cluster = pytest.importorskip("torch_cluster")
    from tactics2d.behavior.smart.primitives import radius_graph

    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(0)
    points = torch.rand((40, 2), generator=generator) * 20.0
    expected = torch_cluster.radius_graph(points, r=5.0, loop=False)
    actual = radius_graph(points, r=5.0, loop=False)
    expected_set = {tuple(edge) for edge in expected.t().tolist()}
    actual_set = {tuple(edge) for edge in actual.t().tolist()}
    assert expected_set == actual_set
