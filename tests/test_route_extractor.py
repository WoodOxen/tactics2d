# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the lane-route extractor."""

import numpy as np
import pytest
from shapely.geometry import LinearRing, LineString

from tactics2d.dataset_parser.route_extractor import (
    extract_all_lane_sequences,
    extract_lane_sequence,
    match_lane_for_state,
)
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

HALF_WIDTH = 1.75
UP = np.pi / 2


def _lane(lane_id, x, y_start, y_end):
    """Build a lane around ``x`` whose centerline runs from ``y_start`` to ``y_end``."""
    return Lane(
        id_=lane_id,
        left_side=LineString([(x + HALF_WIDTH, y_start), (x + HALF_WIDTH, y_end)]),
        right_side=LineString([(x - HALF_WIDTH, y_start), (x - HALF_WIDTH, y_end)]),
        custom_tags={"centerline": np.array([[x, y_start], [x, y_end]], dtype=float)},
    )


def _corridor_map():
    """Lane A feeding into lane B, both running along +y at x = 0."""
    map_ = Map(name="corridor")
    map_.add_lane(_lane("A", 0.0, 0.0, 50.0))
    map_.add_lane(_lane("B", 0.0, 50.0, 100.0))
    map_.lanes["A"].successors = {"B"}
    map_.lanes["B"].predecessors = {"A"}
    return map_


def _two_way_map():
    """Two lanes on the same centerline, driving in opposite directions."""
    map_ = Map(name="two_way")
    map_.add_lane(_lane("up", 0.0, 0.0, 50.0))
    reversed_lane = _lane("down", 0.0, 0.0, 50.0)
    reversed_lane.custom_tags = {"centerline": np.array([[0.0, 50.0], [0.0, 0.0]])}
    reversed_lane._centerline = {}
    map_.add_lane(reversed_lane)
    return map_


def _vehicle(agent_id, frames, x=0.0, y=0.0, heading=UP, speed=20.0):
    """Build a vehicle driving straight at ``speed`` over ``frames``.

    Frames are milliseconds, so frame ``100 * i`` sits ``speed * 0.1 * i`` metres along.
    """
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for frame in frames:
        travelled = speed * frame / 1000.0
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


def _corridor_vehicle(agent_id=0):
    """A vehicle crossing from lane A into lane B over 51 frames."""
    return _vehicle(agent_id, range(0, 5100, 100))


@pytest.mark.dataset_parser
def test_match_lane_for_state_needs_a_map_with_lanes():
    """An absent or empty map has nothing to match against."""
    assert match_lane_for_state(None, 0.0, 0.0, UP) is None
    assert match_lane_for_state(Map(name="empty"), 0.0, 0.0, UP) is None


@pytest.mark.dataset_parser
def test_match_lane_for_state_returns_the_lane_under_the_state():
    """A state on a lane matches that lane, not its neighbour."""
    map_ = _corridor_map()

    assert match_lane_for_state(map_, 0.0, 10.0, UP) == "A"
    assert match_lane_for_state(map_, 0.0, 60.0, UP) == "B"


@pytest.mark.dataset_parser
def test_match_lane_for_state_returns_none_beyond_the_match_radius():
    """A state too far from every lane is left unmatched."""
    map_ = _corridor_map()

    assert match_lane_for_state(map_, 8.0, 10.0, UP) is None
    assert match_lane_for_state(map_, 8.0, 10.0, UP, lane_match_radius=10.0) == "A"


@pytest.mark.dataset_parser
def test_match_lane_for_state_prefers_the_heading_consistent_lane():
    """When two lanes coincide, the one pointing the right way wins."""
    map_ = _two_way_map()

    assert match_lane_for_state(map_, 0.0, 10.0, UP) == "up"
    assert match_lane_for_state(map_, 0.0, 10.0, -UP) == "down"


@pytest.mark.dataset_parser
def test_match_lane_for_state_accepts_a_lane_with_only_a_centerline():
    """A lane carrying no boundary geometry is still matchable."""
    map_ = Map(name="centerline_only")
    map_.add_lane(
        Lane(id_="C", custom_tags={"centerline": np.array([[0.0, 0.0], [0.0, 50.0]], dtype=float)})
    )

    assert map_.lanes["C"].geometry is None
    assert match_lane_for_state(map_, 0.0, 10.0, UP) == "C"


@pytest.mark.dataset_parser
def test_match_lane_for_state_defaults_the_heading_error_without_a_heading():
    """A lane whose centerline has no direction is scored on distance alone."""
    map_ = Map(name="degenerate")
    lane = _lane("Z", 0.0, 0.0, 50.0)
    lane.custom_tags = {"centerline": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)}
    lane._centerline = {}
    map_.add_lane(lane)

    assert match_lane_for_state(map_, 0.0, 0.0, UP) == "Z"


@pytest.mark.dataset_parser
def test_match_lane_for_state_scores_a_boundary_only_lane_by_distance():
    """A lane given as a boundary ring has no centerline but is still matchable."""
    map_ = Map(name="boundary_only")
    map_.add_lane(
        Lane(
            id_="R",
            geometry=LinearRing(
                [(HALF_WIDTH, 0.0), (HALF_WIDTH, 50.0), (-HALF_WIDTH, 50.0), (-HALF_WIDTH, 0.0)]
            ),
        )
    )

    assert map_.lanes["R"].centerline() is None
    assert match_lane_for_state(map_, 0.0, 10.0, UP) == "R"


@pytest.mark.dataset_parser
def test_match_lane_for_state_ignores_lanes_without_any_geometry():
    """Lanes with neither a centerline nor a boundary are passed over."""
    map_ = _corridor_map()
    map_.add_lane(Lane(id_="empty"))

    assert match_lane_for_state(map_, 0.0, 10.0, UP) == "A"

    only_empty = Map(name="empty_only")
    only_empty.add_lane(Lane(id_="empty"))
    assert match_lane_for_state(only_empty, 0.0, 10.0, UP) is None


@pytest.mark.dataset_parser
def test_match_lane_for_state_ignores_unknown_neighbour_ids():
    """Topology entries pointing at lanes outside the map are skipped."""
    map_ = _corridor_map()
    map_.lanes["A"].successors = {"ghost"}

    assert match_lane_for_state(map_, 0.0, 10.0, UP) == "A"


@pytest.mark.dataset_parser
def test_extract_lane_sequence_records_lane_transitions():
    """A run through two lanes yields both ids, in order, without repeats."""
    sequence = extract_lane_sequence(_corridor_vehicle(), _corridor_map())

    assert sequence == ["A", "B"]


@pytest.mark.dataset_parser
def test_extract_lane_sequence_drops_lanes_held_too_briefly():
    """The dwell threshold filters out lanes the vehicle only clips."""
    vehicle = _corridor_vehicle()

    # Lane A is held for 26 frames and lane B for 25.
    assert extract_lane_sequence(vehicle, _corridor_map(), min_dwell_frames=30) == []
    assert extract_lane_sequence(vehicle, _corridor_map(), min_dwell_frames=1000) == []
    assert extract_lane_sequence(vehicle, _corridor_map(), min_dwell_frames=25) == ["A", "B"]


@pytest.mark.dataset_parser
def test_extract_lane_sequence_honours_the_frame_window():
    """The start and end frames bound the part of the trajectory that is projected."""
    vehicle = _corridor_vehicle()
    map_ = _corridor_map()

    assert extract_lane_sequence(vehicle, map_, start_frame=3000) == ["B"]
    assert extract_lane_sequence(vehicle, map_, end_frame=2400) == ["A"]
    assert extract_lane_sequence(vehicle, map_, end_frame=0) == []


@pytest.mark.dataset_parser
def test_extract_all_lane_sequences_trims_to_the_reference_frame():
    """The route keeps only the lanes at and after the reference frame."""
    participants = {0: _corridor_vehicle()}
    map_ = _corridor_map()

    assert extract_all_lane_sequences(participants, map_, 0) == {0: ("A", "B")}
    assert extract_all_lane_sequences(participants, map_, 4000) == {0: ("B",)}
    assert extract_all_lane_sequences(participants, map_, 4000, agent_ids=[]) == {}


@pytest.mark.dataset_parser
def test_extract_all_lane_sequences_skips_non_vehicles_and_unroutable_vehicles():
    """Non-vehicles, absent frames and off-map vehicles get no route entry."""
    map_ = _corridor_map()
    absent = _vehicle(2, range(600, 1600, 100))
    off_map = _vehicle(3, range(0, 5100, 100), x=500.0)
    participants = {0: _corridor_vehicle(), 1: object(), 2: absent, 3: off_map}

    routes = extract_all_lane_sequences(participants, map_, 4000)

    assert set(routes) == {0}
    assert routes[0] == ("B",)


@pytest.mark.dataset_parser
def test_extract_all_lane_sequences_falls_back_without_a_takeover_lane():
    """A vehicle that has left the matched lanes keeps its full sequence."""
    participants = {0: _corridor_vehicle()}
    map_ = _corridor_map()
    # Moving lane B away leaves frame 3000 (y = 60 m) unmatched, so there is no lane to trim to.
    map_.lanes["B"].custom_tags = {"centerline": np.array([[9.0, 50.0], [9.0, 100.0]])}
    map_.lanes["B"].left_side = LineString([(10.75, 50.0), (10.75, 100.0)])
    map_.lanes["B"].right_side = LineString([(7.25, 50.0), (7.25, 100.0)])
    map_.lanes["B"]._centerline = {}

    assert match_lane_for_state(map_, 0.0, 60.0, UP) is None
    assert extract_all_lane_sequences(participants, map_, 3000) == {0: ("A",)}
