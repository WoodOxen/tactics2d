# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the planner-facing semantic map queries."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from tactics2d.map.element import Area, Junction, Lane, Map, Regulatory, RoadLine
from tactics2d.map.query import SemanticMapQuery, StopTarget

HALF_WIDTH = 1.75


def _lane(lane_id, x, y_start, y_end, **kwargs):
    """Build a lane along +y at ``x`` with its centerline in ``custom_tags``."""
    tags = {"centerline": np.array([[x, y_start], [x, y_end]], dtype=float)}
    tags.update(kwargs.pop("custom_tags", {}))
    return Lane(
        id_=lane_id,
        left_side=LineString([(x + HALF_WIDTH, y_start), (x + HALF_WIDTH, y_end)]),
        right_side=LineString([(x - HALF_WIDTH, y_start), (x - HALF_WIDTH, y_end)]),
        custom_tags=tags,
        **kwargs,
    )


def _lane_x(lane_id, y, x_start, x_end, half_width=HALF_WIDTH):
    """Build a lane along +x at ``y``."""
    return Lane(
        id_=lane_id,
        left_side=LineString([(x_start, y + half_width), (x_end, y + half_width)]),
        right_side=LineString([(x_start, y - half_width), (x_end, y - half_width)]),
        custom_tags={"centerline": np.array([[x_start, y], [x_end, y]], dtype=float)},
    )


def _corridor_map():
    """Lane A feeding into lane B, both running along +y at x = 0."""
    map_ = Map(name="corridor")
    map_.add_lane(_lane("A", 0.0, 0.0, 50.0))
    map_.add_lane(_lane("B", 0.0, 50.0, 100.0))
    map_.lanes["A"].successors = {"B"}
    map_.lanes["B"].predecessors = {"A"}
    return map_


def _cross_map():
    """An eastbound lane and a northbound lane crossing at (50, 0)."""
    map_ = Map(name="cross")
    map_.add_lane(_lane_x("E", 0.0, 0.0, 100.0))
    map_.add_lane(_lane("N", 50.0, -50.0, 50.0))
    return map_


def _traffic_light():
    """A traffic light bound to lane A with a red and a green state."""
    return Regulatory(
        id_="tl1",
        subtype="traffic_light",
        position=(0.0, 40.0),
        custom_tags={
            "lane_id": "A",
            "states": [
                {"time_ms": 1000, "state": "red", "stop_point": (0.0, 42.0)},
                {"time_ms": 5000, "state": "green"},
            ],
        },
    )


def _map_with_regulations():
    """The corridor map plus a traffic light and a stop sign on lane A."""
    map_ = _corridor_map()
    map_.add_regulatory(_traffic_light())
    map_.add_regulatory(
        Regulatory(
            id_="ss1", subtype="stop_sign", position=(0.0, 30.0), custom_tags={"lane_id": "A"}
        )
    )
    return map_


def _junction(lane_ids, junction_id="J"):
    """Build a junction that names ``lane_ids`` through its custom tags."""
    return Junction(id_=junction_id, custom_tags={"lane_ids": lane_ids})


@pytest.mark.map_element
def test_get_reference_path_returns_none_for_an_unknown_lane():
    """Only lanes present in the map can start a reference path."""
    query = SemanticMapQuery(_corridor_map())

    assert query.get_reference_path("ghost") is None


@pytest.mark.map_element
def test_get_reference_path_concatenates_the_lookahead_route():
    """The path follows successors and spans their joint centerline."""
    query = SemanticMapQuery(_corridor_map())

    path = query.get_reference_path("A", lookahead_lanes=3)

    assert path.lane_ids == ("A", "B")
    assert path.path.length == pytest.approx(100.0)
    assert path.lane_width == pytest.approx(2 * HALF_WIDTH)
    assert path.path.distance(Point(0.0, 50.0)) == pytest.approx(0.0)


@pytest.mark.map_element
def test_get_reference_path_stops_at_a_cycle_and_at_the_lookahead_limit():
    """Cyclic successors and the lookahead count both end the walk."""
    map_ = _corridor_map()
    map_.lanes["B"].successors = {"A"}
    query = SemanticMapQuery(map_)

    assert query.get_reference_path("A").lane_ids == ("A", "B")
    assert query.get_reference_path("A", lookahead_lanes=1).lane_ids == ("A",)


@pytest.mark.map_element
def test_get_reference_path_stops_at_successors_outside_the_map():
    """A successor id that is not a lane ends the walk."""
    map_ = _corridor_map()
    map_.lanes["A"].successors = {"ghost"}
    query = SemanticMapQuery(map_)

    assert query.get_reference_path("A").lane_ids == ("A",)


@pytest.mark.map_element
def test_get_reference_path_uses_the_requested_route():
    """An explicit route is filtered to the lanes the map actually has."""
    query = SemanticMapQuery(_corridor_map())

    path = query.get_reference_path("A", route_lane_ids=["A", "ghost", "B"])

    assert path.lane_ids == ("A", "B")


@pytest.mark.map_element
def test_get_reference_path_joins_a_route_that_does_not_touch():
    """A route whose lanes do not share an endpoint keeps both centerlines."""
    map_ = _corridor_map()
    map_.add_lane(_lane("R", 0.0, 50.0, 100.0))
    reversed_lane = _lane("rev", 0.0, 100.0, 50.0)
    map_.add_lane(reversed_lane)
    query = SemanticMapQuery(map_)

    path = query.get_reference_path("A", route_lane_ids=["A", "rev"])

    assert path.path.length == pytest.approx(150.0)
    assert path.path.distance(Point(0.0, 100.0)) == pytest.approx(0.0)


@pytest.mark.map_element
def test_get_reference_path_rejects_a_route_without_usable_centerlines():
    """Lanes without a centerline leave nothing to build a path from."""
    map_ = _corridor_map()
    map_.add_lane(Lane(id_="collapsed", custom_tags={"centerline": np.zeros((2, 2))}))
    query = SemanticMapQuery(map_)

    assert query.get_reference_path("collapsed") is None
    assert query.get_reference_path("A", route_lane_ids=["collapsed"]) is None


@pytest.mark.map_element
def test_get_lanes_in_region_filters_by_geometry():
    """Lanes with geometry inside the region are listed; others are not."""
    map_ = _corridor_map()
    centerline_only = _lane("C", 0.0, 0.0, 50.0)
    centerline_only.left_side = None
    centerline_only.right_side = None
    centerline_only.geometry = None
    map_.add_lane(centerline_only)
    query = SemanticMapQuery(map_)

    # The region covers lane A and the centerline-only lane C, but not lane B.
    assert query.get_lanes_in_region((-3.0, 3.0, -1.0, 45.0)) == ["A"]
    assert query.get_lanes_in_region((10.0, 20.0, 10.0, 20.0)) == []
    assert query.get_lanes_in_region(box(-3.0, -1.0, 3.0, 45.0)) == ["A"]


@pytest.mark.map_element
def test_get_stop_targets_falls_back_to_the_static_position():
    """Without a red state the light reports its static position and state."""
    query = SemanticMapQuery(_map_with_regulations())

    targets = query.get_stop_targets("A", include_stop_signs=False)

    assert len(targets) == 1
    assert targets[0].point.distance(Point(0.0, 40.0)) == pytest.approx(0.0)
    assert targets[0].state == "green"


@pytest.mark.map_element
def test_get_stop_line_is_perpendicular_to_the_lane():
    """A stop line spans the lane width across the projected point."""
    query = SemanticMapQuery(_corridor_map())

    stop_line = query.get_stop_line("A", point=(0.0, 30.0))

    assert stop_line.source_id is None
    assert stop_line.reason is None
    assert stop_line.virtual is True
    coordinates = np.asarray(stop_line.geometry.coords)
    np.testing.assert_allclose(coordinates, [[HALF_WIDTH, 30.0], [-HALF_WIDTH, 30.0]], atol=1e-9)


@pytest.mark.map_element
def test_get_stop_line_accepts_an_explicit_width_and_target():
    """The width override and the stop target both reach the stop line."""
    query = SemanticMapQuery(_map_with_regulations())
    target = StopTarget(lane_id="A", point=Point(0.0, 25.0), reason="virtual", source_id="manual")

    stop_line = query.get_stop_line("A", stop_target=target, width=2.0)

    assert stop_line.source_id == "manual"
    assert stop_line.reason == "virtual"
    assert stop_line.geometry.length == pytest.approx(2.0)
    assert stop_line.geometry.distance(Point(0.0, 25.0)) == pytest.approx(0.0)


@pytest.mark.map_element
def test_get_stop_line_falls_back_to_the_mapped_stop_targets():
    """Without an explicit point the first stop target of the lane is used."""
    query = SemanticMapQuery(_map_with_regulations())

    stop_line = query.get_stop_line("A", width=4.0)

    assert stop_line.geometry.length == pytest.approx(4.0)
    assert stop_line.geometry.distance(Point(0.0, 30.0)) == pytest.approx(0.0)


@pytest.mark.map_element
def test_get_stop_line_returns_none_without_a_lane_or_a_target():
    """An unknown lane and a lane without regulations produce no stop line."""
    query = SemanticMapQuery(_corridor_map())

    assert query.get_stop_line("ghost") is None
    assert query.get_stop_line("A") is None
    assert query.get_stop_line_geometry("A") is None


@pytest.mark.map_element
def test_get_stop_line_rejects_a_lane_that_cannot_be_projected():
    """A lane whose centerline has no length has no stop line."""
    map_ = Map(name="collapsed")
    map_.add_lane(Lane(id_="Z", custom_tags={"centerline": np.zeros((2, 2))}))
    query = SemanticMapQuery(map_)

    assert query.get_stop_line("Z", point=(0.0, 0.0)) is None


@pytest.mark.map_element
def test_get_stop_line_uses_the_default_width_for_a_centerline_only_lane():
    """A lane without side boundaries falls back to the default width."""
    map_ = Map(name="centerline_only")
    map_.add_lane(
        Lane(id_="C", custom_tags={"centerline": np.array([[0.0, 0.0], [0.0, 50.0]], dtype=float)})
    )
    query = SemanticMapQuery(map_)

    geometry = query.get_stop_line_geometry("C", point=(0.0, 20.0))

    assert geometry.length == pytest.approx(3.6)


@pytest.mark.map_element
def test_get_lane_change_permission_needs_the_requested_neighbour():
    """A lane change is refused when the side has no neighbouring lane."""
    map_ = _corridor_map()
    query = SemanticMapQuery(map_)

    assert query.get_lane_change_permission("ghost", "left") is False
    assert query.get_lane_change_permission("A", "left") is False
    assert query.get_lane_change_permission("A", "right") is False


@pytest.mark.map_element
def test_get_lane_change_permission_rejects_an_unknown_direction():
    """Only the left and right direction spellings are accepted."""
    query = SemanticMapQuery(_corridor_map())

    with pytest.raises(ValueError, match="direction must be"):
        query.get_lane_change_permission("A", "up")


@pytest.mark.map_element
def test_get_lane_change_permission_ignores_missing_or_silent_roadlines():
    """A roadline outside the map or without a permission leaves the lane open."""
    map_ = _corridor_map()
    map_.lanes["A"].right_neighbors = {"B"}
    map_.lanes["A"].line_ids = {"right": ["ghost"]}
    query = SemanticMapQuery(map_)

    assert query.get_lane_change_permission("A", "right") is True

    map_.add_roadline(
        RoadLine(id_="r2", geometry=LineString([(0.0, 0.0), (0.0, 50.0)]), lane_change=None)
    )
    map_.lanes["A"].line_ids = {"right": ["r2"]}

    assert query.get_lane_change_permission("A", "right") is True


@pytest.mark.map_element
def test_get_lane_change_permission_uses_the_boundary_segment_at_s():
    """A position inside a boundary segment selects that segment's roadline."""
    map_ = _corridor_map()
    map_.lanes["A"].right_neighbors = {"B"}
    map_.lanes["A"].line_ids = {"right": ["outer"]}
    map_.lanes["A"].custom_tags["boundary_segments"] = {
        "right": [{"roadline_id": "closed", "start_s": 0.0, "end_s": 10.0}]
    }
    map_.add_roadline(
        RoadLine(
            id_="closed", geometry=LineString([(0.0, 0.0), (0.0, 10.0)]), lane_change=(False, True)
        )
    )
    map_.add_roadline(
        RoadLine(
            id_="outer", geometry=LineString([(0.0, 0.0), (0.0, 50.0)]), lane_change=(True, True)
        )
    )
    query = SemanticMapQuery(map_)

    assert query.get_lane_change_permission("A", "right", s=5.0) is False
    assert query.get_lane_change_permission("A", "right", s=20.0) is True


@pytest.mark.map_element
def test_is_lane_in_junction_uses_tags_and_subtype():
    """Junction lanes are recognised from tags, subtype or junction membership."""
    map_ = _corridor_map()
    map_.add_lane(_lane("T1", 0.0, 0.0, 50.0, custom_tags={"is_intersection": True}))
    map_.add_lane(_lane("T2", 0.0, 0.0, 50.0, custom_tags={"junction": True}))
    map_.add_lane(_lane("T3", 0.0, 0.0, 50.0, subtype="junction"))
    map_.add_lane(_lane("T4", 0.0, 0.0, 50.0, subtype="Intersection"))
    map_.add_junction(_junction(["A"]))
    query = SemanticMapQuery(map_)

    assert query.is_lane_in_junction("ghost") is False
    assert query.is_lane_in_junction("T1") is True
    assert query.is_lane_in_junction("T2") is True
    assert query.is_lane_in_junction("T3") is True
    assert query.is_lane_in_junction("T4") is True
    assert query.is_lane_in_junction("A") is True
    assert query.is_lane_in_junction("B") is False


@pytest.mark.map_element
def test_get_junction_by_lane_reads_tags_links_and_connections():
    """Every lane named by a junction or its connections resolves back to it."""
    map_ = _corridor_map()
    junction = Junction(
        id_="J",
        lane_links=[("A", "B")],
        connections={"c1": Junction(id_="c1", lane_links=[("C", "D")])},
        custom_tags={
            "lane_ids": ["L1"],
            "incoming_lanes": ["L2"],
            "inside_lanes": ["L3"],
            "outgoing_lanes": None,
        },
    )
    map_.add_junction(junction)
    query = SemanticMapQuery(map_)

    for lane_id in ("A", "B", "C", "D", "L1", "L2", "L3"):
        assert query.get_junction_by_lane(lane_id) is junction
    assert query.get_junction_by_lane("ghost") is None


@pytest.mark.map_element
def test_get_junction_area_prefers_the_junction_shape():
    """A junction carrying a shape reports it as a polygon."""
    map_ = _corridor_map()
    map_.add_junction(
        Junction(
            id_="J", custom_tags={"shape": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]}
        )
    )
    query = SemanticMapQuery(map_)

    area = query.get_junction_area(junction_id="J")

    assert area.area == pytest.approx(100.0)
    assert query.get_junction_area(junction_id="ghost") is None


@pytest.mark.map_element
def test_get_junction_area_falls_back_to_an_intersection_area():
    """Without a shape, an intersection-like area is reported instead."""
    map_ = _corridor_map()
    map_.add_junction(_junction(["A"]))
    map_.add_area(Area(id_="area1", geometry=box(-1.0, -1.0, 1.0, 51.0), subtype="intersection"))
    query = SemanticMapQuery(map_)

    assert query.get_junction_area(lane_id="A") is not None
    assert query.get_junction_area() is not None


@pytest.mark.map_element
def test_get_junction_area_requires_the_lane_to_touch_the_area():
    """When a lane is given, only areas overlapping its geometry qualify."""
    map_ = _corridor_map()
    map_.add_area(Area(id_="area1", geometry=box(20.0, 20.0, 30.0, 30.0), subtype="junction"))
    query = SemanticMapQuery(map_)

    assert query.get_junction_area(lane_id="A") is None
    assert query.get_junction_area(lane_id="ghost") is None
    assert query.get_junction_area() is not None


@pytest.mark.map_element
def test_has_conflict_detects_crossing_centerlines():
    """Lanes that cross at a wide angle conflict with each other."""
    query = SemanticMapQuery(_cross_map())

    assert query.has_conflict("E", "N") is True
    assert query.has_conflict("E", "ghost") is False
    assert [point.coords[0] for point in query.get_conflict_points("E", "N")] == [(50.0, 0.0)]
    assert query.get_conflict_points("E", "ghost") == []


@pytest.mark.map_element
def test_has_conflict_ignores_directly_connected_lanes():
    """Topology neighbours never conflict, however their centerlines cross."""
    map_ = _cross_map()
    query = SemanticMapQuery(map_)

    assert query.has_conflict("E", "N") is True

    map_.lanes["E"].left_neighbors = {"N"}

    assert query.has_conflict("E", "N") is False
    assert query.has_conflict("E", "N") is False


@pytest.mark.map_element
def test_has_conflict_ignores_a_shallow_crossing_angle():
    """A crossing below the angle threshold does not count as a conflict."""
    map_ = Map(name="shallow")
    map_.add_lane(_lane_x("E", 0.0, 0.0, 100.0))
    shallow = Lane(
        id_="S",
        left_side=LineString([(-HALF_WIDTH, -5.0), (100.0 - HALF_WIDTH, 5.0)]),
        right_side=LineString([(HALF_WIDTH, -5.0), (100.0 + HALF_WIDTH, 5.0)]),
        custom_tags={"centerline": np.array([[0.0, -5.0], [100.0, 5.0]])},
    )
    map_.add_lane(shallow)
    query = SemanticMapQuery(map_)

    assert query.get_conflict_points("E", "S")
    assert query.has_conflict("E", "S") is False
    assert query.has_conflict("E", "S", angle_threshold=np.deg2rad(2.0)) is True


@pytest.mark.map_element
def test_has_conflict_needs_overlapping_geometry_without_a_crossing():
    """Centerlines that miss each other only conflict when the lanes overlap."""
    map_ = Map(name="overlapping")
    map_.add_lane(_lane_x("E", 0.0, 0.0, 100.0))
    # A wide lane 6 m to the north whose right boundary reaches into lane E's band.
    map_.add_lane(_lane_x("Z", 6.0, 50.0, 150.0, half_width=8.0))
    map_.add_lane(_lane_x("far", 20.0, 50.0, 150.0, half_width=1.0))
    query = SemanticMapQuery(map_)

    assert query.get_conflict_points("E", "Z") == []
    assert query.has_conflict("E", "Z") is True
    assert query.has_conflict("E", "far") is False


@pytest.mark.map_element
def test_has_conflict_needs_a_centerline():
    """Lanes without any centerline cannot be compared."""
    map_ = _corridor_map()
    map_.add_lane(Lane(id_="N1"))
    map_.add_lane(Lane(id_="N2"))
    query = SemanticMapQuery(map_)

    assert query.has_conflict("N1", "N2") is False
    assert query.has_conflict("ghost", "N1") is False


@pytest.mark.map_element
def test_has_conflict_needs_lane_geometry_beyond_the_centerline():
    """Centerlines that miss each other with no lane geometry are not compared."""
    map_ = Map(name="centerlines_only")
    map_.add_lane(Lane(id_="P1", custom_tags={"centerline": np.array([[0.0, 0.0], [0.0, 50.0]])}))
    map_.add_lane(Lane(id_="P2", custom_tags={"centerline": np.array([[9.0, 0.0], [9.0, 50.0]])}))
    query = SemanticMapQuery(map_)

    assert query.has_conflict("P1", "P2") is False


@pytest.mark.map_element
def test_get_conflict_points_handles_overlapping_and_multiple_crossings():
    """Collinear overlap and a double crossing both reduce to representative points."""
    map_ = Map(name="overlaps")
    map_.add_lane(
        Lane(id_="long", custom_tags={"centerline": np.array([[0.0, 0.0], [100.0, 0.0]])})
    )
    map_.add_lane(
        Lane(id_="short", custom_tags={"centerline": np.array([[20.0, 0.0], [80.0, 0.0]])})
    )
    map_.add_lane(
        Lane(
            id_="vee",
            custom_tags={"centerline": np.array([[0.0, 5.0], [50.0, -5.0], [100.0, 5.0]])},
        )
    )
    map_.add_lane(
        Lane(
            id_="mixed",
            custom_tags={
                "centerline": np.array([[20.0, 0.0], [80.0, 0.0], [90.0, 10.0], [85.0, 0.0]])
            },
        )
    )
    query = SemanticMapQuery(map_)

    # The overlap is a two-point line, and its middle vertex is the second endpoint.
    overlap = query.get_conflict_points("long", "short")
    assert len(overlap) == 1
    assert tuple(overlap[0].coords[0]) == pytest.approx((80.0, 0.0))

    crossings = query.get_conflict_points("long", "vee")
    assert sorted(point.x for point in crossings) == pytest.approx([25.0, 75.0])

    mixed = query.get_conflict_points("long", "mixed")
    assert sorted(tuple(point.coords[0]) for point in mixed) == pytest.approx(
        [(80.0, 0.0), (85.0, 0.0)]
    )


@pytest.mark.map_element
def test_get_junction_conflicts_skips_itself_and_unknown_candidates():
    """Conflicts are reported once per other lane, never for the lane itself."""
    query = SemanticMapQuery(_cross_map())

    conflicts = query.get_junction_conflicts("E")

    assert [(conflict.lane_id, conflict.conflict_lane_id) for conflict in conflicts] == [("E", "N")]
    assert conflicts[0].points
    assert query.get_junction_conflicts("E", candidate_lane_ids=["E", "ghost"]) == []
    assert query.get_conflict_lanes("E") == ["N"]
    assert query.get_conflict_lanes("E", candidate_lane_ids=[]) == []


@pytest.mark.map_element
def test_points_from_geometry_ignores_unsupported_geometry():
    """An area-like geometry contributes no representative point."""
    query = SemanticMapQuery(_corridor_map())

    assert query._points_from_geometry(Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])) == []
