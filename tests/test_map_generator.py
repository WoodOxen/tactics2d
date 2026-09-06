# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map generator."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tactics2d.display.renderers import MatplotlibRenderer
from tactics2d.display.sensor import BEVCamera
from tactics2d.geometry import heading_unit
from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.map.generator.road_segment import (
    EntranceRamp,
    ExitRamp,
    Fork,
    Intersection,
    LaneAdapter,
    Merge,
    OneWay,
    Roundabout,
    TwoWay,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort
from tactics2d.participant.trajectory import State

_RENDER_MAPS = os.environ.get("TACTICS2D_RENDER_MAPS", "0") == "1"


def _render(map_: Map, save_to: os.PathLike) -> None:
    """Render a map to a png file (only when TACTICS2D_RENDER_MAPS=1)."""
    if not _RENDER_MAPS:
        return
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
    renderer = MatplotlibRenderer(
        resolution=(800, 800), xlim=(boundary[0], boundary[1]), ylim=(boundary[2], boundary[3])
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=save_to)
    renderer.destroy()


def _assert_result_element_ids_unique(result: RoadModuleResult) -> None:
    """Assert that one module result has no duplicate element ids."""
    elements = [*result.lanes, *result.roadlines, *result.junctions, *result.areas]
    ids = [element.id_ for element in elements]
    assert len(ids) == len(set(ids))


def _assert_lane_line_ids_resolved(result: RoadModuleResult) -> None:
    """Assert that lane line_ids reference existing RoadLine ids."""
    roadline_ids = {rl.id_ for rl in result.roadlines}
    for lane in result.lanes:
        assert "left" in lane.line_ids
        assert "right" in lane.line_ids
        for side in ("left", "right"):
            for line_id in lane.line_ids.get(side, []):
                assert line_id in roadline_ids


def _add_result(map_: Map, result: RoadModuleResult) -> None:
    """Add a RoadModuleResult into a Map, asserting structural integrity first."""
    _assert_result_element_ids_unique(result)
    _assert_lane_line_ids_resolved(result)
    for lane in result.lanes:
        map_.add_lane(lane)
    for roadline in result.roadlines:
        map_.add_roadline(roadline)
    for junction in result.junctions:
        map_.add_junction(junction)
    for area in result.areas:
        map_.add_area(area)


def _make_port(
    x: float,
    y: float,
    heading: float,
    lane_num: int,
    lane_width: float = 3.5,
    speed_limit: float = 50.0,
) -> RoadPort:
    return RoadPort(
        point=np.array([x, y], dtype=float),
        heading=heading,
        lane_num=lane_num,
        lane_width=lane_width,
        speed_limit=speed_limit,
    )


def _port_from_start_length(
    start: np.ndarray,
    heading: float,
    length: float,
    lane_num: int,
    lane_width: float = 3.5,
    speed_limit: float = 50.0,
) -> tuple[RoadPort, RoadPort]:
    """Create two collinear ports separated by length."""
    start_port = RoadPort(
        point=start,
        heading=heading,
        lane_num=lane_num,
        lane_width=lane_width,
        speed_limit=speed_limit,
    )
    end_port = RoadPort(
        point=start + length * heading_unit(heading),
        heading=heading,
        lane_num=lane_num,
        lane_width=lane_width,
        speed_limit=speed_limit,
    )
    return start_port, end_port


def _two_way_from_port(
    port: RoadPort, *, length: float, id_offset: int, backward_lane_num: int | None = None
) -> RoadModuleResult:
    """Attach a two-way road to an intersection/roundabout outward port."""
    end_port = RoadPort(
        point=port.point + length * heading_unit(port.heading),
        heading=port.heading,
        lane_num=port.lane_num,
        lane_width=port.lane_width,
        speed_limit=port.speed_limit,
    )
    return TwoWay().build(
        port,
        end_port,
        forward_lane_num=port.lane_num,
        backward_lane_num=port.lane_num if backward_lane_num is None else backward_lane_num,
        lane_width=port.lane_width,
        speed_limit=port.speed_limit,
        id_offset=id_offset,
    )


def _attach_outgoing_two_way_roads(
    map_: Map, result: RoadModuleResult, length: float = 30.0
) -> None:
    """Attach a short two-way road to every outward port of a junction module."""
    id_offset = result.id_counter + 1000
    for key, port in sorted(result.ports.items()):
        if key.endswith("_out"):
            parts = key.split("_")
            sibling = f"arm_{parts[1]}_in" if len(parts) == 3 and parts[0] == "arm" else None
            backward_lane_num = result.ports[sibling].lane_num if sibling in result.ports else None
            road_result = _two_way_from_port(
                port, length=length, id_offset=id_offset, backward_lane_num=backward_lane_num
            )
            id_offset = road_result.id_counter
            _add_result(map_, road_result)


def _assert_ramp_ports(result: RoadModuleResult) -> None:
    """Assert that ramp result exposes the required port interface."""
    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "ramp" in result.ports
    assert result.id_counter > 0


@pytest.mark.map_generator
def test_parking_lot_generator(runtime_dir):
    map_generator = ParkingLotGenerator()
    map_ = Map(name="parking_lot", scenario_type="parking")
    start_state, target_area, target_heading = map_generator.generate(map_)
    _render(map_, runtime_dir / "parking_lot.png")
    assert isinstance(start_state, State)
    assert isinstance(target_area, Area)
    assert isinstance(target_heading, float)


@pytest.mark.map_generator
def test_racing_track_generator(runtime_dir):
    map_generator = RacingTrackGenerator()
    map_ = Map(name="racing_track", scenario_type="racing")
    map_generator.generate(map_)
    _render(map_, runtime_dir / "racing_track.png")
    assert isinstance(map_.customs["start_state"], State)


@pytest.mark.map_generator
def test_one_way_straight(runtime_dir):
    map_ = Map(name="one_way_straight")
    start_port, end_port = _port_from_start_length(
        start=np.array([0.0, 0.0]), heading=0.0, length=60.0, lane_num=3, speed_limit=50.0
    )
    result = OneWay().build(start_port, end_port, lane_num=3, id_offset=0)
    _add_result(map_, result)
    _render(map_, runtime_dir / "one_way_straight.png")
    assert len(map_.lanes) == 3
    assert len(map_.roadlines) >= 4
    assert "entry" in result.ports
    assert "exit" in result.ports
    assert result.ports["exit"].lane_num == 3


@pytest.mark.map_generator
def test_one_way_curved(runtime_dir):
    map_ = Map(name="one_way_curved")
    result = OneWay().build(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(30.0, 15.0, 0.65, 2, speed_limit=50.0),
        lane_num=2,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / "one_way_curved.png")
    assert len(map_.lanes) == 2
    assert result.ports["exit"].lane_num == 2
    assert result.ports["exit"].heading == pytest.approx(0.65)
    np.testing.assert_allclose(result.ports["exit"].point, [30.0, 15.0], atol=1e-9)


@pytest.mark.map_generator
def test_two_way_straight(runtime_dir):
    map_ = Map(name="two_way_straight")
    start_port, end_port = _port_from_start_length(
        start=np.array([0.0, 0.0]), heading=0.0, length=60.0, lane_num=2, speed_limit=50.0
    )
    result = TwoWay().build(
        start_port, end_port, forward_lane_num=2, backward_lane_num=2, id_offset=0
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / "two_way_straight.png")
    assert len(map_.lanes) == 4
    assert "forward_in" in result.ports
    assert "forward_out" in result.ports
    assert "backward_in" in result.ports
    assert "backward_out" in result.ports


@pytest.mark.map_generator
def test_two_way_curved(runtime_dir):
    map_ = Map(name="two_way_curved")
    result = TwoWay().build(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(55.0, 25.0, 0.45, 2, speed_limit=50.0),
        forward_lane_num=2,
        backward_lane_num=2,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / "two_way_curved.png")
    assert len(map_.lanes) == 4


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "start_n,end_n,change_side",
    [(2, 3, "right"), (3, 2, "right"), (2, 3, "left"), (3, 2, "left"), (2, 2, "right")],
    ids=["expand_right", "reduce_right", "expand_left", "reduce_left", "no_change"],
)
def test_lane_adapter(runtime_dir, start_n: int, end_n: int, change_side: str) -> None:
    map_ = Map(name=f"lane_adapter_{start_n}to{end_n}_{change_side}")
    result = LaneAdapter(change_side=change_side).build(
        _make_port(0.0, 0.0, 0.0, start_n, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, end_n, speed_limit=50.0),
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / f"lane_adapter_{start_n}to{end_n}_{change_side}.png")
    max_n = max(start_n, end_n)
    assert "entry" in result.ports
    assert "exit" in result.ports
    assert result.ports["entry"].lane_num == start_n
    assert result.ports["exit"].lane_num == end_n
    assert len(result.lanes) == max_n
    assert len(result.roadlines) == max_n + 1


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "fork_side,main_n,branch_n,main_out_x,branch_x,branch_y,branch_h,taper_length,branch_length",
    [
        ("right", 3, 1, 130.0, 95.0, -45.0, -0.55, 35.0, 55.0),
        ("left", 3, 1, 130.0, 95.0, 45.0, 0.55, 35.0, 55.0),
        ("right", 4, 2, 150.0, 105.0, -55.0, -0.55, 40.0, 65.0),
    ],
    ids=["right_1lane", "left_1lane", "right_2lane"],
)
def test_fork(
    runtime_dir,
    fork_side: str,
    main_n: int,
    branch_n: int,
    main_out_x: float,
    branch_x: float,
    branch_y: float,
    branch_h: float,
    taper_length: float,
    branch_length: float,
) -> None:
    map_ = Map(name=f"fork_{fork_side}_{main_n}m_{branch_n}b")
    result = Fork(
        fork_side=fork_side, taper_length=taper_length, branch_length=branch_length
    ).build(
        _make_port(0.0, 0.0, 0.0, main_n, speed_limit=60.0),
        _make_port(main_out_x, 0.0, 0.0, main_n, speed_limit=60.0),
        _make_port(branch_x, branch_y, branch_h, branch_n, speed_limit=40.0),
        branch_lane_num=branch_n,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / f"fork_{fork_side}_{main_n}m_{branch_n}b.png")
    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "branch_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == main_n
    assert len(result.ports["main_out"].lane_ids) == main_n
    assert len(result.ports["branch_out"].lane_ids) == branch_n
    assert len(result.lanes) == main_n + branch_n
    assert result.id_counter > 0


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "merge_side,main_n,branch_n,main_out_x,branch_x,branch_y,branch_h,taper_length,branch_length",
    [
        ("right", 3, 1, 130.0, 35.0, -45.0, 0.55, 35.0, 55.0),
        ("left", 3, 1, 130.0, 35.0, 45.0, -0.55, 35.0, 55.0),
        ("right", 4, 2, 150.0, 45.0, -55.0, 0.55, 40.0, 65.0),
    ],
    ids=["right_1lane", "left_1lane", "right_2lane"],
)
def test_merge(
    runtime_dir,
    merge_side: str,
    main_n: int,
    branch_n: int,
    main_out_x: float,
    branch_x: float,
    branch_y: float,
    branch_h: float,
    taper_length: float,
    branch_length: float,
) -> None:
    map_ = Map(name=f"merge_{merge_side}_{main_n}m_{branch_n}b")
    result = Merge(
        merge_side=merge_side, taper_length=taper_length, branch_length=branch_length
    ).build(
        _make_port(0.0, 0.0, 0.0, main_n, speed_limit=60.0),
        _make_port(branch_x, branch_y, branch_h, branch_n, speed_limit=40.0),
        _make_port(main_out_x, 0.0, 0.0, main_n, speed_limit=60.0),
        branch_lane_num=branch_n,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / f"merge_{merge_side}_{main_n}m_{branch_n}b.png")
    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == main_n
    assert len(result.ports["branch_in"].lane_ids) == branch_n
    assert len(result.ports["main_out"].lane_ids) == main_n
    assert len(result.lanes) == main_n + branch_n
    assert result.id_counter > 0


@pytest.mark.map_generator
def test_merge_tail(runtime_dir) -> None:
    """Late-merge variant: merge_s_ratio near 1 forces branch join close to main_out."""
    map_ = Map(name="merge_tail")
    result = Merge(merge_side="right", taper_length=20.0, branch_length=50.0).build(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(95.0, -45.0, np.pi / 2 - 0.4, 1, speed_limit=40.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        merge_s_ratio=0.92,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / "merge_tail.png")
    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports
    assert len(result.lanes) == 4


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "name,headings",
    [
        ("intersection_cross", [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]),
        ("intersection_t", [0.0, np.pi / 2, np.pi]),
    ],
    ids=["cross", "t"],
)
def test_intersection_uniform_arms(runtime_dir, name: str, headings: list[float]) -> None:
    """Build the two supported intersection layouts and assemble their approaches."""
    map_ = Map(name=name)
    arms = [{"heading": h, "lane_num": 2} for h in headings]
    result = Intersection(radius=10.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)
    _render(map_, runtime_dir / f"{name}.png")
    assert sum(1 for key in result.ports if key.endswith("_out")) == len(arms)
    assert len(map_.junctions) == 1


@pytest.mark.map_generator
def test_intersection_mixed_arm_lane_counts(runtime_dir) -> None:
    """Assemble a cross intersection whose connected arms have unequal widths."""
    map_ = Map(name="intersection_mixed_arm_lane_counts")
    lane_counts = [1, 2, 3, 1]
    arms = [
        {"heading": heading, "lane_num": lane_num, "lane_width": 3.5}
        for heading, lane_num in zip([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], lane_counts)
    ]
    result = Intersection(radius=12.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)
    _render(map_, runtime_dir / "intersection_mixed_arm_lane_counts.png")

    for arm_idx, lane_num in enumerate(lane_counts):
        assert result.ports[f"arm_{arm_idx}_in"].lane_num == lane_num
        assert result.ports[f"arm_{arm_idx}_out"].lane_num == lane_num
    assert len(result.lanes) == 12
    assert len(map_.junctions) == 1


def _asymmetric_cross_arms() -> list[dict[str, float | int]]:
    """Return four arms with independently configured incoming and outgoing lanes."""
    lane_counts = [(3, 2), (2, 3), (2, 2), (1, 1)]
    headings = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    return [
        {"heading": heading, "in_lane_num": in_lane_num, "out_lane_num": out_lane_num}
        for heading, (in_lane_num, out_lane_num) in zip(headings, lane_counts)
    ]


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "name,headings,lane_counts,expected_lane_count,expected_map_lane_count",
    [
        (
            "intersection_asymmetric_cross",
            [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
            [(3, 2), (2, 3), (2, 2), (1, 1)],
            12,
            28,
        ),
        ("intersection_asymmetric_t", [0.0, np.pi / 2, np.pi], [(2, 1), (3, 2), (1, 2)], 6, 17),
    ],
    ids=["cross", "t"],
)
def test_intersection_asymmetric_arms(
    runtime_dir,
    name: str,
    headings: list[float],
    lane_counts: list[tuple[int, int]],
    expected_lane_count: int,
    expected_map_lane_count: int,
) -> None:
    """Build asymmetric arms and verify directional port and connector contracts."""
    map_ = Map(name=name)
    arms = [
        {"heading": heading, "in_lane_num": in_lane_num, "out_lane_num": out_lane_num}
        for heading, (in_lane_num, out_lane_num) in zip(headings, lane_counts)
    ]
    result = Intersection(radius=12.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)
    _render(map_, runtime_dir / f"{name}.png")

    assert len(result.lanes) == expected_lane_count
    assert len(map_.lanes) == expected_map_lane_count
    for arm_idx, (in_lane_num, out_lane_num) in enumerate(lane_counts):
        assert result.ports[f"arm_{arm_idx}_in"].lane_num == in_lane_num
        assert result.ports[f"arm_{arm_idx}_out"].lane_num == out_lane_num
        assert len(result.ports[f"arm_{arm_idx}_in"].lane_ids) == len(headings) - 1
        assert len(result.ports[f"arm_{arm_idx}_out"].lane_ids) == len(headings) - 1

    for lane in result.lanes:
        tags = lane.custom_tags
        assert tags["incoming_lane_index"] < lane_counts[tags["from_arm"]][0]
        assert tags["outgoing_lane_index"] < lane_counts[tags["to_arm"]][1]


@pytest.mark.map_generator
def test_intersection_asymmetric_arm_geometry() -> None:
    """Keep asymmetric arm edges aligned with their attached two-way road."""
    arms = _asymmetric_cross_arms()
    result = Intersection(radius=12.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    map_ = Map(name="intersection_asymmetric_arm_geometry")
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)

    arm_idx = 0
    in_port = result.ports[f"arm_{arm_idx}_in"]
    out_port = result.ports[f"arm_{arm_idx}_out"]
    normal = np.array([-np.sin(in_port.heading), np.cos(in_port.heading)])
    out_edge = out_port.point + out_port.lane_num * out_port.lane_width * normal
    in_edge = in_port.point - in_port.lane_num * in_port.lane_width * normal
    shape = np.asarray(result.junctions[0].custom_tags["shape"], dtype=float)

    assert np.any(np.all(np.isclose(shape, out_edge), axis=1))
    assert np.any(np.all(np.isclose(shape, in_edge), axis=1))

    forward_edge = next(
        roadline
        for roadline in map_.roadlines.values()
        if roadline.custom_tags.get("module") == "two_way"
        and roadline.custom_tags.get("direction") == "forward"
        and roadline.custom_tags.get("boundary_index") == out_port.lane_num
        and np.allclose(np.asarray(roadline.geometry.coords)[0], out_edge)
    )
    backward_edge = next(
        roadline
        for roadline in map_.roadlines.values()
        if roadline.custom_tags.get("module") == "two_way"
        and roadline.custom_tags.get("direction") == "backward"
        and roadline.custom_tags.get("boundary_index") == in_port.lane_num
        and np.allclose(np.asarray(roadline.geometry.coords)[-1], in_edge)
    )
    assert forward_edge is not None
    assert backward_edge is not None


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "ring_lane_num,expected_lane_count", [(1, 12), (2, 22)], ids=["one_ring_lane", "two_ring_lanes"]
)
def test_roundabout_asymmetric_arms(
    runtime_dir, ring_lane_num: int, expected_lane_count: int
) -> None:
    """Use independent arm directions when linking to the roundabout ring."""
    arms = _asymmetric_cross_arms()
    lane_counts = [(arm["in_lane_num"], arm["out_lane_num"]) for arm in arms]
    map_ = Map(name=f"roundabout_asymmetric_{ring_lane_num}")
    result = Roundabout(ring_radius=12.0, ring_lane_num=ring_lane_num).build(
        center=np.array([0.0, 0.0]), arms=arms, id_offset=0
    )
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)
    _render(map_, runtime_dir / f"roundabout_asymmetric_{ring_lane_num}.png")

    assert len(result.lanes) == expected_lane_count
    for arm_idx, (in_lane_num, out_lane_num) in enumerate(lane_counts):
        in_port = result.ports[f"arm_{arm_idx}_in"]
        out_port = result.ports[f"arm_{arm_idx}_out"]
        assert in_port.lane_num == in_lane_num
        assert out_port.lane_num == out_lane_num
        assert len(in_port.lane_ids) == min(in_lane_num, ring_lane_num)
        assert len(out_port.lane_ids) == min(out_lane_num, ring_lane_num)


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "arm,match",
    [
        ({"heading": 0.0, "in_lane_num": 2}, "both 'in_lane_num'"),
        ({"heading": 0.0, "out_lane_num": 2}, "both 'in_lane_num'"),
        ({"heading": 0.0, "lane_num": 2, "in_lane_num": 2, "out_lane_num": 2}, "not combine"),
        ({"heading": 0.0}, "lane_num' or both"),
        ({"heading": 0.0, "in_lane_num": 0, "out_lane_num": 2}, "in_lane_num must be"),
        ({"heading": 0.0, "in_lane_num": 2, "out_lane_num": -1}, "out_lane_num must be"),
    ],
    ids=["only_in", "only_out", "mixed_forms", "missing_counts", "zero_in", "negative_out"],
)
def test_junction_invalid_asymmetric_arm_keys(arm: dict[str, int | float], match: str) -> None:
    """Intersection and roundabout reject ambiguous or invalid arm counts consistently."""
    arms = [arm, {"heading": np.pi / 2, "lane_num": 2}, {"heading": np.pi, "lane_num": 2}]

    for generator in (Intersection(), Roundabout()):
        with pytest.raises(ValueError, match=match):
            generator.build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)


@pytest.mark.map_generator
def test_roadport_arms_remain_symmetric() -> None:
    """RoadPort arms retain their single-count, bidirectional contract."""
    ports = [
        _make_port(12.0, 0.0, 0.0, 1),
        _make_port(0.0, 12.0, np.pi / 2, 2),
        _make_port(-12.0, 0.0, np.pi, 3),
        _make_port(0.0, -12.0, 3 * np.pi / 2, 2),
    ]
    for generator in (Intersection(), Roundabout()):
        result = generator.build(center=np.array([0.0, 0.0]), arms=ports, id_offset=0)
        for arm_idx, port in enumerate(ports):
            assert result.ports[f"arm_{arm_idx}_in"].lane_num == port.lane_num
            assert result.ports[f"arm_{arm_idx}_out"].lane_num == port.lane_num


@pytest.mark.map_generator
@pytest.mark.parametrize("module", ["intersection", "roundabout"])
def test_asymmetric_junction_has_no_id_collision(module: str) -> None:
    """Repeated asymmetric modules must allocate disjoint element identifiers."""
    arms = _asymmetric_cross_arms()
    if module == "intersection":
        generator = Intersection(radius=12.0)
    else:
        generator = Roundabout(ring_radius=12.0, ring_lane_num=2)

    first = generator.build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    second = generator.build(center=np.array([100.0, 0.0]), arms=arms, id_offset=first.id_counter)
    first_ids = {element.id_ for element in [*first.lanes, *first.roadlines, *first.junctions]}
    second_ids = {element.id_ for element in [*second.lanes, *second.roadlines, *second.junctions]}

    assert first_ids.isdisjoint(second_ids)
    assert second.id_counter > first.id_counter


@pytest.mark.map_generator
def test_intersection_cross_curved(runtime_dir):
    map_ = Map(name="intersection_cross_curved")
    arms = [
        {"heading": np.pi / 2, "lane_num": 2, "curvature": 0.0, "radius": 12.0},
        {"heading": np.pi, "lane_num": 2, "curvature": 0.02, "radius": 12.0},
        {"heading": -np.pi / 2, "lane_num": 2, "curvature": 0.0, "radius": 12.0},
        {"heading": 0.0, "lane_num": 2, "curvature": -0.02, "radius": 12.0},
    ]
    result = Intersection().build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)
    _render(map_, runtime_dir / "intersection_cross_curved.png")
    assert sum(1 for k in result.ports if k.endswith("_out")) == 4
    assert len(map_.junctions) == 1


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "arm_headings,arm_num",
    [([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], 4), ([0.0, 2 * np.pi / 3, 4 * np.pi / 3], 3)],
    ids=["4arm", "3arm"],
)
def test_roundabout(runtime_dir, arm_headings: list, arm_num: int) -> None:
    map_ = Map(name=f"roundabout_{arm_num}arm")
    arms = [{"heading": h, "lane_num": 2} for h in arm_headings]
    result = Roundabout(ring_radius=12.0, ring_lane_num=2).build(
        center=np.array([0.0, 0.0]), arms=arms, id_offset=0
    )
    _add_result(map_, result)
    _attach_outgoing_two_way_roads(map_, result)
    _render(map_, runtime_dir / f"roundabout_{arm_num}arm.png")
    assert sum(1 for k in result.ports if k.endswith("_out")) == arm_num
    assert len(result.junctions) == 1


@pytest.mark.map_generator
def test_roundabout_curved_approach(runtime_dir):
    map_ = Map(name="roundabout_curved_approach")
    arms = [
        {"heading": 0.0, "lane_num": 2, "curvature": 0.03},
        {"heading": np.pi / 2, "lane_num": 2, "curvature": 0.0},
        {"heading": np.pi, "lane_num": 2, "curvature": -0.03},
        {"heading": 3 * np.pi / 2, "lane_num": 2, "curvature": 0.0},
    ]
    result = Roundabout(ring_radius=12.0, ring_lane_num=2).build(
        center=np.array([0.0, 0.0]), arms=arms, id_offset=0
    )
    _add_result(map_, result)

    id_off = result.id_counter + 1000
    out_ports = [(k, p) for k, p in sorted(result.ports.items()) if k.endswith("_out")]
    for i, (_, port) in enumerate(out_ports):
        h = float(port.heading)
        normal = np.array([-np.sin(h), np.cos(h)], dtype=float)
        curve_sign = 1.0 if i % 2 == 0 else -1.0
        end_port = RoadPort(
            point=port.point + 32.0 * heading_unit(h) + curve_sign * 7.0 * normal,
            heading=h + curve_sign * 0.22,
            lane_num=port.lane_num,
            lane_width=port.lane_width,
            speed_limit=port.speed_limit,
        )
        road_result = TwoWay().build(
            port,
            end_port,
            forward_lane_num=port.lane_num,
            backward_lane_num=port.lane_num,
            lane_width=port.lane_width,
            speed_limit=port.speed_limit,
            id_offset=id_off,
        )
        id_off = road_result.id_counter
        _add_result(map_, road_result)

    _render(map_, runtime_dir / "roundabout_curved_approach.png")
    assert sum(1 for k in result.ports if k.endswith("_out")) == 4
    assert len(result.junctions) == 1


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "kind,ramp_side,ramp_y,ramp_h",
    [
        ("exit", "right", -95.0, -0.95),
        ("exit", "left", 95.0, 0.95),
        ("entrance", "right", -95.0, 0.95),
        ("entrance", "left", 95.0, -0.95),
    ],
    ids=["exit_right", "exit_left", "entrance_right", "entrance_left"],
)
def test_freeway_ramp(runtime_dir, kind: str, ramp_side: str, ramp_y: float, ramp_h: float) -> None:
    ramp_x = 330.0 if kind == "exit" else 90.0
    main_in = _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0)
    main_out = _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0)
    ramp_port = _make_port(ramp_x, ramp_y, ramp_h, 1, speed_limit=50.0)

    if kind == "exit":
        result = ExitRamp(
            main_road_type="freeway", ramp_side=ramp_side, taper_length=70.0, parallel_length=90.0
        ).build(main_in, main_out, ramp_port, id_offset=0)
    else:
        result = EntranceRamp(
            main_road_type="freeway", ramp_side=ramp_side, taper_length=70.0, parallel_length=90.0
        ).build(main_in, main_out, ramp_port, id_offset=0)

    map_ = Map(name=f"freeway_{kind}_{ramp_side}")
    _add_result(map_, result)
    _render(map_, runtime_dir / f"freeway_{kind}_{ramp_side}.png")

    _assert_ramp_ports(result)
    assert "backward_in" not in result.ports
    assert "backward_out" not in result.ports
    expected_ramp_kind = "ramp_out" if kind == "exit" else "ramp_in"
    assert result.ports["ramp"].kind == expected_ramp_kind


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "kind,ramp_x,ramp_y,ramp_h",
    [("exit", 255.0, -75.0, -0.90), ("entrance", 65.0, -75.0, 0.90)],
    ids=["exit", "entrance"],
)
def test_urban_ramp(runtime_dir, kind: str, ramp_x: float, ramp_y: float, ramp_h: float) -> None:
    main_in = _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0)
    main_out = _make_port(320.0, 16.0, 0.04, 2, speed_limit=80.0)
    ramp_port = _make_port(ramp_x, ramp_y, ramp_h, 1, speed_limit=40.0)

    if kind == "exit":
        result = ExitRamp(
            main_road_type="urban", backward_lane_num=1, taper_length=55.0, parallel_length=75.0
        ).build(main_in, main_out, ramp_port, id_offset=0)
    else:
        result = EntranceRamp(
            main_road_type="urban", backward_lane_num=1, taper_length=55.0, parallel_length=75.0
        ).build(main_in, main_out, ramp_port, id_offset=0)

    map_ = Map(name=f"urban_{kind}")
    _add_result(map_, result)
    _render(map_, runtime_dir / f"urban_{kind}.png")

    _assert_ramp_ports(result)
    assert "backward_in" in result.ports
    assert "backward_out" in result.ports
    expected_ramp_kind = "ramp_out" if kind == "exit" else "ramp_in"
    assert result.ports["ramp"].kind == expected_ramp_kind


@pytest.mark.map_generator
def test_ramp_no_id_collision() -> None:
    r1 = ExitRamp(
        main_road_type="freeway", ramp_side="right", taper_length=70.0, parallel_length=90.0
    ).build(
        _make_port(0.0, 0.0, 0.0, 3),
        _make_port(420.0, 0.0, 0.0, 3),
        _make_port(330.0, -95.0, -0.95, 1),
        id_offset=0,
    )
    r2 = EntranceRamp(
        main_road_type="freeway", ramp_side="right", taper_length=70.0, parallel_length=90.0
    ).build(
        _make_port(0.0, 120.0, 0.0, 3),
        _make_port(420.0, 120.0, 0.0, 3),
        _make_port(90.0, 25.0, 0.95, 1),
        id_offset=r1.id_counter,
    )
    assert r2.id_counter > r1.id_counter
    ids1 = {e.id_ for e in [*r1.lanes, *r1.roadlines, *r1.junctions, *r1.areas]}
    ids2 = {e.id_ for e in [*r2.lanes, *r2.roadlines, *r2.junctions, *r2.areas]}
    assert len(ids1 & ids2) == 0


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "fork_side,branch_lane_num,match",
    [("center", 1, "fork_side must be"), ("right", 4, "branch_lane_num <= main_lane_num")],
    ids=["bad_fork_side", "branch_gt_main"],
)
def test_fork_invalid_args(fork_side: str, branch_lane_num: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Fork(fork_side=fork_side, taper_length=35.0, branch_length=55.0).build(
            _make_port(0.0, 0.0, 0.0, 3),
            _make_port(130.0, 0.0, 0.0, 3),
            _make_port(95.0, -45.0, -0.55, 1),
            branch_lane_num=branch_lane_num,
            id_offset=0,
        )


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "merge_side,branch_lane_num,match",
    [("up", 1, "merge_side must be"), ("right", 5, "branch_lane_num <= main_lane_num")],
    ids=["bad_merge_side", "branch_gt_main"],
)
def test_merge_invalid_args(merge_side: str, branch_lane_num: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Merge(merge_side=merge_side, taper_length=35.0, branch_length=55.0).build(
            _make_port(0.0, 0.0, 0.0, 3),
            _make_port(35.0, -45.0, 0.55, 1),
            _make_port(130.0, 0.0, 0.0, 3),
            branch_lane_num=branch_lane_num,
            id_offset=0,
        )


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "start_n,end_n,change_side,step_size,match",
    [
        (0, 2, "right", 0.5, "start_lane_num must be >= 1"),
        (2, 0, "right", 0.5, "end_lane_num must be >= 1"),
        (2, 2, "diagonal", 0.5, "change_side must be"),
        (1, 3, "right", 0.5, "lane count difference of 1"),
        (2, 2, "right", 0.0, "step_size must be positive"),
    ],
    ids=["start_zero", "end_zero", "bad_side", "delta_too_large", "zero_step"],
)
def test_lane_adapter_invalid_args(
    start_n: int, end_n: int, change_side: str, step_size: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        LaneAdapter(change_side=change_side, step_size=step_size).build(
            _make_port(0.0, 0.0, 0.0, start_n), _make_port(60.0, 0.0, 0.0, end_n), id_offset=0
        )


@pytest.mark.map_generator
def test_ramp_urban_left_side_raises() -> None:
    """Urban ramp only supports ramp_side='right'; 'left' must raise."""
    with pytest.raises(ValueError, match="urban ramp currently supports only ramp_side='right'"):
        ExitRamp(
            main_road_type="urban", ramp_side="left", taper_length=55.0, parallel_length=75.0
        ).build(
            _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0),
            _make_port(320.0, 0.0, 0.0, 2, speed_limit=80.0),
            _make_port(255.0, 75.0, -0.90, 1, speed_limit=40.0),
            id_offset=0,
        )


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "arm_dicts,match",
    [
        (
            [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2]],
            "intersection requires 3 or 4 arms",
        ),
        (
            [
                {"heading": h, "lane_num": 2}
                for h in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4]
            ],
            "intersection requires 3 or 4 arms",
        ),
        ([{"heading": h, "lane_num": 0} for h in [0.0, np.pi / 2, np.pi]], "lane_num must be"),
    ],
    ids=["too_few_arms", "too_many_arms", "zero_lane_num"],
)
def test_intersection_invalid_arms(arm_dicts: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Intersection().build(center=np.array([0.0, 0.0]), arms=arm_dicts, id_offset=0)


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "init_kwargs,build_kwargs,match",
    [
        ({}, {"arms": [{"heading": 0.0, "lane_num": 2}]}, "at least 2 arms"),
        ({"ring_lane_num": 0}, {}, "ring_lane_num must be"),
        ({"ring_radius": -1.0}, {}, "ring_radius must be positive"),
        ({"step_size": 0.0}, {}, "step_size must be positive"),
    ],
    ids=["one_arm", "zero_ring_lanes", "negative_radius", "zero_step"],
)
def test_roundabout_invalid_args(init_kwargs: dict, build_kwargs: dict, match: str) -> None:
    init_params = {"ring_radius": 12.0, "ring_lane_num": 1, "step_size": 0.1, **init_kwargs}
    build_params = {
        "center": np.array([0.0, 0.0]),
        "arms": [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi]],
        "id_offset": 0,
        **build_kwargs,
    }
    with pytest.raises(ValueError, match=match):
        Roundabout(**init_params).build(**build_params)


@pytest.mark.map_generator
def test_intersection_no_id_collision() -> None:
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]]
    result_1 = Intersection(radius=10.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    result_2 = Intersection(radius=10.0).build(
        center=np.array([100.0, 0.0]), arms=arms, id_offset=result_1.id_counter
    )
    ids_1 = {e.id_ for e in [*result_1.lanes, *result_1.roadlines, *result_1.junctions]}
    ids_2 = {e.id_ for e in [*result_2.lanes, *result_2.roadlines, *result_2.junctions]}
    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter
