# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map generator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest

logging.basicConfig(level=logging.INFO)

_RENDER_MAPS = os.environ.get("TACTICS2D_RENDER_MAPS", "0") == "1"

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
    JunctionApproach,
    LaneAdapter,
    Merge,
    OneWay,
    Roundabout,
    TwoWay,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort
from tactics2d.participant.trajectory import State


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


def _two_way_from_port(port: RoadPort, *, length: float, id_offset: int) -> RoadModuleResult:
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
        backward_lane_num=port.lane_num,
        lane_width=port.lane_width,
        speed_limit=port.speed_limit,
        id_offset=id_offset,
    )


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


def _junction_approach_centerline(start, heading, length, step_size=0.1):
    """Generate a straight centreline for JunctionApproach tests."""
    from tactics2d.map.generator.road_segment.reference_line import sample_centerline

    return sample_centerline(start, heading, length, curvature=0.0, step_size=step_size)


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "start_n,end_n,change_side,taper_len",
    [
        (2, 3, "right", 30.0),
        (3, 2, "right", 30.0),
        (2, 3, "left", 30.0),
        (3, 2, "left", 30.0),
        (2, 2, "both", 30.0),
        (1, 3, "both", 40.0),
        (3, 1, "both", 40.0),
    ],
    ids=["expand_right", "reduce_right", "expand_left", "reduce_left", "no_change", "1to3", "3to1"],
)
def test_junction_approach(
    runtime_dir, start_n: int, end_n: int, change_side: str, taper_len: float
) -> None:
    """Visual smoke test: renders each transition scenario for manual inspection.

    Run with ``TACTICS2D_RENDER_MAPS=1 pytest ...`` to generate PNGs under
    ``build/test_maps/``.
    """
    map_ = Map(name=f"junction_approach_{start_n}to{end_n}_{change_side}")
    cl = _junction_approach_centerline(np.array([0.0, 0.0]), 0.0, taper_len)
    result = JunctionApproach(change_side=change_side).build(
        centerline=cl,
        start_lane_num=start_n,
        end_lane_num=end_n,
        end_boundary_offsets=np.array([end_n / 2.0 * 3.5 - i * 3.5 for i in range(end_n + 1)]),
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / f"junction_approach_{start_n}to{end_n}_{change_side}.png")
    max_n = max(start_n, end_n)
    assert "entry" in result.ports
    assert "exit" in result.ports
    assert result.ports["entry"].lane_num == start_n
    assert result.ports["exit"].lane_num == end_n
    assert len(result.lanes) == max_n
    assert len(result.roadlines) == max_n + 1


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "taper_len,expect_warning", [(30.0, False), (3.0, True)], ids=["adequate_length", "too_short"]
)
def test_junction_approach_taper_length(caplog, taper_len: float, expect_warning: bool) -> None:
    """Short centreline should emit a warning about taper stiffness."""
    caplog.set_level(
        logging.WARNING, logger="tactics2d.map.generator.road_segment.junction_approach"
    )
    cl = _junction_approach_centerline(np.array([0.0, 0.0]), 0.0, taper_len)
    JunctionApproach().build(
        centerline=cl,
        start_lane_num=2,
        end_lane_num=3,
        end_boundary_offsets=np.array([5.25, 3.5, 1.75, 0.0]),
        id_offset=0,
    )
    if expect_warning:
        assert any("shorter than recommended" in rec.message for rec in caplog.records)
    else:
        assert not any("shorter than recommended" in rec.message for rec in caplog.records)


@pytest.mark.map_generator
def test_junction_approach_invalid_args() -> None:
    """Invalid arguments should raise ValueError."""
    cl = _junction_approach_centerline(np.array([0.0, 0.0]), 0.0, 20.0)

    with pytest.raises(ValueError, match="start_lane_num must be >= 1"):
        JunctionApproach().build(
            centerline=cl,
            start_lane_num=0,
            end_lane_num=2,
            end_boundary_offsets=np.array([3.5, 1.75, 0.0]),
            id_offset=0,
        )
    with pytest.raises(ValueError, match="end_lane_num must be >= 1"):
        JunctionApproach().build(
            centerline=cl,
            start_lane_num=2,
            end_lane_num=0,
            end_boundary_offsets=np.array([0.0]),
            id_offset=0,
        )
    with pytest.raises(ValueError, match="lane_width must be positive"):
        JunctionApproach().build(
            centerline=cl,
            start_lane_num=2,
            end_lane_num=2,
            end_boundary_offsets=np.array([3.5, 1.75, 0.0]),
            lane_width=0.0,
            id_offset=0,
        )
    with pytest.raises(ValueError, match="end_boundary_offsets must have"):
        JunctionApproach().build(
            centerline=cl,
            start_lane_num=2,
            end_lane_num=2,
            end_boundary_offsets=np.array([3.5, 0.0]),  # should be 3 elements
            id_offset=0,
        )
    with pytest.raises(ValueError, match="centreline must have at least 2 points"):
        JunctionApproach().build(
            centerline=np.array([[0.0, 0.0]]),
            start_lane_num=2,
            end_lane_num=2,
            end_boundary_offsets=np.array([3.5, 1.75, 0.0]),
            id_offset=0,
        )


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
def test_intersection_cross(runtime_dir):
    map_ = Map(name="intersection_cross")
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]]
    result = Intersection(radius=10.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    id_off = result.id_counter + 1000
    for key, port in sorted(result.ports.items()):
        if key.endswith("_out"):
            road_result = _two_way_from_port(port, length=30.0, id_offset=id_off)
            id_off = road_result.id_counter
            _add_result(map_, road_result)
    _render(map_, runtime_dir / "intersection_cross.png")
    assert sum(1 for k in result.ports if k.endswith("_out")) == 4
    assert len(map_.junctions) == 1


@pytest.mark.map_generator
def test_intersection_t(runtime_dir):
    map_ = Map(name="intersection_t")
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi]]
    result = Intersection(radius=10.0).build(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    id_off = result.id_counter + 1000
    for key, port in sorted(result.ports.items()):
        if key.endswith("_out"):
            road_result = _two_way_from_port(port, length=30.0, id_offset=id_off)
            id_off = road_result.id_counter
            _add_result(map_, road_result)
    _render(map_, runtime_dir / "intersection_t.png")
    assert sum(1 for k in result.ports if k.endswith("_out")) == 3
    assert len(map_.junctions) == 1


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
    id_off = result.id_counter + 1000
    for key, port in sorted(result.ports.items()):
        if key.endswith("_out"):
            road_result = _two_way_from_port(port, length=30.0, id_offset=id_off)
            id_off = road_result.id_counter
            _add_result(map_, road_result)
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
    id_off = result.id_counter + 1000
    for key, port in sorted(result.ports.items()):
        if key.endswith("_out"):
            road_result = _two_way_from_port(port, length=30.0, id_offset=id_off)
            id_off = road_result.id_counter
            _add_result(map_, road_result)
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
    [([{"heading": h, "lane_num": 0} for h in [0.0, np.pi / 2, np.pi]], "lane_num must be")],
    ids=["zero_lane_num"],
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


# ---------------------------------------------------------------------------
# JunctionApproach curvature and integration tests
# ---------------------------------------------------------------------------


def _compute_curvature(pts: np.ndarray) -> np.ndarray:
    """Discrete signed curvature along a polyline using finite differences.

    Args:
        pts: Points with shape ``(N, 2)``.

    Returns:
        Curvature array of length ``N``.  Boundary points use one-sided
        differences; interior points use central differences.
    """
    if len(pts) < 3:
        return np.zeros(len(pts))

    d1 = np.diff(pts, axis=0)  # first differences, shape (N-1, 2)
    ds = np.linalg.norm(d1, axis=1)
    ds = np.where(ds < 1e-12, 1e-12, ds)
    tangents = d1 / ds[:, None]  # shape (N-1, 2)

    # second differences via tangent differences at midpoints
    d_tangents = np.diff(tangents, axis=0)  # shape (N-2, 2)
    mid_ds = (ds[:-1] + ds[1:]) / 2.0  # shape (N-2,)
    mid_ds = np.where(mid_ds < 1e-12, 1e-12, mid_ds)
    curvature = np.zeros(len(pts))
    curvature[1:-1] = np.linalg.norm(d_tangents, axis=1) / mid_ds

    # one-sided for endpoints
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return curvature


@pytest.mark.map_generator
def test_junction_approach_curvature_endpoints() -> None:
    """Quintic smoothstep boundaries must have near-zero curvature at both ends.

    This is the key numerical guarantee that fixes the "stiff taper" issue:
    cubic smoothstep produces curvature jumps at endpoints; quintic eliminates
    them.  We verify by generating a taper on a curved centreline with a
    lane-count change and checking that every boundary's curvature decays to
    near zero within a few samples of each endpoint.
    """
    from tactics2d.map.generator.road_segment.reference_line import sample_centerline

    # curved centreline — a gentle left-turning arc
    cl = sample_centerline(np.array([0.0, 0.0]), heading=0.0, length=40.0, curvature=0.005)
    end_n = 3
    end_offs = np.array([end_n / 2.0 * 3.5 - i * 3.5 for i in range(end_n + 1)])

    result = JunctionApproach(change_side="both").build(
        centerline=cl,
        start_lane_num=2,
        end_lane_num=end_n,
        end_boundary_offsets=end_offs,
        lane_width=3.5,
        id_offset=0,
    )

    # extract boundary polylines
    for rl in result.roadlines:
        pts = np.asarray(rl.geometry.coords)
        if len(pts) < 5:
            continue  # skip very short boundaries
        kappa = _compute_curvature(pts)
        # first and last 3 samples should be small (< 50% of max curvature)
        max_k = max(float(np.max(np.abs(kappa))), 1e-6)
        entry_curv = np.abs(kappa[:3]).max()
        exit_curv = np.abs(kappa[-3:]).max()
        assert (
            entry_curv / max_k < 0.5
        ), f"high entry curvature: {entry_curv:.4f} vs max {max_k:.4f}"
        assert exit_curv / max_k < 0.5, f"high exit curvature: {exit_curv:.4f} vs max {max_k:.4f}"


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "change_side,taper_len,label",
    [
        ("both", 20.0, "curved_both"),
        ("left", 20.0, "curved_left"),
        ("right", 20.0, "curved_right"),
        ("both", 5.0, "short_both"),
    ],
    ids=["curved_both", "curved_left", "curved_right", "short_both"],
)
def test_junction_approach_complex(
    runtime_dir, change_side: str, taper_len: float, label: str
) -> None:
    """Curved centreline + non-uniform port boundaries — realistic junction approach.

    This exercises the three fixes together:
    - Quintic smoothstep on a curved reference line
    - change_side controlling asymmetric lane addition
    - min_taper_length warning for the "short" variant

    The junction port simulates a 3-lane arm where the left-turn lane is
    narrower (3.0 m) and the through/right lanes are 3.5 m.
    """
    from tactics2d.map.generator.road_segment.reference_line import sample_centerline

    # curved approach at 30° heading
    cl = sample_centerline(
        np.array([0.0, 0.0]), heading=np.deg2rad(30), length=taper_len, curvature=0.008
    )
    # non-uniform port boundaries: [leftmost=5.25, 2.75, 0.0, -3.5] (rightmost)
    end_offsets = np.array([5.25, 2.75, 0.0, -3.5])

    map_ = Map(name=f"junction_approach_{label}")
    result = JunctionApproach(change_side=change_side).build(
        centerline=cl,
        start_lane_num=2,
        end_lane_num=3,
        end_boundary_offsets=end_offsets,
        lane_width=3.5,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, runtime_dir / f"junction_approach_{label}.png")

    assert len(result.lanes) == 3
    assert len(result.roadlines) == 4
    # all lanes must have valid line references
    _assert_lane_line_ids_resolved(result)


# ---------------------------------------------------------------------------
# Bug 2 verification — lane matching with change_side
# ---------------------------------------------------------------------------


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "start_n,end_n,change_side,expected_roles",
    [
        # 2→3: one lane added
        (2, 3, "left", ["added", "through", "through"]),
        (2, 3, "right", ["through", "through", "added"]),
        # 3→2: one lane dropped
        (3, 2, "left", ["dropped", "through", "through"]),
        (3, 2, "right", ["through", "through", "dropped"]),
        # 2→4: two lanes added on the specified side
        (2, 4, "left", ["added", "added", "through", "through"]),
        (2, 4, "right", ["through", "through", "added", "added"]),
        # 4→2: two lanes dropped on the specified side
        (4, 2, "left", ["dropped", "dropped", "through", "through"]),
        (4, 2, "right", ["through", "through", "dropped", "dropped"]),
        # no change — all through
        (2, 2, "left", ["through", "through"]),
        (2, 2, "right", ["through", "through"]),
    ],
    ids=[
        "expand_left_2to3",
        "expand_right_2to3",
        "reduce_left_3to2",
        "reduce_right_3to2",
        "expand_left_2to4",
        "expand_right_2to4",
        "reduce_left_4to2",
        "reduce_right_4to2",
        "no_change_left",
        "no_change_right",
    ],
)
def test_junction_approach_lane_roles(
    start_n: int, end_n: int, change_side: str, expected_roles: list[str]
) -> None:
    """Verify change_side correctly positions added/dropped lanes.

    This is the direct validation of bug 2 (lane matching).  Each lane's
    ``custom_tags["lane_role"]`` must match the expected side-preference.
    """
    from tactics2d.map.generator.road_segment.reference_line import sample_centerline

    cl = sample_centerline(np.array([0.0, 0.0]), 0.0, 30.0)
    end_offs = np.array([end_n / 2.0 * 3.5 - i * 3.5 for i in range(end_n + 1)])

    result = JunctionApproach(change_side=change_side).build(
        centerline=cl,
        start_lane_num=start_n,
        end_lane_num=end_n,
        end_boundary_offsets=end_offs,
        id_offset=0,
    )

    actual_roles = [lane.custom_tags.get("lane_role", "?") for lane in result.lanes]
    assert actual_roles == expected_roles, (
        f"change_side={change_side!r}, {start_n}→{end_n}: "
        f"expected {expected_roles}, got {actual_roles}"
    )


# ---------------------------------------------------------------------------
# Bug 3 verification — roadline deduplication
# ---------------------------------------------------------------------------


def _make_dummy_roadline(id_: str, start, end, step_size=0.1):
    """Create a RoadLine with a straight segment from *start* to *end*."""
    from shapely.geometry import LineString

    from tactics2d.map.element import RoadLine

    return RoadLine(
        id_=str(id_),
        geometry=LineString([tuple(start), tuple(end)]),
        type_="virtual",
        subtype="solid",
        color="white",
    )


@pytest.mark.map_generator
def test_deduplicate_roadlines_merges_identical() -> None:
    """Two roadlines with nearly identical geometry should be merged."""
    from shapely.geometry import LineString

    from tactics2d.map.element import Lane, RoadLine
    from tactics2d.map.generator.road_segment.element_builder import deduplicate_roadlines

    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])

    rl_a = _make_dummy_roadline("a", start, end)
    rl_b = _make_dummy_roadline("b", start + 0.01, end + 0.01)  # nearly identical
    rl_c = _make_dummy_roadline("c", start, np.array([10.0, 5.0]))  # different end

    # Lane that references both duplicates
    lane = Lane(
        id_="lane_1",
        left_side=LineString([(0.0, 1.75), (10.0, 1.75)]),
        right_side=LineString([(0.0, -1.75), (10.0, -1.75)]),
        line_ids={"left": ["a"], "right": ["b", "c"]},
    )

    roadlines = [rl_a, rl_b, rl_c]
    deduped, updated_lanes = deduplicate_roadlines(roadlines, [lane], step_size=0.1)

    # rl_b should be merged into rl_a
    assert len(deduped) < len(roadlines), "expected at least one roadline to be merged"
    remaining_ids = {rl.id_ for rl in deduped}
    assert "a" in remaining_ids, "kept roadline 'a' should remain"
    assert "b" not in remaining_ids, "duplicate roadline 'b' should be removed"
    assert "c" in remaining_ids, "non-duplicate roadline 'c' should remain"

    # Lane references should be updated
    left_ids = updated_lanes[0].line_ids["left"]
    right_ids = updated_lanes[0].line_ids["right"]
    assert left_ids == ["a"], f"left refs should be ['a'], got {left_ids}"
    assert "b" not in right_ids, f"'b' should not appear in right refs: {right_ids}"
    assert "a" in right_ids, f"'a' should replace 'b' in right refs: {right_ids}"
    assert "c" in right_ids, f"'c' should be kept in right refs: {right_ids}"


@pytest.mark.map_generator
def test_deduplicate_roadlines_no_false_positive() -> None:
    """Roadlines with clearly different geometry should NOT be merged."""
    from shapely.geometry import LineString

    from tactics2d.map.element import Lane, RoadLine
    from tactics2d.map.generator.road_segment.element_builder import deduplicate_roadlines

    rl_a = _make_dummy_roadline("a", np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    rl_b = _make_dummy_roadline("b", np.array([0.0, 0.0]), np.array([10.0, 5.0]))  # different
    rl_c = _make_dummy_roadline("c", np.array([5.0, 0.0]), np.array([15.0, 0.0]))  # shifted
    # Same endpoints and length within the old tolerance, but not the same path.
    rl_d = RoadLine(
        id_="d",
        geometry=LineString([(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)]),
        type_="virtual",
        subtype="solid",
        color="white",
    )

    lane = Lane(
        id_="lane_1",
        left_side=LineString([(0.0, 1.0), (10.0, 1.0)]),
        right_side=LineString([(0.0, -1.0), (10.0, -1.0)]),
        line_ids={"left": ["a"], "right": ["b", "c"]},
    )

    roadlines = [rl_a, rl_b, rl_c, rl_d]
    deduped, _ = deduplicate_roadlines(roadlines, [lane], step_size=0.1)

    assert len(deduped) == 4, "no roadlines should be merged — all are distinct"


# ---------------------------------------------------------------------------
# LaneAdapter lane_side parameter tests
# ---------------------------------------------------------------------------


def _adapter_lane_centers(result: RoadModuleResult) -> list[float]:
    """Return lane centre offsets at the start end (midpoint of left/right boundaries)."""
    centers: list[float] = []
    for lane in result.lanes:
        left_pts = np.asarray(lane.left_side.coords)
        right_pts = np.asarray(lane.right_side.coords)
        # measure offset at the first point of each boundary
        dx_l = left_pts[0, 0] - left_pts[-1, 0]
        dy_l = left_pts[0, 1] - left_pts[-1, 1]
        dx_r = right_pts[0, 0] - right_pts[-1, 0]
        dy_r = right_pts[0, 1] - right_pts[-1, 1]
        # lateral offset at start ≈ perpendicular distance from centreline start
        # For a horizontal road heading=0, y is the lateral coordinate
        start_y = (left_pts[0, 1] + right_pts[0, 1]) / 2.0
        centers.append(round(start_y, 6))
    return centers


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "lane_side,start_n,end_n,expected_signs",
    [
        ("center", 2, 3, ["mixed", "mixed", "mixed"]),
        ("right", 2, 3, ["right", "right", "right"]),
        ("left", 2, 3, ["left", "left", "left"]),
        ("right", 3, 2, ["right", "right", "right"]),
        ("left", 3, 2, ["left", "left", "left"]),
        ("center", 2, 2, ["mixed", "mixed"]),
        ("right", 2, 2, ["right", "right"]),
    ],
    ids=[
        "center_2to3",
        "right_2to3",
        "left_2to3",
        "right_3to2",
        "left_3to2",
        "center_no_change",
        "right_no_change",
    ],
)
def test_lane_adapter_lane_side_position(
    lane_side: str, start_n: int, end_n: int, expected_signs: list[str]
) -> None:
    """LaneAdapter lane_side must correctly position lanes relative to centreline.

    For a straight, horizontal (heading=0) adapter, the centreline is on y=0.
    "right" means all lane centre y values are negative;
    "left" means all are positive;
    "center" means lanes straddle both sides.

    Zero-width ("added" / "dropped") lanes may have a centre at y=0; they are
    excluded from the sign check because the lane does not yet exist at the
    collapsed end.
    """
    result = LaneAdapter(change_side="right", lane_side=lane_side).build(
        _make_port(0.0, 0.0, 0.0, start_n),
        _make_port(30.0, 0.0, 0.0, end_n),
        start_lane_num=start_n,
        end_lane_num=end_n,
        id_offset=0,
    )

    assert len(result.lanes) == max(start_n, end_n)

    for lane, expected in zip(result.lanes, expected_signs):
        left_y = np.asarray(lane.left_side.coords)[0, 1]
        right_y = np.asarray(lane.right_side.coords)[0, 1]
        center_y = (left_y + right_y) / 2.0
        width = abs(left_y - right_y)

        # collapsed (zero-width) lanes may sit at y=0 — skip sign check
        if width < 1e-6:
            continue

        if expected == "right":
            assert center_y < -1e-6, f"lane {lane.id_} centre {center_y} should be < 0 (right)"
            assert left_y <= 1e-6, f"lane {lane.id_} left boundary {left_y} should be ≤ 0"
        elif expected == "left":
            assert center_y > 1e-6, f"lane {lane.id_} centre {center_y} should be > 0 (left)"
            assert right_y >= -1e-6, f"lane {lane.id_} right boundary {right_y} should be ≥ 0"
        # "mixed" — no assertion, symmetric distribution


@pytest.mark.map_generator
def test_lane_adapter_per_end_lane_side() -> None:
    """Per-end lane_side overrides allow asymmetric configurations.

    TwoWay forward → intersection: start_side="right" (road), end_side="center"
    (intersection).
    """
    result = LaneAdapter(change_side="right", lane_side="right").build(
        _make_port(0.0, 0.0, 0.0, 2),
        _make_port(30.0, 0.0, 0.0, 3),
        start_lane_num=2,
        end_lane_num=3,
        end_lane_side="center",  # intersection side stays symmetric
        id_offset=0,
    )

    # start end (road side): all lanes on the right
    for lane in result.lanes:
        left_start_y = np.asarray(lane.left_side.coords)[0, 1]
        right_start_y = np.asarray(lane.right_side.coords)[0, 1]
        assert left_start_y <= 1e-6, f"start: lane {lane.id_} left should be ≤ 0"

    # end end (intersection side): centred
    end_centers = []
    for lane in result.lanes:
        left_end_y = np.asarray(lane.left_side.coords)[-1, 1]
        right_end_y = np.asarray(lane.right_side.coords)[-1, 1]
        end_centers.append((left_end_y + right_end_y) / 2.0)

    # after shifting at start only, the end should still be symmetric
    assert any(c > 1e-6 for c in end_centers), "at least one lane centre should be > 0"
    assert any(c < -1e-6 for c in end_centers), "at least one lane centre should be < 0"


@pytest.mark.map_generator
def test_junction_approach_lane_side() -> None:
    """JunctionApproach lane_side shifts road-end lanes while keeping junction end fixed."""
    from tactics2d.map.generator.road_segment.reference_line import sample_centerline

    cl = sample_centerline(np.array([0.0, 0.0]), 0.0, 30.0)
    end_n = 3
    end_offs = np.array([end_n / 2.0 * 3.5 - i * 3.5 for i in range(end_n + 1)])

    result = JunctionApproach(change_side="both", lane_side="right").build(
        centerline=cl,
        start_lane_num=2,
        end_lane_num=end_n,
        end_boundary_offsets=end_offs,
        id_offset=0,
    )

    # road-end (start): all lanes on the right side
    for lane in result.lanes:
        left_start_y = np.asarray(lane.left_side.coords)[0, 1]
        right_start_y = np.asarray(lane.right_side.coords)[0, 1]
        center_start_y = (left_start_y + right_start_y) / 2.0
        assert (
            center_start_y < -1e-6 or abs(center_start_y) < 1e-6
        ), f"road-end lane centre {center_start_y} should be ≤ 0 (right side)"

    # junction end keeps the port offsets
    _assert_lane_line_ids_resolved(result)


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "init_kwargs,match", [({"lane_side": "diagonal"}, "lane_side must be")], ids=["bad_lane_side"]
)
def test_lane_adapter_invalid_lane_side(init_kwargs: dict, match: str) -> None:
    """Invalid lane_side must raise ValueError."""
    with pytest.raises(ValueError, match=match):
        LaneAdapter(**init_kwargs)


@pytest.mark.map_generator
def test_lane_adapter_positional_step_size_compatibility() -> None:
    """The pre-existing positional ``step_size`` argument remains supported."""
    adapter = LaneAdapter("left", 0.5)
    assert adapter.change_side == "left"
    assert adapter.step_size == pytest.approx(0.5)
    assert adapter.lane_side == "center"


@pytest.mark.map_generator
def test_junction_approach_lane_side_invalid() -> None:
    """Invalid lane_side in JunctionApproach build() must raise ValueError."""
    from tactics2d.map.generator.road_segment.reference_line import sample_centerline

    cl = sample_centerline(np.array([0.0, 0.0]), 0.0, 20.0)
    with pytest.raises(ValueError, match="lane_side must be"):
        JunctionApproach().build(
            centerline=cl,
            start_lane_num=2,
            end_lane_num=2,
            end_boundary_offsets=np.array([3.5, 1.75, 0.0]),
            start_lane_side="diagonal",
            id_offset=0,
        )


@pytest.mark.map_generator
def test_lane_adapter_centerline_marking():
    """When centerline_marking_token is set, the centreline boundary uses it."""
    result = LaneAdapter(
        change_side="right", lane_side="right", centerline_marking_token="solid_double_yellow"
    ).build(
        _make_port(0.0, 0.0, 0.0, 2),
        _make_port(30.0, 0.0, 0.0, 3),
        start_lane_num=2,
        end_lane_num=3,
        id_offset=0,
    )

    # boundary 0 (leftmost) should use the centreline marking
    rl0 = result.roadlines[0]
    assert rl0.custom_tags.get("marking_token") == "solid_double_yellow"
    assert rl0.custom_tags.get("marking_role") == "centerline"
    assert rl0.color == "yellow"

    # boundary 1 and beyond should use standard one-way markings
    rl1 = result.roadlines[1]
    assert rl1.custom_tags.get("marking_token") != "solid_double_yellow"
