# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map generator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)

_RENDER_MAPS = os.environ.get("TACTICS2D_RENDER_MAPS", "0") == "1"

from tactics2d.geometry import heading_unit
from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.map.generator.road_elements.fork import fork
from tactics2d.map.generator.road_elements.intersection import intersection
from tactics2d.map.generator.road_elements.lane_adapter import lane_adapter
from tactics2d.map.generator.road_elements.merge import merge
from tactics2d.map.generator.road_elements.one_way import one_way
from tactics2d.map.generator.road_elements.ramp import (
    entrance_ramp,
    exit_ramp,
    freeway_entrance_ramp,
    freeway_exit_ramp,
    urban_entrance_ramp,
    urban_exit_ramp,
)
from tactics2d.map.generator.road_elements.roundabout import roundabout
from tactics2d.map.generator.road_elements.two_way import two_way
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort
from tactics2d.participant.trajectory import State
from tactics2d.renderer import MatplotlibRenderer
from tactics2d.sensor import BEVCamera


def _render(map_: Map, save_to: str) -> None:
    """Render a map to a png file (only when TACTICS2D_RENDER_MAPS=1)."""
    if not _RENDER_MAPS:
        return
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=(800, 800), xlim=(boundary[0], boundary[1]), ylim=(boundary[2], boundary[3])
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=save_to)
    renderer.destroy()


def _assert_result_element_ids_unique(result: RoadModuleResult) -> None:
    """Assert that one module result has no duplicate element ids."""
    elements = [*result.lanes, *result.roadlines, *getattr(result, "junctions", [])]
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
    for junction in getattr(result, "junctions", []):
        map_.add_junction(junction)


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


def _two_way_from_interface(iface: dict, *, length: float, id_offset: int) -> RoadModuleResult:
    """Attach a two-way road to an intersection/roundabout interface dict."""
    lane_width = iface.get("lane_width", 3.5)
    speed_limit = iface.get("speed_limit", 50.0)
    start_port = RoadPort(
        point=iface["point"],
        heading=iface["heading"],
        lane_num=iface["lane_num"],
        lane_width=lane_width,
        speed_limit=speed_limit,
    )
    end_port = RoadPort(
        point=iface["point"] + length * heading_unit(iface["heading"]),
        heading=iface["heading"],
        lane_num=iface["lane_num"],
        lane_width=lane_width,
        speed_limit=speed_limit,
    )
    return two_way(
        start_port,
        end_port,
        forward_lane_num=iface["lane_num"],
        backward_lane_num=iface["lane_num"],
        lane_width=lane_width,
        speed_limit=speed_limit,
        id_offset=id_offset,
    )


def _assert_no_yellow_centerline(result: RoadModuleResult) -> None:
    yellow_centerlines = [
        rl
        for rl in result.roadlines
        if rl.color == "yellow" and rl.custom_tags.get("marking_role") == "centerline"
    ]
    assert len(yellow_centerlines) == 0


def _assert_has_yellow_centerline(result: RoadModuleResult) -> None:
    yellow_lines = [rl for rl in result.roadlines if rl.color == "yellow"]
    assert len(yellow_lines) >= 1
    assert any(rl.custom_tags.get("marking_role") == "centerline" for rl in yellow_lines)


def _assert_has_ramp_auxiliary_line(result: RoadModuleResult) -> None:
    aux_lines = [
        rl for rl in result.roadlines if rl.custom_tags.get("marking_token") == "dashed_white_ramp"
    ]
    assert len(aux_lines) >= 1
    assert all(rl.subtype == "dashed" for rl in aux_lines)
    assert all(rl.color == "white" for rl in aux_lines)


def _assert_ramp_ports(result: RoadModuleResult) -> None:
    """Structural ramp assertions: required ports exist and result is accepted."""
    assert isinstance(result, RoadModuleResult)
    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "ramp" in result.ports
    assert result.quality["accepted"] is True
    assert result.id_counter > 0


def _assert_freeway_ramp_common(result: RoadModuleResult, ramp_side: str) -> None:
    _assert_ramp_ports(result)
    _assert_no_yellow_centerline(result)
    _assert_has_ramp_auxiliary_line(result)
    assert result.quality["ramp_side"] == ramp_side
    assert "backward_in" not in result.ports
    assert "backward_out" not in result.ports


def _assert_urban_ramp_common(result: RoadModuleResult) -> None:
    _assert_ramp_ports(result)
    _assert_has_yellow_centerline(result)
    _assert_has_ramp_auxiliary_line(result)
    assert result.quality["ramp_side"] == "right"
    assert "backward_in" in result.ports
    assert "backward_out" in result.ports


def _assert_roundabout_seam_edges(result: RoadModuleResult) -> None:
    edge_lines = [rl for rl in result.roadlines if rl.custom_tags.get("role") == "arm_outer_edge"]
    expected = 2 * result.quality["arm_num"]
    assert result.quality["arm_outer_edge_count"] == expected
    assert len(edge_lines) == expected
    assert all(rl.type_ != "virtual" for rl in edge_lines)
    assert all(rl.color == "white" for rl in edge_lines)


def _assert_roundabout_connector_boundaries_virtual(result: RoadModuleResult) -> None:
    connector_lines = [
        rl
        for rl in result.roadlines
        if rl.custom_tags.get("role") in {"entry_connection_boundary", "exit_connection_boundary"}
    ]
    assert len(connector_lines) >= 1
    assert all(rl.type_ == "virtual" for rl in connector_lines)


@pytest.mark.map_generator
def test_parking_lot_generator():
    map_generator = ParkingLotGenerator()
    map_ = Map(name="parking_lot", scenario_type="parking")
    start_state, target_area, target_heading = map_generator.generate(map_)
    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    matplotlib_renderer = MatplotlibRenderer(
        resolution=((boundary[1] - boundary[0]) * 100, (boundary[3] - boundary[2]) * 100),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    matplotlib_renderer.update(geometry_data)
    matplotlib_renderer.save_single_frame(save_to="./tests/runtime/parking_lot.png")
    assert isinstance(start_state, State)
    assert isinstance(target_area, Area)
    assert isinstance(target_heading, float)


@pytest.mark.map_generator
def test_racing_track_generator():
    map_generator = RacingTrackGenerator()
    map_ = Map(name="racing_track", scenario_type="racing")
    map_generator.generate(map_)
    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    matplotlib_renderer = MatplotlibRenderer(
        resolution=((boundary[1] - boundary[0]) * 10, (boundary[3] - boundary[2]) * 10),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    matplotlib_renderer.update(geometry_data)
    matplotlib_renderer.save_single_frame(save_to="./tests/runtime/racing_track.png")
    assert isinstance(map_.customs["start_state"], State)


@pytest.mark.map_generator
def test_one_way_straight():
    map_ = Map(name="one_way_straight")
    start_port, end_port = _port_from_start_length(
        start=np.array([0.0, 0.0]), heading=0.0, length=60.0, lane_num=3, speed_limit=50.0
    )
    result = one_way(start_port, end_port, lane_num=3, id_offset=0)
    _add_result(map_, result)
    _render(map_, "./tests/runtime/one_way_straight.png")
    assert len(map_.lanes) == 3
    assert len(map_.roadlines) >= 4
    assert "entry" in result.ports
    assert "exit" in result.ports
    assert result.ports["exit"].lane_num == 3


@pytest.mark.map_generator
def test_one_way_curved():
    map_ = Map(name="one_way_curved")
    result = one_way(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(30.0, 15.0, 0.65, 2, speed_limit=50.0),
        lane_num=2,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, "./tests/runtime/one_way_curved.png")
    assert len(map_.lanes) == 2
    assert isinstance(result.ports["exit"].point, np.ndarray)
    assert isinstance(result.ports["exit"].heading, float)
    assert result.quality["self_intersection"] is False


@pytest.mark.map_generator
def test_two_way_straight():
    map_ = Map(name="two_way_straight")
    start_port, end_port = _port_from_start_length(
        start=np.array([0.0, 0.0]), heading=0.0, length=60.0, lane_num=2, speed_limit=50.0
    )
    result = two_way(start_port, end_port, forward_lane_num=2, backward_lane_num=2, id_offset=0)
    _add_result(map_, result)
    _render(map_, "./tests/runtime/two_way_straight.png")
    assert len(map_.lanes) == 4
    assert "forward_in" in result.ports
    assert "forward_out" in result.ports
    assert "backward_in" in result.ports
    assert "backward_out" in result.ports


@pytest.mark.map_generator
def test_two_way_curved():
    map_ = Map(name="two_way_curved")
    result = two_way(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(55.0, 25.0, 0.45, 2, speed_limit=50.0),
        forward_lane_num=2,
        backward_lane_num=2,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, "./tests/runtime/two_way_curved.png")
    assert len(map_.lanes) == 4
    assert result.quality["self_intersection"] is False


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "start_n,end_n,change_side",
    [(2, 3, "right"), (3, 2, "right"), (2, 3, "left"), (3, 2, "left"), (2, 2, "right")],
    ids=["expand_right", "reduce_right", "expand_left", "reduce_left", "no_change"],
)
def test_lane_adapter(start_n: int, end_n: int, change_side: str) -> None:
    map_ = Map(name=f"lane_adapter_{start_n}to{end_n}_{change_side}")
    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, start_n, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, end_n, speed_limit=50.0),
        change_side=change_side,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, f"./tests/runtime/lane_adapter_{start_n}to{end_n}_{change_side}.png")

    max_n = max(start_n, end_n)
    assert "entry" in result.ports
    assert "exit" in result.ports
    assert result.ports["entry"].lane_num == start_n
    assert result.ports["exit"].lane_num == end_n
    assert len(result.lanes) == max_n
    assert len(result.roadlines) == max_n + 1
    assert result.quality["accepted"] is True


@pytest.mark.map_generator
def test_lane_adapter_no_id_collision() -> None:
    result_1 = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 2),
        _make_port(80.0, 0.0, 0.0, 3),
        change_side="right",
        id_offset=0,
    )
    result_2 = lane_adapter(
        _make_port(0.0, 20.0, 0.0, 3),
        _make_port(80.0, 20.0, 0.0, 2),
        change_side="left",
        id_offset=result_1.id_counter,
    )
    ids_1 = {e.id_ for e in result_1.lanes + result_1.roadlines}
    ids_2 = {e.id_ for e in result_2.lanes + result_2.roadlines}
    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter


@pytest.mark.map_generator
def test_lane_adapter_quality_fields() -> None:
    """Smoke test: quality dict for a representative expand-right case."""
    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, 3, speed_limit=50.0),
        change_side="right",
        id_offset=0,
    )
    assert result.quality["module"] == "lane_adapter"
    assert result.quality["start_lane_num"] == 2
    assert result.quality["end_lane_num"] == 3
    assert result.quality["lane_delta"] == 1
    assert result.quality["change_side"] == "right"
    assert len(result.quality["added_lane_ids"]) == 1
    assert len(result.quality["dropped_lane_ids"]) == 0
    assert len(result.ports["entry"].lane_ids) == 2
    assert len(result.ports["exit"].lane_ids) == 3


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
    result = fork(
        _make_port(0.0, 0.0, 0.0, main_n, speed_limit=60.0),
        _make_port(main_out_x, 0.0, 0.0, main_n, speed_limit=60.0),
        _make_port(branch_x, branch_y, branch_h, branch_n, speed_limit=40.0),
        fork_side=fork_side,
        branch_lane_num=branch_n,
        taper_length=taper_length,
        branch_length=branch_length,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, f"./tests/runtime/fork_{fork_side}_{main_n}m_{branch_n}b.png")

    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "branch_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == main_n
    assert len(result.ports["main_out"].lane_ids) == main_n
    assert len(result.ports["branch_out"].lane_ids) == branch_n
    assert len(result.lanes) == main_n + branch_n
    assert result.quality["accepted"] is True
    assert result.id_counter > 0


@pytest.mark.map_generator
def test_fork_no_id_collision() -> None:
    result_1 = fork(
        _make_port(0.0, 0.0, 0.0, 3),
        _make_port(130.0, 0.0, 0.0, 3),
        _make_port(95.0, -45.0, -0.55, 1),
        fork_side="right",
        id_offset=0,
    )
    result_2 = fork(
        _make_port(0.0, 80.0, 0.0, 3),
        _make_port(130.0, 80.0, 0.0, 3),
        _make_port(95.0, 125.0, 0.55, 1),
        fork_side="left",
        id_offset=result_1.id_counter,
    )
    ids_1 = {e.id_ for e in result_1.lanes + result_1.roadlines}
    ids_2 = {e.id_ for e in result_2.lanes + result_2.roadlines}
    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter


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
    result = merge(
        _make_port(0.0, 0.0, 0.0, main_n, speed_limit=60.0),
        _make_port(branch_x, branch_y, branch_h, branch_n, speed_limit=40.0),
        _make_port(main_out_x, 0.0, 0.0, main_n, speed_limit=60.0),
        merge_side=merge_side,
        branch_lane_num=branch_n,
        taper_length=taper_length,
        branch_length=branch_length,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, f"./tests/runtime/merge_{merge_side}_{main_n}m_{branch_n}b.png")

    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == main_n
    assert len(result.ports["branch_in"].lane_ids) == branch_n
    assert len(result.ports["main_out"].lane_ids) == main_n
    assert len(result.lanes) == main_n + branch_n
    assert result.quality["accepted"] is True
    assert result.id_counter > 0


@pytest.mark.map_generator
def test_merge_tail() -> None:
    """Late-merge variant: merge_s_ratio near 1 forces branch join close to main_out."""
    map_ = Map(name="merge_tail")
    result = merge(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(95.0, -45.0, np.pi / 2 - 0.4, 1, speed_limit=40.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        merge_side="right",
        merge_s_ratio=0.92,
        taper_length=20.0,
        branch_length=50.0,
        id_offset=0,
    )
    _add_result(map_, result)
    _render(map_, "./tests/runtime/merge_tail.png")
    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports
    assert len(result.lanes) == 4
    assert result.quality["accepted"] is True


@pytest.mark.map_generator
def test_merge_no_id_collision() -> None:
    result_1 = merge(
        _make_port(0.0, 0.0, 0.0, 3),
        _make_port(35.0, -45.0, 0.55, 1),
        _make_port(130.0, 0.0, 0.0, 3),
        merge_side="right",
        id_offset=0,
    )
    result_2 = merge(
        _make_port(0.0, 80.0, 0.0, 3),
        _make_port(35.0, 125.0, -0.55, 1),
        _make_port(130.0, 80.0, 0.0, 3),
        merge_side="left",
        id_offset=result_1.id_counter,
    )
    ids_1 = {e.id_ for e in result_1.lanes + result_1.roadlines}
    ids_2 = {e.id_ for e in result_2.lanes + result_2.roadlines}
    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter


@pytest.mark.map_generator
def test_intersection_cross():
    map_ = Map(name="intersection_cross")
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]]
    result = intersection(center=np.array([0.0, 0.0]), arms=arms, radius=10.0, id_offset=0)
    _add_result(map_, result)
    id_off = result.id_counter + 1000
    for iface in result.interfaces:
        road_result = _two_way_from_interface(iface, length=30.0, id_offset=id_off)
        id_off = road_result.id_counter
        _add_result(map_, road_result)
    _render(map_, "./tests/runtime/intersection_cross.png")
    assert result.quality["module"] == "intersection"
    assert result.quality["arm_num"] == 4
    assert len(result.interfaces) == 4
    assert len(map_.junctions) == 1
    assert result.quality["accepted"] is True


@pytest.mark.map_generator
def test_intersection_t():
    map_ = Map(name="intersection_t")
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi]]
    result = intersection(center=np.array([0.0, 0.0]), arms=arms, radius=10.0, id_offset=0)
    _add_result(map_, result)
    id_off = result.id_counter + 1000
    for iface in result.interfaces:
        road_result = _two_way_from_interface(iface, length=30.0, id_offset=id_off)
        id_off = road_result.id_counter
        _add_result(map_, road_result)
    _render(map_, "./tests/runtime/intersection_t.png")
    assert result.quality["module"] == "intersection"
    assert result.quality["arm_num"] == 3
    assert len(result.interfaces) == 3
    assert len(map_.junctions) == 1
    assert result.quality["accepted"] is True


@pytest.mark.map_generator
def test_intersection_cross_curved():
    map_ = Map(name="intersection_cross_curved")
    arms = [
        {"heading": np.pi / 2, "lane_num": 2, "curvature": 0.0, "radius": 12.0},
        {"heading": np.pi, "lane_num": 2, "curvature": 0.02, "radius": 12.0},
        {"heading": -np.pi / 2, "lane_num": 2, "curvature": 0.0, "radius": 12.0},
        {"heading": 0.0, "lane_num": 2, "curvature": -0.02, "radius": 12.0},
    ]
    result = intersection(center=np.array([0.0, 0.0]), arms=arms, id_offset=0)
    _add_result(map_, result)
    id_off = result.id_counter + 1000
    for iface in result.interfaces:
        road_result = _two_way_from_interface(iface, length=30.0, id_offset=id_off)
        id_off = road_result.id_counter
        _add_result(map_, road_result)
    _render(map_, "./tests/runtime/intersection_cross_curved.png")
    assert result.quality["arm_num"] == 4
    assert len(result.interfaces) == 4
    assert len(map_.junctions) == 1


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "arm_headings,arm_num",
    [([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], 4), ([0.0, 2 * np.pi / 3, 4 * np.pi / 3], 3)],
    ids=["4arm", "3arm"],
)
def test_roundabout(arm_headings: list, arm_num: int) -> None:
    map_ = Map(name=f"roundabout_{arm_num}arm")
    arms = [{"heading": h, "lane_num": 2} for h in arm_headings]
    result = roundabout(
        center=np.array([0.0, 0.0]), arms=arms, ring_radius=12.0, ring_lane_num=2, id_offset=0
    )
    _add_result(map_, result)
    id_off = result.id_counter + 1000
    for iface in result.interfaces:
        road_result = _two_way_from_interface(iface, length=30.0, id_offset=id_off)
        id_off = road_result.id_counter
        _add_result(map_, road_result)
    _render(map_, f"./tests/runtime/roundabout_{arm_num}arm.png")

    assert result.quality["module"] == "roundabout"
    assert result.quality["arm_num"] == arm_num
    assert result.quality["ring_radius"] == 12.0
    assert result.quality["ring_lane_num"] == 2
    assert len(result.interfaces) == arm_num
    assert len(result.junctions) == 1
    assert result.quality["accepted"] is True
    _assert_roundabout_seam_edges(result)
    _assert_roundabout_connector_boundaries_virtual(result)
    expected_radius = result.quality["outer_ring_radius"]
    for iface in result.interfaces:
        assert np.isclose(np.linalg.norm(iface["point"]), expected_radius)


@pytest.mark.map_generator
def test_roundabout_curved_approach():
    map_ = Map(name="roundabout_curved_approach")
    arms = [
        {"heading": 0.0, "lane_num": 2, "curvature": 0.03},
        {"heading": np.pi / 2, "lane_num": 2, "curvature": 0.0},
        {"heading": np.pi, "lane_num": 2, "curvature": -0.03},
        {"heading": 3 * np.pi / 2, "lane_num": 2, "curvature": 0.0},
    ]
    result = roundabout(
        center=np.array([0.0, 0.0]), arms=arms, ring_radius=12.0, ring_lane_num=2, id_offset=0
    )
    _add_result(map_, result)
    expected_radius = result.quality["outer_ring_radius"]
    for iface in result.interfaces:
        assert np.isclose(np.linalg.norm(iface["point"]), expected_radius)

    id_off = result.id_counter + 1000
    for i, iface in enumerate(result.interfaces):
        h = iface["heading"]
        lane_width = iface.get("lane_width", 3.5)
        speed_limit = iface.get("speed_limit", 50.0)
        normal = np.array([-np.sin(h), np.cos(h)], dtype=float)
        curve_sign = 1.0 if i % 2 == 0 else -1.0
        start_port = RoadPort(
            point=iface["point"],
            heading=h,
            lane_num=iface["lane_num"],
            lane_width=lane_width,
            speed_limit=speed_limit,
        )
        end_port = RoadPort(
            point=iface["point"] + 32.0 * heading_unit(h) + curve_sign * 7.0 * normal,
            heading=h + curve_sign * 0.22,
            lane_num=iface["lane_num"],
            lane_width=lane_width,
            speed_limit=speed_limit,
        )
        road_result = two_way(
            start_port,
            end_port,
            forward_lane_num=iface["lane_num"],
            backward_lane_num=iface["lane_num"],
            lane_width=lane_width,
            speed_limit=speed_limit,
            id_offset=id_off,
        )
        id_off = road_result.id_counter
        _add_result(map_, road_result)

    _render(map_, "./tests/runtime/roundabout_curved_approach.png")
    assert result.quality["module"] == "roundabout"
    assert result.quality["arm_num"] == 4
    assert len(result.junctions) == 1
    assert result.quality["accepted"] is True
    _assert_roundabout_seam_edges(result)
    _assert_roundabout_connector_boundaries_virtual(result)


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
def test_freeway_ramp(kind: str, ramp_side: str, ramp_y: float, ramp_h: float) -> None:
    ramp_x = 330.0 if kind == "exit" else 90.0
    main_in = _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0)
    main_out = _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0)
    ramp_port = _make_port(ramp_x, ramp_y, ramp_h, 1, speed_limit=50.0)

    if kind == "exit":
        result = freeway_exit_ramp(
            main_in,
            main_out,
            ramp_port,
            ramp_side=ramp_side,
            taper_length=70.0,
            parallel_length=90.0,
            id_offset=0,
        )
    else:
        result = freeway_entrance_ramp(
            main_in,
            main_out,
            ramp_port,
            ramp_side=ramp_side,
            taper_length=70.0,
            parallel_length=90.0,
            id_offset=0,
        )

    map_ = Map(name=f"freeway_{kind}_{ramp_side}")
    _add_result(map_, result)
    _render(map_, f"./tests/runtime/freeway_{kind}_{ramp_side}.png")

    _assert_freeway_ramp_common(result, ramp_side)
    expected_ramp_kind = "ramp_out" if kind == "exit" else "ramp_in"
    assert result.ports["ramp"].kind == expected_ramp_kind


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "kind,ramp_x,ramp_y,ramp_h",
    [("exit", 255.0, -75.0, -0.90), ("entrance", 65.0, -75.0, 0.90)],
    ids=["exit", "entrance"],
)
def test_urban_ramp(kind: str, ramp_x: float, ramp_y: float, ramp_h: float) -> None:
    main_in = _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0)
    main_out = _make_port(320.0, 16.0, 0.04, 2, speed_limit=80.0)
    ramp_port = _make_port(ramp_x, ramp_y, ramp_h, 1, speed_limit=40.0)

    if kind == "exit":
        result = urban_exit_ramp(
            main_in,
            main_out,
            ramp_port,
            backward_lane_num=1,
            taper_length=55.0,
            parallel_length=75.0,
            id_offset=0,
        )
    else:
        result = urban_entrance_ramp(
            main_in,
            main_out,
            ramp_port,
            backward_lane_num=1,
            taper_length=55.0,
            parallel_length=75.0,
            id_offset=0,
        )

    map_ = Map(name=f"urban_{kind}")
    _add_result(map_, result)
    _render(map_, f"./tests/runtime/urban_{kind}.png")

    _assert_urban_ramp_common(result)
    expected_ramp_kind = "ramp_out" if kind == "exit" else "ramp_in"
    assert result.ports["ramp"].kind == expected_ramp_kind


@pytest.mark.map_generator
def test_ramp_unified_interface() -> None:
    """exit_ramp / entrance_ramp route to the correct freeway/urban variant."""
    r_fw = exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(330.0, -95.0, -0.95, 1, speed_limit=50.0),
        main_road_type="freeway",
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )
    _assert_freeway_ramp_common(r_fw, "right")
    assert r_fw.ports["main_in"].lane_num == 3
    assert r_fw.ports["ramp"].lane_num == 1

    r_urb = entrance_ramp(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0),
        _make_port(320.0, 16.0, 0.04, 2, speed_limit=80.0),
        _make_port(65.0, -75.0, 0.90, 1, speed_limit=40.0),
        main_road_type="urban",
        ramp_side="right",
        backward_lane_num=1,
        taper_length=55.0,
        parallel_length=75.0,
        id_offset=0,
    )
    _assert_urban_ramp_common(r_urb)
    assert r_urb.ports["main_in"].lane_num == 2
    assert r_urb.ports["ramp"].lane_num == 1


@pytest.mark.map_generator
def test_ramp_no_id_collision() -> None:
    r1 = freeway_exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3),
        _make_port(420.0, 0.0, 0.0, 3),
        _make_port(330.0, -95.0, -0.95, 1),
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )
    r2 = freeway_entrance_ramp(
        _make_port(0.0, 120.0, 0.0, 3),
        _make_port(420.0, 120.0, 0.0, 3),
        _make_port(90.0, 25.0, 0.95, 1),
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=r1.id_counter,
    )
    ids1 = {e.id_ for e in r1.lanes + r1.roadlines}
    ids2 = {e.id_ for e in r2.lanes + r2.roadlines}
    assert len(ids1 & ids2) == 0


@pytest.mark.map_generator
def test_ramp_result_interface() -> None:
    """Smoke test: quality dict structure for a freeway exit ramp."""
    result = freeway_exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3),
        _make_port(420.0, 0.0, 0.0, 3),
        _make_port(330.0, -95.0, -0.95, 1),
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )
    assert result.quality["module"] == "ramp"
    assert result.quality["kind"] == "exit"
    assert result.quality["main_road_type"] == "freeway"
    assert result.quality["connector_length"] > 0.0
    assert result.quality["main_self_intersection"] is False
    assert result.quality["connector_self_intersection"] is False
    assert isinstance(result.quality["accepted_reasons"], list)
    assert "connector_min_radius" in result.quality
    assert "connector_max_abs_curvature" in result.quality
    assert "connector_max_abs_curvature_rate" in result.quality
    assert isinstance(result.ports["main_out"].point, np.ndarray)
    assert result.ports["main_out"].lane_num == 3
    assert result.ports["ramp"].lane_num == 1
    assert result.id_counter > 0


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "fork_side,branch_lane_num,match",
    [("center", 1, "fork_side must be"), ("right", 4, "branch_lane_num <= main_lane_num")],
    ids=["bad_fork_side", "branch_gt_main"],
)
def test_fork_invalid_args(fork_side: str, branch_lane_num: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fork(
            _make_port(0.0, 0.0, 0.0, 3),
            _make_port(130.0, 0.0, 0.0, 3),
            _make_port(95.0, -45.0, -0.55, 1),
            fork_side=fork_side,
            branch_lane_num=branch_lane_num,
            taper_length=35.0,
            branch_length=55.0,
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
        merge(
            _make_port(0.0, 0.0, 0.0, 3),
            _make_port(35.0, -45.0, 0.55, 1),
            _make_port(130.0, 0.0, 0.0, 3),
            merge_side=merge_side,
            branch_lane_num=branch_lane_num,
            taper_length=35.0,
            branch_length=55.0,
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
        lane_adapter(
            _make_port(0.0, 0.0, 0.0, start_n),
            _make_port(60.0, 0.0, 0.0, end_n),
            change_side=change_side,
            step_size=step_size,
            id_offset=0,
        )


@pytest.mark.map_generator
def test_ramp_urban_left_side_raises() -> None:
    """Urban ramp only supports ramp_side='right'; 'left' must raise."""
    with pytest.raises(ValueError, match="urban ramp currently supports only ramp_side='right'"):
        exit_ramp(
            _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0),
            _make_port(320.0, 0.0, 0.0, 2, speed_limit=80.0),
            _make_port(255.0, 75.0, -0.90, 1, speed_limit=40.0),
            main_road_type="urban",
            ramp_side="left",
            taper_length=55.0,
            parallel_length=75.0,
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
        intersection(center=np.array([0.0, 0.0]), arms=arm_dicts, id_offset=0)


@pytest.mark.map_generator
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"arms": [{"heading": 0.0, "lane_num": 2}]}, "at least 2 arms"),
        ({"ring_lane_num": 0}, "ring_lane_num must be"),
        ({"ring_radius": -1.0}, "ring_radius must be positive"),
        ({"step_size": 0.0}, "step_size must be positive"),
    ],
    ids=["one_arm", "zero_ring_lanes", "negative_radius", "zero_step"],
)
def test_roundabout_invalid_args(kwargs: dict, match: str) -> None:
    base = {
        "center": np.array([0.0, 0.0]),
        "arms": [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi]],
        "ring_radius": 12.0,
        "ring_lane_num": 1,
        "step_size": 0.1,
        "id_offset": 0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        roundabout(**base)


@pytest.mark.map_generator
def test_intersection_no_id_collision() -> None:
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]]
    result_1 = intersection(center=np.array([0.0, 0.0]), arms=arms, radius=10.0, id_offset=0)
    result_2 = intersection(
        center=np.array([100.0, 0.0]), arms=arms, radius=10.0, id_offset=result_1.id_counter
    )
    ids_1 = {e.id_ for e in result_1.lanes + result_1.roadlines}
    ids_2 = {e.id_ for e in result_2.lanes + result_2.roadlines}
    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter
