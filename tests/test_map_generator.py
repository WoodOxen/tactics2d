# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map generator."""

import sys

sys.path.append(".")
sys.path.append("..")

import logging

import numpy as np
import pytest
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)

from tactics2d.map.element import Area, Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.map.generator.geometry.module_geometry import unit
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
    """Render a map to a png file."""
    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=(800, 800), xlim=(boundary[0], boundary[1]), ylim=(boundary[2], boundary[3])
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=save_to)
    renderer.destroy()


def _add_result(map_: Map, result: RoadModuleResult) -> None:
    """Add a RoadModuleResult into a Tactics2D Map."""
    for lane in result.lanes:
        map_.add_lane(lane)
    for roadline in result.roadlines:
        map_.add_roadline(roadline)
    for junction in getattr(result, "junctions", []):
        map_.add_junction(junction)


def _add_elements(map_: Map, lanes, roadlines, junction=None) -> None:
    """Add legacy tuple-style module outputs into a Tactics2D Map."""
    for lane in lanes:
        map_.add_lane(lane)
    for roadline in roadlines:
        map_.add_roadline(roadline)
    if junction is not None:
        map_.add_junction(junction)


def _make_port(
    x: float,
    y: float,
    heading: float,
    lane_num: int,
    lane_width: float = 3.5,
    speed_limit: float = 50.0,
) -> RoadPort:
    """Create a RoadPort."""
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
    """Create two ports from a start pose and a straight-line length."""
    start_port = RoadPort(
        point=start,
        heading=heading,
        lane_num=lane_num,
        lane_width=lane_width,
        speed_limit=speed_limit,
    )
    end_port = RoadPort(
        point=start + length * unit(heading),
        heading=heading,
        lane_num=lane_num,
        lane_width=lane_width,
        speed_limit=speed_limit,
    )
    return start_port, end_port


def _two_way_from_interface(iface: dict, *, length: float, id_offset: int) -> RoadModuleResult:
    """Attach a two-way road to an intersection/roundabout interface."""
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
        point=iface["point"] + length * unit(iface["heading"]),
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
    """Assert that a freeway ramp has no yellow centerline."""
    yellow_centerlines = [
        roadline
        for roadline in result.roadlines
        if roadline.color == "yellow" and roadline.custom_tags.get("marking_role") == "centerline"
    ]
    assert len(yellow_centerlines) == 0


def _assert_has_yellow_centerline(result: RoadModuleResult) -> None:
    """Assert that an urban two-way ramp has a yellow centerline."""
    yellow_lines = [roadline for roadline in result.roadlines if roadline.color == "yellow"]
    assert len(yellow_lines) >= 1
    assert any(
        roadline.custom_tags.get("marking_role") == "centerline"
        or roadline.custom_tags.get("role") == "centerline"
        for roadline in yellow_lines
    )


def _assert_has_ramp_auxiliary_line(result: RoadModuleResult) -> None:
    """Assert that a ramp has a dashed auxiliary merge/diverge line."""
    aux_lines = [
        roadline
        for roadline in result.roadlines
        if roadline.custom_tags.get("marking_token") == "dashed_white_ramp"
    ]
    assert len(aux_lines) >= 1
    assert all(roadline.subtype == "dashed" for roadline in aux_lines)
    assert all(roadline.color == "white" for roadline in aux_lines)


def _assert_ramp_result_basic(result: RoadModuleResult, kind: str, main_road_type: str) -> None:
    """Assert basic ramp result interface and quality fields."""
    assert isinstance(result, RoadModuleResult)
    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "ramp" in result.ports
    assert result.quality["module"] == "ramp"
    assert result.quality["kind"] == kind
    assert result.quality["main_road_type"] == main_road_type
    assert result.quality["connector_length"] > 0.0
    assert result.quality["main_self_intersection"] is False
    assert result.quality["connector_self_intersection"] is False
    assert isinstance(result.quality["accepted_reasons"], list)
    assert result.id_counter > 0


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
    assert len(map_.roadlines) == 6
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
def test_lane_adapter_expand_right():
    map_ = Map(name="lane_adapter_expand_right")

    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, 3, speed_limit=50.0),
        change_side="right",
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/lane_adapter_expand_right.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "lane_adapter"
    assert result.quality["start_lane_num"] == 2
    assert result.quality["end_lane_num"] == 3
    assert result.quality["lane_delta"] == 1
    assert result.quality["change_side"] == "right"
    assert result.quality["accepted"] is True

    assert len(result.lanes) == 3
    assert len(result.roadlines) == 4
    assert len(result.ports["entry"].lane_ids) == 2
    assert len(result.ports["exit"].lane_ids) == 3
    assert len(result.quality["added_lane_ids"]) == 1
    assert len(result.quality["dropped_lane_ids"]) == 0
    assert result.ports["entry"].lane_num == 2
    assert result.ports["exit"].lane_num == 3


@pytest.mark.map_generator
def test_lane_adapter_reduce_right():
    map_ = Map(name="lane_adapter_reduce_right")

    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, 2, speed_limit=50.0),
        change_side="right",
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/lane_adapter_reduce_right.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "lane_adapter"
    assert result.quality["start_lane_num"] == 3
    assert result.quality["end_lane_num"] == 2
    assert result.quality["lane_delta"] == -1
    assert result.quality["change_side"] == "right"
    assert result.quality["accepted"] is True

    assert len(result.lanes) == 3
    assert len(result.roadlines) == 4
    assert len(result.ports["entry"].lane_ids) == 3
    assert len(result.ports["exit"].lane_ids) == 2
    assert len(result.quality["added_lane_ids"]) == 0
    assert len(result.quality["dropped_lane_ids"]) == 1
    assert result.ports["entry"].lane_num == 3
    assert result.ports["exit"].lane_num == 2


@pytest.mark.map_generator
def test_lane_adapter_expand_left():
    map_ = Map(name="lane_adapter_expand_left")

    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, 3, speed_limit=50.0),
        change_side="left",
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/lane_adapter_expand_left.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "lane_adapter"
    assert result.quality["start_lane_num"] == 2
    assert result.quality["end_lane_num"] == 3
    assert result.quality["lane_delta"] == 1
    assert result.quality["change_side"] == "left"
    assert result.quality["accepted"] is True

    assert len(result.lanes) == 3
    assert len(result.roadlines) == 4
    assert len(result.ports["entry"].lane_ids) == 2
    assert len(result.ports["exit"].lane_ids) == 3
    assert len(result.quality["added_lane_ids"]) == 1
    assert len(result.quality["dropped_lane_ids"]) == 0
    assert result.ports["entry"].lane_num == 2
    assert result.ports["exit"].lane_num == 3


@pytest.mark.map_generator
def test_lane_adapter_reduce_left():
    map_ = Map(name="lane_adapter_reduce_left")

    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=50.0),
        _make_port(80.0, 0.0, 0.0, 2, speed_limit=50.0),
        change_side="left",
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/lane_adapter_reduce_left.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "lane_adapter"
    assert result.quality["start_lane_num"] == 3
    assert result.quality["end_lane_num"] == 2
    assert result.quality["lane_delta"] == -1
    assert result.quality["change_side"] == "left"
    assert result.quality["accepted"] is True

    assert len(result.lanes) == 3
    assert len(result.roadlines) == 4
    assert len(result.ports["entry"].lane_ids) == 3
    assert len(result.ports["exit"].lane_ids) == 2
    assert len(result.quality["added_lane_ids"]) == 0
    assert len(result.quality["dropped_lane_ids"]) == 1
    assert result.ports["entry"].lane_num == 3
    assert result.ports["exit"].lane_num == 2


@pytest.mark.map_generator
def test_lane_adapter_no_change():
    map_ = Map(name="lane_adapter_no_change")

    result = lane_adapter(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=50.0),
        _make_port(80.0, 10.0, 0.15, 2, speed_limit=50.0),
        change_side="right",
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/lane_adapter_no_change.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "lane_adapter"
    assert result.quality["start_lane_num"] == 2
    assert result.quality["end_lane_num"] == 2
    assert result.quality["lane_delta"] == 0
    assert result.quality["accepted"] is True

    assert len(result.lanes) == 2
    assert len(result.roadlines) == 3
    assert len(result.ports["entry"].lane_ids) == 2
    assert len(result.ports["exit"].lane_ids) == 2
    assert len(result.quality["added_lane_ids"]) == 0
    assert len(result.quality["dropped_lane_ids"]) == 0


@pytest.mark.map_generator
def test_lane_adapter_no_id_collision():
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

    ids_1 = {element.id_ for element in result_1.lanes + result_1.roadlines}
    ids_2 = {element.id_ for element in result_2.lanes + result_2.roadlines}

    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter


@pytest.mark.map_generator
def test_fork_right():
    map_ = Map(name="fork_right")

    result = fork(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(95.0, -45.0, -0.55, 1, speed_limit=40.0),
        fork_side="right",
        taper_length=35.0,
        branch_length=55.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/fork_right.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "fork"
    assert result.quality["fork_side"] == "right"
    assert result.quality["main_lane_num"] == 3
    assert result.quality["branch_lane_num"] == 1
    assert result.quality["accepted"] is True

    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "branch_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == 3
    assert len(result.ports["main_out"].lane_ids) == 3
    assert len(result.ports["branch_out"].lane_ids) == 1
    assert len(result.lanes) == 4
    assert result.id_counter > 0


@pytest.mark.map_generator
def test_fork_left():
    map_ = Map(name="fork_left")

    result = fork(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(95.0, 45.0, 0.55, 1, speed_limit=40.0),
        fork_side="left",
        taper_length=35.0,
        branch_length=55.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/fork_left.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "fork"
    assert result.quality["fork_side"] == "left"
    assert result.quality["main_lane_num"] == 3
    assert result.quality["branch_lane_num"] == 1
    assert result.quality["accepted"] is True

    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "branch_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == 3
    assert len(result.ports["main_out"].lane_ids) == 3
    assert len(result.ports["branch_out"].lane_ids) == 1
    assert len(result.lanes) == 4


@pytest.mark.map_generator
def test_fork_two_lane_branch():
    map_ = Map(name="fork_two_lane_branch")

    result = fork(
        _make_port(0.0, 0.0, 0.0, 4, speed_limit=60.0),
        _make_port(150.0, 0.0, 0.0, 4, speed_limit=60.0),
        _make_port(105.0, -55.0, -0.55, 2, speed_limit=40.0),
        fork_side="right",
        branch_lane_num=2,
        taper_length=40.0,
        branch_length=65.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/fork_two_lane_branch.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "fork"
    assert result.quality["fork_side"] == "right"
    assert result.quality["main_lane_num"] == 4
    assert result.quality["branch_lane_num"] == 2
    assert result.quality["accepted"] is True
    assert len(result.ports["main_in"].lane_ids) == 4
    assert len(result.ports["main_out"].lane_ids) == 4
    assert len(result.ports["branch_out"].lane_ids) == 2
    assert len(result.lanes) == 6


@pytest.mark.map_generator
def test_fork_no_id_collision():
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

    ids_1 = {element.id_ for element in result_1.lanes + result_1.roadlines}
    ids_2 = {element.id_ for element in result_2.lanes + result_2.roadlines}

    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter


@pytest.mark.map_generator
def test_fork_tail():
    map_ = Map(name="fork_tail")

    result = fork(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(130.0, -45.0, -np.pi / 2, 1, speed_limit=40.0),
        fork_side="right",
        diverge_s_ratio=0.95,
        taper_length=20.0,
        branch_length=50.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/fork_tail.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "fork"
    assert result.quality["accepted"] is True
    assert len(result.lanes) == 4


@pytest.mark.map_generator
def test_merge_right():
    map_ = Map(name="merge_right")

    result = merge(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(35.0, -45.0, 0.55, 1, speed_limit=40.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        merge_side="right",
        taper_length=35.0,
        branch_length=55.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/merge_right.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "merge"
    assert result.quality["merge_side"] == "right"
    assert result.quality["main_lane_num"] == 3
    assert result.quality["branch_lane_num"] == 1
    assert result.quality["accepted"] is True

    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == 3
    assert len(result.ports["branch_in"].lane_ids) == 1
    assert len(result.ports["main_out"].lane_ids) == 3
    assert len(result.lanes) == 4
    assert result.id_counter > 0


@pytest.mark.map_generator
def test_merge_left():
    map_ = Map(name="merge_left")

    result = merge(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=60.0),
        _make_port(35.0, 45.0, -0.55, 1, speed_limit=40.0),
        _make_port(130.0, 0.0, 0.0, 3, speed_limit=60.0),
        merge_side="left",
        taper_length=35.0,
        branch_length=55.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/merge_left.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "merge"
    assert result.quality["merge_side"] == "left"
    assert result.quality["main_lane_num"] == 3
    assert result.quality["branch_lane_num"] == 1
    assert result.quality["accepted"] is True

    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports
    assert len(result.ports["main_in"].lane_ids) == 3
    assert len(result.ports["branch_in"].lane_ids) == 1
    assert len(result.ports["main_out"].lane_ids) == 3
    assert len(result.lanes) == 4


@pytest.mark.map_generator
def test_merge_two_lane_branch():
    map_ = Map(name="merge_two_lane_branch")

    result = merge(
        _make_port(0.0, 0.0, 0.0, 4, speed_limit=60.0),
        _make_port(45.0, -55.0, 0.55, 2, speed_limit=40.0),
        _make_port(150.0, 0.0, 0.0, 4, speed_limit=60.0),
        merge_side="right",
        branch_lane_num=2,
        taper_length=40.0,
        branch_length=65.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/merge_two_lane_branch.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "merge"
    assert result.quality["merge_side"] == "right"
    assert result.quality["main_lane_num"] == 4
    assert result.quality["branch_lane_num"] == 2
    assert result.quality["accepted"] is True
    assert len(result.ports["main_in"].lane_ids) == 4
    assert len(result.ports["branch_in"].lane_ids) == 2
    assert len(result.ports["main_out"].lane_ids) == 4
    assert len(result.lanes) == 6


@pytest.mark.map_generator
def test_merge_no_id_collision():
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

    ids_1 = {element.id_ for element in result_1.lanes + result_1.roadlines}
    ids_2 = {element.id_ for element in result_2.lanes + result_2.roadlines}

    assert len(ids_1 & ids_2) == 0
    assert result_2.id_counter > result_1.id_counter


@pytest.mark.map_generator
def test_merge_tail():
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

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "merge"
    assert result.quality["merge_side"] == "right"
    assert result.quality["accepted"] is True
    assert len(result.lanes) == 4
    assert "main_in" in result.ports
    assert "branch_in" in result.ports
    assert "main_out" in result.ports


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

    assert isinstance(result, RoadModuleResult)
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

    assert isinstance(result, RoadModuleResult)
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

    assert isinstance(result, RoadModuleResult)
    assert result.quality["arm_num"] == 4
    assert len(result.interfaces) == 4
    assert len(map_.junctions) == 1


@pytest.mark.map_generator
def test_roundabout_4arm():
    map_ = Map(name="roundabout_4arm")
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]]

    result = roundabout(
        center=np.array([0.0, 0.0]), arms=arms, ring_radius=12.0, ring_lane_num=2, id_offset=0
    )
    _add_result(map_, result)

    id_off = result.id_counter + 1000
    for iface in result.interfaces:
        road_result = _two_way_from_interface(iface, length=30.0, id_offset=id_off)
        id_off = road_result.id_counter
        _add_result(map_, road_result)

    _render(map_, "./tests/runtime/roundabout_4arm.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "roundabout"
    assert result.quality["arm_num"] == 4
    assert result.quality["ring_radius"] == 12.0
    assert result.quality["ring_lane_num"] == 2
    assert len(result.interfaces) == 4
    assert len(result.junctions) == 1
    assert len(map_.junctions) == 1
    assert result.quality["accepted"] is True

    expected_radius = result.quality["outer_ring_radius"]
    for iface in result.interfaces:
        assert np.isclose(np.linalg.norm(iface["point"]), expected_radius)


@pytest.mark.map_generator
def test_roundabout_3arm():
    map_ = Map(name="roundabout_3arm")
    arms = [{"heading": h, "lane_num": 2} for h in [0.0, 2 * np.pi / 3, 4 * np.pi / 3]]

    result = roundabout(
        center=np.array([0.0, 0.0]), arms=arms, ring_radius=12.0, ring_lane_num=2, id_offset=0
    )
    _add_result(map_, result)

    id_off = result.id_counter + 1000
    for iface in result.interfaces:
        road_result = _two_way_from_interface(iface, length=30.0, id_offset=id_off)
        id_off = road_result.id_counter
        _add_result(map_, road_result)

    _render(map_, "./tests/runtime/roundabout_3arm.png")

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "roundabout"
    assert result.quality["arm_num"] == 3
    assert result.quality["ring_radius"] == 12.0
    assert result.quality["ring_lane_num"] == 2
    assert len(result.interfaces) == 3
    assert len(result.junctions) == 1
    assert len(map_.junctions) == 1
    assert result.quality["accepted"] is True

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
            point=iface["point"] + 32.0 * unit(h) + curve_sign * 7.0 * normal,
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

    assert isinstance(result, RoadModuleResult)
    assert result.quality["module"] == "roundabout"
    assert result.quality["arm_num"] == 4
    assert len(result.interfaces) == 4
    assert len(result.junctions) == 1
    assert len(map_.junctions) == 1
    assert result.quality["accepted"] is True


def _assert_has_yellow_centerline(result: RoadModuleResult) -> None:
    """Assert that an urban two-way ramp has a yellow centerline."""
    yellow_lines = [roadline for roadline in result.roadlines if roadline.color == "yellow"]
    assert len(yellow_lines) >= 1
    assert any(
        roadline.custom_tags.get("marking_role") == "centerline"
        or roadline.custom_tags.get("role") == "centerline"
        for roadline in yellow_lines
    )


def _assert_has_ramp_auxiliary_line(result: RoadModuleResult) -> None:
    """Assert that a ramp has a dashed auxiliary merge/diverge line."""
    aux_lines = [
        roadline
        for roadline in result.roadlines
        if roadline.custom_tags.get("marking_token") == "dashed_white_ramp"
    ]
    assert len(aux_lines) >= 1
    assert all(roadline.subtype == "dashed" for roadline in aux_lines)
    assert all(roadline.color == "white" for roadline in aux_lines)


def _assert_ramp_result_basic(result: RoadModuleResult, kind: str, main_road_type: str) -> None:
    """Assert basic ramp result interface and quality fields."""
    assert isinstance(result, RoadModuleResult)

    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "ramp" in result.ports

    assert result.quality["module"] == "ramp"
    assert result.quality["kind"] == kind
    assert result.quality["main_road_type"] == main_road_type

    assert result.quality["connector_length"] > 0.0
    assert result.quality["main_self_intersection"] is False
    assert result.quality["connector_self_intersection"] is False
    assert isinstance(result.quality["accepted_reasons"], list)

    assert "connector_min_radius" in result.quality
    assert "connector_max_abs_curvature" in result.quality
    assert "connector_max_abs_curvature_rate" in result.quality
    assert result.quality["accepted"] is True
    assert result.id_counter > 0


def _assert_freeway_ramp_common(result: RoadModuleResult, kind: str, ramp_side: str) -> None:
    """Assert common freeway ramp properties."""
    _assert_ramp_result_basic(result, kind, "freeway")
    _assert_no_yellow_centerline(result)
    _assert_has_ramp_auxiliary_line(result)

    assert result.quality["ramp_side"] == ramp_side
    assert "backward_in" not in result.ports
    assert "backward_out" not in result.ports


def _assert_urban_ramp_common(result: RoadModuleResult, kind: str) -> None:
    """Assert common urban ramp properties."""
    _assert_ramp_result_basic(result, kind, "urban")
    _assert_has_yellow_centerline(result)
    _assert_has_ramp_auxiliary_line(result)

    assert result.quality["ramp_side"] == "right"
    assert "backward_in" in result.ports
    assert "backward_out" in result.ports


@pytest.mark.map_generator
def test_freeway_exit_ramp_right():
    map_ = Map(name="freeway_exit_ramp_right")

    result = freeway_exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(330.0, -95.0, -0.95, 1, speed_limit=50.0),
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/freeway_exit_ramp_right.png")

    _assert_freeway_ramp_common(result, "exit", "right")
    assert result.ports["ramp"].kind == "ramp_out"


@pytest.mark.map_generator
def test_freeway_entrance_ramp_right():
    map_ = Map(name="freeway_entrance_ramp_right")

    result = freeway_entrance_ramp(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(90.0, -95.0, 0.95, 1, speed_limit=50.0),
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/freeway_entrance_ramp_right.png")

    _assert_freeway_ramp_common(result, "entrance", "right")
    assert result.ports["ramp"].kind == "ramp_in"


@pytest.mark.map_generator
def test_freeway_exit_ramp_left():
    map_ = Map(name="freeway_exit_ramp_left")

    result = freeway_exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(330.0, 95.0, 0.95, 1, speed_limit=50.0),
        ramp_side="left",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/freeway_exit_ramp_left.png")

    _assert_freeway_ramp_common(result, "exit", "left")
    assert result.ports["ramp"].kind == "ramp_out"


@pytest.mark.map_generator
def test_freeway_entrance_ramp_left():
    map_ = Map(name="freeway_entrance_ramp_left")

    result = freeway_entrance_ramp(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(90.0, 95.0, -0.95, 1, speed_limit=50.0),
        ramp_side="left",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/freeway_entrance_ramp_left.png")

    _assert_freeway_ramp_common(result, "entrance", "left")
    assert result.ports["ramp"].kind == "ramp_in"


@pytest.mark.map_generator
def test_urban_exit_ramp_right():
    map_ = Map(name="urban_exit_ramp_right")

    result = urban_exit_ramp(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0),
        _make_port(320.0, 16.0, 0.04, 2, speed_limit=80.0),
        _make_port(255.0, -75.0, -0.90, 1, speed_limit=40.0),
        backward_lane_num=1,
        taper_length=55.0,
        parallel_length=75.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/urban_exit_ramp_right.png")

    _assert_urban_ramp_common(result, "exit")
    assert result.ports["ramp"].kind == "ramp_out"


@pytest.mark.map_generator
def test_urban_entrance_ramp_right():
    map_ = Map(name="urban_entrance_ramp_right")

    result = urban_entrance_ramp(
        _make_port(0.0, 0.0, 0.0, 2, speed_limit=80.0),
        _make_port(320.0, 16.0, 0.04, 2, speed_limit=80.0),
        _make_port(65.0, -75.0, 0.90, 1, speed_limit=40.0),
        backward_lane_num=1,
        taper_length=55.0,
        parallel_length=75.0,
        id_offset=0,
    )

    _add_result(map_, result)
    _render(map_, "./tests/runtime/urban_entrance_ramp_right.png")

    _assert_urban_ramp_common(result, "entrance")
    assert result.ports["ramp"].kind == "ramp_in"


@pytest.mark.map_generator
def test_exit_ramp_unified_interface_freeway():
    result = exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(420.0, 0.0, 0.0, 3, speed_limit=100.0),
        _make_port(330.0, -95.0, -0.95, 1, speed_limit=50.0),
        main_road_type="freeway",
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )

    _assert_freeway_ramp_common(result, "exit", "right")
    assert result.ports["main_in"].lane_num == 3
    assert result.ports["main_out"].lane_num == 3
    assert result.ports["ramp"].lane_num == 1


@pytest.mark.map_generator
def test_entrance_ramp_unified_interface_urban():
    result = entrance_ramp(
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

    _assert_urban_ramp_common(result, "entrance")
    assert result.ports["main_in"].lane_num == 2
    assert result.ports["main_out"].lane_num == 2
    assert result.ports["ramp"].lane_num == 1


@pytest.mark.map_generator
def test_ramp_no_id_collision():
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

    ids1 = {element.id_ for element in r1.lanes + r1.roadlines}
    ids2 = {element.id_ for element in r2.lanes + r2.roadlines}

    assert len(ids1 & ids2) == 0


@pytest.mark.map_generator
def test_ramp_result_interface():
    result = freeway_exit_ramp(
        _make_port(0.0, 0.0, 0.0, 3),
        _make_port(420.0, 0.0, 0.0, 3),
        _make_port(330.0, -95.0, -0.95, 1),
        ramp_side="right",
        taper_length=70.0,
        parallel_length=90.0,
        id_offset=0,
    )

    assert "main_in" in result.ports
    assert "main_out" in result.ports
    assert "ramp" in result.ports

    assert isinstance(result.ports["main_out"].point, np.ndarray)
    assert isinstance(result.ports["main_out"].heading, float)
    assert result.ports["main_out"].lane_num == 3

    assert isinstance(result.ports["ramp"].point, np.ndarray)
    assert isinstance(result.ports["ramp"].heading, float)
    assert result.ports["ramp"].lane_num == 1

    assert "connector_min_radius" in result.quality
    assert "accepted_reasons" in result.quality
    assert result.id_counter > 0
