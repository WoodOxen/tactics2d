# Copyright (C) 2022, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map parser."""


import sys

sys.path.append(".")
sys.path.append("..")

import logging

import pytest

from tactics2d.display.renderers import MatplotlibRenderer
from tactics2d.display.sensor import BEVCamera
from tactics2d.map.map_config import *
from tactics2d.map.parser import NetXMLParser, OSMParser, XODRParser
from tactics2d.utils.common import get_absolute_path

_MAX_IMG_PX = 8000


def _make_resolution(boundary, default_scale=100.0):
    """Compute an adaptive resolution that stays within safe pixel limits.

    Args:
        boundary (tuple): (x_min, x_max, y_min, y_max).
        default_scale (float): Pixels per metre when the map is small enough.
            Defaults to 100.0.

    Returns:
        tuple: (width_px, height_px).
    """
    w_m = boundary[1] - boundary[0]
    h_m = boundary[3] - boundary[2]
    scale = min(default_scale, _MAX_IMG_PX / max(w_m, h_m, 0.01))
    return (int(w_m * scale), int(h_m * scale))


@pytest.mark.map_parser
def test_osm_parser(runtime_dir):
    map_path = get_absolute_path("./tactics2d/data/map/SJTU/raw.osm")
    map_parser = OSMParser()

    map_ = map_parser.parse(map_path)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = None
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    matplotlib_renderer.update(geometry_data)
    matplotlib_renderer.save_single_frame(save_to=runtime_dir / "raw.png")


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "map_folder, map_configs",
    [
        ("./tactics2d/data/map/DLP", DLP_MAP_CONFIG),
        ("./tactics2d/data/map/exiD", EXID_MAP_CONFIG),
        ("./tactics2d/data/map/highD", HIGHD_MAP_CONFIG),
        ("./tactics2d/data/map/inD", IND_MAP_CONFIG),
        ("./tactics2d/data/map/INTERACTION", INTERACTION_MAP_CONFIG),
        ("./tactics2d/data/map/rounD", ROUND_MAP_CONFIG),
    ],
)
def test_lanelet2_parser(runtime_dir, map_folder, map_configs):
    """Test whether the current parser can manage to parse the provided maps.

    [TODO] split this test to two part:
        One for testing the correctness of the provided maps' notations;
        One for testing the parser's ability to parse the lanelet2 format maps.
    """
    map_parser = OSMParser(lanelet2=True)
    parsed_map_set = set()

    for map_name, map_config in map_configs.items():
        logging.info(f"Parsing map {map_name}.")

        try:
            file_name = map_config["osm_file"]
            map_path = get_absolute_path(f"{map_folder}/{file_name}")
            map_ = map_parser.parse(map_path, map_config)
            parsed_map_set.add(map_.name)

            boundary = map_.boundary
            camera = BEVCamera(1, map_)
            position = None
            geometry_data, _, _ = camera.update(0, None, None, None, None, position)

            matplotlib_renderer = MatplotlibRenderer(
                resolution=_make_resolution(boundary, default_scale=10.0),
                xlim=(boundary[0], boundary[1]),
                ylim=(boundary[2], boundary[3]),
            )

            matplotlib_renderer.update(geometry_data)
            matplotlib_renderer.save_single_frame(save_to=runtime_dir / f"{map_name}.png")

            matplotlib_renderer.destroy()

        except SyntaxError as err:
            logging.error(err)
        except KeyError as err:
            logging.error(err)
        except FileNotFoundError as err:
            raise err


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "map_path, img_name",
    [
        ("./tests/cases/XodrSamples/cross.xodr", "cross.png"),
        ("./tests/cases/XodrSamples/ring.xodr", "ring.png"),
        ("./tests/cases/XodrSamples/LargeParkingLot.xodr", "LargeParkingLot.png"),
        ("./tests/cases/XodrSamples/FourWayStop.xodr", "FourWayStop.png"),
        ("./tests/cases/XodrSamples/SimpleBankedRoad.xodr", "SimpleBankedRoad.png"),
        ("./tests/cases/XodrSamples/SimpleFreewayRamps.xodr", "SimpleFreewayRamps.png"),
    ],
)
def test_xodr_parser(runtime_dir, map_path, img_name):
    map_path = get_absolute_path(map_path)
    map_parser = XODRParser()
    map_ = map_parser.parse(map_path)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = None
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary, default_scale=10.0),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )

    matplotlib_renderer.update(geometry_data)
    matplotlib_renderer.save_single_frame(save_to=runtime_dir / img_name)


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "map_path, img_name",
    [
        ("./tests/cases/NetXMLSamples/net.net.xml", "net.png"),
        ("./tests/cases/NetXMLSamples/lefthand.net.xml", "lefthand.png"),
        ("./tests/cases/NetXMLSamples/roundabout.net.xml", "roundabout.png"),
    ],
)
def test_net_xml_parser(runtime_dir, map_path, img_name):
    map_path = get_absolute_path(map_path)
    map_parser = NetXMLParser()
    map_ = map_parser.parse(map_path)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = None
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary, default_scale=10.0),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )

    matplotlib_renderer.update(geometry_data)
    matplotlib_renderer.save_single_frame(save_to=runtime_dir / img_name)
    matplotlib_renderer.destroy()


def _write_lanelet2_osm(path, nodes, ways, relations):
    """Write a minimal lanelet2 OSM file from (id, ...) tuples."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for node_id, lat, lon in nodes:
        lines.append(f'  <node id="{node_id}" lat="{lat}" lon="{lon}"/>')
    for way_id, node_ids in ways:
        lines.append(f'  <way id="{way_id}">')
        lines.extend(f'    <nd ref="{node_id}"/>' for node_id in node_ids)
        lines.append('    <tag k="type" v="line_thin"/>')
        lines.append("  </way>")
    for relation_id, members, tags in relations:
        lines.append(f'  <relation id="{relation_id}">')
        for member_type, ref, role in members:
            lines.append(f'    <member type="{member_type}" ref="{ref}" role="{role}"/>')
        for key, value in tags:
            lines.append(f'    <tag k="{key}" v="{value}"/>')
        lines.append("  </relation>")
    lines.append("</osm>")
    path.write_text("\n".join(lines))


@pytest.mark.map_parser
def test_lanelet2_parser_remaps_colliding_ids(tmp_path):
    """Node/way/relation ids live in independent OSM namespaces (issue #291)."""
    osm = tmp_path / "collide.osm"
    # Node 1, way 1, and relation 1 coexist; ways 2/3 also collide with nodes.
    _write_lanelet2_osm(
        osm,
        nodes=[(1, 50.0, 6.0), (2, 50.0001, 6.0), (3, 50.0, 6.0002), (4, 50.0001, 6.0002)],
        ways=[(1, [1, 3]), (2, [2, 4])],
        relations=[
            (
                1,
                [("way", 1, "left"), ("way", 2, "right"), ("relation", 5, "regulatory_element")],
                [("type", "lanelet"), ("subtype", "road")],
            ),
            (
                5,
                [("relation", 1, "refers")],
                [("type", "regulatory_element"), ("subtype", "speed_limit")],
            ),
        ],
    )

    map_ = OSMParser(lanelet2=True).parse(str(osm))

    assert len(map_.roadlines) == 2
    assert len(map_.lanes) == 1
    assert len(map_.regulations) == 1
    lane = next(iter(map_.lanes.values()))
    # Member references follow the remapped way ids.
    assert all(line_id in map_.roadlines for side in ("left", "right") for line_id in lane.line_ids[side])
    # The regulatory element still points at the (remapped) lane relation.
    regulatory = next(iter(map_.regulations.values()))
    assert set(regulatory.relations) == {lane.id_}
    assert lane.regulatory_ids == {regulatory.id_}


@pytest.mark.map_parser
def test_lanelet2_parser_stitches_out_of_order_members(tmp_path):
    """Relation members are not guaranteed head-to-tail; stitch by endpoints."""
    osm = tmp_path / "unordered.osm"
    # Left side is split into two ways listed in reverse order: 20 then 10.
    _write_lanelet2_osm(
        osm,
        nodes=[
            (101, 50.0, 6.0),
            (102, 50.0, 6.0001),
            (103, 50.0, 6.0002),
            (104, 50.0001, 6.0),
            (105, 50.0001, 6.0002),
        ],
        ways=[(10, [101, 102]), (20, [102, 103]), (30, [104, 105])],
        relations=[
            (
                40,
                [("way", 20, "left"), ("way", 10, "left"), ("way", 30, "right")],
                [("type", "lanelet"), ("subtype", "road")],
            )
        ],
    )

    map_ = OSMParser(lanelet2=True).parse(str(osm))

    assert len(map_.lanes) == 1
    lane = next(iter(map_.lanes.values()))
    assert len(lane.left_side.coords) == 3  # both left segments joined


@pytest.mark.map_parser
def test_lanelet2_parser_skips_broken_elements(tmp_path):
    """Missing node references degrade to skipped elements, not a failed map."""
    osm = tmp_path / "broken.osm"
    _write_lanelet2_osm(
        osm,
        nodes=[
            (101, 50.0, 6.0),
            (102, 50.0, 6.0002),
            (103, 50.0001, 6.0),
            (104, 50.0001, 6.0002),
            (105, 50.0002, 6.0),
        ],
        ways=[
            (10, [101, 102]),
            (20, [103, 104]),
            (30, [105, 999999]),  # dangling node reference
        ],
        relations=[
            (
                40,
                [("way", 10, "left"), ("way", 20, "right")],
                [("type", "lanelet"), ("subtype", "road")],
            ),
            (
                41,
                [("way", 10, "left"), ("way", 30, "right")],  # references the broken way
                [("type", "lanelet"), ("subtype", "road")],
            ),
            (
                42,
                [],  # no members at all
                [("type", "lanelet"), ("subtype", "road")],
            ),
        ],
    )

    map_ = OSMParser(lanelet2=True).parse(str(osm))

    # Way 30 degenerates to a single point and is skipped; lane 40 survives.
    assert 30 not in map_.roadlines
    assert set(map_.lanes) == {40}
