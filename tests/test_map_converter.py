# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map format converters."""

import logging
import os
import sys

sys.path.append(".")
sys.path.append("..")

import pytest
from shapely.geometry import Point

from tactics2d.map.converter import (
    Net2OsmConverter,
    Net2XodrConverter,
    Osm2NetConverter,
    Osm2XodrConverter,
    Xodr2NetConverter,
    Xodr2OsmConverter,
)
from tactics2d.map.parser import NetXMLParser, OSMParser, XODRParser
from tactics2d.renderer import MatplotlibRenderer
from tactics2d.sensor import BEVCamera
from tactics2d.utils.common import get_absolute_path

logging.disable(logging.WARNING)

_MAX_RENDER_PX = 8000
_PX_PER_METRE = 10


def _make_resolution(boundary):
    w = (boundary[1] - boundary[0]) * _PX_PER_METRE
    h = (boundary[3] - boundary[2]) * _PX_PER_METRE
    scale = min(1.0, _MAX_RENDER_PX / max(w, h, 1))
    return (max(1, int(w * scale)), max(1, int(h * scale)))


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/NetXMLSamples/net.net.xml",
            "./tests/runtime/net2xodr_net.xodr",
            "./tests/runtime/net2xodr_net.png",
        ),
        (
            "./tests/cases/NetXMLSamples/lefthand.net.xml",
            "./tests/runtime/net2xodr_lefthand.xodr",
            "./tests/runtime/net2xodr_lefthand.png",
        ),
        (
            "./tests/cases/NetXMLSamples/roundabout.net.xml",
            "./tests/runtime/net2xodr_roundabout.xodr",
            "./tests/runtime/net2xodr_roundabout.png",
        ),
    ],
)
def test_net2xodr(input_path, output_path, img_path):
    input_path = get_absolute_path(input_path)

    converter = Net2XodrConverter()
    result = converter.convert(input_path, output_path)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0

    original = NetXMLParser().parse(input_path)
    converted = XODRParser().parse(result)

    assert len(converted.lanes) == len(
        original.lanes
    ), f"Lane count mismatch: original={len(original.lanes)}, converted={len(converted.lanes)}"
    assert len(converted.junctions) == len(
        original.junctions
    ), f"Junction count mismatch: original={len(original.junctions)}, converted={len(converted.junctions)}"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/XodrSamples/cross.xodr",
            "./tests/runtime/xodr2net_cross.net.xml",
            "./tests/runtime/xodr2net_cross.png",
        ),
        (
            "./tests/cases/XodrSamples/FourWayStop.xodr",
            "./tests/runtime/xodr2net_FourWayStop.net.xml",
            "./tests/runtime/xodr2net_FourWayStop.png",
        ),
    ],
)
def test_xodr2net(input_path, output_path, img_path):
    input_path = get_absolute_path(input_path)

    converter = Xodr2NetConverter()
    result = converter.convert(input_path, output_path)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0

    original = XODRParser().parse(input_path)
    converted = NetXMLParser().parse(result)

    assert len(converted.lanes) > 0, "Converted net.xml has no lanes"
    assert len(converted.junctions) == len(
        original.junctions
    ), f"Junction count mismatch: original={len(original.junctions)}, converted={len(converted.junctions)}"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/OsmSamples/cross.osm",
            "./tests/runtime/osm2xodr_cross.xodr",
            "./tests/runtime/osm2xodr_cross.png",
        ),
        (
            "./tests/cases/OsmSamples/FourWayStop.osm",
            "./tests/runtime/osm2xodr_FourWayStop.xodr",
            "./tests/runtime/osm2xodr_FourWayStop.png",
        ),
    ],
)
def test_osm2xodr(input_path, output_path, img_path):
    input_path = get_absolute_path(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = Osm2XodrConverter()
    result = converter.convert(input_path, output_path)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0

    original = OSMParser(lanelet2=True).parse(input_path)
    converted = XODRParser().parse(result)

    assert len(converted.lanes) > 0, "Converted XODR has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/XodrSamples/cross.xodr",
            "./tests/runtime/xodr2osm_cross.osm",
            "./tests/runtime/xodr2osm_cross.png",
        ),
        (
            "./tests/cases/XodrSamples/FourWayStop.xodr",
            "./tests/runtime/xodr2osm_FourWayStop.osm",
            "./tests/runtime/xodr2osm_FourWayStop.png",
        ),
    ],
)
def test_xodr2osm(input_path, output_path, img_path):
    input_path = get_absolute_path(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = Xodr2OsmConverter()
    result = converter.convert(input_path, output_path)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0

    converted = OSMParser(lanelet2=True).parse(result)

    assert len(converted.lanes) > 0, "Converted OSM has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/NetXMLSamples/net.net.xml",
            "./tests/runtime/net2osm_net.osm",
            "./tests/runtime/net2osm_net.png",
        ),
        (
            "./tests/cases/NetXMLSamples/lefthand.net.xml",
            "./tests/runtime/net2osm_lefthand.osm",
            "./tests/runtime/net2osm_lefthand.png",
        ),
        (
            "./tests/cases/NetXMLSamples/roundabout.net.xml",
            "./tests/runtime/net2osm_roundabout.osm",
            "./tests/runtime/net2osm_roundabout.png",
        ),
    ],
)
def test_net2osm(input_path, output_path, img_path):
    input_path = get_absolute_path(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = Net2OsmConverter()
    result = converter.convert(input_path, output_path)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0

    converted = OSMParser(lanelet2=True).parse(result)

    assert len(converted.lanes) > 0, "Converted OSM has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/OsmSamples/cross.osm",
            "./tests/runtime/osm2net_cross.net.xml",
            "./tests/runtime/osm2net_cross.png",
        ),
        (
            "./tests/cases/OsmSamples/FourWayStop.osm",
            "./tests/runtime/osm2net_FourWayStop.net.xml",
            "./tests/runtime/osm2net_FourWayStop.png",
        ),
    ],
)
def test_osm2net(input_path, output_path, img_path):
    input_path = get_absolute_path(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = Osm2NetConverter()
    result = converter.convert(input_path, output_path)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0

    converted = NetXMLParser().parse(result)

    assert len(converted.lanes) > 0, "Converted net.xml has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, Point(0, 0))
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()
