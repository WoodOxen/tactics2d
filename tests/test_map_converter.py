# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map format converters."""

import logging
import sys
from pathlib import Path

sys.path.append(".")
sys.path.append("..")

import pytest

from tactics2d.display.renderers import MatplotlibRenderer
from tactics2d.display.sensor import BEVCamera
from tactics2d.map.converter import (
    Net2OsmConverter,
    Net2XodrConverter,
    Osm2NetConverter,
    Osm2XodrConverter,
    Xodr2NetConverter,
    Xodr2OsmConverter,
)
from tactics2d.map.parser import NetXMLParser, OSMParser, XODRParser
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
    "input_path, output_name, img_name",
    [
        ("./tests/cases/NetXMLSamples/net.net.xml", "net2xodr_net.xodr", "net2xodr_net.png"),
        (
            "./tests/cases/NetXMLSamples/lefthand.net.xml",
            "net2xodr_lefthand.xodr",
            "net2xodr_lefthand.png",
        ),
        (
            "./tests/cases/NetXMLSamples/roundabout.net.xml",
            "net2xodr_roundabout.xodr",
            "net2xodr_roundabout.png",
        ),
    ],
)
def test_net2xodr(runtime_dir, input_path, output_name, img_name):
    input_path = get_absolute_path(input_path)
    output_path = str(runtime_dir / output_name)
    img_path = str(runtime_dir / img_name)

    converter = Net2XodrConverter()
    result = converter.convert(input_path, output_path)

    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0

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
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
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
    "input_path, output_name, img_name",
    [
        ("./tests/cases/XodrSamples/cross.xodr", "xodr2net_cross.net.xml", "xodr2net_cross.png"),
        (
            "./tests/cases/XodrSamples/FourWayStop.xodr",
            "xodr2net_FourWayStop.net.xml",
            "xodr2net_FourWayStop.png",
        ),
    ],
)
def test_xodr2net(runtime_dir, input_path, output_name, img_name):
    input_path = get_absolute_path(input_path)
    output_path = str(runtime_dir / output_name)
    img_path = str(runtime_dir / img_name)

    converter = Xodr2NetConverter()
    result = converter.convert(input_path, output_path)

    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0

    original = XODRParser().parse(input_path)
    converted = NetXMLParser().parse(result)

    assert len(converted.lanes) > 0, "Converted net.xml has no lanes"
    assert len(converted.junctions) == len(
        original.junctions
    ), f"Junction count mismatch: original={len(original.junctions)}, converted={len(converted.junctions)}"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
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
    "input_path, output_name, img_name",
    [
        ("./tests/cases/OsmSamples/cross.osm", "osm2xodr_cross.xodr", "osm2xodr_cross.png"),
        (
            "./tests/cases/OsmSamples/FourWayStop.osm",
            "osm2xodr_FourWayStop.xodr",
            "osm2xodr_FourWayStop.png",
        ),
    ],
)
def test_osm2xodr(runtime_dir, input_path, output_name, img_name):
    input_path = get_absolute_path(input_path)
    output_path = str(runtime_dir / output_name)
    img_path = str(runtime_dir / img_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converter = Osm2XodrConverter()
    result = converter.convert(input_path, output_path)

    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0

    original = OSMParser(lanelet2=True).parse(input_path)
    converted = XODRParser().parse(result)

    assert len(converted.lanes) > 0, "Converted XODR has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
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
    "input_path, output_name, img_name",
    [
        ("./tests/cases/XodrSamples/cross.xodr", "xodr2osm_cross.osm", "xodr2osm_cross.png"),
        (
            "./tests/cases/XodrSamples/FourWayStop.xodr",
            "xodr2osm_FourWayStop.osm",
            "xodr2osm_FourWayStop.png",
        ),
    ],
)
def test_xodr2osm(runtime_dir, input_path, output_name, img_name):
    input_path = get_absolute_path(input_path)
    output_path = str(runtime_dir / output_name)
    img_path = str(runtime_dir / img_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converter = Xodr2OsmConverter()
    result = converter.convert(input_path, output_path)

    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0

    converted = OSMParser(lanelet2=True).parse(result)

    assert len(converted.lanes) > 0, "Converted OSM has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
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
    "input_path, output_name, img_name",
    [
        ("./tests/cases/NetXMLSamples/net.net.xml", "net2osm_net.osm", "net2osm_net.png"),
        (
            "./tests/cases/NetXMLSamples/lefthand.net.xml",
            "net2osm_lefthand.osm",
            "net2osm_lefthand.png",
        ),
        (
            "./tests/cases/NetXMLSamples/roundabout.net.xml",
            "net2osm_roundabout.osm",
            "net2osm_roundabout.png",
        ),
    ],
)
def test_net2osm(runtime_dir, input_path, output_name, img_name):
    input_path = get_absolute_path(input_path)
    output_path = str(runtime_dir / output_name)
    img_path = str(runtime_dir / img_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converter = Net2OsmConverter()
    result = converter.convert(input_path, output_path)

    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0

    converted = OSMParser(lanelet2=True).parse(result)

    assert len(converted.lanes) > 0, "Converted OSM has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
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
    "input_path, output_name, img_name",
    [
        ("./tests/cases/OsmSamples/cross.osm", "osm2net_cross.net.xml", "osm2net_cross.png"),
        (
            "./tests/cases/OsmSamples/FourWayStop.osm",
            "osm2net_FourWayStop.net.xml",
            "osm2net_FourWayStop.png",
        ),
    ],
)
def test_osm2net(runtime_dir, input_path, output_name, img_name):
    input_path = get_absolute_path(input_path)
    output_path = str(runtime_dir / output_name)
    img_path = str(runtime_dir / img_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converter = Osm2NetConverter()
    result = converter.convert(input_path, output_path)

    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0

    converted = NetXMLParser().parse(result)

    assert len(converted.lanes) > 0, "Converted net.xml has no lanes"

    boundary = converted.boundary
    camera = BEVCamera(1, converted)
    geometry_data, _, _ = camera.update(0, None, None, None, None, None)
    renderer = MatplotlibRenderer(
        resolution=_make_resolution(boundary),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()
