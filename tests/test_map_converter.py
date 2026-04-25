# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Net2XodrConverter."""

import sys

sys.path.append(".")
sys.path.append("..")

import os

import pytest
from shapely.geometry import Point

from tactics2d.map.converter import Net2XodrConverter
from tactics2d.map.parser import NetXMLParser, XODRParser
from tactics2d.renderer import MatplotlibRenderer
from tactics2d.sensor import BEVCamera
from tactics2d.utils.common import get_absolute_path


@pytest.mark.map_converter
@pytest.mark.parametrize(
    "input_path, output_path, img_path",
    [
        (
            "./tests/cases/NetXMLSamples/net.net.xml",
            "./tests/runtime/net_converted.xodr",
            "./tests/runtime/net_converted.png",
        ),
        (
            "./tests/cases/NetXMLSamples/lefthand.net.xml",
            "./tests/runtime/lefthand_converted.xodr",
            "./tests/runtime/lefthand_converted.png",
        ),
        (
            "./tests/cases/NetXMLSamples/roundabout.net.xml",
            "./tests/runtime/roundabout_converted.xodr",
            "./tests/runtime/roundabout_converted.png",
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
        resolution=((boundary[1] - boundary[0]) * 10, (boundary[3] - boundary[2]) * 10),
        xlim=(boundary[0], boundary[1]),
        ylim=(boundary[2], boundary[3]),
    )
    renderer.update(geometry_data)
    renderer.save_single_frame(save_to=img_path)
    renderer.destroy()
