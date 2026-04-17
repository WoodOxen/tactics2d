# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map converter."""

import sys

sys.path.append(".")
sys.path.append("..")

import os

import pytest

from tactics2d.map.converter import Net2XodrConverter, Osm2XodrConverter, Xodr2NetConverter, Xodr2OsmConverter
from tactics2d.utils.common import get_absolute_path


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "input_path, output_path",
    [
        (
            "./tests/cases/NetXMLSamples/net.net.xml",
            "./tests/runtime/net.xodr",
        ),
        (
            "./tests/cases/NetXMLSamples/lefthand.net.xml",
            "./tests/runtime/lefthand.xodr",
        ),
    ],
)
def test_net2xodr(input_path, output_path):
    input_path = get_absolute_path(input_path)
    converter = Net2XodrConverter()
    result = converter.convert(input_path, output_path)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "input_path, output_path",
    [
        (
            "./tests/cases/XodrSamples/cross.xodr",
            "./tests/runtime/cross.net.xml",
        ),
        (
            "./tests/cases/XodrSamples/FourWayStop.xodr",
            "./tests/runtime/FourWayStop.net.xml",
        ),
    ],
)
def test_xodr2net(input_path, output_path):
    input_path = get_absolute_path(input_path)
    converter = Xodr2NetConverter()
    result = converter.convert(input_path, output_path)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "input_path, output_path",
    [
        (
            "./tests/cases/OsmSamples/sample1.osm",
            "./tests/runtime/sample1.xodr",
        ),
    ],
)
def test_osm2xodr(input_path, output_path):
    input_path = get_absolute_path(input_path)
    converter = Osm2XodrConverter()
    result = converter.convert(input_path, output_path)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "input_path, output_path",
    [
        (
            "./tests/cases/XodrSamples/cross.xodr",
            "./tests/runtime/cross.osm",
        ),
    ],
)
def test_xodr2osm(input_path, output_path):
    input_path = get_absolute_path(input_path)
    converter = Xodr2OsmConverter()
    result = converter.convert(input_path, output_path)
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0
