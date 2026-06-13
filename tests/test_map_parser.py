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
