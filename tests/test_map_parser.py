##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_map_parser.py
# @Description: This script is used to test the parsers and converters in the map module.
# @Author: Yueyuan Li
# @Version: 1.0.0


import sys

sys.path.append(".")
sys.path.append("..")

import logging

import pytest
from shapely.geometry import Point

from tactics2d.map.map_config import *
from tactics2d.map.parser import GISParser, OSMParser, XODRParser
from tactics2d.renderer import MatplotlibRenderer
from tactics2d.sensor import BEVCamera
from tactics2d.utils.common import get_absolute_path


@pytest.mark.map_parser
def test_osm_parser():
    map_path = get_absolute_path("./tactics2d/data/map/SJTU/raw.osm")
    map_parser = OSMParser()

    map_ = map_parser.parse(map_path)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = Point(0, 0)
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        (boundary[0], boundary[1]),
        (boundary[2], boundary[3]),
        resolution=((boundary[1] - boundary[0]) * 100, (boundary[3] - boundary[2]) * 100),
    )
    matplotlib_renderer.update(geometry_data, (position.x, position.y))
    matplotlib_renderer.save_single_frame(save_to="./tests/runtime/raw.png")


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
def test_lanelet2_parser(map_folder, map_configs):
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
            position = Point(0, 0)
            geometry_data, _, _ = camera.update(0, None, None, None, None, position)

            matplotlib_renderer = MatplotlibRenderer(
                (boundary[0], boundary[1]),
                (boundary[2], boundary[3]),
                resolution=((boundary[1] - boundary[0]) * 10, (boundary[3] - boundary[2]) * 10),
            )

            matplotlib_renderer.update(geometry_data, [position.x, position.y])
            matplotlib_renderer.save_single_frame(save_to=f"./tests/runtime/{map_name}.png")

            matplotlib_renderer.destroy()

        except SyntaxError as err:
            logging.error(err)
        except KeyError as err:
            logging.error(err)
        except FileNotFoundError as err:
            raise err


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "map_path, img_path",
    [
        ("./tests/cases/XodrSamples/cross.xodr", "./tests/runtime/cross.png"),
        ("./tests/cases/XodrSamples/ring.xodr", "./tests/runtime/ring.png"),
        ("./tests/cases/XodrSamples/LargeParkingLot.xodr", "./tests/runtime/LargeParkingLot.png"),
        ("./tests/cases/XodrSamples/FourWayStop.xodr", "./tests/runtime/FourWayStop.png"),
        ("./tests/cases/XodrSamples/SimpleBankedRoad.xodr", "./tests/runtime/SimpleBankedRoad.png"),
        (
            "./tests/cases/XodrSamples/SimpleFreewayRamps.xodr",
            "./tests/runtime/SimpleFreewayRamps.png",
        ),
    ],
)
def test_xodr_parser(map_path, img_path):
    map_path = get_absolute_path(map_path)
    map_parser = XODRParser()
    map_ = map_parser.parse(map_path)

    boundary = map_.boundary
    camera = BEVCamera(1, map_)
    position = Point(0, 0)
    geometry_data, _, _ = camera.update(0, None, None, None, None, position)

    matplotlib_renderer = MatplotlibRenderer(
        (boundary[0], boundary[1]),
        (boundary[2], boundary[3]),
        resolution=((boundary[1] - boundary[0]) * 10, (boundary[3] - boundary[2]) * 10),
    )

    matplotlib_renderer.update(geometry_data, [position.x, position.y])
    matplotlib_renderer.save_single_frame(save_to=img_path)


@pytest.mark.map_parser
@pytest.mark.parametrize("map_path, img_path", [()])
def test_gis_parse(map_path, img_path):
    map_path = "./tactics2d/data/map/"

    matploblib_renderer = MatplotlibRenderer()
