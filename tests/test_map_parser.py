##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_map_parser.py
# @Description: This script is used to test the parsers and converters in the map module.
# @Author: Yueyuan Li
# @Version: 1.0.0


import sys

sys.path.append(".")
sys.path.append("..")

import json
import logging
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from tactics2d.map.parser import OSMParser, XODRParser
from tactics2d.traffic import ScenarioDisplay
from tactics2d.utils.get_absolute_path import get_absolute_path


@pytest.mark.map_parser
def test_osm_parser():
    map_path = get_absolute_path("./tactics2d/data/map/SJTU/raw.osm")
    map_parser = OSMParser()

    map_ = map_parser.parse(map_path)

    fig, ax = plt.subplots()
    fig.set_layout_engine("none")
    ax.set_aspect("equal")
    ax.set_axis_off()

    scenario_display = ScenarioDisplay()
    scenario_display.display_map(map_, ax)
    ax.plot()
    fig.savefig("./tests/runtime/raw.png")
    plt.close(fig)


@pytest.mark.map_parser
def test_lanelet2_parser():
    """Test whether the current parser can manage to parse the provided maps.

    [TODO] split this test to two part:
        One for testing the correctness of the provided maps' notations;
        One for testing the parser's ability to parse the lanelet2 format maps.
    """

    data_path = "./tactics2d/data/map"
    config_path = "./tactics2d/dataset_parser/map.config"

    map_parser = OSMParser(lanelet2=True)
    scenario_display = ScenarioDisplay()

    with open(config_path) as f:
        configs = json.load(f)

    parsed_map_set = set()

    for map_name, map_config in configs.items():
        if map_config["dataset"] in ["inD", "rounD", "uniD", "exiD", "NuPlan"]:
            continue
        logging.info(f"Parsing map {map_name}.")

        try:
            map_path = get_absolute_path(
                "{}/{}/{}.osm".format(data_path, map_config["dataset"], map_name)
            )
            map_ = map_parser.parse(map_path, map_config)
            parsed_map_set.add(map_.name)

            fig, ax = plt.subplots()
            fig.set_layout_engine("none")
            ax.set_axis_off()
            scenario_display.reset()
            scenario_display.display_map(map_, ax)
            ax.set_aspect("equal")
            ax.plot()
            fig.savefig(f"./tests/runtime/{map_name}.png", dpi=300)
            plt.close(fig)

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

    fig, ax = plt.subplots()
    fig.set_layout_engine("none")
    ax.set_aspect("equal")
    ax.set_axis_off()

    scenario_display = ScenarioDisplay()
    scenario_display.display_map(map_, ax)
    ax.plot()
    fig.savefig(img_path, facecolor="black")
    plt.close(fig)


@pytest.mark.map_parser
def test_gis_parse():
    map_path = "./tactics2d/data/map/"
