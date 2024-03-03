import sys

sys.path.append(".")
sys.path.append("..")

import json
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
import pytest

from tactics2d.map.parser import OSMParser
from tactics2d.traffic import ScenarioDisplay


@pytest.mark.map_parser
def test_osm_parser():
    data_path = "./tactics2d/data/map/SJTU/raw.osm"
    map_parser = OSMParser()

    map_root = ET.parse(data_path).getroot()
    map_ = map_parser.parse(map_root)

    fig, ax = plt.subplots()
    fig.set_layout_engine("none")
    ax.set_aspect("equal")
    ax.set_axis_off()

    scenario_display = ScenarioDisplay()
    scenario_display.display_map(map_, ax)
    ax.plot()
    fig.savefig("./tests/runtime/raw.png")


@pytest.mark.map_parser
def test_lanelet2_parser():
    """Test whether the current parser can manage to parse the provided maps.

    [TODO] split this test to two part:
        One for testing the correctness of the provided maps' notations;
        One for testing the parser's ability to parse the lanelet2 format maps.
    """

    data_path = "./tactics2d/data/map"
    config_path = "./tactics2d/data/map/map.config"

    map_parser = OSMParser(lanelet2=True)
    scenario_display = ScenarioDisplay()

    with open(config_path) as f:
        configs = json.load(f)

    parsed_map_set = set()

    for map_name, map_config in configs.items():
        if map_config["dataset"] in ["uniD", "exiD", "NuPlan"]:
            continue
        logging.info(f"Parsing map {map_name}.")

        try:
            map_path = "{}/{}/{}.osm".format(data_path, map_config["dataset"], map_name)
            map_root = ET.parse(map_path).getroot()
            map_ = map_parser.parse(
                map_root, map_config["project_rule"], map_config["gps_origin"], map_config
            )
            parsed_map_set.add(map_.name)

            fig, ax = plt.subplots()
            fig.set_layout_engine("none")
            ax.set_axis_off()
            scenario_display.reset()
            scenario_display.display_map(map_, ax)
            ax.set_aspect("equal")
            ax.plot()
            fig.savefig(f"./tests/runtime/{map_name}.png", dpi=300)

        except SyntaxError as err:
            logging.error(err)
        except KeyError as err:
            logging.error(err)
        except FileNotFoundError as err:
            raise err
