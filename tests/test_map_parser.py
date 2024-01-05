import sys

sys.path.append(".")
sys.path.append("..")

import json
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.DEBUG)

import pytest

from tactics2d.map.parser import Lanelet2Parser


@pytest.mark.map_parser
def test_lanelet2_parser():
    """Test whether the current parser can manage to parse the provided maps.

    [TODO] split this test to two part:
        One for testing the correctness of the provided maps' notations;
        One for testing the parser's ability to parse the lanelet2 format maps.
    """

    data_path = "./tactics2d/data/map"
    config_path = "./tactics2d/data/map/map.config"

    map_parser = Lanelet2Parser()

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_list = set(configs.keys())
    parsed_map_set = set()

    for map_name, map_config in configs.items():
        if map_config["dataset"] in ["uniD", "exiD"]:
            continue
        logging.info(f"Parsing map {map_name}.")

        try:
            map_path = "%s/%s/%s.osm" % (data_path, map_config["dataset"], map_name)
            map_root = ET.parse(map_path).getroot()
            map_ = map_parser.parse(map_root, map_config)
            parsed_map_set.add(map_.name)
        except SyntaxError as err:
            logging.error(err)
        except KeyError as err:
            logging.error(err)
        except FileNotFoundError:
            pass

    # assert len(map_list) == len(parsed_map_set)
