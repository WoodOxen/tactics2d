import sys
sys.path.append(".")
sys.path.append("..")
import json
import xml.etree.ElementTree as ET

import pytest

from tactics2d.map.parser.lanelet2_parser import Lanelet2Parser
from tactics2d.map.parser import MapParseError
from tactics2d.trajectory.parser.parse_dlp import DLPParser
from tactics2d.trajectory.parser.parse_interaction import InteractionParser
from tactics2d.trajectory.parser.parse_levelx import LevelXParser


@pytest.mark.map
def test_lanelet2_parser():
    """Test whether the current parser can manage to parse the provided maps.
    
    [TODO] split this test to two part: 
        One for testing the correctness of the provided maps' notations;
        One for testing the parser's ability to parse the lanelet2 format maps.
    """
    map_path = "../tactics2d/data/map_default/"
    config_path = "../tactics2d/data/map_default.config"

    map_parser = Lanelet2Parser()

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_list = set(configs.keys())
    parsed_map_set = set()

    for idx, map_config in configs.items():
        print(f"Parsing map {map_config['map_name']}.")
        try:
            map_root = ET.parse(map_path+map_config["map_name"]).getroot()
            _ = map_parser.parse(map_root, map_config)
            parsed_map_set.add(idx)
        except MapParseError as err:
            print(err)

    assert len(map_list) == len(parsed_map_set)


@pytest.mark.trajectory
@pytest.mark.parametrize(
    "dataset, file_id, stamp_range, expected",
    [
    ("highD", 1, (100., 400.), (1047, 386)),
    ("inD", 0, (100., 400.), (384, 134)),
    ("rounD", 0, (100., 400.), (348, 134)),
    ("exiD", 0, (100., 300.), (362, 134)),
    ("uniD", 0, (100., 300.), (362, 134))
    ],
)
def test_levelx_parser(dataset, file_id, stamp_range, expected):
    file_path = f"../tactics2d/data/trajectory_sample/{dataset}/data/"

    trajectory_parser = LevelXParser(dataset)

    # test the dataset with full range
    participants = trajectory_parser.parse(file_id, file_path)
    assert len(participants) == expected[0]

    # test the dataset with limited range
    participants = trajectory_parser.parse(file_id, file_path, stamp_range)
    assert len(participants) == expected[1]


@pytest.mark.trajectory
def test_interaction_parser():
    return


@pytest.mark.trajectory
@pytest.mark.parametrize(
    "file_id, file_path, stamp_range, processed, expected",
    [
    (12, "../tactics2d/data/trajectory_sample/DLP/", (100., 400.), False, (342, 317))
    ]
)
def test_dlp_parser(file_id, file_path, stamp_range, processed, expected):
    # processed_file_path = "../tactics2d/data/trajectory_test_processed/DLP/"

    trajectory_parser = DLPParser()

    # test parsing the dataset with full range
    participants = trajectory_parser.parse(
        file_id, file_path, processed = processed)
    assert (len(participants)) == expected[0]

    # test parsing the dataset with limited range
    participants = trajectory_parser.parse(
        file_id, file_path, stamp_range=stamp_range, processed = processed)
    assert (len(participants)) == expected[1]