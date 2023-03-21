import sys

sys.path.append(".")
sys.path.append("..")

import json
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)

from tactics2d.map.parser import Lanelet2Parser
from tactics2d.map.parser import MapParseError
from tactics2d.trajectory.parser import DLPParser, InteractionParser, LevelXParser
from tactics2d.participant.guess_type import GuessType

@pytest.mark.map_parser
def test_lanelet2_parser():
    """Test whether the current parser can manage to parse the provided maps.

    [TODO] split this test to two part:
        One for testing the correctness of the provided maps' notations;
        One for testing the parser's ability to parse the lanelet2 format maps.
    """
    
    map_path = "./tactics2d/data/map_default/"
    config_path = "./tactics2d/data/map_default.config"

    map_parser = Lanelet2Parser()

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_list = set(configs.keys())
    parsed_map_set = set()

    for idx, map_config in configs.items():
        logging.info(f"Parsing map {map_config['map_name']}.")
        try:
            map_root = ET.parse(map_path + map_config["map_name"]).getroot()
            _ = map_parser.parse(map_root, map_config)
            parsed_map_set.add(idx)
        except MapParseError as err:
            logging.error(err)
        except KeyError as err:
            logging.error(err)

    assert len(map_list) == len(parsed_map_set)


@pytest.mark.trajectory_parser
@pytest.mark.parametrize(
    "dataset, file_id, stamp_range, expected",
    [
        ("highD", 1, (-float("inf"), float("inf")), 1047),
        ("inD", 0, (-float("inf"), float("inf")), 384),
        ("rounD", 0, (-float("inf"), float("inf")), 348),
        ("exiD", 1, (-float("inf"), float("inf")), 871),
        ("uniD", 0, (-float("inf"), float("inf")), 362),
        ("highD", 1, (100.0, 400.0), 386),
        ("inD", 0, (100.0, 400.0), 134),
        ("rounD", 0, (100.0, 400.0), 113),
        ("exiD", 1, (100.0, 300.0), 312),
        ("uniD", 0, (100.0, 300.0), 229),
    ],
)
def test_levelx_parser(dataset: str, file_id: int, stamp_range: tuple, expected: int):
    file_path = f"./tactics2d/data/trajectory_sample/{dataset}/data/"

    trajectory_parser = LevelXParser(dataset)

    participants = trajectory_parser.parse(file_id, file_path, stamp_range)
    assert len(participants) == expected


@pytest.mark.trajectory_parser
@pytest.mark.parametrize(
    "file_id, stamp_range, expected",
    [(0, (-float("inf"), float("inf")), 97), (0, (100.0, 400.0), 12)],
)
def test_interaction_parser(file_id: int, stamp_range: tuple, expected: int):
    folder_path = "./tactics2d/data/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0"
    clf = GuessType.get_svm_model("svm_model2.m")
    trajectory_parser = InteractionParser()

    participants = trajectory_parser.parse(file_id, folder_path, clf, stamp_range)
    print(len(participants))
    assert len(participants) == expected


@pytest.mark.trajectory_parser
@pytest.mark.parametrize(
    "file_id, stamp_range, expected",
    [(12, (-float("inf"), float("inf")), 342), (12, (100.0, 400.0), 317)],
)
def test_dlp_parser(file_id: int, stamp_range: tuple, expected: int):
    zip_path = "./tactics2d/data/trajectory_sample/DLP.zip"
    unzip_path = "./tactics2d/data/trajectory_sample"
    file_path = "./tactics2d/data/trajectory_sample/DLP"

    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(unzip_path)

    trajectory_parser = DLPParser()

    participants = trajectory_parser.parse(file_id, file_path, stamp_range)
    assert (len(participants)) == expected
