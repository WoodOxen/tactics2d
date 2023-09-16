import sys

sys.path.append(".")
sys.path.append("..")

import json
import xml.etree.ElementTree as ET
import logging
from PIL import Image

from tactics2d.map.parser import Lanelet2Parser
from tactics2d.sensor import TopDownCamera

config = {
    "name": "Dragon Lake Parking Lot",
    "osm_path": "../data/map/DLP/DLP.osm",
    "country": "USA",
    "scenario": "parking",
    "dataset": "DLP",
    "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
    "gps_origin": [0, 0],
    "track_files": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    ],
}

map_file = config["osm_path"]

map_parser = Lanelet2Parser()
map_root = ET.parse(map_file).getroot()
map_ = map_parser.parse(map_root, config)

camera = TopDownCamera(1, map_, (20, 20, 20, 20), window_size=(600, 600))
camera.update(None, [])
observation = camera.get_observation()
img = Image.fromarray(observation)
img = img.rotate(180)
img.save("./test_viz.jpg")
