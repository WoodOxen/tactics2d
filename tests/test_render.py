import sys
sys.path.append(".")
sys.path.append("..")
import json
import xml.etree.ElementTree as ET

import pygame

from tactics2d.map.parser import Lanelet2Parser
from tactics2d.render.render_manager import RenderManager
from tactics2d.render.sensors import TopDownCamera


if __name__ == "__main__":
    map_path = "./tactics2d/data/map_default/I_0_inD_DEU.osm"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_0"])


    render_manager = RenderManager(map_, {})

    camera = TopDownCamera(1, map_, window_size=(800, 800))
    render_manager.add_sensor(camera)