import sys
sys.path.append(".")
sys.path.append("..")
import json
import xml.etree.ElementTree as ET

from tactics2d.map.parser import Lanelet2Parser


if __name__ == "__main__":

    map_parser = Lanelet2Parser()

    map_path = "../tactics2d/data/map_default/"
    config_path = "../tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    for map_config in configs.values():
        print(f"Parsing map {map_config['map_name']}.")
        map_root = ET.parse(map_path+map_config["map_name"]).getroot()
        try:
            map_ = map_parser.parse(map_root, map_config)
            del map_
        except Exception as err:
            print(f"Failed to parse map {map_config['map_name']}.")
            print(err)
