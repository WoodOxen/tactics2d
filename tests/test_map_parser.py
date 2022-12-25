import sys
sys.path.append(".")
sys.path.append("..")
import os
import json
import xml.etree.ElementTree as ET

from tactics2d.map_parser.lanelet2_parser import Lanelet2Parser


if __name__ == "__main__":

    map_parser = Lanelet2Parser()

    map_dir = "../data/maps/defaults/"
    config_dir = "../data/maps/configs/"
    config_list = os.listdir(config_dir)

    for config_file in config_list:
        map_configs = json.load(open(config_dir+config_file, "r"))
        for map_name, map_config in map_configs.items():
            print("Parsing map %s" % map_name)
            map_file = map_name + ".osm"
            map_root = ET.parse(map_dir+map_file).getroot()
            map = map_parser.parse(map_root, map_config)