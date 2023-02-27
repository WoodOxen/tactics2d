import sys
sys.path.append(".")
sys.path.append("..")
import logging
logging.basicConfig(level=logging.DEBUG)
import xml.etree.ElementTree as ET
import json
import time

from tactics2d.map.parser import Lanelet2Parser
# from tactics2d.trajectory.parser import LevelXParser
from tactics2d.trajectory.parser import InteractionParser
from tactics2d.scenario.render_manager import RenderManager
from tactics2d.sensor import TopDownCamera


def test_camera():
    # map_path = "./tactics2d/data/map_default/I_0_inD_DEU.osm"
    map_path = "./tactics2d/data/map_default/I_5_INTERACTION_USA.osm"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_5"])


if __name__ == "__main__":

    map_path = "./tactics2d/data/map_default/I_5_INTERACTION_USA.osm"
    trajectory_path = "./tactics2d/data/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_5"])

    # trajectory_parser = LevelXParser("inD")
    trajectory_parser = InteractionParser()
    participants = trajectory_parser.parse(0, trajectory_path, (0., 200.))

    render_manager = RenderManager(
        map_, fps=200, windows_size=(1200, 1200), layout_style="hierarchical")

    perception_range = (30, 30, 45, 15)
    main_camera = TopDownCamera(1, map_, window_size=(800, 800))
    camera1 = TopDownCamera(2, map_, perception_range=perception_range, window_size=(200, 200))
    camera2 = TopDownCamera(3, map_, perception_range=perception_range, window_size=(200, 200))

    render_manager.add(main_camera, main_sensor=True)
    # render_manager.add(camera1)
    # render_manager.add(camera2)
    # render_manager.bind(2, 2)
    # render_manager.bind(3, 3)

    t1 = time.time()
    for step in range(1, 200):
        active_participant = set()
        for participant in participants.values():
            if participant.is_alive(int(step * 100)):
                active_participant.add(participant.id_)
            else:
                active_participant.discard(participant.id_)

        frame = int(step * 100)
        render_manager.update(participants, frame)
        render_manager.render()
    t2 = time.time()
    print("Average fps: ", 1//((t2 - t1)/200))