import sys
sys.path.append(".")
sys.path.append("..")
import logging
logging.basicConfig(level=logging.DEBUG)
import xml.etree.ElementTree as ET
import json

from tactics2d.map.parser import Lanelet2Parser
from tactics2d.trajectory.parser import LevelXParser
from tactics2d.scenario.render_manager import RenderManager
from tactics2d.sensor import TopDownCamera


def test_camera():
    map_path = "./tactics2d/data/map_default/I_0_inD_DEU.osm"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_0"])




if __name__ == "__main__":

    map_path = "./tactics2d/data/map_default/I_0_inD_DEU.osm"
    trajectory_path = "./tactics2d/data/trajectory_sample/inD/data"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_0"])

    trajectory_parser = LevelXParser("inD")
    participants = trajectory_parser.parse(0, trajectory_path, (0., 100.))

    render_manager = RenderManager(
        map_, fps=1, windows_size=(800, 800), layout_style="hierarchical")

    perception_range = (30, 30, 45, 15)
    main_camera = TopDownCamera(1, map_, window_size=(800, 800))
    camera1 = TopDownCamera(2, map_, perception_range=perception_range, window_size=(200, 200))
    camera2 = TopDownCamera(3, map_, perception_range=perception_range, window_size=(200, 200))

    render_manager.add(main_camera, main_sensor=True)
    render_manager.add(camera1)
    render_manager.add(camera2)
    render_manager.bind(2, 1)
    render_manager.bind(3, 2)


    for step in range(100):
        frame = int(step / 25 * 1000)
        render_manager.update(participants, frame)
        render_manager.render()

    # perception_range = 100
    # camera1 = TopDownCamera(
    #     1, map_, perception_range=perception_range, window_size=(400, 400))

    # trajectory = Trajectory(1)
    # trajectory.append_state(State(0, position[0], position[1], heading))
    # vehicle = Vehicle(1, trajectory=trajectory)
    # print(map_.boundary, position, vehicle.pose)

    # participants = {1: vehicle}
    # camera2 = TopDownCamera(2, map_, window_size=(800, 800))
    # render_manager.add_sensor(camera1)
    # render_manager.bind(1, 1)
    # render_manager.add_sensor(camera2)
    # while True:
    #     render_manager.update(participants)
    #     render_manager.render()
