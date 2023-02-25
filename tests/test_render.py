import sys
sys.path.append(".")
sys.path.append("..")
import json
import xml.etree.ElementTree as ET

from tactics2d.participant.element import Vehicle
from tactics2d.trajectory.element import State, Trajectory
from tactics2d.map.parser import Lanelet2Parser
from tactics2d.sensor import TopDownCamera
from tactics2d.scenario.render_manager import RenderManager


if __name__ == "__main__":
    map_path = "./tactics2d/data/map_default/I_0_inD_DEU.osm"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_0"])

    render_manager = RenderManager(map_, fps=1, windows_size=(800, 800), layout_style="modular")

    position = ((map_.boundary[0] + map_.boundary[1]) / 2, (map_.boundary[2] + map_.boundary[3]) / 2)
    heading = 0

    perception_range = 100
    camera1 = TopDownCamera(1, map_, perception_range=perception_range, window_size=(400, 400))

    trajectory = Trajectory(1)
    trajectory.append_state(State(0, position[0], position[1], heading))
    vehicle = Vehicle(1, trajectory=trajectory)
    print(map_.boundary, position, vehicle.pose)

    participants = {1: vehicle}
    camera2 = TopDownCamera(2, map_, window_size=(800, 800))
    render_manager.add_sensor(camera1)
    render_manager.bind(1, 1)
    render_manager.add_sensor(camera2)
    while True:
        render_manager.update(participants)
        render_manager.render()