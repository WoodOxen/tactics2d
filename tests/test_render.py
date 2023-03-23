import sys

sys.path.append(".")
sys.path.append("..")

import json
import time
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.DEBUG)

import numpy as np

from tactics2d.map.parser import Lanelet2Parser
from tactics2d.trajectory.parser import LevelXParser
from tactics2d.scenario.render_manager import RenderManager
from tactics2d.sensor import TopDownCamera, SingleLineLidar


def test_camera():
    camera = TopDownCamera(1, None, window_size=(600, 600))


def test_lidar():
    lidar = SingleLineLidar(1, None, window_size=(600, 600))


def test_render_manager(map_, participants, layout_style):
    """This function tests the following functions in RenderManager:
        _rearrange_layout, add, is_bound, bind, unbind, remove_sensor, update, render, close

    Args:
        map_ (_type_): _description_
        participants (_type_): _description_
        layout_style (_type_): _description_
    """
    render_manager = RenderManager(
        fps=100, windows_size=(600, 600), layout_style=layout_style
    )

    perception_range = (30, 30, 45, 15)
    main_camera = TopDownCamera(1, map_, window_size=(600, 600))
    camera1 = TopDownCamera(
        2, map_, perception_range=perception_range, window_size=(200, 200)
    )
    camera2 = TopDownCamera(
        3, map_, perception_range=perception_range, window_size=(200, 200)
    )

    render_manager.add(main_camera, main_sensor=True)
    render_manager.add(camera1)
    render_manager.add(camera2)

    def auto_bind_camera(camera, participant_ids, bind_target):
        bind_id = render_manager.is_bound(camera.id_)
        if bind_id is None:
            render_manager.bind(camera.id_, participant_ids[bind_target])
        elif not participants[bind_id].is_active(frame):
            render_manager.unbind(camera.id_)
            render_manager.bind(camera.id_, participant_ids[bind_target])

    t1 = time.time()

    for frame in np.arange(0, 50 * 1000, 40):
        participant_ids = [
            participant.id_
            for participant in participants.values()
            if participant.is_active(frame)
        ]

        auto_bind_camera(camera1, participant_ids, 0)
        auto_bind_camera(camera2, participant_ids, 1)

        render_manager.update(participants, participant_ids, frame)
        render_manager.render()

    render_manager.close()

    t2 = time.time()

    average_fps = 1 // ((t2 - t1) / (50 * 1000 / 40))

    logging.debug(f"The average FPS is {average_fps}")


def test_off_screen_rendering():
    return


if __name__ == "__main__":
    map_path = "./tactics2d/data/map_default/I_0_inD_DEU.osm"
    trajectory_path = "./tactics2d/data/trajectory_sample/inD/data/"
    config_path = "./tactics2d/data/map_default.config"

    with open(config_path, "r") as f:
        configs = json.load(f)

    map_parser = Lanelet2Parser()
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(map_root, configs["I_0"])

    trajectory_parser = LevelXParser("inD")
    participants = trajectory_parser.parse(0, trajectory_path, (0.0, 200.0))

    test_camera()

    test_lidar()

    test_render_manager(map_, participants, "modular")
    test_render_manager(map_, participants, "hierarchical")

    test_off_screen_rendering()
