import sys

sys.path.append(".")
sys.path.append("..")

import json
import logging
import os
import platform
import time
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pygame
import pytest
from PIL import Image
from shapely.geometry import Point

from tactics2d.dataset_parser import LevelXParser
from tactics2d.map.parser import OSMParser
from tactics2d.sensor import RenderManager, SingleLineLidar, TopDownCamera


@pytest.mark.render
@pytest.mark.parametrize("follow_view", [True, False])
def test_camera(follow_view: bool):
    map_path = "./tactics2d/data/map/inD/inD_4.osm"
    trajectory_path = "./tactics2d/data/trajectory_sample/inD/data/"
    config_path = "./tactics2d/data/map/map.config"

    with open(config_path) as f:
        configs = json.load(f)

    map_parser = OSMParser(lanelet2=True)
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(
        map_root, configs["inD_4"]["project_rule"], configs["inD_4"]["gps_origin"], configs["inD_4"]
    )

    frame = 40
    dataset_parser = LevelXParser("inD")
    participants, _ = dataset_parser.parse_trajectory(0, trajectory_path, (0, 10000))
    participant_ids = [
        participant.id_ for participant in participants.values() if participant.is_active(frame)
    ]

    if follow_view:
        camera = TopDownCamera(1, map_, window_size=(600, 600))
        camera.update(participants, participant_ids, frame)
    else:
        camera = TopDownCamera(1, map_, (30, 30, 45, 15), window_size=(600, 600))
        state = participants[participant_ids[0]].get_state(frame)
        camera.update(participants, participant_ids, frame, Point(state.location), state.heading)
    observation = camera.get_observation()
    logging.info(f"observation.shape: {observation.shape}")

    img = Image.fromarray(observation)
    img = img.rotate(270)

    if not os.path.exists("./tests/runtime"):
        os.makedirs("./tests/runtime")
    if follow_view:
        img.save("./tests/runtime/test_camera_follow_view.jpg")
    else:
        img.save("./tests/runtime/test_camera.jpg")


@pytest.mark.render
@pytest.mark.parametrize("perception_range", [12.0, 30.0, 45.0, 100.0])
def test_lidar(perception_range):
    map_path = "./tactics2d/data/map/inD/inD_4.osm"
    trajectory_path = "./tactics2d/data/trajectory_sample/inD/data/"
    config_path = "./tactics2d/data/map/map.config"

    with open(config_path) as f:
        configs = json.load(f)

    map_parser = OSMParser(lanelet2=True)
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(
        map_root, configs["inD_4"]["project_rule"], configs["inD_4"]["gps_origin"], configs["inD_4"]
    )

    frame = 40
    dataset_parser = LevelXParser("inD")
    participants, _ = dataset_parser.parse_trajectory(0, trajectory_path, (0, 10000))
    participant_ids = [
        participant.id_ for participant in participants.values() if participant.is_active(frame)
    ]

    lidar = SingleLineLidar(1, map_, perception_range, window_size=(600, 600), off_screen=False)

    state = participants[participant_ids[0]].get_state(frame)
    lidar.update(participants, participant_ids[1:], frame, Point(state.location), state.heading)
    _ = lidar.get_observation()

    observation = pygame.surfarray.array3d(lidar.surface)
    img = Image.fromarray(observation)
    img = img.rotate(270)

    if not os.path.exists("./tests/runtime"):
        os.makedirs("./tests/runtime")
    img.save(f"./tests/runtime/test_lidar_{int(perception_range)}.jpg")


@pytest.mark.render
@pytest.mark.skipif(platform.system() == "Darwin", reason="This test is not supported on MacOS.")
@pytest.mark.parametrize(
    "layout_style, off_screen",
    [("block", False), ("hierarchy", False), ("block", True), ("hierarchy", True)],
)
def test_render_manager(layout_style, off_screen):
    """This function tests the following functions in RenderManager:
    _rearrange_layout, add, is_bound, bind, unbind, remove_sensor, update, render, reset, close
    """
    map_path = "./tactics2d/data/map/inD/inD_4.osm"
    trajectory_path = "./tactics2d/data/trajectory_sample/inD/data/"
    config_path = "./tactics2d/data/map/map.config"

    with open(config_path) as f:
        configs = json.load(f)

    map_parser = OSMParser(lanelet2=True)
    map_root = ET.parse(map_path).getroot()
    map_ = map_parser.parse(
        map_root, configs["inD_4"]["project_rule"], configs["inD_4"]["gps_origin"], configs["inD_4"]
    )

    dataset_parser = LevelXParser("inD")
    participants, _ = dataset_parser.parse_trajectory(0, trajectory_path, (0, 10000))

    render_manager = RenderManager(
        fps=100, windows_size=(600, 600), layout_style=layout_style, off_screen=off_screen
    )

    perception_range = (30, 30, 45, 15)
    main_camera = TopDownCamera(1, map_, window_size=(600, 600), off_screen=off_screen)
    camera1 = TopDownCamera(
        2, map_, perception_range=perception_range, window_size=(200, 200), off_screen=off_screen
    )
    camera2 = TopDownCamera(
        3, map_, perception_range=perception_range, window_size=(200, 200), off_screen=off_screen
    )

    render_manager.add_sensor(main_camera, main_sensor=True)
    render_manager.add_sensor(camera1)
    render_manager.add_sensor(camera2)

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
            participant.id_ for participant in participants.values() if participant.is_active(frame)
        ]

        if len(participant_ids) == 1:
            auto_bind_camera(camera1, participant_ids, 0)
            render_manager.unbind(camera2.id_)
        elif len(participant_ids) >= 2:
            auto_bind_camera(camera1, participant_ids, 0)
            auto_bind_camera(camera2, participant_ids, 1)

        render_manager.update(participants, participant_ids, frame)
        if not "DISPLAY" in os.environ:
            render_manager.render()

    render_manager.remove_sensor(0)
    render_manager.remove_sensor(1)
    render_manager.reset()
    render_manager.close()

    t2 = time.time()

    average_fps = 1 // ((t2 - t1) / (50 * 1000 / 40))

    logging.debug(f"The average FPS is {average_fps}")


@pytest.mark.skip(reason="This test is not implemented yet.")
def test_off_screen_rendering():
    return
