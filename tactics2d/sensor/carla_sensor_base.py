##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: camera.py
# @Description: This file implements a pseudo camera with top-down view RGB semantic segmentation image.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, Union

import numpy as np
import pygame
from shapely.affinity import affine_transform
from shapely.geometry import Point

from tactics2d.map.element import Area, Lane, Map, RoadLine
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle

from .render_template import COLOR_PALETTE, DEFAULT_COLOR
from .sensor_base import SensorBase


class CarlaSensorBase(SensorBase):
    """This class implements a CarlaSensorBase.

    Attributes:
        id_ (int): The unique identifier of the sensor.
        map_ (Map): The map that the camera is attached to.
        perception_range (Union[float, Tuple[float]]): The distance from the camera to its maximum detection range in (left, right, front, back). When this value is undefined, the camera is assumed to detect the whole map. Defaults to None.
        window_size (Tuple[int, int]): The size of the rendering window. Defaults to (200, 200).
        off_screen (bool): Whether to render the camera off screen. Defaults to True.
        scale (float): The scale of the rendering window.
        bind_id (int): The unique identifier of the participant that the sensor is bound to.
        surface (pygame.Surface): The rendering surface of the sensor. This attribute is **read-only**.
        heading (float): The heading of the sensor. This attribute is **read-only**.
        position (Point): The position of the sensor. This attribute is **read-only**.
        max_perception_distance (float): The maximum detection range of the sensor. This attribute is **read-only**.
    """

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: Union[float, Tuple[float]] = None,
        window_size: Tuple[int, int] = (200, 200),
        off_screen: bool = True,
    ):
        """Initialize the top-down camera.

        Args:
            id_ (int): The unique identifier of the sensor.
            map_ (Map): The map that the sensor is attached to.
            perception_range (Union[float, tuple], optional): The distance from the camera to its maximum detection range in (left, right, front, back). When this value is undefined, the camera is assumed to detect the whole map.
            window_size (Tuple[int, int], optional): The size of the rendering window.
            off_screen (bool, optional): Whether to render the sensor off screen.
        """
        super().__init__(id_, map_, perception_range, window_size, off_screen)


    def update(
        self,
        participants,
        participant_ids: list,
        frame: int = None,
        position: Point = None,
        heading: float = None,
    ):
        """This function is used to update the camera's location and observation.

        Args:
            participants (_type_): The participants in the scenario.
            participant_ids (list): The ids of the participants in the scenario.
            frame (int, optional): The frame of the scenario. If None, the sensor will update to the current frame.
            position (Point, optional): The position of the sensor.
            heading (float, optional): The heading of the sensor.
        """
        pass

    def get_observation(self) -> np.ndarray:
        """This function is used to get the observation of the camera from the viewpoint.

        Returns:
            The observation of the sensor.
        """
        return pygame.surfarray.array3d(self._surface)
