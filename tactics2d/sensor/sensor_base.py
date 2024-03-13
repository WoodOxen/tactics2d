##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: sensor_base.py
# @Description: This file defines the base class for all the pseudo sensors provided in tactics2d.
# @Author: Yueyuan Li
# @Version: 1.0.0

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import pygame
from shapely.geometry import Point

from tactics2d.map.element import Map


class SensorBase(ABC):
    """This class define an interface for all the pseudo sensors provided in tactics2d.

    Attributes:
        id_ (int): The unique identifier of the sensor.
        map_ (Map): The map that the sensor is attached to.
        perception_range (Union[float, Tuple[float]]): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None.
        window_size (Tuple[int, int]): The size of the rendering window. Defaults to (200, 200).
        off_screen (bool): Whether to render the sensor off screen. Defaults to True.
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
        """Initialize the sensor.

        Args:
            id_ (int): The unique identifier of the sensor.
            map_ (Map): The map that the sensor is attached to.
            perception_range (Union[float, Tuple[float]], optional): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map.
            window_size (Tuple[int, int], optional): The size of the rendering window.
            off_screen (bool, optional): Whether to render the sensor off screen.
        """
        self.id_ = id_
        self.map_ = map_
        self.off_screen = off_screen
        self.window_size = window_size
        self._surface = pygame.Surface(self.window_size)
        self._bind_id = None
        self._heading = None
        self._position = None

        if perception_range is None:
            width = (map_.boundary[1] - map_.boundary[0]) / 2
            height = (map_.boundary[3] - map_.boundary[2]) / 2
            self.perception_range = (width, width, height, height)
        elif isinstance(perception_range, float) or isinstance(perception_range, int):
            self.perception_range = (
                perception_range,
                perception_range,
                perception_range,
                perception_range,
            )
        else:
            self.perception_range = perception_range

        perception_width = self.perception_range[0] + self.perception_range[1]
        perception_height = self.perception_range[2] + self.perception_range[3]

        scale_width = window_size[0] / perception_width
        scale_height = window_size[1] / perception_height
        self.scale = max(scale_width, scale_height)

        if scale_width != scale_height:
            logging.warning(
                "The x-y proportion of the perception range and the rendering window is inconsistent. "
            )

    @property
    def heading(self):
        return self._heading

    @property
    def position(self):
        return self._position

    @property
    def max_perception_distance(self):
        return np.max(self.perception_range)

    @property
    def surface(self):
        return self._surface

    @property
    def bind_id(self):
        return self._bind_id

    def set_bind_id(self, bind_id):
        """This function is used to set the bind id of the sensor.

        Args:
            bind_id (int): The unique identifier of the participant that the sensor is bound to.
        """
        self._bind_id = bind_id

    @abstractmethod
    def update(self, participants, position: Point = None, heading: float = None):
        """This function is used to update the sensor's location and observation.

        Args:
            participants (Dict[int, Participant]): The participants in the scenario.
            position (Point, optional): The position of the sensor. Defaults to None.
            heading (float, optional): The heading of the sensor. Defaults to None.
        """

    @abstractmethod
    def get_observation(self):
        """This function is used to get the observation of the sensor from the viewpoint."""
