from typing import Tuple, Union
from abc import ABC, abstractmethod
import warnings

import numpy as np
from shapely.geometry import Point
import pygame

from tactics2d.map.element import Map


class SensorBase(ABC):
    """This class define an interface for all the pseudo sensors provided in tactics2d.

    Attributes:
        id_ (int): The unique identifier of the sensor.
        map_ (Map): The map that the sensor is attached to.
        perception_range (Union[float, tuple]): The distance from the sensor to its maximum detection range in
            (left, right, front, back). When this value is undefined, the camera is assumed to
            detect the whole map. Defaults to None.
        window_size (Tuple[int, int]): The size of the rendering window. Defaults to (200, 200).
    """

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: Union[float, Tuple[float]] = None,
        window_size: Tuple[int, int] = (200, 200),
        off_screen: bool = True,
    ):
        self.id_ = id_
        self.map_ = map_
        self.off_screen = off_screen

        if perception_range is None:
            width = (map_.boundary[1] - map_.boundary[0]) / 2
            height = (map_.boundary[3] - map_.boundary[2]) / 2
            self.perception_range = (width, width, height, height)
        elif isinstance(perception_range, float):
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
            warnings.warn(
                "The x-y proportion of the perception range and the rendering window is inconsistent. "
            )

        self.max_perception_distance = np.max(self.perception_range)

        self.window_size = window_size
        self.surface = pygame.Surface(self.window_size)

    @abstractmethod
    def update(self, participants, position: Point = None, heading: float = None):
        """Sync the sensor to a new viewpoint. Update the observation of the sensor."""

    @abstractmethod
    def get_observation(self):
        """Get the observation of the sensor from the viewpoint."""
