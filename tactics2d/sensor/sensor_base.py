from abc import ABC, abstractmethod
from typing import Iterable

from shapely.geometry import Point
from pygame.colordict import THECOLORS

from tactics2d.map.element import Map


class SensorBase(ABC):
    """This class define an interface for all the pseudo sensors provided in tactics2d.

    Attributes:
        sensor_id (str): The unique identifier of the sensor.
        map_ (Map): The map that the sensor is attached to.
    """

    BG_COLOR = THECOLORS["white"]

    def __init__(self, sensor_id, map_: Map):
        self.sensor_id = sensor_id
        self.map_ = map_

    @abstractmethod
    def update(self, participants, position: Point = None, heading: float = None):
        """Sync the sensor to a new viewpoint. Update the observation of the sensor."""

    @abstractmethod
    def get_observation(self):
        """Get the observation of the sensor from the viewpoint."""
