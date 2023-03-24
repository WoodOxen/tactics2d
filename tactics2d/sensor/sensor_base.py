from abc import ABC, abstractmethod

from shapely.geometry import Point

from tactics2d.map.element import Map


class SensorBase(ABC):
    """This class define an interface for all the pseudo sensors provided in tactics2d.

    Attributes:
        id_ (int): The unique identifier of the sensor.
        map_ (Map): The map that the sensor is attached to.
    """

    def __init__(self, id_: int, map_: Map):
        self.id_ = id_
        self.map_ = map_

    @abstractmethod
    def update(self, participants, position: Point = None, heading: float = None):
        """Sync the sensor to a new viewpoint. Update the observation of the sensor."""

    @abstractmethod
    def get_observation(self):
        """Get the observation of the sensor from the viewpoint."""
