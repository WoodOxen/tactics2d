from abc import ABC, abstractmethod

from tactics2d.map.element import Map


class SensorBase(ABC):
    """The base of all the pseudo sensors provided in tactics2d.

    Attributes:
    """

    def __init__(self, sensor_id, map_: Map):
        self.sensor_id = sensor_id
        self.map_ = map_

    @abstractmethod
    def _render_areas(self, areas: list):
        """_summary_

        Args:
            areas (_type_): _description_
        """

    @abstractmethod
    def _render_lanes(self, lanes: list):
        """_summary_

        Args:
            lanes (_type_): _description_
        """

    @abstractmethod
    def _render_roadlines(self, roadlines: list):
        """_summary_

        Args:
            roadlines (_type_): _description_
        """

    @abstractmethod
    def _render_participants(self, vehicles: list):
        """_summary_

        Args:
            vehicles (_type_): _description_
        """

    @abstractmethod
    def update(self):
        """_summary_"""
