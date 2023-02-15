from abc import ABC, abstractmethod

import pygame


class SensorBase(ABC):
    """The base of all the pseudo sensors provided in tactics2d.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, sensor_id, position) -> None:
        self.renderer_id = sensor_id
        self.position = position

    def _transform_to_position(self, geometry):
        """Transform a point or a list of points around the sensor's position.

        Args:
            geometry (_type_): _description_
        """
        return
    
    def _map_to_surface_coords(self, geometry):
        """Map a point or a list of points to the pygame.surface.

        Args:
            geometry (_type_): _description_
        """
        return

    @abstractmethod
    def _render_lane(self, lanes):
        """_summary_

        Args:
            lanes (_type_): _description_
        """
    
    @abstractmethod
    def _render_area(self, areas):
        """_summary_

        Args:
            areas (_type_): _description_
        """

    @abstractmethod
    def _render_roadline(self, roadlines):
        """_summary_

        Args:
            roadlines (_type_): _description_
        """

    @abstractmethod
    def _render_vehicle(self, vehicles):
        """_summary_

        Args:
            vehicles (_type_): _description_
        """

    @abstractmethod
    def _render_pedestrian(self, pedestrians):
        """_summary_

        Args:
            pedestrians (_type_): _description_
        """

    @abstractmethod
    def update(self):
        """_summary_
        """


class TopDownCamera(SensorBase):
    """Pseudo camera that provides a bird-eye view RGB semantic segmentation image of 
    the map and all the moving objects.

    Args:
        SensorBase (_type_): _description_
    """
    def __init__(self, sensor_id, position) -> None:
        super().__init__(sensor_id, position)

    def _render_lane(self, lanes):

        return

    def _render_area(self, areas):
        return

    def _render_roadline(self, roadlines):
        return

    def _render_vehicle(self, vehicles):
        return

    def _render_pedestrian(self, pedestrians):
        return

    def update(self, map):
        self._render_lane()
        self._render_area()
        self._render_roadline()
        self._render_vehicle()
        self._render_pedestrian()
