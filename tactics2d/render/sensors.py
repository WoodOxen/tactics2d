from abc import ABC, abstractmethod
from typing import Tuple, Union
import warnings

import numpy as np
from shapely.geometry import Point
from shapely.affinity import affine_transform
import pygame

from tactics2d.map.element import Map
from .defaults import LANE_COLOR, AREA_COLOR, ROADLINE_COLOR
from .defaults import VEHICLE_COLOR, PEDESTRIAN_COLOR


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
    def _render_vehicles(self, vehicles: list):
        """_summary_

        Args:
            vehicles (_type_): _description_
        """

    @abstractmethod
    def _render_pedestrians(self, pedestrians: list):
        """_summary_

        Args:
            pedestrians (_type_): _description_
        """

    @abstractmethod
    def update(self):
        """_summary_
        """


class TopDownCamera(SensorBase):
    """Pseudo camera that provides a top-down view RGB semantic segmentation image of 
    the map and all the moving objects.

    Attributes:
        perception_range (Union[float, tuple]): The distance from the sensor to its (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None.
    """
    def __init__(
            self, sensor_id, map_: Map, perception_range: Union[float, Tuple[float]] = None,
            window_size: Tuple[int, int] = (200, 200), 
        ):

        super().__init__(sensor_id, map_)

        if perception_range is None:
            width = (map_.boundary[1] - map_.boundary[0]) / 2
            height = (map_.boundary[3] - map_.boundary[2]) / 2
            self.perception_range = (width, width, height, height)
        else:
            if isinstance(perception_range, tuple):
                self.perception_range = perception_range
            else:
                self.perception_range = \
                    (perception_range, perception_range, perception_range, perception_range)

        self.perception_width = self.perception_range[0] + self.perception_range[1]
        self.perception_height = self.perception_range[2] + self.perception_range[3]
        self.max_perception_distance = np.linalg.norm([
            max(self.perception_range[0], self.perception_range[1]),
            max(self.perception_range[2], self.perception_range[3])
        ])

        self.window_size = window_size

        if perception_range is not None:
            if self.window_size[0] / self.perception_width != self.window_size[1] / self.perception_height:
                warnings.warn("The height-width proportion of the perception and the image is inconsistent.")

        self.surface = pygame.Surface(window_size)
        self.position = None
        self.heading = None

    def _update_transform_matrix(self):
        if None in [self.position, self.heading]:
            if not hasattr(self, "transform_matrix"):
                scale = min(
                    self.window_size[0] / self.perception_width, 
                    self.window_size[1] / self.perception_height
                )

                x_center = 0.5 * (self.map_.boundary[0] + self.map_.boundary[1])
                y_center = 0.5 * (self.map_.boundary[2] + self.map_.boundary[3])

                self.transform_matrix = [
                    scale, 0, 0, -scale, 
                    0.5 * self.window_size[0] - scale * x_center,
                    0.5 * self.window_size[1] + scale * y_center
                ]
        else:
            scale = min(
                self.window_size[0] / self.perception_width, 
                self.window_size[1] / self.perception_height
            )
            theta = 3*np.pi/ 2 - self.heading if self.heading < 3*np.pi/2 else 5*np.pi/ 2 - self.heading

            self.transform_matrix = [theta]

    def _in_perception_range(self, geometry) -> bool:
        return geometry.distance(self.position) > self.max_perception_distance

    def _render_areas(self):
        for area in self.map_.areas.values():
            if self.position is not None:
                if self._in_perception_range(area.polygon):
                    continue

            color = AREA_COLOR[area.subtype] \
                if area.subtype in AREA_COLOR else AREA_COLOR["default"]
            color = color if area.color is None else area.color
            polygon = affine_transform(area.polygon, self.transform_matrix)
            outer_points = list(polygon.exterior.coords)
            inner_list = list(polygon.interiors)

            pygame.draw.polygon(self.surface, color, outer_points)
            for inner_points in inner_list:
                pygame.draw.polygon(self.surface, AREA_COLOR["hole"], inner_points)

    def _render_lanes(self):
        for lane in self.map_.lanes.values():
            if self.position is not None:
                if self._in_perception_range(lane.polygon):
                    continue

            color = LANE_COLOR[lane.subtype] \
                if lane.subtype in LANE_COLOR else LANE_COLOR["default"]
            color = color if lane.color is None else lane.color
            points = list(affine_transform(lane.polygon, self.transform_matrix).coords)

            pygame.draw.polygon(self.surface, color, points)

    def _render_roadlines(self):
        for roadline in self.map_.roadlines.values():
            if self.position is not None:
                if self._in_perception_range(roadline.linestring):
                    continue

            color = ROADLINE_COLOR[roadline.color] \
                if roadline.subtype in ROADLINE_COLOR else ROADLINE_COLOR["default"]
            color = color if roadline.color is None else roadline.color
            points = list(affine_transform(roadline.linestring, self.transform_matrix).coords)
            width = 2 if roadline.type_ == "line_thick" else 1

            pygame.draw.lines(self.surface, color, False, points, width)

    def _render_vehicles(self, vehicles):
        for vehicle in vehicles:
            if self.position is not None:
                if vehicle.pose.distance(self.position) > self.max_perception_distance:
                    continue

            color = VEHICLE_COLOR["default"] if vehicle.color is None else vehicle.color
            points = list(affine_transform(vehicle.pose))

            pygame.draw.polygon(self.surface, color, points)

    def _render_pedestrians(self, pedestrians):
        radius = max(1., min(
                self.window_size[0] / self.perception_width, 
                self.window_size[1] / self.perception_height
            ) / 10)
        for pedestrian in pedestrians:
            if self.position is not None:
                if pedestrian.pose.distance(self.position) > self.max_perception_distance:
                    continue
            
            color = PEDESTRIAN_COLOR["default"] \
                if pedestrian.color is None else pedestrian.color
            point = affine_transform(Point(pedestrian.location))

            pygame.draw.circle(self.surface, color, point, radius)

    def update(self, vehicles, pedestrians, position = None, location = None):
        self.position = position
        self.location = location
        self._update_transform_matrix()

        self.surface.fill((255, 255, 255))
        self._render_areas()
        self._render_lanes()
        self._render_roadlines()
        self._render_vehicles(vehicles)
        self._render_pedestrians(pedestrians)


class SingleLineLidar(SensorBase):
    def __init__(self, sensor_id):
        super().__init__(sensor_id)