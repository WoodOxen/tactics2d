from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from shapely.geometry import Point
from shapely.affinity import affine_transform
import pygame

from .defaults import LANE_COLOR, AREA_COLOR, ROADLINE_COLOR
from .defaults import VEHICLE_COLOR, PEDESTRIAN_COLOR


class SensorBase(ABC):
    """The base of all the pseudo sensors provided in tactics2d.

    Attributes:
    """
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id

    @abstractmethod
    def _render_lane(self, lanes: list):
        """_summary_

        Args:
            lanes (_type_): _description_
        """
    
    @abstractmethod
    def _render_areas(self, areas: list):
        """_summary_

        Args:
            areas (_type_): _description_
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
        perception_range (Union[float, list]): [front, back, left, right]. Defaults to None.
    """
    def __init__(
            self, sensor_id, perception_range: Union[float, Tuple[float]] = None,
            window_size: Union[int, Tuple[int]] = (200, 200), 
        ):

        super().__init__(sensor_id)

        if perception_range is not None:
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

        if isinstance(window_size, tuple):
            self.window_size = window_size
        else:
            self.window_size = (window_size, window_size)

        if perception_range is not None:
            if self.window_size[0] / self.perception_width != self.window_size[1] / self.perception_height:
                print("The proportion of the perception and the image is not equal in width and height.")

        self.position = None
        self.heading = None


    def _update_transform_matrix(self, map_):
        if None in [self.perception_range, self.position, self.heading]:
            if not hasattr(self, "transform_matrix"):
                self.perception_width = map_.boundary[1]-map_.boundary[0]
                self.perception_height = map_.boundary[3]-map_.boundary[2]
                scale = min(
                    self.window_size[0] / self.perception_width, 
                    self.window_size[1] / self.perception_height
                )

                self.transform_matrix = [
                    scale, 0, 0, scale, 
                    0.5 * (-self.perception_width+self.window_size[0]),
                    0.5 * (-self.perception_height-self.window_size[1])
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

    def _render_areas(self, areas):
        for area in areas:
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

    def _render_lanes(self, lanes: list):
        for lane in lanes:
            if self.position is not None:
                if self._in_perception_range(lanes.polygon):
                    continue

            color = LANE_COLOR[lane.subtype] \
                if lane.subtype in LANE_COLOR else LANE_COLOR["default"]
            color = color if lane.color is None else lane.color
            points = list(affine_transform(lanes.polygon, self.transform_matrix).coords)

            pygame.draw.polygon(self.surface, color, points)

    def _render_roadlines(self, roadlines):
        for roadline in roadlines:
            if self.position is not None:
                if self._in_perception_range(roadline.linestring):
                    continue

            color = ROADLINE_COLOR[roadline.color] \
                if roadline.subtype in ROADLINE_COLOR else ROADLINE_COLOR["default"]
            color = color if roadline.color is None else roadline.color
            points = list(affine_transform(roadline.linestring))
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

    def update(self, map_, vehicles, pedestrians, position = None, location = None):
        self.position = position
        self.location = location
        self._update_transform_matrix(map_)

        self._render_areas(map_.areas.values())
        self._render_lanes(map_.lanes.values())
        self._render_roadlines(map_.roadlines.values())
        self._render_vehicles(vehicles)
        self._render_pedestrians(pedestrians)


class SingleLineLidar(SensorBase):
    def __init__(self, sensor_id):
        super().__init__(sensor_id)