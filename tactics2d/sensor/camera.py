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


class TopDownCamera(SensorBase):
    """This class implements a pseudo camera with top-down view RGB semantic segmentation image.

    Attributes:
        id_ (int): The unique identifier of the camera.
        map_ (Map): The map that the camera is attached to.
        perception_range (Union[float, Tuple[float]]): The distance from the camera to its maximum detection range in (left, right, front, back). When this value is undefined, the camera is assumed to detect the whole map. Defaults to None.
        window_size (Tuple[int, int]): The size of the rendering window. Defaults to (200, 200).
        off_screen (bool): Whether to render the camera off screen. Defaults to True.
        scale (float): The scale of the rendering window.
        bind_id (int): The unique identifier of the participant that the sensor is bound to.
        surface (pygame.Surface): The rendering surface of the sensor. This attribute is **read-only**.
        heading (float): The heading of the camera. This attribute is **read-only**.
        position (Point): The position of the camera. This attribute is **read-only**.
        max_perception_distance (float): The maximum detection range of the camera. This attribute is **read-only**.
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
            id_ (int): The unique identifier of the camera.
            map_ (Map): The map that the camera is attached to.
            perception_range (Union[float, tuple], optional): The distance from the camera to its maximum detection range in (left, right, front, back). When this value is undefined, the camera is assumed to detect the whole map.
            window_size (Tuple[int, int], optional): The size of the rendering window.
            off_screen (bool, optional): Whether to render the camera off screen.
        """
        super().__init__(id_, map_, perception_range, window_size, off_screen)

    def _update_transform_matrix(self):
        if None in [self._position, self._heading]:
            if not hasattr(self, "transform_matrix"):
                x_center = 0.5 * (self.map_.boundary[0] + self.map_.boundary[1])
                y_center = 0.5 * (self.map_.boundary[2] + self.map_.boundary[3])

                self.transform_matrix = np.array(
                    [
                        self.scale,
                        0,
                        0,
                        -self.scale,
                        0.5 * self.window_size[0] - self.scale * x_center,
                        0.5 * self.window_size[1] + self.scale * y_center,
                    ]
                )
        else:
            theta = self._heading - np.pi / 2

            self.transform_matrix = self.scale * np.array(
                [
                    np.cos(theta),
                    np.sin(theta),
                    np.sin(theta),
                    -np.cos(theta),
                    self.perception_range[0]
                    - self._position.x * np.cos(theta)
                    - self._position.y * np.sin(theta),
                    self.perception_range[2]
                    - self._position.x * np.sin(theta)
                    + self._position.y * np.cos(theta),
                ]
            )

    def _in_perception_range(self, geometry) -> bool:
        return geometry.distance(self._position) > self.max_perception_distance * 2

    def _get_color(self, element):
        if element.color in COLOR_PALETTE:
            return pygame.Color(COLOR_PALETTE[element.color])

        if element.color is None:
            if hasattr(element, "subtype") and element.subtype in DEFAULT_COLOR:
                return pygame.Color(DEFAULT_COLOR[element.subtype])
            if hasattr(element, "type_") and element.type_ in DEFAULT_COLOR:
                return pygame.Color(DEFAULT_COLOR[element.type_])
            elif isinstance(element, Area):
                return pygame.Color(DEFAULT_COLOR["area"])
            elif isinstance(element, Lane):
                return pygame.Color(DEFAULT_COLOR["lane"])
            elif isinstance(element, RoadLine):
                return pygame.Color(DEFAULT_COLOR["roadline"])

        return element.color

    def _render_areas(self):
        for area in self.map_.areas.values():
            if self._position is not None:
                if self._in_perception_range(area.geometry):
                    continue

            color = self._get_color(area)
            polygon = affine_transform(area.geometry, self.transform_matrix)
            outer_points = list(polygon.exterior.coords)
            inner_list = list(polygon.interiors)

            pygame.draw.polygon(self._surface, color, outer_points)
            for inner_points in inner_list:
                pygame.draw.polygon(
                    self._surface, pygame.Color(DEFAULT_COLOR["hole"]), inner_points
                )

    def _render_lanes(self):
        for lane in self.map_.lanes.values():
            if self._position is not None:
                if self._in_perception_range(lane.geometry):
                    continue

            color = self._get_color(lane)
            points = list(affine_transform(lane.geometry, self.transform_matrix).coords)

            pygame.draw.polygon(self._surface, color, points)

    def _render_roadlines(self):
        for roadline in self.map_.roadlines.values():
            if self._position is not None:
                if self._in_perception_range(roadline.geometry):
                    continue

            color = self._get_color(roadline)
            points = list(affine_transform(roadline.geometry, self.transform_matrix).coords)

            if roadline.type_ == "line_thick":
                width = max(2, 0.2 * self.scale)
            else:
                width = max(1, 0.1 * self.scale)

            pygame.draw.aalines(self._surface, color, False, points, width)

    def _render_vehicle(self, vehicle: Vehicle, frame: int = None):
        color = self._get_color(vehicle)
        points = np.array(affine_transform(vehicle.get_pose(frame), self.transform_matrix).coords)
        triangle = [
            (points[0] + points[1]) / 2,
            (points[1] + points[2]) / 2,
            (points[3] + points[0]) / 2,
        ]

        pygame.draw.polygon(self._surface, color, points)
        pygame.draw.polygon(self._surface, (0, 0, 0), triangle, width=1)

    def _render_cyclist(self, cyclist: Cyclist, frame: int = None):
        color = self._get_color(cyclist)
        points = list(affine_transform(cyclist.get_pose(frame), self.transform_matrix).coords)

        pygame.draw.polygon(self._surface, color, points)

    def _render_pedestrian(self, pedestrian: Pedestrian, frame: int = None):
        color = self._get_color(pedestrian)
        point = affine_transform(
            Point(pedestrian.trajectory.get_state(frame).location), self.transform_matrix
        )
        radius = max(1, 0.5 * self.scale)

        pygame.draw.circle(self._surface, color, point, radius)

    def _render_participants(self, participants: dict, participant_ids: list, frame: int = None):
        for participant_id in participant_ids:
            participant = participants[participant_id]

            state = participant.trajectory.get_state(frame)
            if self._position is not None:
                if self._in_perception_range(Point(state.location)):
                    continue

            if isinstance(participant, Vehicle):
                self._render_vehicle(participant, frame)
            elif isinstance(participant, Pedestrian):
                self._render_pedestrian(participant, frame)
            elif isinstance(participant, Cyclist):
                self._render_cyclist(participant, frame)

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
            frame (int, optional): The frame of the scenario. If None, the camera will update to the current frame.
            position (Point, optional): The position of the camera.
            heading (float, optional): The heading of the camera.
        """
        self._position = position
        self._heading = heading
        self._update_transform_matrix()

        self._surface.fill(pygame.Color(COLOR_PALETTE["white"]))
        self._render_areas()
        self._render_lanes()
        self._render_roadlines()
        self._render_participants(participants, participant_ids, frame)

    def get_observation(self) -> np.ndarray:
        """This function is used to get the observation of the camera from the viewpoint.

        Returns:
            The observation of the camera.
        """
        return pygame.surfarray.array3d(self._surface)
