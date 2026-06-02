# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pygame top-down camera implementation."""


from typing import Tuple, Union

import numpy as np
import pygame
from shapely.affinity import affine_transform
from shapely.geometry import Point

from tactics2d.map.element import Area, Lane, Map, RoadLine
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle

from .pygame_sensor import PygameSensorBase
from .render_template import COLOR_PALETTE, DEFAULT_COLOR


class TopDownCamera(PygameSensorBase):
    """Top-down semantic camera rendered with pygame."""

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: Union[float, Tuple[float]] = None,
        window_size: Tuple[int, int] = (200, 200),
        off_screen: bool = True,
    ):
        super().__init__(id_, map_, perception_range, window_size, off_screen)

        self.map_surface = pygame.Surface(self.window_size)
        self.map_rendered = False

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

    def _out_of_perception_range(self, geometry) -> bool:
        return geometry.distance(self._position) > self.max_perception_distance * 2

    def _get_color(self, element):
        if element.color in COLOR_PALETTE:
            return pygame.Color(COLOR_PALETTE[element.color])

        if element.color is None:
            if hasattr(element, "subtype") and element.subtype in DEFAULT_COLOR:
                return pygame.Color(DEFAULT_COLOR[element.subtype])
            if hasattr(element, "type_") and element.type_ in DEFAULT_COLOR:
                return pygame.Color(DEFAULT_COLOR[element.type_])
            if isinstance(element, Area):
                return pygame.Color(DEFAULT_COLOR["area"])
            if isinstance(element, Lane):
                return pygame.Color(DEFAULT_COLOR["lane"])
            if isinstance(element, RoadLine):
                return pygame.Color(DEFAULT_COLOR["roadline"])

        return element.color

    def _render_areas(self):
        for area in self.map_.areas.values():
            if self._position is not None and self._out_of_perception_range(area.geometry):
                continue

            color = self._get_color(area)
            polygon = affine_transform(area.geometry, self.transform_matrix)
            outer_points = list(polygon.exterior.coords)
            inner_list = list(polygon.interiors)

            pygame.draw.polygon(self.map_surface, color, outer_points)
            for inner_points in inner_list:
                pygame.draw.polygon(
                    self.map_surface, pygame.Color(DEFAULT_COLOR["hole"]), list(inner_points.coords)
                )

    def _render_lanes(self):
        for lane in self.map_.lanes.values():
            if self._position is not None and self._out_of_perception_range(lane.geometry):
                continue

            color = self._get_color(lane)
            points = list(affine_transform(lane.geometry, self.transform_matrix).coords)

            pygame.draw.polygon(self.map_surface, color, points)

    def _render_roadlines(self):
        for roadline in self.map_.roadlines.values():
            if self._position is not None and self._out_of_perception_range(roadline.geometry):
                continue

            color = self._get_color(roadline)
            points = list(affine_transform(roadline.geometry, self.transform_matrix).coords)

            if roadline.type_ == "line_thick":
                width = max(2, int(0.2 * self.scale))
            else:
                width = max(1, int(0.1 * self.scale))

            pygame.draw.lines(self.map_surface, color, False, points, width)

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

        pygame.draw.circle(self._surface, color, (point.x, point.y), radius)

    def _render_participants(self, participants: dict, participant_ids: list, frame: int = None):
        for participant_id in participant_ids:
            participant = participants[participant_id]

            state = participant.trajectory.get_state(frame)
            if self._position is not None and self._out_of_perception_range(Point(state.location)):
                continue

            if isinstance(participant, Vehicle):
                self._render_vehicle(participant, frame)
            elif isinstance(participant, Pedestrian):
                self._render_pedestrian(participant, frame)
            elif isinstance(participant, Cyclist):
                self._render_cyclist(participant, frame)

    def _render_map(self):
        self.map_surface.fill(pygame.Color(COLOR_PALETTE["white"]))
        self._render_areas()
        self._render_lanes()
        self._render_roadlines()

    def update(
        self,
        participants,
        participant_ids: list,
        frame: int = None,
        position: Point = None,
        heading: float = None,
    ):
        self._position = position
        self._heading = heading
        self._update_transform_matrix()

        if None in [self._position, self._heading]:
            if not self.map_rendered:
                self._render_map()
                self.map_rendered = True
        else:
            self._render_map()

        self._surface.fill(pygame.Color(COLOR_PALETTE["white"]))
        self._surface.blit(self.map_surface, (0, 0))
        self._render_participants(participants, participant_ids, frame)

    def get_observation(self) -> np.ndarray:
        return pygame.surfarray.array3d(self._surface)
