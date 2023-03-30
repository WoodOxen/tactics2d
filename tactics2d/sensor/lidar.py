from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform
import pygame
from pygame.colordict import THECOLORS

from .sensor_base import SensorBase
from tactics2d.map.element import Map


class SingleLineLidar(SensorBase):
    """This class implements a pseudo single line lidar.

    The default parameters are from the lidar STL-06P.

    Attributes:
        id_ (int): The unique identifier of the sensor.
        map_ (Map): The map that the sensor is attached to.
        perception_range (float): The distance from the sensor to its maximum detection
            range. Defaults to 12.0 meters.
        freq_scan (float): The frequency of the lidar scanning a full round. Defaults to
            10.0 Hz.
        freq_detect (float): The frequency of the lidar emitting and receiving the signal.
            Defaults to 5000.0 Hz.
        window_size (Tuple[int, int]): The size of the rendering window. Defaults to (200, 200).
    """

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: float = 12.0,
        freq_scan: float = 10.0,
        freq_detect: float = 5000.0,
        window_size: Tuple[int, int] = (200, 200),
        visualize: bool = False,
    ):
        super().__init__(id_, map_, perception_range, window_size)

        self.perception_range = perception_range
        self.freq_scan = freq_scan
        self.freq_detect = freq_detect
        self.visualize = visualize

        self.point_density = int(self.freq_detect / self.freq_scan)
        self.angle_resolution = 2 * np.pi / self.point_density

        self.position = None
        self.heading = None

    def _update_transform_matrix(self):
        theta = self.heading - np.pi / 2

        self.transform_matrix = self.scale * np.array(
            [
                np.cos(theta),
                np.sin(theta),
                np.sin(theta),
                -np.cos(theta),
                self.perception_range
                - self.position.x * np.cos(theta)
                - self.position.y * np.sin(theta),
                self.perception_range
                - self.position.x * np.sin(theta)
                + self.position.y * np.cos(theta),
            ]
        )

    def _estimate_line_idx_range(self, polygon) -> Tuple[int, int]:
        # estimate the lidar idx range that an obstacle may fall in
        bound = polygon.bounds
        angles = [
            np.arctan2(bound[1] - self.position.y, bound[0] - self.position.x),
            np.arctan2(bound[1] - self.position.y, bound[2] - self.position.x),
            np.arctan2(bound[3] - self.position.y, bound[0] - self.position.x),
            np.arctan2(bound[3] - self.position.y, bound[2] - self.position.x),
        ]
        angles = [angle if angle >= 0 else angle + 2 * np.pi for angle in angles]
        angle_range = (min(angles), max(angles))

        line_idx_range = (
            int(np.floor((angle_range[0] - self.heading) / self.angle_resolution)),
            int(np.ceil((angle_range[1] - self.heading) / self.angle_resolution)),
        )

        return line_idx_range

    def _update_scan_line(self, point: Point, line_idx: int):
        distance = self.position.distance(point)
        self.scan_result[line_idx] = min(self.scan_result[line_idx], distance)

    def _scan_obstacles(
        self, participants: dict, participant_ids: list, frame: int = None
    ):
        lidar_lines = [
            LineString(
                [
                    self.position,
                    (
                        self.position.x
                        + self.perception_range
                        * np.cos(i * self.angle_resolution + self.heading),
                        self.position.y
                        + self.perception_range
                        * np.sin(i * self.angle_resolution + self.heading),
                    ),
                ]
            )
            for i in range(self.point_density)
        ]
        self.scan_result = [float("inf")] * self.point_density

        if "obstacles" in self.map_.customs:
            for obstacle in self.map_.customs["obstacles"]:
                shape = obstacle.shape
                line_idx_range = self._estimate_line_idx_range(shape)

                for i in range(line_idx_range[0], line_idx_range[1]):
                    intersection = shape.intersection(lidar_lines[i])
                    self._update_scan_line(intersection, i)

        for participant_id in participant_ids:
            shape = participants[participant_id].get_pose(frame)
            line_idx_range = self._estimate_line_idx_range(shape)

            for i in range(line_idx_range[0], line_idx_range[1] + 1):
                intersection = shape.intersection(lidar_lines[i])
                self._update_scan_line(intersection, i)

    def _render_lidar_points(self):
        self.surface.fill(THECOLORS["black"])

        lidar_lines = [
            LineString(
                [
                    self.position,
                    (
                        self.position.x
                        + self.perception_range
                        * np.cos(i * self.angle_resolution + self.heading),
                        self.position.y
                        + self.perception_range
                        * np.sin(i * self.angle_resolution + self.heading),
                    ),
                ]
            )
            for i in range(self.point_density)
        ]

        for distance, lidar_line in zip(self.scan_result, lidar_lines):
            if distance == float("inf"):
                continue
            point = lidar_line.interpolate(distance)
            point_ = affine_transform(point, self.transform_matrix)

            pygame.draw.circle(self.surface, THECOLORS["white"], point_.coords[0], 1)

    def update(
        self,
        participants: dict,
        participant_ids: list,
        frame: int = None,
        position: Point = None,
        heading: float = None,
    ):
        self.position = position
        self.heading = heading
        if None in [self.position, self.heading]:
            self.position = Point(
                0.5 * (self.map_.boundary[0] + self.map_.boundary[1]),
                0.5 * (self.map_.boundary[2] + self.map_.boundary[3]),
            )
            self.heading = np.pi / 2

        self._update_transform_matrix()

        self._scan_obstacles(participants, participant_ids, frame)
        if self.visualize:
            self._render_lidar_points()

    def get_observation(self):
        return self.scan_result
