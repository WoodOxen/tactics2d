from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

from .sensor_base import SensorBase
from tactics2d.map.element import Map


class Lidar(SensorBase):
    """This class implements a pseudo single line lidar.

    The default parameters are from the lidar STL-06P.

    Attributes:
        sensor_id (str): The unique identifier of the sensor.
        map_ (Map): The map that the sensor is attached to.
        perception_range (float): The distance from the sensor to its maximum detection
            range. Defaults to 12.0 meters.
        freq_scan (float): The frequency of the lidar scanning a full round. Defaults to
            10.0 Hz.
        freq_detect (float): The frequency of the lidar emitting and receiving the signal.
            Defaults to 5000.0 Hz.
    """

    def __init__(
        self,
        sensor_id,
        map_: Map,
        perception_range: float = 12.0,
        freq_scan: float = 10.0,
        freq_detect: float = 5000.0,
    ):
        super().__init__(sensor_id, map_)

        self.perception_range = perception_range
        self.freq_scan = freq_scan
        self.freq_detect = freq_detect

        self.point_density = int(self.freq_detect / self.freq_scan)
        self.angle_resolution = 2 * np.pi / self.point_density

        self.position = Point(
            (self.map_.boundary[1] - self.map_.boundary[0])
            / 2(self.map_.boundary[3] - self.map_.boundary[2])
            / 2
        )
        self.heading = 0.0

    def _estimate_line_idx_range(self, polygon) -> Tuple[int, int]:
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
            np.floor((angle_range[0] - self.heading) / self.angle_resolution),
            np.ceil((angle_range[1] - self.heading) / self.angle_resolution) + 1,
        )

        return line_idx_range

    def _update_scan_line(self, point: Point, line_idx: int):
        distance = self.position.distance(point)
        self.scan_result[line_idx] = min(self.scan_result[line_idx], distance)

    def _scan_obstacles(self, participants, frame: int = None):
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

        for obstacle in self.map_.obstacles:
            shape = obstacle.shape
            line_idx_range = self._estimate_line_idx_range(shape)

            for i in range(line_idx_range[0], line_idx_range[1]):
                intersection = shape.intersection(lidar_lines[i])
                self._update_scan_line(intersection, i)

        for participant in participants:
            shape = participant.get_pose(frame)
            line_idx_range = self._estimate_line_idx_range(shape)

            for i in range(line_idx_range[0], line_idx_range[1]):
                intersection = shape.intersection(lidar_lines[i])
                self._update_scan_line(intersection, i)

    def update(
        self,
        participants,
        frame: int = None,
        position: Point = None,
        heading: float = None,
    ):
        self.position = self.position if position is None else position
        self.heading = self.heading if heading is None else heading

        self._scan_obstacles(participants, frame)

    def get_observation(self):
        return self.scan_result
