from typing import Tuple

import numpy as np
import pygame
from pygame.colordict import THECOLORS
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Point, Polygon

from tactics2d.map.element import Map

from .sensor_base import SensorBase


class SingleLineLidar(SensorBase):
    """This class implements a pseudo single line lidar.

    The default parameters refer to lidar STL-06P. This lidar sensor has only one scan line.
    Its documentation is [here](https://www.ldrobot.com/images/2023/03/02/LDROBOT_STL-06P_Datasheet_EN_v1.3_txOyicBl.pdf).

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
        off_screen (bool): Whether to render the sensor off screen. Defaults to True.
    """

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: float = 12.0,
        freq_scan: float = 10.0,
        freq_detect: float = 5000.0,
        window_size: Tuple[int, int] = (200, 200),
        off_screen: bool = True,
    ):
        super().__init__(id_, map_, perception_range, window_size, off_screen)

        self.perception_range = perception_range
        self.freq_scan = freq_scan
        self.freq_detect = freq_detect

        self.point_density = int(self.freq_detect / self.freq_scan)
        self.angle_resolution = 2 * np.pi / self.point_density
        self.scan_result = [float("inf")] * self.point_density

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
            max(int(np.floor((angle_range[0] - self.heading) / self.angle_resolution)), 0),
            min(
                int(np.ceil((angle_range[1] - self.heading) / self.angle_resolution)),
                self.point_density,
            ),
        )

        return line_idx_range

    def _scan_obstacles(self, participants: dict, participant_ids: list, frame: int = None):
        potential_obstacles = []
        for area in self.map_.areas.values():
            if area.type_ == "obstacle":
                if isinstance(area.geometry, Polygon):
                    potential_obstacles.append(area.geometry.exterior)
                elif isinstance(area.geometry, LinearRing):
                    potential_obstacles.append(area.geometry)

        for participant_id in participant_ids:
            if participant_id == self.bind_id:
                continue
            shape = participants[participant_id].get_pose(frame)
            if isinstance(shape, Polygon):
                potential_obstacles.append(shape.exterior)
            elif isinstance(shape, LinearRing):
                potential_obstacles.append(shape)
        considered_obstacles = self._rotate_and_filter_obstacles(
            (self.position.x, self.position.y, self.heading), potential_obstacles
        )

        # Line 1: the lidar ray, ax + by + c = 0
        theta = np.array(
            [a * np.pi / self.point_density * 2 for a in range(self.point_density)]
        )  # (point_density,)
        a = np.sin(theta).reshape(-1, 1)  # (point_density, 1)
        b = -np.cos(theta).reshape(-1, 1)
        c = 0

        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x1s, x2s, y1s, y2s = [], [], [], []
        for obstacle in considered_obstacles:
            obstacle_coords = np.array(obstacle.coords)  # (n+1,2)
            x1s.extend(list(obstacle_coords[:-1, 0]))
            x2s.extend(list(obstacle_coords[1:, 0]))
            y1s.extend(list(obstacle_coords[:-1, 1]))
            y2s.extend(list(obstacle_coords[1:, 1]))
        if len(x1s) == 0:  # no obstacle around
            self.scan_result = np.ones(self.point_density) * float("inf")
            return
        x1s, x2s, y1s, y2s = (
            np.array(x1s).reshape(1, -1),
            np.array(x2s).reshape(1, -1),
            np.array(y1s).reshape(1, -1),
            np.array(y2s).reshape(1, -1),
        )
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1, -1)  # (1,E)
        e = (x1s - x2s).reshape(1, -1)
        f = (y1s * x2s - x1s * y2s).reshape(1, -1)

        # calculate the intersections
        det = a * e - b * d  # (point_density, E)
        parallel_line_pos = det == 0  # (point_density, E)
        det[parallel_line_pos] = 1  # temporarily set "1" to avoid "divided by zero"
        raw_x = (b * f - c * e) / det  # (point_density, E)
        raw_y = (c * d - a * f) / det

        # select the true intersections, set the false positive intersections to inf
        tmp_inf = self.perception_range * 10
        tmp_zero = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        lidar_line_x = (np.cos(theta) * self.perception_range).reshape(-1, 1)  # (point_density, 1)
        lidar_line_y = (np.sin(theta) * self.perception_range).reshape(-1, 1)
        raw_x[raw_x > np.maximum(tmp_zero, lidar_line_x)] = tmp_inf
        raw_x[raw_x < np.minimum(-tmp_zero, lidar_line_x)] = tmp_inf
        raw_y[raw_y > np.maximum(tmp_zero, lidar_line_y)] = tmp_inf
        raw_y[raw_y < np.minimum(-tmp_zero, lidar_line_y)] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x > np.maximum(x1s, x2s)] = tmp_inf
        raw_x[raw_x < np.minimum(x1s, x2s)] = tmp_inf
        raw_y[raw_y > np.maximum(y1s, y2s)] = tmp_inf
        raw_y[raw_y < np.minimum(y1s, y2s)] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        lidar_obs = np.min(np.sqrt(raw_x**2 + raw_y**2), axis=1)  # (point_density,)
        lidar_obs = np.clip(lidar_obs, 0, self.perception_range)
        lidar_obs[lidar_obs == self.perception_range] = float("inf")
        self.scan_result = lidar_obs

    def _rotate_and_filter_obstacles(self, ego_pos: tuple, obstacles: list):
        # Rotate the obstacles around the vehicle and remove the obstacles out of perception range.
        origin = Point((0, 0))
        x, y, theta = ego_pos
        a = np.cos(theta)
        b = np.sin(theta)
        x_off = -x * a - y * b
        y_off = x * b - y * a
        affine_mat = [a, b, -b, a, x_off, y_off]

        rotated_obstacles = []
        for obs in obstacles:
            rotated_obs = affine_transform(obs, affine_mat)
            if rotated_obs.distance(origin) < self.perception_range:
                rotated_obstacles.append(rotated_obs)

        return rotated_obstacles

    def _render_lidar_points(self):
        self.surface.fill(THECOLORS["black"])

        lidar_angles = np.array(
            [a * np.pi / self.point_density * 2 for a in range(self.point_density)]
        )
        lidar_point_angles = lidar_angles[self.scan_result != float("inf")]
        lidar_point_position = self.scan_result[self.scan_result != float("inf")]
        point_x_ego = self.position.x + lidar_point_position * np.cos(
            lidar_point_angles + self.heading
        )
        point_y_ego = self.position.y + lidar_point_position * np.sin(
            lidar_point_angles + self.heading
        )
        a, b, d, e, x_off, y_off = self.transform_matrix
        point_x_render = a * point_x_ego + b * point_y_ego + x_off
        point_y_render = d * point_x_ego + e * point_y_ego + y_off

        for x, y in zip(point_x_render, point_y_render):
            pygame.draw.circle(self.surface, THECOLORS["white"], (x, y), 1)

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

        self.scan_result = [float("inf")] * self.point_density
        self._scan_obstacles(participants, participant_ids, frame)

        if not self.off_screen:
            self._render_lidar_points()

    def get_observation(self) -> np.ndarray:
        """Get the lidar points at current frame. The points are sorted counter clockwise.
        The points are given in the global coordinate system.

        Returns:
            np.ndarray: The lidar points at current frame.
        """
        return np.array(self.scan_result)
