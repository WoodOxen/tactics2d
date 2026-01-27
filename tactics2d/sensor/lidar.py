# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pseudo lidar implementation."""


import time
from typing import Dict, List, Set, Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Point, Polygon

from tactics2d.map.element import Map

from .sensor_base import SensorBase


class SingleLineLidar(SensorBase):
    """This class defines the render paradigm for a single line lidar.

    The default parameters refer to LiDAR STL-06P. This LiDAR sensor has only one scan line. Its documentation is [here](https://www.ldrobot.com/images/2023/03/02/LDROBOT_STL-06P_Datasheet_EN_v1.3_txOyicBl.pdf).

    Attributes:
        id_ (int): The unique identifier of the sensor. This attribute is **read-only** once the instance is initialized.
        map_ (Map): The map that the sensor is attached to. This attribute is **read-only** once the instance is initialized.
        perception_range (Union[float, Tuple[float]]): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None.
        position (Point): The position of the sensor in the global 2D coordinate system.
        bind_id (Any): The unique identifier of object that the sensor is bound to. This attribute is **read-only** and can only be set using the `bind_with` method.
        is_bound (bool): Whether the sensor is bound to an object. This attribute is **read-only** once the instance is initialized.
    """

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: float = 12.0,
        freq_scan: float = 10.0,
        freq_detect: float = 5000.0,
    ):
        """Initialize the single line lidar.

        Args:
            id_ (int): The unique identifier of the LiDAR.
            map_ (Map): The map that the LiDAR is attached to.
            perception_range (float, optional): The distance from the LiDAR to its maximum detection range.
            freq_scan (float, optional): The frequency of the LiDAR scanning a full round.
            freq_detect (float, optional): The frequency of the LiDAR sending and receiving the signal.
        """
        super().__init__(id_, map_, perception_range)

        self._perception_range = perception_range
        self._freq_scan = freq_scan
        self._freq_detect = freq_detect

        self.point_density = max(int(self._freq_detect / self._freq_scan), 1)
        self.angle_resolution = 2 * np.pi / self.point_density
        self.scan_result = np.full(self.point_density, float("inf"))

    @property
    def freq_scan(self) -> float:
        return self._freq_scan

    @property
    def freq_detect(self) -> float:
        return self._freq_detect

    def _estimate_line_idx_range(self, polygon) -> Tuple[int, int]:
        """Estimate the lidar index range that an obstacle may fall within.

        Args:
            polygon: Shapely polygon representing an obstacle.

        Returns:
            Tuple of (start_idx, end_idx) representing the range of lidar beam indices
            that could intersect with the obstacle's bounding box.
        """
        bound = polygon.bounds
        angles = [
            np.arctan2(bound[1] - self._position.y, bound[0] - self._position.x),
            np.arctan2(bound[1] - self._position.y, bound[2] - self._position.x),
            np.arctan2(bound[3] - self._position.y, bound[0] - self._position.x),
            np.arctan2(bound[3] - self._position.y, bound[2] - self._position.x),
        ]
        angles = [angle if angle >= 0 else angle + 2 * np.pi for angle in angles]
        angle_range = (min(angles), max(angles))

        line_idx_range = (
            max(int(np.floor((angle_range[0] - self._heading) / self.angle_resolution)), 0),
            min(
                int(np.ceil((angle_range[1] - self._heading) / self.angle_resolution)),
                self.point_density,
            ),
        )

        return line_idx_range

    def _rotate_and_filter_obstacles(
        self, ego_pos: Tuple[float, float, float], obstacles: List
    ) -> List:
        """Rotate obstacles to vehicle coordinate system and filter by perception range.

        Args:
            ego_pos: Tuple of (x, y, heading) representing vehicle position and orientation.
            obstacles: List of shapely geometry objects (LinearRing or Polygon exterior).

        Returns:
            List of obstacles transformed to vehicle coordinate system and within perception range.
        """
        origin = Point((0, 0))
        x, y, theta = ego_pos
        a = np.cos(theta)
        b = np.sin(theta)
        x_off = -x * a - y * b
        y_off = x * b - y * a
        affine_mat = [a, b, -b, a, x_off, y_off]

        rotated_obstacles = []
        for i, obs in enumerate(obstacles):
            rotated_obs = affine_transform(obs, affine_mat)
            distance = rotated_obs.distance(origin)
            in_range = distance < self.perception_range

        return rotated_obstacles

    def _scan_obstacles(self, frame: int, participants: Dict, participant_ids: List[int]) -> None:
        """Perform lidar scanning to detect obstacles and compute distances.

        Args:
            frame: Current frame number.
            participants: Dictionary of all participants.
            participant_ids: List of participant IDs to consider (excluding self).
        """
        # Collect obstacles
        potential_obstacles = []
        for area in self.map_.areas.values():
            if area.type_ == "obstacle":
                if isinstance(area.geometry, Polygon):
                    potential_obstacles.append(area.geometry.exterior)
                elif isinstance(area.geometry, LinearRing):
                    potential_obstacles.append(area.geometry)

        # Transform to the local coordinate system of the LiDAR
        for participant_id in participant_ids:
            if participant_id == self.bind_id:
                continue
            shape = participants[participant_id].get_pose(frame)
            if isinstance(shape, Polygon):
                potential_obstacles.append(shape.exterior)
            elif isinstance(shape, LinearRing):
                potential_obstacles.append(shape)
        considered_obstacles = self._rotate_and_filter_obstacles(
            (self._position.x, self._position.y, self._heading), potential_obstacles
        )

        # Line 1: the lidar ray, ax + by + c = 0 (c = 0 because ray passes through origin)
        # Compute line parameters for each lidar beam (theta = angle)
        theta = np.linspace(0, 2 * np.pi, self.point_density, endpoint=False)  # (point_density,)
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
            self.scan_result = np.full(self.point_density, float("inf"))
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

        # Calculate intersection points between lidar rays and obstacle edges using determinant method
        det = a * e - b * d  # (point_density, E)
        parallel_line_pos = det == 0  # (point_density, E)
        det[parallel_line_pos] = 1  # temporarily set "1" to avoid "divided by zero"
        raw_x = (b * f - c * e) / det  # (point_density, E)
        raw_y = (c * d - a * f) / det

        # Filter out false positive intersections:
        # - Points not on the lidar ray segment (within perception range)
        # - Points not on the obstacle edge segment
        # - Parallel lines (det = 0)
        tmp_inf = self.perception_range * 10
        tmp_zero = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        lidar_line_x = (np.cos(theta) * self.perception_range).reshape(-1, 1)  # (point_density, 1)
        lidar_line_y = (np.sin(theta) * self.perception_range).reshape(-1, 1)
        raw_x[raw_x > np.maximum(tmp_zero, lidar_line_x) + tmp_zero] = tmp_inf
        raw_x[raw_x < np.minimum(-tmp_zero, lidar_line_x) - tmp_zero] = tmp_inf
        raw_y[raw_y > np.maximum(tmp_zero, lidar_line_y) + tmp_zero] = tmp_inf
        raw_y[raw_y < np.minimum(-tmp_zero, lidar_line_y) - tmp_zero] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x > np.maximum(x1s, x2s) + tmp_zero] = tmp_inf
        raw_x[raw_x < np.minimum(x1s, x2s) - tmp_zero] = tmp_inf
        raw_y[raw_y > np.maximum(y1s, y2s) + tmp_zero] = tmp_inf
        raw_y[raw_y < np.minimum(y1s, y2s) - tmp_zero] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        # Compute minimum distance for each lidar beam (closest intersection)
        lidar_obs = np.min(np.sqrt(raw_x**2 + raw_y**2), axis=1)  # (point_density,)
        lidar_obs = np.clip(lidar_obs, 0, self.perception_range)
        lidar_obs[lidar_obs == self.perception_range] = float("inf")
        self.scan_result = lidar_obs

    def _get_points(self) -> List[List[float]]:
        """Convert lidar scan results to point cloud coordinates in render coordinate system.

        Returns:
            List of [x, y] point coordinates for all lidar beams.
            Points with no detection are placed at max_perception_distance.
        """
        lidar_angles = np.linspace(0, 2 * np.pi, self.point_density, endpoint=False)

        valid_mask = self.scan_result != float("inf")
        valid_points = np.sum(valid_mask)

        valid_distances = self.scan_result[valid_mask]
        valid_angles = lidar_angles[valid_mask]

        point_x_ego = self._position.x + valid_distances * np.cos(valid_angles + self._heading)
        point_y_ego = self._position.y + valid_distances * np.sin(valid_angles + self._heading)

        # TODO: Coordinate transformation to render space may be needed; see self.transform_matrix
        points = np.stack([point_x_ego, point_y_ego], axis=1)

        return points.tolist()

    def update(
        self,
        frame: int,
        participants: Dict,
        participant_ids: List,
        prev_road_id_set: Set = None,
        prev_participant_id_set: Set = None,
        position: Point = None,
        heading: float = None,
    ) -> Tuple[Dict, Set, Set]:
        """This function is used to update the lidar's position and obtain the geometry data under specific rendering paradigm.

        Args:
            frame (int): The frame of the observation.
            participants (Dict): The participants in the scenario.
            participant_ids (List): The list of participant IDs to be rendered.
            prev_road_id_set (Set, optional): The set of road IDs that were rendered in the previous frame. Defaults to None.
            prev_participant_id_set (Set, optional): The set of participant IDs that were rendered in the previous frame. Defaults to None.
            position (Point, optional): The position of the lidar. Defaults to None.
            heading (float, optional): The heading of the object that the lidar is attached to. Defaults to None.

        Returns:
            geometry_data (Dict): The geometry data to be rendered in unified format.
            road_id_set (Set): The set of road IDs that were rendered in the current frame. Always an empty set for lidar.
            participant_id_set (Set): The set of participant IDs that were rendered in the current frame. Always an empty set for lidar.
        """
        self._set_position_heading(position, heading)

        # Use base class methods to setup parameters
        participant_ids, prev_road_id_set, prev_participant_id_set = self._setup_update_parameters(
            participant_ids, prev_road_id_set, prev_participant_id_set
        )

        self.scan_result = np.full(self.point_density, float("inf"))
        self._scan_obstacles(frame, participants, participant_ids)

        # Create empty map_data and participant_data for backward compatibility
        map_data = {
            "road_id_to_remove": list(prev_road_id_set),
            "road_elements": [],  # No road elements for lidar
        }

        # Create point cloud data in standardized format
        point_clouds = [
            {
                "id": "lidar_0",  # Fixed ID as there's only one lidar
                "points": self._get_points(),  # Use existing method to get coordinates
                "type": "lidar_point_cloud",
                "color": "red",
                "point_size": 1.0,
                "alpha": 0.8,
            }
        ]

        participant_data = {
            "participant_id_to_create": [],  # No participants to create
            "participant_id_to_remove": list(prev_participant_id_set),
            "participants": [],  # No participant elements for lidar
            "point_clouds": point_clouds,  # Point clouds for rendering in standardized format
        }

        # Unified geometry data format compatible with camera renderer
        geometry_data = {
            "frame": frame,
            "map_data": map_data,
            "participant_data": participant_data,
            # Additional sensor metadata for backward compatibility
            "metadata": {
                "timestamp": time.time(),
                "perception_range": self._perception_range,
                "sensor_type": "lidar",
                "sensor_id": self.id_,
                "sensor_position": self._position,
                "sensor_yaw": self._heading,
            },
        }

        # Return empty sets for road_id_set and participant_id_set
        road_id_set = set()
        participant_id_set = set()

        return geometry_data, road_id_set, participant_id_set
