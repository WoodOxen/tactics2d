##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: generate_parking_lot.py
# @Description: This file defines a class to generate random parking lot scenarios.
# @Author: Yueyuan Li
# @Version: 1.0.0


import logging
import time
from typing import Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon

from tactics2d.map.element import Area, Map
from tactics2d.participant.trajectory import State


class ParkingLotGenerator:
    """Generate a random bay parking lot scenario with determined start and destination.

    Attributes:
        vehicle_size (Tuple[float, float]): The size of the vehicle. The first element is the length and the second element is the width. This value is used to generate the target parking space. Defaults to (5.3, 2.5).
        type_proportion (float): The proportion of "bay" parking scenario in all generated scenarios. It should be in the range of [0, 1]. If the input is out of the range, it will be clipped to the range. When it is 0, the generator only generates "parallel" parking scenarios. Defaults to 0.5.
        mode (str): The type of the parking scenario. It can be "bay" or "parallel". Defaults to None.
    """

    _origin = Point(0.0, 0.0)
    _scenario_size = 30.0
    _margin = 13.0
    _dist_to_obstacle = (0.8, 1.6)
    _heading_distribution = {
        "bay": (np.pi / 2, np.pi / 54, np.pi * 4 / 9, np.pi * 5 / 9),
        "parallel": (0, np.pi / 54, -np.pi / 18, np.pi / 18),
    }
    _length = {"bay": 7.0, "parallel": 4.5}
    _vehicle_size = (5.3, 2.5)
    _n_parking_lots_bay = 9
    _n_parking_lots_parallel = 7
    _target_color = "#EE766E"

    def __init__(
        self, vehicle_size: Tuple[float, float] = (5.3, 2.5), type_proportion: float = 0.5
    ):
        """Initialize the attributes in the class.

        Args:
            vehicle_size (Tuple[float, float], optional): he size of the vehicle. The first element is the length and the second element is the width. This value is used to generate the target parking space.
            type_proportion (float, optional): The proportion of "bay" parking scenario in all generated scenarios. It should be in the range of [0, 1]. If the input is out of the range, it will be clipped to the range. When it is 0, the generator only generates "parallel" parking scenarios. When it is 1, the generator only generates "bay" parking scenarios.
        """

        if vehicle_size[0] < vehicle_size[1] or vehicle_size[0] <= 0 or vehicle_size[1] <= 0:
            self.vehicle_size = self._vehicle_size
            logging.warning(
                f"The input parking size is invalid. Use default value {self._vehicle_size} instead."
            )
        else:
            self.vehicle_size = vehicle_size

        self.type_proportion = np.clip(type_proportion, 0, 1)
        self.mode = None

    def _truncate_gaussian(self, mean, std, min_, max_, size=None):
        random_numbers = np.random.normal(mean, std, size)
        return np.clip(random_numbers, min_, max_)

    def _get_bbox(
        self, center_point: Point, heading: float, length: float, width: float
    ) -> Polygon:
        bbox = Polygon(
            [
                [0.5 * length, -0.5 * width],  # top_right
                [0.5 * length, 0.5 * width],  # top_left
                [-0.5 * length, 0.5 * width],  # bottom_left
                [-0.5 * length, -0.5 * width],  # bottom_right
            ]
        )
        transform_matrix = [
            np.cos(heading),
            -np.sin(heading),
            np.sin(heading),
            np.cos(heading),
            center_point.x,
            center_point.y,
        ]

        return affine_transform(bbox, transform_matrix)

    def _get_random_position(
        self, origin: Point, angle_range: Tuple[float], radius_range: Tuple[float]
    ) -> Point:
        angle = self._truncate_gaussian(
            np.mean(angle_range), np.std(angle_range), angle_range[0], angle_range[1]
        )
        radius = self._truncate_gaussian(
            np.mean(radius_range), np.std(radius_range), radius_range[0], radius_range[1]
        )

        return Point(origin.x + radius * np.cos(angle), origin.y + radius * np.sin(angle))

    def _get_target_area(self) -> Area:
        heading = self._truncate_gaussian(*self._heading_distribution[self.mode])

        top_right, _, bottom_left, bottom_right, _ = list(
            self._get_bbox(self._origin, heading, *self.vehicle_size).exterior.coords
        )
        if self.mode == "bay":
            y_min = -min(bottom_right[1], bottom_left[1]) + self._dist_to_obstacle[0]
        else:
            y_min = -min(bottom_right[1], top_right[1]) + self._dist_to_obstacle[0]
        center_point = Point(0.0, self._truncate_gaussian(y_min + 0.4, 0.2, y_min, y_min + 0.8))

        shape = self._get_bbox(center_point, heading, *self.vehicle_size)
        area = Area(id_=0, geometry=shape, subtype="target_area", color=self._target_color)

        return area, heading

    def _get_back_wall(self) -> Area:
        wall_width = np.random.uniform(0.5, 1.5)
        wall_center = Point(self._origin.x, self._origin.y - wall_width / 2)
        shape = self._get_bbox(wall_center, 0, self._scenario_size, wall_width)
        obstacle = Area(id_="0000", type_="obstacle", geometry=shape)

        return obstacle

    def _get_left_wall(self, id_: int, target_area: Area, dist_to_obstacle: np.ndarray) -> Area:
        _, top_left, bottom_left, bottom_right, _ = list(target_area.geometry.exterior.coords)

        wall_top_right = self._get_random_position(
            Point(top_left) if self.mode == "bay" else Point(bottom_left),
            (np.pi * 11 / 12, np.pi * 13 / 12),
            dist_to_obstacle,
        )
        wall_bottom_right = self._get_random_position(
            Point(bottom_left) if self.mode == "bay" else Point(bottom_right),
            (np.pi * 11 / 12, np.pi * 13 / 12),
            dist_to_obstacle,
        )

        shape = Polygon(
            [
                wall_top_right,
                wall_bottom_right,
                (self._origin.x - self._scenario_size / 2, self._origin.y),
                (self._origin.x - self._scenario_size / 2, wall_top_right.y),
            ]
        )
        obstacle = Area(id_="%04d" % id_, type_="obstacle", geometry=shape)
        return obstacle

    def _get_right_wall(self, id_: int, target_area: Area, dist_to_obstacle: np.ndarray) -> Area:
        top_right, top_left, _, bottom_right, _ = list(target_area.geometry.exterior.coords)

        wall_bottom_left = self._get_random_position(
            Point(bottom_right) if self.mode == "bay" else Point(top_right),
            (-np.pi * 1 / 12, np.pi * 1 / 12),
            dist_to_obstacle,
        )
        wall_top_left = self._get_random_position(
            Point(top_right) if self.mode == "bay" else Point(top_left),
            (-np.pi * 1 / 12, np.pi * 1 / 12),
            dist_to_obstacle,
        )

        shape = Polygon(
            [
                (self._origin.x + self._scenario_size / 2, top_left[1]),
                (self._origin.x + self._scenario_size / 2, self._origin.y),
                wall_bottom_left,
                wall_top_left,
            ]
        )
        obstacle = Area(id_="%04d" % id_, type_="obstacle", geometry=shape)
        return obstacle

    def _get_side_vehicle(
        self, id_: int, dist_to_obstacle: np.ndarray, left_side: bool = True
    ) -> Area:
        heading = self._truncate_gaussian(*self._heading_distribution[self.mode])

        side_factor = -1 if left_side else 1
        # get x coordinate of the side vehicle
        if self.mode == "bay":
            x = self._origin.x + side_factor * (
                self.vehicle_size[1] + np.random.uniform(*dist_to_obstacle)
            )
        else:
            x = self._origin.x + side_factor * (
                self.vehicle_size[0] + np.random.uniform(*dist_to_obstacle)
            )

        # get y coordinate of the side vehicle
        top_right, _, bottom_left, bottom_right, _ = list(
            self._get_bbox(Point(x, self._origin.y), heading, *self.vehicle_size).exterior.coords
        )

        if self.mode == "bay":
            min_left_y = -min(bottom_right[1], bottom_left[1]) + self._dist_to_obstacle[0]
        else:
            min_left_y = -min(bottom_right[1], top_right[1]) + self._dist_to_obstacle[0]
        y = self._truncate_gaussian(min_left_y + 0.4, 0.2, min_left_y, min_left_y + 0.8)

        shape = self._get_bbox(Point(x, y), heading, *self.vehicle_size)
        obstacle = Area(id_="%04d" % id_, type_="obstacle", geometry=shape)
        return obstacle

    def _verify_obstacles(
        self, target_area: Area, obstacles: list, dist_target_to_obstacle: Tuple[float, float]
    ) -> bool:
        target_polygon = target_area.geometry
        for obstacle in obstacles:
            if target_polygon.intersects(obstacle.geometry):
                return False

        if any(dist_target_to_obstacle) < self._dist_to_obstacle[0]:
            return False

        if self.mode == "bay" and sum(dist_target_to_obstacle) < 0.85:
            return False
        elif self.mode == "parallel" and sum(dist_target_to_obstacle) < self.vehicle_size[0] / 4:
            return False

        return True

    def _get_start_state(self, x_range: tuple, y_range: tuple) -> State:
        location = Point(np.random.uniform(*x_range), np.random.uniform(*y_range))
        heading = self._truncate_gaussian(*self._heading_distribution["parallel"])
        state = State(0, x=location.x, y=location.y, heading=heading, vx=0.0, vy=0.0, accel=0.0)
        return state

    def _verify_start_state(self, state: State, obstacles: list, target_area: Area) -> bool:
        state_shape = self._get_bbox(Point(state.location), state.heading, *self.vehicle_size)
        for obstacle in obstacles:
            if state_shape.intersects(obstacle.geometry):
                return False

        return not state_shape.intersects(target_area.geometry)

    def generate(self, map_: Map):
        """Generate a random parking scenario.

        Args:
            map_ (Map): The map instance to store the generated parking scenario.

        Returns:
            start_state (State): The start state of the vehicle.
            target_area (Area): The target area of the parking scenario.
            target_heading (float): The heading of the target area.
        """
        t1 = time.time()

        if map_.name is None:
            map_.name = "parking_lot"

        if map_.scenario_type is None:
            map_.scenario_type = "parking"

        self.mode = "bay" if np.random.rand() < self.type_proportion else "parallel"
        logging.info(f"Start generating a {self.mode} parking scenario.")

        obstacles = []
        valid_obstacles = False
        while not valid_obstacles:
            # get the target area
            target_area, target_heading = self._get_target_area()
            map_.areas = {target_area.id_: target_area}

            back_wall = self._get_back_wall()

            # generate a wall / static vehicle as an obstacle on the left side of the target area
            dist_to_obstacle = np.array(
                (self._dist_to_obstacle[0] + 0.1, self._dist_to_obstacle[1])
            )
            if np.random.uniform() < 0.2:
                left_obstacle = self._get_left_wall(1, target_area, dist_to_obstacle)
            else:
                left_obstacle = self._get_side_vehicle(1, dist_to_obstacle, True)
                # generate other vehicle on left
                next_vehicle_distance = (
                    self.vehicle_size[1] if self.mode == "bay" else self.vehicle_size[0]
                )
                parking_lot_num = (
                    self._n_parking_lots_bay
                    if self.mode == "bay"
                    else self._n_parking_lots_parallel
                )
                for i in range(
                    (parking_lot_num - 3) // 2
                ):  # 3 is the number of obstacles already generated
                    dist_to_obstacle += next_vehicle_distance + self._dist_to_obstacle[0]
                    left_obstacle_ = self._get_side_vehicle(2 * i + 3, dist_to_obstacle, True)
                    obstacles.append(left_obstacle_)

            # generate a wall / static vehicle as an obstacle on the right side of the target area
            dist_target_to_left_obstacle = target_area.geometry.distance(left_obstacle.geometry)
            if self.mode == "bay":
                min_dist_to_obstacle = (
                    max(0.85 - dist_target_to_left_obstacle, 0) + self._dist_to_obstacle[0]
                )
            else:
                min_dist_to_obstacle = (
                    max(0.25 * self.vehicle_size[0] - dist_target_to_left_obstacle, 0)
                    + self._dist_to_obstacle[0]
                )

            dist_to_obstacle = np.array((min_dist_to_obstacle, self._dist_to_obstacle[1]))
            if np.random.uniform() < 0.2:
                right_obstacle = self._get_right_wall(2, target_area, dist_to_obstacle)
            else:
                right_obstacle = self._get_side_vehicle(2, dist_to_obstacle, False)
                # generate other vehicle on right
                next_vehicle_distance = (
                    self.vehicle_size[1] if self.mode == "bay" else self.vehicle_size[0]
                )
                parking_lot_num = (
                    self._n_parking_lots_bay
                    if self.mode == "bay"
                    else self._n_parking_lots_parallel
                )
                for i in range((parking_lot_num - 3) // 2):
                    dist_to_obstacle += next_vehicle_distance + self._dist_to_obstacle[0]
                    right_obstacle_ = self._get_side_vehicle(2 * i + 4, dist_to_obstacle, False)
                    obstacles.append(right_obstacle_)

            dist_target_to_right = target_area.geometry.distance(right_obstacle.geometry)
            valid_obstacles = self._verify_obstacles(
                target_area,
                [back_wall, left_obstacle, right_obstacle],
                (dist_target_to_left_obstacle, dist_target_to_right),
            )

        obstacles.append(back_wall)
        obstacles.append(left_obstacle)
        obstacles.append(right_obstacle)

        # generate obstacles out of start range
        y_max_obstacle = (
            max(
                [
                    np.max(np.array(obstacle.geometry.exterior.coords)[:, 1])
                    for obstacle in obstacles
                ]
            )
            + self._dist_to_obstacle[0]
        )
        if np.random.uniform() < 0.2:
            width = np.random.uniform(0.0, 0.2)
            shape = self._get_bbox(
                Point(self._origin.x, y_max_obstacle + self._length[self.mode]),
                0,
                self._scenario_size,
                width,
            )
            obstacle = Area(id_="0003", type_="obstacle", geometry=shape)
            obstacles.append(obstacle)
        else:
            bbox = self._get_bbox(
                Point(self._origin.x, y_max_obstacle + self._length[self.mode] + 4),
                0,
                self._scenario_size,
                8,
            )
            x_range = (
                self._origin.x - self._scenario_size / 2,
                self._origin.x + self._scenario_size / 2,
            )
            y_range = (
                y_max_obstacle + self._length[self.mode] + 2,
                y_max_obstacle + self._length[self.mode] + 6,
            )

            id_ = len(obstacles) + 1
            for _ in range(3):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                heading = np.random.uniform() * 2 * np.pi
                shape = np.array(
                    list(self._get_bbox(Point(x, y), heading, *self.vehicle_size).exterior.coords)[
                        :4
                    ]
                )
                shape = Polygon(shape + 0.5 * np.random.uniform(size=shape.shape))

                if Polygon(bbox).contains(shape):
                    obstacle = Area(id_="%04d" % id_, type_="obstacle", geometry=shape)
                    obstacles.append(obstacle)
                    id_ += 1

        # randomly drop the obstacles
        obstacles = [obstacle for obstacle in obstacles if np.random.uniform() >= 0.05]

        # store obstacles in map
        for obstacle in obstacles:
            map_.add_area(obstacle)

        # get the start state
        valid_start_state = False
        while not valid_start_state:
            start_state = self._get_start_state(
                (-self._scenario_size / 4, self._scenario_size / 4),
                (
                    y_max_obstacle + self._dist_to_obstacle[0] + 1,
                    y_max_obstacle + self._length[self.mode] - 1,
                ),
            )
            valid_start_state = self._verify_start_state(start_state, obstacles, target_area)

        # flip the orientation of start pose
        target_box_center = np.mean(np.array(target_area.geometry.exterior.coords[:-1]), axis=0)
        target_x = target_box_center[0]
        target_y = target_box_center[1]
        if np.random.rand() > 0.5:
            start_x, start_y, start_heading = (start_state.x, start_state.y, start_state.heading)
            start_box = self._get_bbox(
                Point(start_state.location), start_state.heading, *self.vehicle_size
            )
            start_box_center = np.mean(np.array(start_box.exterior.coords[:-1]), axis=0)
            start_x = 2 * start_box_center[0] - start_x
            start_y = 2 * start_box_center[1] - start_y
            start_heading += np.pi
            start_state = State(
                0, x=start_x, y=start_y, heading=start_heading, vx=0.0, vy=0.0, accel=0.0
            )
            if self.mode == "parallel":  # flip the target pose
                target_heading += np.pi
                target_shape = self._get_bbox(
                    Point(target_x, target_y), target_heading, *self.vehicle_size
                )
                target_area = Area(
                    id_=0, geometry=target_shape, subtype="target_area", color=self._target_color
                )
                map_.add_area(target_area)

        xmin = np.floor(min(start_state.x, target_x) - self._margin)
        xmax = np.ceil(max(start_state.x, target_x) + self._margin)
        ymin = np.floor(min(start_state.y, target_y) - self._margin)
        ymax = np.ceil(max(start_state.y, target_y) + self._margin)
        map_.set_boundary((xmin, xmax, ymin, ymax))

        # record time cost
        t2 = time.time()
        logging.info("The generating process takes %.4fs." % (t2 - t1))

        return start_state, target_area, target_heading
