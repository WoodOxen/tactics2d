from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing, Point
from shapely.affinity import affine_transform

from tactics2d.math.random import truncate_gaussian
from tactics2d.map.element import Area, Map
from tactics2d.participant.element import Other
from tactics2d.trajectory.element import State


ORIGIN = Point(0.0, 0.0)
SCENARIO_SIZE = (15.0, 15.0)
HEADING_PARAMS = {
    "bay": (np.pi / 2, np.pi / 36, np.pi * 5 / 12, np.pi * 7 / 12),
    "parallel": (0, np.pi / 36, -np.pi / 12, np.pi / 12),
}
DIST_TO_OBSTACLE = (0.1, 1.0)
DIST_TO_WALL = {"bay": 7.0, "parallel": 4.5}


def _get_random_position(
    origin: Point, angle_range: Tuple[float], radius_range: Tuple[float]
) -> Point:
    """Get a random position in a circle with given origin and radius range.

    Args:
        origin (Point): The origin of the circle.
        angle_range (Tuple[float, float]): The range of angle.
        radius_range (Tuple[float, float]): The range of radius.

    Returns:
        Point: The random position.
    """
    angle = truncate_gaussian(
        np.mean(angle_range), np.std(angle_range), angle_range[0], angle_range[1]
    )
    radius = truncate_gaussian(
        np.mean(radius_range), np.std(radius_range), radius_range[0], radius_range[1]
    )

    return Point(origin.x + radius * np.cos(angle), origin.y + radius * np.sin(angle))


def _get_bbox(
    center_point: Point, heading: float, length: float, width: float
) -> LinearRing:
    """Generate a bounding box."""
    bbox = LinearRing(
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


class ParkingLotGenerator:
    """Generate a random bay parking lot scenario with determined start and destination.

    Attributes:
        vehicle_size
        scenario_proportion (float): The proportion of "parallel" parking scenario in all
            generated scenarios. It should be in the range of [0, 1]. When it is 0, the
            generator only generates bay parking scenario. When it is 1, the generator only
            generates parallel parking scenario.
    """

    def __init__(self, vehicle_size: Tuple[float, float], proportion: float = 0.5):
        self.vehicle_size = vehicle_size
        self.proportion = proportion
        self.mode = None

    def _get_target_area(self) -> Area:
        heading = truncate_gaussian(*HEADING_PARAMS[self.mode])

        top_right, _, bottom_left, bottom_right = list(
            _get_bbox(ORIGIN, heading, *self.vehicle_size)
        ).coords
        if self.mode == "bay":
            y_min = -min(bottom_right[1], bottom_left[1]) + DIST_TO_OBSTACLE[0]
        else:
            y_min = -min(bottom_right[1], top_right[1]) + DIST_TO_WALL[self.mode]
        center_point = Point(0.0, truncate_gaussian(y_min + 0.4, 0.2, y_min, y_min + 0.8))

        shape = _get_bbox(center_point, heading, *self.vehicle_size)
        area = Area(id_=0, polygon=shape)

        return area

    def _get_back_wall(self) -> Other:
        shape = _get_bbox(
            self.origin, 0, self.bay_parking_size[1] / 2, np.random.uniform(0.5, 1.5)
        )
        obstacle = Other(id_=0, type_="obstacle", shape=shape)

        return obstacle

    def _get_left_wall(
        self, id_: int, target_area: Area, dist_to_obstacle: Tuple[float, float]
    ) -> Other:
        _, top_left, bottom_left, bottom_right = list(target_area.polygon.exterior).coords

        wall_top_right = _get_random_position(
            Point(top_left) if self.mode == "bay" else Point(bottom_left),
            (np.pi * 11 / 12, np.pi * 13 / 12),
            dist_to_obstacle,
        )
        wall_bottom_right = _get_random_position(
            Point(bottom_left) if self.mode == "bay" else Point(bottom_right),
            (np.pi * 11 / 12, np.pi * 13 / 12),
            dist_to_obstacle,
        )

        shape = LinearRing(
            [
                wall_top_right,
                wall_bottom_right,
                (ORIGIN.x - SCENARIO_SIZE[0], ORIGIN.y),
                (ORIGIN.x - SCENARIO_SIZE[0], wall_top_right.x),
            ]
        )
        obstacle = Other(id_=id_, type_="obstacle", shape=shape)
        return obstacle

    def _get_right_wall(
        self, id_: int, target_area: Area, dist_to_obstacle: Tuple[float, float]
    ) -> Other:
        top_right, top_left, _, bottom_right = list(target_area.polygon.exterior).coords

        wall_bottom_left = _get_random_position(
            Point(bottom_right) if self.mode == "bay" else Point(top_right),
            (-np.pi * 1 / 12, np.pi * 1 / 12),
            dist_to_obstacle,
        )
        wall_top_left = _get_random_position(
            Point(top_right) if self.mode == "bay" else Point(top_left),
            (-np.pi * 1 / 12, np.pi * 1 / 12),
            dist_to_obstacle,
        )

        shape = LinearRing(
            [
                (ORIGIN.x + SCENARIO_SIZE[0], top_left.y),
                (ORIGIN.x + SCENARIO_SIZE[0], ORIGIN.y),
                wall_bottom_left,
                wall_top_left,
            ]
        )
        obstacle = Other(id_=id_, type_="obstacle", shape=shape)
        return obstacle

    def _get_side_vehicle(
        self, id_: int, dist_to_obstacle: Tuple[float, float], left_side: bool = True
    ) -> Other:
        heading = truncate_gaussian(*HEADING_PARAMS[self.mode])

        side_factor = -1 if left_side else 1
        # get x coordinate of the side vehicle
        if self.mode == "bay":
            x = ORIGIN.x + side_factor * (
                self.vehicle_size[0] + np.random.uniform(*dist_to_obstacle)
            )
        else:
            x = ORIGIN.x + side_factor * (
                self.vehicle_size[1] + np.random.uniform(*dist_to_obstacle)
            )

        # get y coordinate of the side vehicle
        top_right, _, bottom_left, bottom_right = list(
            _get_bbox(Point(x, ORIGIN.y), heading, *self.vehicle_size)
        ).coords

        if self.mode == "bay":
            min_left_y = min(bottom_right[1], bottom_left[1]) + DIST_TO_OBSTACLE[0]
        else:
            min_left_y = min(bottom_right[1], top_right[1]) + DIST_TO_OBSTACLE[0]
        y = truncate_gaussian(min_left_y + 0.4, 0.2, min_left_y, min_left_y + 0.8)

        shape = _get_bbox(Point(x, y), heading, *self.vehicle_size)
        obstacle = Other(id_=id_, type_="obstacle", shape=shape)
        return obstacle

    def _verify_obstacles(
        self,
        target_area: Area,
        obstacles: list,
        dist_target_to_obstacle: Tuple[float, float],
    ) -> bool:
        target_polyon = target_area.polygon.exterior
        for obstacle in obstacles:
            if target_polyon.intersects(obstacle.shape):
                return False

        if any(dist_target_to_obstacle) < DIST_TO_OBSTACLE[0]:
            return False

        if self.mode == "bay" and sum(dist_target_to_obstacle) < 0.85:
            return False
        elif (
            self.mode == "parallel"
            and sum(dist_target_to_obstacle) < self.vehicle_size[0] / 4
        ):
            return False

        return True

    def _get_start_state(self) -> State:
        x_range = (-SCENARIO_SIZE[0] / 2, SCENARIO_SIZE[0] / 2)
        y_range = ()
        location = Point(np.random.uniform(*x_range), np.random.uniform(*y_range))
        heading = truncate_gaussian(*HEADING_PARAMS[self.mode])
        state = State(0, location.x, location.y, heading, 0.0, 0.0)
        return state

    def _verify_start_state(
        self, state: State, obstacles: list, target_area: Area
    ) -> bool:
        state_shape = _get_bbox(Point(state.location), state.heading, *self.vehicle_size)
        for obstacle in obstacles:
            if state_shape.intersects(obstacle.shape):
                return False

        return not state_shape.intersects(target_area.polygon.exterior)

    def generate(self, map_: Map):
        self.mode = "bay" if np.random.rand() < self.proportion else "parallel"

        target_area = self._get_target_area()

        obstacles = []
        valid_obstacles = False

        while not valid_obstacles:
            back_wall = self._get_back_wall()

            # generate a wall / static vehicle as an obstacle on the left side of the target area
            dist_to_obstacle = (DIST_TO_OBSTACLE[0] + 0.4, DIST_TO_OBSTACLE[1])
            left_obstacle = (
                self._get_side_vehicle(1, dist_to_obstacle)
                if np.random.uniform() < 0.5
                else self._get_left_wall(1, target_area, dist_to_obstacle)
            )

            # generate a wall / static vehicle as an obstacle on the right side of the target area
            dist_target_to_left_obstacle = target_area.polygon.exterior.distance(
                left_obstacle.shape
            )
            if self.mode == "bay":
                min_dist_to_obstacle = (
                    max(0.85 - dist_target_to_left_obstacle, 0) + DIST_TO_OBSTACLE[0]
                )
            else:
                min_dist_to_obstacle = (
                    max(0.25 * self.vehicle_size[0] - dist_target_to_left_obstacle, 0)
                    + DIST_TO_WALL[self.mode]
                )
            dist_to_obstacle = (min_dist_to_obstacle, DIST_TO_OBSTACLE[1])

            right_obstacle = (
                self._get_side_vehicle(2, dist_to_obstacle, False)
                if np.random.uniform() < 0.5
                else self._get_right_wall(2, target_area, dist_target_to_left_obstacle)
            )

            dist_target_to_right = target_area.polygon.exterior.distance(
                right_obstacle.shape
            )
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
            max([np.max(np.array(obstacle.coords)[:, 1]) for obstacle in obstacles])
            + DIST_TO_OBSTACLE[0]
        )
        if np.random.uniform() < 0.2:
            width = np.random.uniform(0.0, 0.2)
            shape = _get_bbox(
                Point(ORIGIN.x, y_max_obstacle + 0.7 + 0.5 * width),
                0,
                SCENARIO_SIZE[0],
                width,
            )
            obstacle = Other(id_=3, type_="obstacle", shape=shape)
            obstacles.append(obstacle)
        else:
            bbox = _get_bbox(
                Point(ORIGIN.x, y_max_obstacle + 0.7), 0, SCENARIO_SIZE[0], 8
            )
            x_range = (ORIGIN.x - SCENARIO_SIZE[0], ORIGIN.x + SCENARIO_SIZE[0])
            y_range = (y_max_obstacle + 0.7 + 2, y_max_obstacle + 0.7 + 6)

            id_ = 3
            for _ in range(3):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                heading = np.random.uniform() * 2 * np.pi
                shape = np.array(
                    list(_get_bbox(Point(x, y), heading, *self.vehicle_size).coords)
                )
                shape = LinearRing(shape + 0.5 * np.random.uniform(size=shape.shape))

                if bbox.contains(shape):
                    obstacle = Other(id_=id_, type_="obstacle", shape=shape)
                    obstacles.append(obstacle)
                    id_ += 1

        # randomly drop the obstacles
        for obstacle in obstacles:
            if np.random.uniform() < 0.1:
                obstacles.remove(obstacle)

        valid_start_state = False
        while not valid_start_state:
            start_state = self._get_start_state()
            valid_start_state = self._verify_start_state(
                start_state, obstacles, target_area
            )

        return obstacles
