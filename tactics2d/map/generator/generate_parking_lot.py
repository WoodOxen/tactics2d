from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing, Point
from shapely.affinity import affine_transform

from tactics2d.math.random import truncate_gaussian
from tactics2d.trajectory.element import State


MIN_DIST_TO_OBST = 0.1
MIN_PARA_PARK_LOT_LEN = LENGTH * 1.25
MIN_BAY_PARK_LOT_WIDTH = WIDTH + 1.2
# the distance that the obstacles out of driving area is from dest
BAY_PARK_WALL_DIST = 7.0
PARA_PARK_WALL_DIST = 4.5
origin = (0.0, 0.0)
bay_half_len = 18.0


def get_random_position(
        origin: Point, angle_range: Tuple[float], radius_range: Tuple[float]
     ):
    """Get a random position in a circle with given origin and radius range.

    Args:
        origin (Point): The origin of the circle.
        angle_range (Tuple[float, float]): The range of angle.
        radius_range (Tuple[float, float]): The range of radius.

    Returns:
        Point: The random position.
    """
    angle = truncate_gaussian(
        np.mean(angle_range), np.std(angle_range), angle_range[0], angle_range[1])
    radius = truncate_gaussian(
        np.mean(radius_range), np.std(radius_range), radius_range[0], radius_range[1])

    return Point(origin.x + radius * np.cos(angle), origin.y + radius * np.sin(angle))


def generate_bbox(center_point: Point, heading: float, length: float, width: float):
    """Generate a bounding box.
    """
    bbox = LinearRing(
        [
            [0.5 * length, -0.5 * width], [0.5 * length, 0.5 * width],
            [-0.5 * length, 0.5 * width], [-0.5 * length, -0.5 * width],
        ]
    )
    transform_matrix = [
        np.cos(heading), -np.sin(heading),
        np.sin(heading), np.cos(heading),
        center_point.x, center_point.y,
    ]

    return affine_transform(bbox, transform_matrix)


class ParkingLotGenerator:
    """Generate a random bay parking lot scenario with determined start and destination.
    """

    origin = Point(0., 0.)
    # The size of the parking scenario. All the obstacles are generated within this area.
    parking_size = [15.0, 15.0]
    mode = {"bay", "parallel", "mixed"}

    def _create_back_wall(self):
        return generate_bbox(
            self.origin, 0, self.bay_parking_size[1] / 2, np.random.uniform(0.5, 1.5))

    def _get_target_area(self):
        heading = truncate_gaussian(np.pi/2, np.pi/36, np.pi * 5/12, np.pi * 7/12)
        center_point = Point(
            0.0, 
        )

    def generate(self):

        return
