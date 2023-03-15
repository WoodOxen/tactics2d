from typing import Tuple

import numpy as np
from shapely.geometry import LinearRing, Point
from shapely.affinity import affine_transform

from tactics2d.math.random import truncate_gaussian
from tactics2d.map.element import Map


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
        np.mean(angle_range), np.std(angle_range), angle_range[0], angle_range[1]
    )
    radius = truncate_gaussian(
        np.mean(radius_range), np.std(radius_range), radius_range[0], radius_range[1]
    )

    return Point(origin.x + radius * np.cos(angle), origin.y + radius * np.sin(angle))


def generate_bbox(center_point: Point, heading: float, length: float, width: float):
    """Generate a bounding box."""
    bbox = LinearRing(
        [
            [0.5 * length, -0.5 * width],
            [0.5 * length, 0.5 * width],
            [-0.5 * length, 0.5 * width],
            [-0.5 * length, -0.5 * width],
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
    """Generate a random bay parking lot scenario with determined start and destination."""

    origin = Point(0.0, 0.0)
    scenario_size = [15.0, 15.0]
    modes = {"bay", "parallel", "mixed"}

    def __init__(self, vehicle_size: Tuple[float, float], mode: str = "bay"):
        if mode not in self.modes:
            raise NotImplementedError
        self.mode = mode

    def _get_target_area(self):
        heading = truncate_gaussian(
            np.pi / 2, np.pi / 36, np.pi * 5 / 12, np.pi * 7 / 12
        )
        center_point = Point(
            0.0,
        )

    def _create_back_wall(self):
        return generate_bbox(
            self.origin, 0, self.bay_parking_size[1] / 2, np.random.uniform(0.5, 1.5)
        )

    def generate(self, map_: Map):
        return
