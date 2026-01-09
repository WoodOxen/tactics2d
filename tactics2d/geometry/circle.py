# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Circle implementation."""


from typing import Tuple

import numpy as np

from cpp_geometry import Circle as cpp_Circle
from tactics2d.geometry.direction import RelativeDirection


class Circle:
    """This class implement some common operations on circle."""

    @staticmethod
    def get_circle(**kwargs) -> Tuple[np.ndarray, float]:
        """Create a circle from various input parameter combinations.

        Supported usages:

            1. From two points. The first point is the center, and the second point is a point on the circumference.
                get_circle(point1, point2)

            2. From three points on the circumference.
                get_circle(point1, point2, point3)

            3. From a tangent vector and radius.
                get_circle(tangent_point, tangent_heading, radius, side)

        Args:
            point1 (ArrayLike): The first point, either the center of the circle or a point on the circumference. The shape should be (2,).
            point2 (ArrayLike): The second point, a point on the circumference. The shape should be (2,).
            point3 (ArrayLike): The third point on the circumference. The shape should be (2,).
            tangent_point (ArrayLike): A point on the circumference where the tangent is applied. The shape should be (2,).
            tangent_heading (float): The heading of the tangent line at the tangent point, in radians.
            radius (float): The radius of the circle.
            side (Union[str, RelativeDirection]): The side of the tangent line to consider. Can be "L"/"R" or a RelativeDirection enum value (LEFT/RIGHT).

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the center of the circle as a numpy array and the radius as a float. The center is an array of shape (2,)

        Examples:
            >>> Circle.get_circle(point1=np.array([0, 0]), point2=np.array([1, 0]))
            (array([0.5, 0. ]), 0.5)

            >>> Circle.get_circle(point1=np.array([0, 0]), point2=np.array([1, 0]), point3=np.array([0, 1]))
            (array([0.5, 0.5]), 0.7071067811865476)

            >>> Circle.get_circle(tangent_point=np.array([1, 0]), tangent_heading=np.pi/2, radius=1, side="L")
            (array([1., 1.]), 1.0)
        """

        if len(kwargs) == 2 and "point1" in kwargs and "point2" in kwargs:
            center = np.asarray(kwargs["point1"])
            radius = np.linalg.norm(kwargs["point2"] - center)
            return center, radius

        elif len(kwargs) == 3 and "point1" in kwargs and "point2" in kwargs and "point3" in kwargs:
            p1 = np.asarray(kwargs["point1"])
            p2 = np.asarray(kwargs["point2"])
            p3 = np.asarray(kwargs["point3"])

            center, radius = cpp_Circle.get_circle_by_three_points(
                p1.tolist(), p2.tolist(), p3.tolist()
            )

            return np.array(center), radius

        elif (
            len(kwargs) == 4
            and "tangent_point" in kwargs
            and "tangent_heading" in kwargs
            and "radius" in kwargs
            and "side" in kwargs
        ):
            tangent_point = np.asarray(kwargs["tangent_point"])
            tangent_heading = kwargs["tangent_heading"]
            radius = kwargs["radius"]
            side = kwargs["side"]

            if isinstance(side, str):
                side = RelativeDirection.from_string(side)
            if not side in [RelativeDirection.LEFT, RelativeDirection.RIGHT]:
                raise ValueError(
                    f"Invalid side: {side}. "
                    "Must be 'L'/'R' or a RelativeDirection enum value (LEFT/RIGHT)."
                )

            center, radius = cpp_Circle.get_circle_by_tangent_vector(
                tangent_point.tolist(), tangent_heading, radius, side.value
            )

            return np.array(center), radius

        else:
            raise ValueError(
                "Invalid arguments. The supported parameter combinations are:\n"
                "1. get_circle(point1, point2)\n"
                "2. get_circle(point1, point2, point3)\n"
                "3. get_circle(tangent_point, tangent_heading, radius, side)"
            )

    @staticmethod
    def get_arc(
        center_point: np.ndarray,
        radius: float,
        delta_angle: float,
        start_angle: float,
        clockwise: bool = True,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """This function gets the points on an arc curve line.

        Args:
            center_point (np.ndarray): The center of the arc. The shape is (2,).
            radius (float): The radius of the arc.
            delta_angle (float): The angle of the arc. This values is expected to be positive. The unit is radian.
            start_angle (float): The start angle of the arc. The unit is radian.
            clockwise (bool): The direction of the arc. True represents clockwise. False represents counterclockwise.
            step_size (float): The step size of the arc. The unit is radian.

        Returns:
            arc_points(np.ndarray): The points on the arc. The shape is (int(radius * delta / step_size), 2).
        """
        if clockwise:
            angles = np.array(
                np.arange(start_angle, (start_angle - delta_angle), -step_size / radius)
            )
        else:
            angles = np.array(
                np.arange(start_angle, (start_angle + delta_angle), step_size / radius)
            )

        arc_points = center_point + np.array([np.cos(angles), np.sin(angles)]).T * radius

        return arc_points
