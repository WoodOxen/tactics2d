##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: circle.py
# @Description: This file implements some frequently operations on circle.
# @Author: Tactics2D Team
# @Version: 1.0.0

from enum import Enum
from typing import Tuple, Union

import numpy as np
from cpp_function import Circle as cpp_circle


class Circle:
    """This class implement some frequently operations on circle.

    !!! note "TODO"
        To improve the performance, we will rewrite the methods in C++ in the future.
    """

    class ConstructBy(Enum):
        """The method to derive a circle.

        Attributes:
            ThreePoints (int): Derive a circle by three points.
            TangentVector (int): Derive a circle by a tangent point, a heading and a radius.
        """

        ThreePoints = 1
        TangentVector = 2

    @staticmethod
    def get_circle_by_three_points(
        pt1: Union[list, np.ndarray], pt2: Union[list, np.ndarray], pt3: Union[list, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """This function gets a circle by three points.

        Args:
            pt1 (Union[list, np.ndarray]): The first point. The shape is (2,).
            pt2 (Union[list, np.ndarray]): The second point. The shape is (2,).
            pt3 (Union[list, np.ndarray]): The third point. The shape is (2,).

        Returns:
            center (np.ndarray): The center of the circle. The shape is (2,).
            radius (float): The radius of the circle.
        """
        ## Verify input
        assert len(pt1) == 2
        assert len(pt2) == 2
        assert len(pt3) == 2

        ## Derive the circle
        d = -np.linalg.det(
            [
                [pt1[0] ** 2 + pt1[1] ** 2, pt1[1], 1],
                [pt2[0] ** 2 + pt2[1] ** 2, pt2[1], 1],
                [pt3[0] ** 2 + pt3[1] ** 2, pt3[1], 1],
            ]
        )
        e = np.linalg.det(
            [
                [pt1[0] ** 2 + pt1[1] ** 2, pt1[0], 1],
                [pt2[0] ** 2 + pt2[1] ** 2, pt2[0], 1],
                [pt3[0] ** 2 + pt3[1] ** 2, pt3[0], 1],
            ]
        )
        det = np.linalg.det([[pt1[0], pt1[1], 1], [pt2[0], pt2[1], 1], [pt3[0], pt3[1], 1]])

        D = d / det
        E = e / det

        center = [-D / 2, -E / 2]
        radius = np.linalg.norm(pt1 - center)

        return center, radius

    @staticmethod
    def get_circle_by_tangent_vector(
        tangent_point: Union[list, np.ndarray], heading: float, radius: float, side: str
    ) -> Tuple[np.ndarray, float]:
        """This function gets a circle by a tangent point, a heading and a radius.

        Args:
            tangent_point (np.ndarray): The tangent point on the circle. The shape is (2,).
            heading (float): The heading of the tangent point. The unit is radian.
            radius (float): The radius of the circle.
            side (int): The location of circle center relative to the tangent point. "L" represents left. "R" represents right.

        Returns:
            center (np.ndarray): The center of the circle. The shape is (2,).
            radius (float): The radius of the circle.
        """
        if side == "R":
            vec = np.array([np.cos(heading - np.pi / 2), np.sin(heading - np.pi / 2)]) * radius
        elif side == "L":
            vec = np.array([np.cos(heading + np.pi / 2), np.sin(heading + np.pi / 2)]) * radius
        return tangent_point + vec, radius

    @staticmethod
    def get_circle(method: ConstructBy, *args: tuple):
        """This function gets a circle by different given conditions.

        Args:
            method (ConstructBy): The method to derive a circle. The available choices are Circle.ConstructBy.ThreePoints) and Circle.ConstructBy.TangentVector).
            *args (tuple): The arguments of the method.

        Returns:
            center (np.ndarray): The center of the circle. The shape is (2,).
            radius (float): The radius of the circle.

        Raises:
            NotImplementedError: The input method id is not an available choice.
        """
        if method == Circle.ConstructBy.ThreePoints:
            # return Circle.get_circle_by_three_points(*args)
            center, radius = cpp_circle.get_circle_by_three_points(*args)
            return np.array(center), radius
        elif method == Circle.ConstructBy.TangentVector:
            # return Circle.get_circle_by_tangent_vector(*args)
            center, radius = cpp_circle.get_circle_by_tangent_vector(*args)
            return np.array(center), radius

        raise NotImplementedError

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

        # TODO:Frequent conversions to numpy arrays can slow down the runtime.
        # arc_points = np.array(cpp_circle.get_arc(center_point, radius, delta_angle, start_angle, clockwise, step_size))
        return arc_points
