from typing import Tuple, Union
from enum import Enum

import numpy as np


class Circle:
    class ConstructBy(Enum):
        ThreePoints = 1
        TangentVector = 2

    @staticmethod
    def get_circle_by_three_points(
        pt1: Union[list, np.ndarray], pt2: Union[list, np.ndarray], pt3: Union[list, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Derive a circle by three points.

        Args:
            pt1 (Union[list, np.ndarray]): The first point. The shape is (2,).
            pt2 (Union[list, np.ndarray]): The second point. The shape is (2,).
            pt3 (Union[list, np.ndarray]): The third point. The shape is (2,).
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
        """Derive a circle by a tangent point, a heading and a radius. The circle is tangent to the tangent point.

        Args:
            tangent_point (np.ndarray): The tangent point on the circle. The shape is (2,).
            heading (float): The heading of the tangent point. The unit is radian.
            radius (float): The radius of the circle.
            direction (int): The location of circle center relative to the tangent point. "L" represents left. "R" represents right.
        """
        if side == "R":
            vec = np.array([np.cos(heading - np.pi / 2), np.sin(heading - np.pi / 2)]) * radius
        elif side == "L":
            vec = np.array([np.cos(heading + np.pi / 2), np.sin(heading + np.pi / 2)]) * radius
        return tangent_point + vec, radius

    @staticmethod
    def get_circle(method: ConstructBy, *args):
        """Derive a circle by different given conditions."""
        if method == Circle.ConstructBy.ThreePoints:
            return Circle.get_circle_by_three_points(*args)
        elif method == Circle.ConstructBy.TangentVector:
            return Circle.get_circle_by_tangent_vector(*args)
        else:
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
        """Derive an arc by the center, radius, start angle and end angle.

        Args:
            center_point (np.ndarray): The center of the arc. The shape is (2,).
            radius (float): The radius of the arc.
            delta_angle (float): The angle of the arc. This values is expected to be positive. The unit is radian.
            start_angle (float): The start angle of the arc. The unit is radian.
            clockwise (bool): The direction of the arc. True represents clockwise. False represents counterclockwise.
            step_size (float): The step size of the arc. The unit is radian.

        Returns:
            np.ndarray: The points on the arc. The shape is (int(radius * delta / step_size), 2).
        """
        if clockwise:
            angles = np.array(
                np.arange(start_angle, (start_angle - delta_angle), -step_size / radius)
            )
        else:
            angles = np.array(
                np.arange(start_angle, (start_angle + delta_angle), step_size / radius)
            )
        print(angles)
        arc_points = center_point + np.array([np.cos(angles), np.sin(angles)]).T * radius
        return arc_points
