from typing import Tuple

import numpy as np


class Vector:
    """_summary_"""

    @staticmethod
    def angle(vec1, vec2) -> float:
        """Get the angle between two vectors.

        Returns:
            angle (float): The angle between two vectors in radians. The value is in the range [0, pi].
        """
        dot_product = np.dot(vec1, vec2)
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        cos_angle = dot_product / (vec1_norm * vec2_norm)
        angle = np.arccos(cos_angle)
        return angle


class Circle:
    """_summary_"""

    @staticmethod
    def get_circle(pt1, pt2, pt3) -> Tuple[list, float]:
        """Derive a circle by three points.

        Returns:
            center (list): The center of the circle
            radius (float): The radius of the circle
        """
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
        det = np.linalg.det(
            [[pt1[0], pt1[1], 1], [pt2[0], pt2[1], 1], [pt3[0], pt3[1], 1]]
        )

        D = d / det
        E = e / det

        center = [-D / 2, -E / 2]
        radius = np.linalg.norm(pt1 - center)

        return center, radius
