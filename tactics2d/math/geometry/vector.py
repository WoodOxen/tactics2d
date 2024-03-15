##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: vector.py
# @Description: This file implements some frequently operations on vector.
# @Author: Yueyuan Li
# @Version: 1.0.0

import numpy as np


class Vector:
    """This class implement some frequently operations on vector.

    !!! note "TODO"
        To improve the performance, we will rewrite the methods in C++ in the future.
    """

    @staticmethod
    def angle(vec1, vec2) -> float:
        """This method calculate the angle between two vectors.

        Args:
            vec1 (np.ndarray): The first vector. The shape is (n,).
            vec2 (np.ndarray): The second vector. The shape is (n,).

        Returns:
            angle (float): The angle between two vectors in radians. The value is in the range [0, pi].
        """
        dot_product = np.dot(vec1, vec2)
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        cos_angle = dot_product / (vec1_norm * vec2_norm)
        angle = np.arccos(cos_angle)
        return angle
