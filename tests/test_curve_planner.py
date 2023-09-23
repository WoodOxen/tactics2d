import sys

sys.path.append(".")
sys.path.append("..")

import time
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import pyest

from tactics2d.math.curves import *


class DumbBezier:
    def __init__(self, order: int):
        self.order = order

    def get_curve(self, control_points: np.ndarray, n_interpolation: int):
        return np.array([[0, 0]])


@pytest.mark.math
def test_bezier():
    for order in range(1, 5):
        control_points = np.random.random_integers(0, 100, size=(order + 1, 2))
        interpolator_to_test = Bezier(order)
        interpolator = DumbBezier(order)

        t1 = time.time()
        bezier_curve_to_test = interpolator_to_test.get_curve(control_points, 500)
        t2 = time.time()
        bezier_curve = interpolator.get_curve(control_points, 500)
        t3 = time.time()

        assert ()
