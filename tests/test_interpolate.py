import sys

sys.path.append(".")
sys.path.append("..")

import time
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import pyest

from tactics2d.math.interpolate import *


class DumbBezier:
    def __init__(self, order: int):
        self.order = order

    def get_curve(self, control_points: np.ndarray, n_interpolation: int):
        return np.array([[0, 0]])


@pytest.mark.math
def test_bezier():
    bezier1 = Bezier(1)
    bezier2 = Bezier(2)
    bezier3 = Bezier(3)
    bezier4 = Bezier(4)

    control_points = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])

    curve_points1 = bezier1.get_curve(control_points[:2], 100)
    curve_points2 = bezier2.get_curve(control_points[:3], 100)
    curve_points3 = bezier3.get_curve(control_points[:4], 100)
    curve_points4 = bezier4.get_curve(control_points[:5], 100)

    import matplotlib.pyplot as plt

    plt.plot(control_points[:, 0], control_points[:, 1], "ro:")
    plt.plot(curve_points1[:, 0], curve_points1[:, 1], "b")
    plt.plot(curve_points2[:, 0], curve_points2[:, 1], "g")
    plt.plot(curve_points3[:, 0], curve_points3[:, 1], "y")
    plt.plot(curve_points4[:, 0], curve_points4[:, 1], "r")
    plt.show()
    plt.savefig("bezier.png")


@pytest.mark.math
def test_cubic_spline():
    control_points = np.array([[4, 4], [4.5, 6], [4.6, 6.6], [4.7, 6.2], [5.2, 4.9], [6, 4.7]])
    # control_points = []
    # for x in range(10):
    #     control_points.append([x, np.sin(x)])
    # control_points = np.array(control_points)

    cubic_spline1 = CubicSpline(CubicSpline.BoundaryType.Natural)
    curve_points1 = cubic_spline1.get_curve(control_points, n_interpolation=100)
    cubic_spline2 = CubicSpline(CubicSpline.BoundaryType.Clamped)
    curve_points2 = cubic_spline2.get_curve(control_points, xx=(0, 0), n_interpolation=100)
    cubic_spline3 = CubicSpline(CubicSpline.BoundaryType.NotAKnot)
    curve_points3 = cubic_spline3.get_curve(control_points, n_interpolation=100)

    import matplotlib.pyplot as plt

    plt.plot(curve_points1[:, 0], curve_points1[:, 1], "b")
    plt.plot(curve_points2[:, 0], curve_points2[:, 1], "r")
    plt.plot(curve_points3[:, 0], curve_points3[:, 1], "g")

    plt.show()
    plt.savefig("cubic_spline.png")
    plt.close()

    from scipy.interpolate import CubicSpline as SciCubic

    cs1 = SciCubic(control_points[:, 0], control_points[:, 1], bc_type="natural")
    cs2 = SciCubic(control_points[:, 0], control_points[:, 1], bc_type="clamped")
    cs3 = SciCubic(control_points[:, 0], control_points[:, 1], bc_type="not-a-knot")
    xs = np.arange(4, 6, 0.01)
    # xs = np.arange(0, 9, 0.01)
    plt.plot(xs, cs1(xs), "b")
    plt.plot(xs, cs2(xs), "r")
    plt.plot(xs, cs3(xs), "g")
    plt.show()
    plt.savefig("scipy_cubic_spline.png")
    plt.close()
