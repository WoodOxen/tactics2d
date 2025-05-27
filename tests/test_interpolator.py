import logging
import sys
import time

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline
from scipy.interpolate import CubicSpline as SciCubic
from scipy.spatial.distance import directed_hausdorff

sys.path.append(".")
sys.path.append("..")

from tactics2d.interpolator import Bezier, BSpline, CubicSpline, Dubins, ReedsShepp, Spiral


def compare_similarity(curve1: np.ndarray, curve2: np.ndarray, diff: float = 0.001) -> bool:
    len1 = np.linalg.norm(curve1[1:] - curve1[:-1], axis=1).sum()
    len2 = np.linalg.norm(curve2[1:] - curve2[:-1], axis=1).sum()
    len_diff = abs(len1 - len2)
    ratio_len_diff = len_diff / max(min(len1, len2), 1e-8)

    hausdorff_dist = max(
        directed_hausdorff(curve1, curve2)[0], directed_hausdorff(curve2, curve1)[0]
    )
    ratio_shape_diff = hausdorff_dist / max(min(len1, len2), 1e-8)

    if ratio_len_diff >= diff or ratio_shape_diff >= diff:
        logging.warning(f"Hausdorff dist: {hausdorff_dist}, Length diff: {len_diff}")
    return ratio_len_diff < diff and ratio_shape_diff < diff


@pytest.mark.math
@pytest.mark.parametrize(
    "degree, control_points, knots, n_interpolation",
    [
        (0, None, None, 100),
        (1, None, None, 1000),
        (
            2,
            np.array([[-1, 0], [-0.5, 0.5], [0.5, -0.5], [1, 0]]),
            np.array([0, 0, 0, 1, 2, 2, 2]),
            100,
        ),
        (
            2,
            np.array(
                [
                    [-1, 0],
                    [-0.5, 0.5],
                    [0.5, -0.5],
                    [1, 0],
                    [-1, 0],
                    [-0.5, 0.5],
                    [0.5, -0.5],
                ]
            ),
            np.array(np.arange(0, 10)),
            100,
        ),
        (2, np.random.uniform(-10, 10, (3, 3)), None, 100),
        (2, np.random.uniform(-10, 10, (3, 2)), np.array(np.arange(0, 4)), 100),
        (2, np.random.uniform(-10, 10, (3, 2)), np.array(np.arange(0, 10)), 100),
        (3, None, None, 1000),
        (4, None, None, 1000),
    ],
)
def test_b_spline(degree, control_points, knots, n_interpolation):
    if control_points is None:
        n_control_point = np.random.randint(5, 100)
        control_points = np.zeros((n_control_point, 2))
        for i in range(1, n_control_point):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0.1, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    if knots is None and np.random.rand() < 0.3:
        knots = np.zeros((len(control_points) + degree + 1,))
        for i in range(1, len(knots) - 1):
            knots[i] = knots[i - 1] + np.random.randint(0, 3)

    # Construct B-spline instance

    # Call our BSpline implementation
    try:
        t1 = time.time()
        my_curve = BSpline.get_curve(control_points, knots, degree, n_interpolation)
        t2 = time.time()
    except ValueError as err:
        if degree < 0:
            assert (
                err.args[0] == "Degree must be non-negative."
            ), "Incorrect error message for invalid degree"

        elif control_points.shape[-1] != 2:
            assert (
                err.args[0] == "Control points should have shape (n, 2)."
            ), "Unexpected error message for invalid control points shape"

        elif knots is not None and len(knots) != len(control_points) + degree + 1:
            expected = len(control_points) + degree + 1
            assert (
                err.args[0] == f"Expected {expected} knots, got {len(knots)}."
            ), "Unexpected error message for invalid knot count"

        elif knots is not None and np.any(np.diff(knots) < 0):
            assert (
                err.args[0] == "Knot vector must be non-decreasing."
            ), "Unexpected error for non-monotonic knots"

        else:
            raise
        return

    # Create scipy BSpline for ground truth
    if knots is None:
        knots = np.linspace(0, 1, len(control_points) + degree + 1)

    t3 = time.time()
    scipy_bspline = SciBSpline(knots, control_points, degree)
    us = np.linspace(knots[degree], knots[-degree - 1], n_interpolation, endpoint=False)
    scipy_curve = np.array([scipy_bspline(u) for u in us])
    t4 = time.time()

    assert compare_similarity(
        my_curve, scipy_curve
    ), "The curve output differs from SciPy's BSpline"

    t_custom = t2 - t1
    t_scipy = t4 - t3
    if t_custom > t_scipy:
        logging.warning(f"Our B-spline is ~{t_custom / (t_scipy + 1e-8):.2f}x slower than SciPy's")


@pytest.mark.math
@pytest.mark.parametrize(
    "order, control_points, n_interpolation",
    [
        (1, None, 100),
        (2, None, 1000),
        (3, np.random.uniform(1, 5, (5, 2)), 100),
        (3, np.random.uniform(1, 5, (4, 3)), 100),
        (4, None, 1000),
        (4, np.random.uniform(1, 5, (4, 2)), 100),
        (5, None, 1000),
    ],
)
def test_bezier(order: int, control_points: np.ndarray, n_interpolation: int):
    if control_points is None:
        control_points = np.zeros((order + 1, 2))
        for i in range(1, order + 1):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    # Test curve generation
    try:
        t1 = time.time()
        my_curve = Bezier.get_curve(control_points, n_interpolation, order)
        t2 = time.time()

    except ValueError as err:
        if order < 1:
            assert (
                err.args[0] == "Order must be greater than or equal to one."
            ), "Test failed: error handling for invalid order."

        elif len(control_points.shape) != 2 or control_points.shape[1] != 2:
            assert (
                err.args[0] == "Control points should have shape (n, 2)."
            ), "Test failed: error handling for invalid shape of control points."

        elif len(control_points) != order + 1:
            assert (
                err.args[0] == "Number of control points must be equal to order plus one."
            ), "Test failed: error handling for invalid number of control points."

        else:
            raise err
        return

    # Reference result (bezier package)
    try:
        import bezier
    except ImportError:
        logging.warning("Skipping reference test: 'bezier' package not installed.")
        return

    t3 = time.time()
    curve = bezier.Curve(control_points.T, degree=order).evaluate_multi(
        np.linspace(0.0, 1.0, n_interpolation)
    )
    t4 = time.time()

    assert compare_similarity(my_curve, curve.T), "Bezier curve output mismatch."

    t_custom = t2 - t1
    t_bezier = t4 - t3  # avoid zero-division

    if t_custom > t_bezier:
        logging.warning(f"Our Bezier is ~{t_custom / (t_bezier + 1e-8):.2f}x slower than bezier's")


@pytest.mark.math
@pytest.mark.parametrize(
    "boundary_type, n, control_points, n_interpolation",
    [
        ("hello-world", 2, None, 100),
        ("natural", 2, None, 100),
        ("clamped", None, np.random.uniform(-10, 10, (3, 3)), 100),
        ("not-a-knot", None, np.random.uniform(-10, 10, (3, 1)), 100),
        ("natural", None, None, 10),
        ("natural", None, None, 100),
        ("clamped", None, None, 10),
        ("clamped", None, None, 100),
        ("not-a-knot", None, None, 10),
        ("not-a-knot", None, None, 100),
    ],
)
def test_cubic_spline(boundary_type: str, n: int, control_points: np.ndarray, n_interpolation: int):
    if boundary_type == "natural":
        my_cubic_spline = CubicSpline.BoundaryType.Natural
    elif boundary_type == "clamped":
        my_cubic_spline = CubicSpline.BoundaryType.Clamped
    elif boundary_type == "not-a-knot":
        my_cubic_spline = CubicSpline.BoundaryType.NotAKnot
    else:
        my_cubic_spline = boundary_type

    if control_points is None:
        n = np.random.randint(3, 1000) if n is None else n
        control_points = np.zeros((n, 2))
        for i in range(1, n):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    try:
        t1 = time.time()
        curve = CubicSpline.get_curve(
            control_points, n_interpolation=n_interpolation, boundary_type=my_cubic_spline
        )
        t2 = time.time()

    except ValueError as err:
        if boundary_type not in ["natural", "clamped", "not-a-knot"]:
            assert (
                err.args[0]
                == f"Invalid boundary type: {boundary_type}. Available options are: {', '.join(CubicSpline.BoundaryType.__members__.keys())} or an integer value from 1 to {len(CubicSpline.BoundaryType.__members__)}."
            )

        elif len(control_points.shape) != 2 or control_points.shape[1] != 2:
            assert (
                err.args[0] == "Control points should have shape (n, 2)."
            ), "Test failed: error handling for invalid shape of control points."

        elif len(control_points) < 3:
            assert (
                err.args[0] == "Need at least 3 control points."
            ), "Test failed: error handling for insufficient number of control points."

        elif np.any(np.diff(control_points[:, 0]) <= 0):
            assert (
                err.args[0] == "x-values must be strictly increasing."
            ), "Test failed: error handling for non-increasing x-values."

        else:
            raise err
        return

    sci_cubic = SciCubic(control_points[:, 0], control_points[:, 1], bc_type=boundary_type)
    t3 = time.time()
    xs = np.linspace(control_points[:-1, 0], control_points[1:, 0], n_interpolation)
    xs = np.concatenate(xs.T, axis=None)
    ys = sci_cubic(xs)
    t4 = time.time()

    assert compare_similarity(
        curve, np.array([xs, ys]).T
    ), "The curve output of the Cubic Spline interpolator is incorrect."

    t_custom = t2 - t1
    t_scipy = t4 - t3
    if t_custom > t_scipy:
        logging.warning(f"Our Cubic is ~{t_custom / (t_scipy + 1e-8):.2f}x slower than SciPy's")


@pytest.mark.math
@pytest.mark.parametrize(
    "radius, start_point, start_heading, end_point, end_heading, step_size",
    [
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
        (7.5, np.array([10, 10]), 0.5, np.array([5, 5]), 2, 0.01),
        (-7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
    ],
)
def test_dubins(radius, start_point, start_heading, end_point, end_heading, step_size):
    # t1 = time.time()
    try:
        dubins = Dubins(radius)
    except ValueError as err:
        if radius <= 0:
            assert (
                err.args[0] == "The minimum turning radius must be positive."
            ), "Test failed: error handling for invalid radius."
        else:
            raise err
        return

    path = dubins.get_curve(start_point, start_heading, end_point, end_heading, step_size)
    curve = path.curve

    curve_length = np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum()
    assert abs(path.length - curve_length) / min(path.length, curve_length) < 0.01


@pytest.mark.math
@pytest.mark.parametrize(
    "radius, start_point, start_heading, end_point, end_heading, step_size",
    [
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
        (7.5, np.array([10, 10]), 0.5, np.array([5, 5]), 2, 0.01),
        (-7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
        (7.5, np.array([0, 0]), 0, np.array([0, 0]), np.pi, 0.01),  # Same point, flipped heading
        # (7.5, np.array([0, 0]), 0, np.array([1e-5, 1e-5]), 0, 0.01),   # Near-zero displacement
        (7.5, np.array([10, 10]), 7.0, np.array([-20, -10]), -3.0, 0.01),  # Out-of-range angles
    ],
)
def test_reeds_shepp(radius, start_point, start_heading, end_point, end_heading, step_size):
    try:
        rs = ReedsShepp(radius)
    except ValueError as err:
        if radius <= 0:
            assert (
                err.args[0] == "The minimum turning radius must be positive."
            ), "Test failed: error handling for invalid radius."
        else:
            raise err
        return

    path = rs.get_curve(start_point, start_heading, end_point, end_heading, step_size)
    curve = path.curve

    curve_length = np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum()
    assert abs(path.length - curve_length) / min(path.length, curve_length) < 0.01


@pytest.mark.math
@pytest.mark.parametrize(
    "length, n_interpolation, gamma, start_curvature",
    [
        (5.0, 500, 0.1, 0.0),  # 正常螺旋
        (10.0, 1000, 0.0, 0.1),  # 圆弧
        (8.0, 800, 0.0, 0.0),  # 直线
        (105.0, 10000, 0.05, 0.05),  # 长曲线
        (2.0, 200, -0.1, 0.2),  # 负 gamma（反向螺旋）
    ],
)
def test_spiral(length, n_interpolation, gamma, start_curvature):
    spiral = Spiral()
    start_point = np.array([0.0, 0.0])
    heading = np.random.uniform(0, 2 * np.pi)

    t1 = time.time()
    curve = spiral.get_curve(length, start_point, heading, start_curvature, gamma)
    t2 = time.time()

    # Ensure correct number of points
    assert len(curve) == n_interpolation, f"Expected {n_interpolation+1}, got {len(curve)}"

    # Check that the start point matches
    np.testing.assert_allclose(curve[0], start_point, atol=1e-6, err_msg="Start point mismatch")

    # Check curve length
    segment_lengths = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
    curve_length = segment_lengths.sum()
    rel_error = abs(curve_length - length) / max(length, curve_length)
    assert rel_error < 0.01, f"Length error too high: {rel_error:.4f}"

    logging.info(
        f"Spiral test (len={length}, gamma={gamma}) passed in {t2 - t1:.4f}s. "
        f"Computed len={curve_length:.3f}, err={rel_error:.4%}"
    )
