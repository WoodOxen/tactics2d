# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for interpolator."""

import logging
import sys
import time

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline
from scipy.interpolate import CubicSpline as SciCubic
from scipy.spatial.distance import directed_hausdorff
from scipy.special import comb

sys.path.append(".")
sys.path.append("..")

from tactics2d.interpolator import Bezier, BSpline, CubicSpline, Dubins, ReedsShepp, Spiral


def compare_similarity(curve1: np.ndarray, curve2: np.ndarray, diff: float = 0.001) -> bool:
    """Compare similarity between two curves using length ratio and Hausdorff distance.

    Args:
        curve1: First curve as numpy array of shape (n_points, 2).
        curve2: Second curve as numpy array of shape (n_points, 2).
        diff: Maximum allowed relative difference for length and shape.

    Returns:
        True if both length difference ratio and Hausdorff distance ratio are less than diff.
    """
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
                [[-1, 0], [-0.5, 0.5], [0.5, -0.5], [1, 0], [-1, 0], [-0.5, 0.5], [0.5, -0.5]]
            ),
            np.array(np.arange(0, 10)),
            100,
        ),
        (2, np.random.uniform(-10, 10, (3, 3)), None, 100),
        (2, np.random.uniform(-10, 10, (3, 2)), np.array(np.arange(0, 4)), 100),
        (2, np.random.uniform(-10, 10, (3, 2)), np.array(np.arange(0, 10)), 100),
        (3, None, None, 1000),
        (4, None, None, 1000),
        # Edge case test: single control point (minimal length)
        (1, np.array([[0, 0]]), np.array([0, 0, 1, 1]), 10),
        # Edge case test: all control points identical (zero-length curve)
        (2, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), None, 50),
        # Edge case test: extreme curvature control points
        (2, np.array([[0, 0], [0.1, 100], [0.2, -100], [0.3, 0]]), None, 100),
        # Edge case test: minimal interpolation points
        (2, np.array([[0, 0], [1, 0], [2, 1], [3, 0]]), None, 2),
        # Edge case test: non-monotonic knot vector (should raise error)
        (2, np.array([[0, 0], [1, 0], [2, 1]]), np.array([0, 2, 1, 3, 4, 5]), 100),
        # Edge case test: invalid degree (should raise error)
        (-1, np.array([[0, 0], [1, 0], [2, 1]]), None, 100),
    ],
)
def test_b_spline(degree, control_points, knots, n_interpolation):
    """Test B-spline interpolation against SciPy implementation.

    This test validates B-spline curve generation for various degrees,
    control points, and knot vectors. It compares the custom implementation
    with SciPy's BSpline as reference and includes edge case tests.

    Args:
        degree: Degree of the B-spline.
        control_points: Control points as numpy array of shape (n, 2).
        knots: Knot vector as numpy array, or None for uniform knots.
        n_interpolation: Number of interpolation points.
    """
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
        # Edge case test: all control points identical (zero-length curve)
        (2, np.array([[0, 0], [0, 0], [0, 0]]), 50),
        # Edge case test: first and last control points coincide (closed curve)
        (3, np.array([[0, 0], [1, 1], [-1, 1], [0, 0]]), 100),
        # Edge case test: extreme curvature control points
        (2, np.array([[0, 0], [0.5, 100], [1, 0]]), 100),
        # Edge case test: minimal interpolation points
        (2, np.array([[0, 0], [1, 0], [2, 1]]), 2),
        # Edge case test: single control point (order 0? actual order 1 requires at least 2 points)
        (1, np.array([[0, 0], [0, 0]]), 10),
        # Edge case test: invalid order (should raise error)
        (0, np.array([[0, 0], [1, 0]]), 100),
        # Edge case test: mismatched control points count and order
        (2, np.array([[0, 0], [1, 0], [2, 1], [3, 0]]), 100),  # 4 points correspond to order 3
        # Edge case test: 1D control points (should raise error)
        (2, np.array([0, 1, 2]), 100),
    ],
)
def test_bezier(order: int, control_points: np.ndarray, n_interpolation: int):
    """Test Bézier curve interpolation against reference Bernstein polynomial implementation.

    This test validates Bézier curve generation for various orders and control points.
    It checks endpoint interpolation, convex hull property, and compares with
    a reference implementation using Bernstein polynomials.

    Args:
        order: Order of the Bézier curve (degree = order).
        control_points: Control points as numpy array of shape (n, 2).
        n_interpolation: Number of interpolation points.
    """
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

    # Mathematical properties verification
    # 1. Endpoint interpolation
    np.testing.assert_allclose(
        my_curve[0],
        control_points[0],
        atol=1e-10,
        err_msg="Bezier curve must interpolate first control point",
    )
    np.testing.assert_allclose(
        my_curve[-1],
        control_points[-1],
        atol=1e-10,
        err_msg="Bezier curve must interpolate last control point",
    )

    # 2. Convex hull property (simplified: check bounding box)
    min_ctrl = control_points.min(axis=0)
    max_ctrl = control_points.max(axis=0)
    # All curve points should be within the bounding box of control points (with small tolerance)
    assert np.all(
        my_curve >= min_ctrl - 1e-10
    ), "Curve point outside control points bounding box (min)"
    assert np.all(
        my_curve <= max_ctrl + 1e-10
    ), "Curve point outside control points bounding box (max)"

    # 3. Reference implementation using Bernstein polynomials
    n = order  # degree of Bezier curve
    t = np.linspace(0, 1, n_interpolation)
    t3 = time.time()
    reference_curve = np.zeros((n_interpolation, 2))

    for i in range(n + 1):
        # Bernstein polynomial: B_{i,n}(t) = C(n,i) * t^i * (1-t)^{n-i}
        bernstein = comb(n, i) * (t**i) * ((1 - t) ** (n - i))
        reference_curve += bernstein[:, np.newaxis] * control_points[i]

    t4 = time.time()

    # Verify against reference implementation
    assert compare_similarity(
        my_curve, reference_curve
    ), "Bezier curve implementation differs from reference Bernstein polynomial implementation"

    t_custom = t2 - t1
    t_reference = t4 - t3

    if t_custom > t_reference:
        logging.warning(
            f"Our Bezier is ~{t_custom/(t_reference + 1e-8):.2f}x slower than reference implementation"
        )


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
        # Edge case test: minimal number of control points (exactly 3)
        ("natural", 3, np.array([[0, 0], [1, 1], [2, 0]]), 50),
        # Edge case test: all control points have same y-value (straight line)
        ("clamped", None, np.array([[0, 0], [1, 0], [2, 0], [3, 0]]), 100),
        # Edge case test: extreme curvature control points
        ("natural", None, np.array([[0, 0], [1, 100], [2, -100], [3, 0]]), 100),
        # Edge case test: minimal interpolation points
        ("not-a-knot", None, np.array([[0, 0], [1, 1], [2, 0], [3, 1]]), 2),
        # Edge case test: non-increasing x-values (should raise error)
        ("natural", None, np.array([[0, 0], [2, 1], [1, 2], [3, 0]]), 100),
        # Edge case test: insufficient control points (less than 3)
        ("natural", 2, np.array([[0, 0], [1, 1]]), 100),
        # Edge case test: duplicate x-values with different y-values (should raise error)
        ("clamped", None, np.array([[0, 0], [1, 1], [1, 2], [2, 0]]), 100),
        # Edge case test: minimal x-interval
        ("natural", None, np.array([[0, 0], [1e-10, 1], [2, 0]]), 100),
    ],
)
def test_cubic_spline(boundary_type: str, n: int, control_points: np.ndarray, n_interpolation: int):
    """Test cubic spline interpolation against SciPy implementation.

    This test validates cubic spline curve generation for various boundary conditions
    (natural, clamped, not-a-knot) and control points. It compares the custom
    implementation with SciPy's CubicSpline as reference and includes edge case tests.

    Args:
        boundary_type: Boundary condition type as string ("natural", "clamped", "not-a-knot").
        n: Optional number of control points for random generation.
        control_points: Control points as numpy array of shape (n, 2).
        n_interpolation: Number of interpolation points between each control point pair.
    """
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
        # Edge case test: same start and end points, same heading (zero length)
        (7.5, np.array([0, 0]), 0, np.array([0, 0]), 0, 0.01),
        # Edge case test: same start and end points, different heading (in-place turn)
        (7.5, np.array([0, 0]), 0, np.array([0, 0]), np.pi / 2, 0.01),
        # Edge case test: same start and end points, opposite heading
        (7.5, np.array([0, 0]), 0, np.array([0, 0]), np.pi, 0.01),
        # Edge case test: minimal turning radius
        (0.1, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.001),
        # Edge case test: maximal turning radius
        (100.0, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.1),
        # Edge case test: zero radius (should raise error)
        (0.0, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.01),
        # Edge case test: minimal step size
        (7.5, np.array([0, 0]), 0, np.array([10, 0]), 0, 1e-6),
        # Edge case test: maximal step size
        (7.5, np.array([0, 0]), 0, np.array([10, 0]), 0, 1.0),
        # Edge case test: extreme heading values (beyond 2π)
        (7.5, np.array([0, 0]), 10 * np.pi, np.array([10, 0]), 20 * np.pi, 0.01),
    ],
)
def test_dubins(radius, start_point, start_heading, end_point, end_heading, step_size):
    """Test Dubins path generation for basic correctness.

    This test validates Dubins path generation for various start/end configurations
    and radius values. It checks that the computed curve length matches the
    reported path length within tolerance.

    Args:
        radius: Minimum turning radius (positive for forward, negative for backward).
        start_point: Starting position as numpy array (x, y).
        start_heading: Starting heading angle in radians.
        end_point: Ending position as numpy array (x, y).
        end_heading: Ending heading angle in radians.
        step_size: Step size for curve discretization.
    """
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
    if min(path.length, curve_length) == 0:
        assert abs(path.length - curve_length) < 1e-10
    else:
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
        (7.5, np.array([0, 0]), 0, np.array([1e-5, 1e-5]), 0, 0.01),  # Near-zero displacement
        (7.5, np.array([10, 10]), 7.0, np.array([-20, -10]), -3.0, 0.01),  # Out-of-range angles
        # Edge case test: same start and end points, same heading (zero length)
        (7.5, np.array([0, 0]), 0, np.array([0, 0]), 0, 0.01),
        # Edge case test: same start and end points, different heading (in-place turn)
        (7.5, np.array([0, 0]), 0, np.array([0, 0]), np.pi / 2, 0.01),
        # Edge case test: minimal turning radius
        (0.1, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.001),
        # Edge case test: maximal turning radius
        (100.0, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.1),
        # Edge case test: zero radius (should raise error)
        (0.0, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.01),
        # Edge case test: minimal step size
        (7.5, np.array([0, 0]), 0, np.array([10, 0]), 0, 1e-6),
        # Edge case test: maximal step size
        (7.5, np.array([0, 0]), 0, np.array([10, 0]), 0, 1.0),
        # Edge case test: extreme heading values (beyond 2π)
        (7.5, np.array([0, 0]), 10 * np.pi, np.array([10, 0]), 20 * np.pi, 0.01),
        # Edge case test: reverse driving scenario (specific to ReedsShepp)
        (7.5, np.array([0, 0]), np.pi, np.array([10, 0]), np.pi, 0.01),  # reverse driving
        # Edge case test: minimal negative radius (should raise error)
        (-0.1, np.array([0, 0]), 0, np.array([10, 0]), 0, 0.01),
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
    if min(path.length, curve_length) == 0:
        assert abs(path.length - curve_length) < 1e-10
    else:
        assert abs(path.length - curve_length) / min(path.length, curve_length) < 0.01


@pytest.mark.math
@pytest.mark.parametrize(
    "length, n_interpolation, gamma, start_curvature",
    [
        (5.0, 500, 0.1, 0.0),  # normal spiral
        (10.0, 1000, 0.0, 0.1),  # circular arc
        (8.0, 800, 0.0, 0.0),  # straight line
        (105.0, 10000, 0.05, 0.05),  # long curve
        (2.0, 200, -0.1, 0.2),  # negative gamma (reverse spiral)
        # Edge case test: zero length
        (0.0, 10, 0.1, 0.0),
        # Edge case test: minimal length
        (1e-6, 10, 0.1, 0.0),
        # Edge case test: maximal length
        (1000.0, 10000, 0.001, 0.0),
        # Edge case test: extreme curvature values
        (10.0, 1000, 10.0, 0.0),  # large gamma
        (10.0, 1000, -10.0, 0.0),  # small gamma
        (10.0, 1000, 0.0, 10.0),  # large start curvature
        (10.0, 1000, 0.0, -10.0),  # small start curvature
        # Edge case test: minimal interpolation points
        (5.0, 2, 0.1, 0.0),
        # Edge case test: negative length (should raise error?)
        (-5.0, 500, 0.1, 0.0),
        # Edge case test: zero interpolation points (should raise error?)
        (5.0, 0, 0.1, 0.0),
        # Edge case test: both gamma and start curvature zero (straight line)
        (10.0, 1000, 0.0, 0.0),
    ],
)
def test_spiral(length, n_interpolation, gamma, start_curvature):
    spiral = Spiral()
    start_point = np.array([0.0, 0.0])
    heading = np.random.uniform(0, 2 * np.pi)

    # Handle negative length (should raise ValueError)
    if length < 0:
        with pytest.raises(ValueError, match="Length must be non-negative."):
            spiral.get_curve(length, start_point, heading, start_curvature, gamma, n_interpolation)
        return

    t1 = time.time()
    curve = spiral.get_curve(length, start_point, heading, start_curvature, gamma, n_interpolation)
    t2 = time.time()

    # Ensure correct number of points
    assert len(curve) == n_interpolation, f"Expected {n_interpolation}, got {len(curve)}"

    # Skip further checks for empty curve
    if n_interpolation == 0:
        logging.info(
            f"Spiral test (len={length}, gamma={gamma}) passed with zero interpolation points"
        )
        return

    # Check that the start point matches
    np.testing.assert_allclose(curve[0], start_point, atol=1e-6, err_msg="Start point mismatch")

    # Check curve length
    segment_lengths = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
    curve_length = segment_lengths.sum()

    # Handle zero-length case to avoid division by zero
    if max(length, curve_length) == 0:
        rel_error = 0.0
    else:
        rel_error = abs(curve_length - length) / max(length, curve_length)

    # Use larger tolerance for extreme gamma values due to numerical stability issues
    if abs(gamma) > 5.0:
        max_error = 0.05  # 5% tolerance for extreme gamma values
    elif n_interpolation <= 2:
        # Very few interpolation points cannot accurately represent curves
        max_error = 0.2  # 20% tolerance for minimal interpolation points
    else:
        max_error = 0.01  # 1% tolerance for normal cases

    assert (
        rel_error < max_error
    ), f"Length error too high: {rel_error:.4f} (max allowed: {max_error})"

    logging.info(
        f"Spiral test (len={length}, gamma={gamma}) passed in {t2 - t1:.4f}s. "
        f"Computed len={curve_length:.3f}, err={rel_error:.4%}"
    )
