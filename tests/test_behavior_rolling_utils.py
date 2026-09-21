# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for shared closed-loop collision and progress primitives."""

import numpy as np
import pytest

from tactics2d.behavior.rolling_utils import (
    COLLISION_MARGIN,
    PROGRESS_END,
    PROGRESS_START,
    REAR_TOL,
    SIDE_TOL,
    bodies_overlap,
    classify_collision,
    collision_kind,
    first_observed_frame,
    progress_window,
)
from tactics2d.geometry import spatial

_MARGIN = 0.7


def _poses(rng, count, steps, spread=12.0):
    """Build pose arrays with x, y, yaw drawn from a small window."""

    arrays = {}
    for index in range(count):
        array = np.zeros((steps, 4))
        array[:, 0] = rng.uniform(-spread, spread, steps)
        array[:, 1] = rng.uniform(-spread, spread, steps)
        array[:, 3] = rng.uniform(-np.pi, np.pi, steps)
        arrays["a%d" % index] = array
    return arrays


# ------------------------------------------------------------------
# Frozen references: the pre-refactor expressions, kept verbatim so the
# extraction stays pinned to the behaviour each model had before.
# ------------------------------------------------------------------


def _legacy_factor_overlap(pose_a, dims_a, pose_b, dims_b):
    """InterSim's check: spatial.boxes_overlap on full extents."""

    box_a = (pose_a[0], pose_a[1], pose_a[3], dims_a[0], dims_a[1])
    box_b = (pose_b[0], pose_b[1], pose_b[3], dims_b[0], dims_b[1])
    return spatial.boxes_overlap(box_a, box_b, _MARGIN)


def _legacy_linear_overlap(pose_a, dims_a, pose_b, dims_b):
    """SMART's check: shaped boxes with the margin subtracted from each extent."""

    body_a = spatial.oriented_box(
        float(pose_a[0]),
        float(pose_a[1]),
        float(pose_a[3]),
        dims_a[0] - _MARGIN,
        dims_a[1] - _MARGIN,
    )
    body_b = spatial.oriented_box(
        float(pose_b[0]),
        float(pose_b[1]),
        float(pose_b[3]),
        dims_b[0] - _MARGIN,
        dims_b[1] - _MARGIN,
    )
    return body_a.intersects(body_b)


def _legacy_collision_kind(poses, dims, ego_id, index, overlap):
    """The per-runner collision loop as it stood before the extraction."""

    pose_ego = poses[ego_id][index]
    if pose_ego[0] == -1:
        return None
    for other_id in poses:
        if other_id == ego_id:
            continue
        pose_other = poses[other_id][index]
        if pose_other[0] == -1:
            continue
        if not overlap(pose_ego, dims[ego_id], pose_other, dims[other_id]):
            continue
        diff = abs(spatial.normalize_angle(float(pose_ego[3]) - float(pose_other[3])))
        if diff < REAR_TOL:
            return 2
        if diff > SIDE_TOL:
            return 1
        return 0
    return None


def _legacy_progress(poses, agent_id, end_index, steps):
    """The progress window as it stood before the extraction."""

    array = poses[agent_id]
    total = 0.0
    counted = 0
    for index in range(12, 80):
        if index >= end_index:
            break
        if index + 1 >= steps:
            break
        pose_i = array[index]
        pose_j = array[index + 1]
        if pose_i[0] == -1 or pose_j[0] == -1:
            break
        distance = float(np.hypot(pose_i[0] - pose_j[0], pose_i[1] - pose_j[1]))
        if distance >= 20.0:
            continue
        total += distance
        counted += 1
    return total, counted


# ------------------------------------------------------------------
# bodies_overlap
# ------------------------------------------------------------------


@pytest.mark.math
def test_bodies_overlap_factor_matches_boxes_overlap():
    """Test the factor convention reproduces the InterSim expression."""

    rng = np.random.default_rng(11)
    pose_a = np.array([0.3, -0.2, 0.4, 0.0])
    pose_b = np.array([2.1, 0.15, -1.2, 0.0])
    dims_a = (4.8, 1.9)
    dims_b = (4.2, 1.8)
    mismatches = 0
    for _ in range(300):
        pose_a[:2] = rng.uniform(-4.0, 4.0, 2)
        pose_b[:2] = rng.uniform(-4.0, 4.0, 2)
        expected = _legacy_factor_overlap(pose_a, dims_a, pose_b, dims_b)
        assert bodies_overlap(pose_a, dims_a, pose_b, dims_b, "factor") == expected
        mismatches += expected
    assert mismatches > 0, "the sample must include overlapping pairs"


@pytest.mark.math
def test_bodies_overlap_linear_matches_oriented_box():
    """Test the linear convention reproduces the SMART expression."""

    rng = np.random.default_rng(12)
    pose_a = np.zeros(4)
    pose_b = np.zeros(4)
    for dims in ((4.8, 1.9), (0.6, 0.6), (1.0, 0.5)):
        for _ in range(200):
            pose_a[:3] = rng.uniform(-3.0, 3.0, 3)
            pose_a[3] = rng.uniform(-np.pi, np.pi)
            pose_b[:3] = rng.uniform(-3.0, 3.0, 3)
            pose_b[3] = rng.uniform(-np.pi, np.pi)
            expected = _legacy_linear_overlap(pose_a, dims, pose_b, dims)
            assert bodies_overlap(pose_a, dims, pose_b, dims, "linear") == expected


@pytest.mark.math
def test_bodies_overlap_conventions_differ():
    """Test the two shrink conventions are genuinely different rules.

    The extraction deliberately did not unify them: InterSim scales the full
    extents by the margin while the SMART port subtracts it, so a pair can be a
    collision under one and clear under the other. This test is the guard
    against a future "cleanup" that quietly collapses them.
    """

    # Longitudinally tight: the linear box keeps more length, so it is the
    # stricter test here and the factor box already misses the contact.
    pose_a = np.array([0.0, 0.0, 0.0, 0.0])
    pose_b = np.array([3.7, 0.0, 0.0, 0.0])
    dims = (4.8, 1.9)
    assert bodies_overlap(pose_a, dims, pose_b, dims, "linear") is True
    assert bodies_overlap(pose_a, dims, pose_b, dims, "factor") is False

    # Laterally tight: the factor box is wider, so it catches what the linear
    # box lets through.
    pose_b = np.array([0.0, 1.28, 0.0, 0.0])
    assert bodies_overlap(pose_a, dims, pose_b, dims, "factor") is True
    assert bodies_overlap(pose_a, dims, pose_b, dims, "linear") is False


@pytest.mark.math
def test_bodies_overlap_is_symmetric():
    """Test the overlap test does not depend on argument order."""

    rng = np.random.default_rng(13)
    pose_a = np.zeros(4)
    pose_b = np.zeros(4)
    dims_a = (4.8, 1.9)
    dims_b = (0.8, 0.7)
    for _ in range(200):
        pose_a[:3] = rng.uniform(-3.0, 3.0, 3)
        pose_a[3] = rng.uniform(-np.pi, np.pi)
        pose_b[:3] = rng.uniform(-3.0, 3.0, 3)
        pose_b[3] = rng.uniform(-np.pi, np.pi)
        for shrink in ("factor", "linear"):
            forward = bodies_overlap(pose_a, dims_a, pose_b, dims_b, shrink)
            backward = bodies_overlap(pose_b, dims_b, pose_a, dims_a, shrink)
            assert forward == backward


# ------------------------------------------------------------------
# classify_collision
# ------------------------------------------------------------------


@pytest.mark.math
@pytest.mark.parametrize(
    "diff,expected",
    [
        (0.0, 2),  # same heading, rear-ended
        (REAR_TOL - 1e-6, 2),
        (REAR_TOL + 1e-6, 0),
        (np.pi / 2.0, 0),
        (SIDE_TOL - 1e-6, 0),
        (SIDE_TOL + 1e-6, 1),
        (np.pi, 1),  # head-on
    ],
)
def test_classify_collision_thresholds(diff, expected):
    """Test the relative-heading thresholds map to the upstream classes."""

    assert classify_collision(0.0, diff) == expected
    assert classify_collision(0.0, -diff) == expected


# ------------------------------------------------------------------
# collision_kind
# ------------------------------------------------------------------


@pytest.mark.math
@pytest.mark.parametrize("shrink", ["factor", "linear"])
def test_collision_kind_matches_legacy_loop(shrink):
    """Test the extracted loop reproduces both pre-refactor loops exactly."""

    rng = np.random.default_rng(14)
    legacy_overlap = _legacy_factor_overlap if shrink == "factor" else _legacy_linear_overlap
    dims = {"a%d" % index: (4.8, 1.9) for index in range(4)}
    hits = 0
    for _ in range(120):
        poses = _poses(rng, 4, 6, spread=4.0)
        for index in range(6):
            expected = _legacy_collision_kind(poses, dims, "a0", index, legacy_overlap)
            assert collision_kind(poses, dims, "a0", index, shrink) == expected
            hits += expected is not None
    assert hits > 0, "the sample must include colliding steps"


@pytest.mark.math
def test_collision_kind_returns_none_when_ego_invalid():
    """Test an unoccupied ego slot reports no collision."""

    poses = {"ego": np.array([[5.0, 0.0, 0.0, 0.0]]), "other": np.array([[5.0, 0.0, 0.0, 0.0]])}
    poses["ego"][0, 0] = -1
    dims = {"ego": (4.8, 1.9), "other": (4.8, 1.9)}
    assert collision_kind(poses, dims, "ego", 0) is None


@pytest.mark.math
def test_collision_kind_skips_invalid_other():
    """Test an unoccupied other slot does not count as a collision."""

    poses = {"ego": np.array([[0.0, 0.0, 0.0, 0.0]]), "ghost": np.array([[-1.0, -1.0, 0.0, 0.0]])}
    dims = {"ego": (4.8, 1.9), "ghost": (4.8, 1.9)}
    assert collision_kind(poses, dims, "ego", 0) is None


@pytest.mark.math
def test_collision_kind_scan_order_decides_class():
    """Test the first overlapping agent in insertion order decides the class."""

    poses = {
        "ego": np.array([[0.0, 0.0, 0.0, 0.0]]),
        # Same heading -> rear. Both sit inside the factor convention's reach.
        "rear": np.array([[3.0, 0.0, 0.0, 0.0]]),
        # Opposite heading -> side.
        "head_on": np.array([[0.0, 1.28, 0.0, np.pi]]),
    }
    dims = {key: (4.8, 1.9) for key in poses}
    assert collision_kind(poses, dims, "ego", 0, "factor") == 2

    reordered = {"ego": poses["ego"], "head_on": poses["head_on"], "rear": poses["rear"]}
    assert collision_kind(reordered, dims, "ego", 0, "factor") == 1


# ------------------------------------------------------------------
# progress_window
# ------------------------------------------------------------------


@pytest.mark.math
def test_progress_window_matches_legacy():
    """Test the extracted window reproduces the pre-refactor sum."""

    rng = np.random.default_rng(15)
    poses = {"a0": np.zeros((91, 4))}
    poses["a0"][:, 0] = np.cumsum(rng.uniform(0.0, 3.0, 91))
    poses["a0"][:, 1] = rng.uniform(-0.2, 0.2, 91)
    for end_index in (40, 80, 91):
        expected = _legacy_progress(poses, "a0", end_index, 91)
        assert progress_window(poses, "a0", end_index, 91) == pytest.approx(expected)


@pytest.mark.math
def test_progress_window_defaults_match_module_constants():
    """Test the window bounds and step cap are the documented ones."""

    assert (PROGRESS_START, PROGRESS_END) == (12, 80)
    assert COLLISION_MARGIN == _MARGIN
    rng = np.random.default_rng(16)
    poses = {"a0": np.zeros((91, 4))}
    poses["a0"][:, 0] = np.cumsum(rng.uniform(0.0, 3.0, 91))
    assert progress_window(poses, "a0", 91, 91) == _legacy_progress(poses, "a0", 91, 91)


def _ramp_poses():
    """Build one agent advancing exactly 1 m per step, over 91 steps."""

    poses = {"a0": np.zeros((91, 4))}
    poses["a0"][:, 0] = np.arange(91, dtype=float)
    return poses


@pytest.mark.math
def test_progress_window_drops_implausible_step():
    """Test a step longer than the cap is skipped rather than ending the scan."""

    poses = _ramp_poses()
    poses["a0"][20, 0] = 90.0
    # The two steps touching the spike exceed the cap and are dropped; the
    # remaining window is still summed, so the scan does not end early.
    total, counted = progress_window(poses, "a0", 91, 91)
    assert counted == PROGRESS_END - PROGRESS_START - 2
    assert total == pytest.approx(float(counted))


@pytest.mark.math
def test_progress_window_breaks_on_invalid_pose():
    """Test the scan stops at the first unoccupied step."""

    poses = _ramp_poses()
    poses["a0"][30, 0] = -1.0
    # Step 29 reads the invalid slot as its successor and stops there.
    total, counted = progress_window(poses, "a0", 91, 91)
    assert counted == 30 - PROGRESS_START - 1
    assert total == pytest.approx(float(counted))


@pytest.mark.math
def test_progress_window_stops_at_end_index():
    """Test the window never runs past the last simulated step."""

    poses = _ramp_poses()
    total, counted = progress_window(poses, "a0", 20, 91)
    assert counted == 20 - PROGRESS_START
    assert total == pytest.approx(float(20 - PROGRESS_START))


# ------------------------------------------------------------------
# first_observed_frame
# ------------------------------------------------------------------


@pytest.mark.math
def test_first_observed_frame_takes_the_minimum(make_trajectory):
    """Test the anchor is the earliest frame across participants."""

    participants = {
        "early": _Participant(make_trajectory(first_frame=900, last_frame=5000)),
        "late": _Participant(make_trajectory(first_frame=1500, last_frame=5000)),
    }
    assert first_observed_frame(participants) == 900


@pytest.mark.math
def test_first_observed_frame_skips_missing(make_trajectory):
    """Test participants without an observed first frame are ignored."""

    participants = {
        "empty": _Participant(make_trajectory(first_frame=None, last_frame=None)),
        "late": _Participant(make_trajectory(first_frame=1500, last_frame=5000)),
    }
    assert first_observed_frame(participants) == 1500


@pytest.mark.math
def test_first_observed_frame_raises_without_trajectories(make_trajectory):
    """Test a scenario with no observed frame is rejected."""

    participants = {"empty": _Participant(make_trajectory(first_frame=None, last_frame=None))}
    with pytest.raises(ValueError, match="no observed frames"):
        first_observed_frame(participants)


class _Participant:
    """Minimal stand-in carrying only the trajectory the helper reads."""

    def __init__(self, trajectory):
        self.trajectory = trajectory


@pytest.fixture
def make_trajectory():
    """Build a stand-in trajectory exposing the two frame attributes."""

    def _make(first_frame, last_frame):
        return type("TrajectoryStub", (), {"first_frame": first_frame, "last_frame": last_frame})

    return _make
