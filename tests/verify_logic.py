#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: verify_logic.py
# @Description: This file implements geometric alignment verification between
#               the DriveInsightD map and participant trajectories.
# @Author: Zexi Chen
# @Version: 1.0.0

import sys

sys.path.append(".")
sys.path.append("..")

import logging
import os
import statistics

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from tactics2d.dataset_parser.parse_driveinsightd import DriveInsightDParser


DRIVEINSIGHTD_FOLDER   = os.environ.get(
    "DRIVEINSIGHTD_FOLDER",
    "/root/autodl-tmp/driveinsightD/database/cz_zlin",
)
DRIVEINSIGHTD_MAP      = os.environ.get("DRIVEINSIGHTD_MAP",      "cz_zlin.xodr")
DRIVEINSIGHTD_SCENARIO = os.environ.get("DRIVEINSIGHTD_SCENARIO", "106")

# Verification thresholds
_MIN_IN_LANE_RATE    = 0.85   # >= 85% of observations must fall inside a lane polygon
_MAX_FALLBACK_MEAN_M = 2.5    # mean distance to centreline for junction vehicles (metres)
_MAX_FALLBACK_MAX_M  = 6.0    # hard ceiling on the same distance (metres)
_MAX_CENTROID_BIAS_M = 50.0   # max tolerated offset between vehicle and road centroids
_N_SAMPLE_FRAMES     = 20     # number of frames sampled uniformly from the scenario


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _lane_polygon(lane):
    """Build a Shapely Polygon from a lane's left and right boundaries."""
    try:
        left  = list(lane.left_side.coords)
        right = list(lane.right_side.coords)
    except AttributeError:
        return None

    if len(left) < 2 or len(right) < 2:
        return None

    ring = left + right[::-1]
    if len(ring) < 3:
        return None

    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def _lane_centreline(lane):
    """Return the midpoint LineString of a lane."""
    try:
        left  = list(lane.left_side.coords)
        right = list(lane.right_side.coords)
    except AttributeError:
        return None

    n = min(len(left), len(right))
    if n < 2:
        return None

    mid = [
        ((left[i][0] + right[i][0]) / 2, (left[i][1] + right[i][1]) / 2)
        for i in range(n)
    ]
    return LineString(mid)


def _collect_timestamps(participants: dict) -> list:
    """Collect all trajectory timestamps from all participants."""
    all_ts = set()
    for p in participants.values():
        frames = getattr(p.trajectory, "frames", None)
        if frames is not None:
            all_ts.update(frames)
    return sorted(all_ts)


# ---------------------------------------------------------------------------
# Pytest fixture: pre-built geometry structures (computed once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenario_data():
    """Load scenario and pre-build lane polygons and centrelines."""
    if not os.path.isdir(DRIVEINSIGHTD_FOLDER):
        pytest.skip(f"Dataset folder not found: {DRIVEINSIGHTD_FOLDER}")

    parser   = DriveInsightDParser()
    scenario = parser.parse(
        scenario_id=DRIVEINSIGHTD_SCENARIO,
        folder=DRIVEINSIGHTD_FOLDER,
        map_name=DRIVEINSIGHTD_MAP,
    )

    map_  = scenario["map"]
    lanes = list(map_.lanes.values() if isinstance(map_.lanes, dict) else map_.lanes)

    lane_polys = []
    lane_lines = []
    for lane in lanes:
        poly = _lane_polygon(lane)
        if poly is not None:
            lane_polys.append((lane.id_, poly))
        cl = _lane_centreline(lane)
        if cl is not None:
            lane_lines.append((lane.id_, cl))

    road_surface = unary_union([p for _, p in lane_polys])

    all_ts       = _collect_timestamps(scenario["participants"])
    step         = max(1, len(all_ts) // _N_SAMPLE_FRAMES)
    sample_times = all_ts[::step][:_N_SAMPLE_FRAMES]

    return {
        "scenario":     scenario,
        "lane_polys":   lane_polys,
        "lane_lines":   lane_lines,
        "road_surface": road_surface,
        "sample_times": sample_times,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.map_parser
def test_lane_polygon_coverage(scenario_data):
    """Verify that >= _MIN_IN_LANE_RATE of vehicle observations fall inside
    a lane polygon, with fallback distance checks for junction-area vehicles."""
    participants = scenario_data["scenario"]["participants"]
    lane_polys   = scenario_data["lane_polys"]
    lane_lines   = scenario_data["lane_lines"]
    sample_times = scenario_data["sample_times"]

    assert len(lane_polys) > 0, "No valid lane polygons were built."
    assert len(sample_times) > 0, "No trajectory timestamps found."

    in_lane_count      = 0
    fallback_count     = 0
    fallback_ok_count  = 0
    total_count        = 0
    fallback_distances = []

    for t in sample_times:
        for p in participants.values():
            try:
                state = p.trajectory.get_state(t)
            except (KeyError, AttributeError):
                continue
            if state is None:
                continue

            total_count += 1
            pt = Point(state.x, state.y)

            if any(poly.contains(pt) for _, poly in lane_polys):
                in_lane_count += 1
                continue

            fallback_count += 1
            min_dist = min(
                (cl.distance(pt) for _, cl in lane_lines),
                default=float("inf"),
            )
            fallback_distances.append(min_dist)
            if min_dist <= _MAX_FALLBACK_MEAN_M:
                fallback_ok_count += 1

    assert total_count > 0, (
        "No vehicle states were found across all sampled frames. "
        "Check that timestamps align with trajectory data."
    )

    combined_ok_rate = (in_lane_count + fallback_ok_count) / total_count
    mean_fb_dist     = statistics.mean(fallback_distances) if fallback_distances else 0.0
    max_fb_dist      = max(fallback_distances)             if fallback_distances else 0.0

    logging.info(f"Total observations   : {total_count}")
    logging.info(f"Inside lane polygon  : {in_lane_count} ({in_lane_count/total_count*100:.1f}%)")
    logging.info(f"Fallback (junction)  : {fallback_count}")
    logging.info(f"Combined OK rate     : {combined_ok_rate*100:.1f}%")
    logging.info(f"Mean fallback dist   : {mean_fb_dist:.2f} m")
    logging.info(f"Max  fallback dist   : {max_fb_dist:.2f} m")

    assert combined_ok_rate >= _MIN_IN_LANE_RATE, (
        f"Combined OK rate {combined_ok_rate*100:.1f}% < threshold {_MIN_IN_LANE_RATE*100:.0f}%."
    )
    assert mean_fb_dist <= _MAX_FALLBACK_MEAN_M, (
        f"Mean fallback distance {mean_fb_dist:.2f} m > threshold {_MAX_FALLBACK_MEAN_M} m."
    )
    assert max_fb_dist <= _MAX_FALLBACK_MAX_M, (
        f"Max fallback distance {max_fb_dist:.2f} m > threshold {_MAX_FALLBACK_MAX_M} m."
    )


@pytest.mark.map_parser
def test_spatial_alignment(scenario_data):
    """Verify that the vehicle trajectory centroid is close to the road surface
    centroid, catching systematic coordinate-frame mismatches."""
    participants  = scenario_data["scenario"]["participants"]
    road_surface  = scenario_data["road_surface"]
    sample_times  = scenario_data["sample_times"]

    vehicle_pts = []
    for t in sample_times:
        for p in participants.values():
            try:
                state = p.trajectory.get_state(t)
            except (KeyError, AttributeError):
                continue
            if state is not None:
                vehicle_pts.append((state.x, state.y))

    assert len(vehicle_pts) > 0, "No vehicle positions collected for centroid check."

    veh_cx  = statistics.mean(pt[0] for pt in vehicle_pts)
    veh_cy  = statistics.mean(pt[1] for pt in vehicle_pts)
    road_cx = road_surface.centroid.x
    road_cy = road_surface.centroid.y
    bias    = ((veh_cx - road_cx) ** 2 + (veh_cy - road_cy) ** 2) ** 0.5

    logging.info(f"Vehicle centroid : ({veh_cx:.1f}, {veh_cy:.1f})")
    logging.info(f"Road centroid    : ({road_cx:.1f}, {road_cy:.1f})")
    logging.info(f"Centroid offset  : {bias:.2f} m")

    assert bias <= _MAX_CENTROID_BIAS_M, (
        f"Centroid offset {bias:.2f} m > threshold {_MAX_CENTROID_BIAS_M} m. "
        "This may indicate a coordinate frame mismatch."
    )