#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_driveinsightd.py
# @Description: This file implements the test cases for the DriveInsightD dataset parser.
# @Author: Zexi Chen
# @Version: 1.0.0

import sys

sys.path.append(".")
sys.path.append("..")

import logging
import os
import time

import pytest

from tactics2d.dataset_parser.parse_driveinsightd import DriveInsightDParser


# ---------------------------------------------------------------------------
# Dataset root is resolved from an environment variable so the test suite
# can be run on any machine without modifying source code:
#
#   export DRIVEINSIGHTD_FOLDER=/path/to/driveinsightD/database/cz_zlin
#   pytest tests/test_driveinsightd.py -v -m dataset_parser
# ---------------------------------------------------------------------------
DRIVEINSIGHTD_FOLDER = os.environ.get(
    "DRIVEINSIGHTD_FOLDER",
    "./tactics2d/data/trajectory_sample/DriveInsightD/cz_zlin",
)
DRIVEINSIGHTD_MAP = os.environ.get("DRIVEINSIGHTD_MAP", "cz_zlin.xodr")


# ---------------------------------------------------------------------------
# Trajectory parser tests
# ---------------------------------------------------------------------------

@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "scenario_id, stamp_range, min_participants",
    [
        ("106", None,                          1),
        ("106", (-float("inf"), float("inf")), 1),
        ("106", (0, 5000),                     1),
    ],
)
def test_driveinsightd_trajectory_parser(
    scenario_id: str, stamp_range: tuple, min_participants: int
):
    if not os.path.isdir(DRIVEINSIGHTD_FOLDER):
        pytest.skip(f"Dataset folder not found: {DRIVEINSIGHTD_FOLDER}")

    parser = DriveInsightDParser()

    t1 = time.time()
    participants, time_range = parser.parse_trajectory(
        file=scenario_id, folder=DRIVEINSIGHTD_FOLDER, stamp_range=stamp_range
    )
    t2 = time.time()

    assert len(participants) >= min_participants, (
        f"Expected >= {min_participants} participants, got {len(participants)}."
    )

    t_start, t_end = time_range
    assert t_start <= t_end, (
        f"Invalid time range: {t_start} ms -> {t_end} ms."
    )

    # Every participant must have at least one trajectory state
    for name, p in participants.items():
        assert len(p.trajectory.frames) >= 1, (
            f"Participant '{name}' has no trajectory states."
        )

    logging.info(f"Participants  : {len(participants)}")
    logging.info(f"Time range    : {t_start} ms -> {t_end} ms")
    logging.info(f"Parse time    : {t2 - t1:.3f}s")


# ---------------------------------------------------------------------------
# Full scenario (trajectory + map + metadata) tests
# ---------------------------------------------------------------------------

@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "scenario_id, min_participants, min_lanes",
    [
        ("106", 1, 10),
    ],
)
def test_driveinsightd_full_scenario(
    scenario_id: str, min_participants: int, min_lanes: int
):
    if not os.path.isdir(DRIVEINSIGHTD_FOLDER):
        pytest.skip(f"Dataset folder not found: {DRIVEINSIGHTD_FOLDER}")

    map_path = os.path.join(DRIVEINSIGHTD_FOLDER, DRIVEINSIGHTD_MAP)
    if not os.path.exists(map_path):
        pytest.skip(f"Map file not found: {map_path}")

    parser = DriveInsightDParser()

    t1 = time.time()
    scenario = parser.parse(
        scenario_id=scenario_id,
        folder=DRIVEINSIGHTD_FOLDER,
        map_name=DRIVEINSIGHTD_MAP,
    )
    t2 = time.time()

    # Participants
    participants = scenario["participants"]
    assert len(participants) >= min_participants, (
        f"Expected >= {min_participants} participants, got {len(participants)}."
    )

    # Time range
    t_start, t_end = scenario["time_range"]
    assert t_start < t_end, f"Invalid time range: {t_start} ms -> {t_end} ms."

    # Map
    n_lanes = len(scenario["map"].lanes)
    assert n_lanes >= min_lanes, f"Expected >= {min_lanes} lanes, got {n_lanes}."

    # Metadata keys
    required_metadata_keys = ("time", "weather", "precipitation", "friction")
    for key in required_metadata_keys:
        assert key in scenario["metadata"], f"Missing metadata key: '{key}'."

    # Friction must be in a physically plausible range
    friction = scenario["metadata"]["friction"]
    assert 0.0 <= friction <= 2.0, f"Friction {friction} out of plausible range [0, 2]."

    logging.info(f"Participants  : {len(participants)}")
    logging.info(f"Time range    : {t_start} ms -> {t_end} ms")
    logging.info(f"Lanes         : {n_lanes}")
    logging.info(f"Roadlines     : {len(scenario['map'].roadlines)}")
    logging.info(f"Junctions     : {len(scenario['map'].junctions)}")
    logging.info(f"Weather       : {scenario['metadata']['weather']}")
    logging.info(f"Total time    : {t2 - t1:.3f}s")