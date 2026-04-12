#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_xodr_load.py
# @Description: This file implements the test cases for the OpenDRIVE map parser.
# @Author: Zexi Chen
# @Version: 1.0.0

import sys

sys.path.append(".")
sys.path.append("..")

import logging
import os
import time

import pytest

from tactics2d.map.parser.parse_xodr import XODRParser


DRIVEINSIGHTD_XODR_PATH = os.environ.get(
    "DRIVEINSIGHTD_XODR_PATH",
    "/root/autodl-tmp/driveinsightD/database/cz_zlin/cz_zlin.xodr",
)


@pytest.mark.map_parser
@pytest.mark.parametrize(
    "xodr_path, min_lanes, min_roadlines, min_junctions",
    [
        (DRIVEINSIGHTD_XODR_PATH, 10, 10, 1),
    ],
)
def test_xodr_parser(xodr_path: str, min_lanes: int, min_roadlines: int, min_junctions: int):
    if not os.path.exists(xodr_path):
        pytest.skip(f"Map file not found: {xodr_path}")

    parser = XODRParser()

    t1 = time.time()
    parsed_map = parser.parse(xodr_path)
    t2 = time.time()

    n_lanes     = len(parsed_map.lanes)
    n_roadlines = len(parsed_map.roadlines)
    n_junctions = len(parsed_map.junctions)

    assert n_lanes     >= min_lanes,     f"Expected >= {min_lanes} lanes, got {n_lanes}."
    assert n_roadlines >= min_roadlines, f"Expected >= {min_roadlines} roadlines, got {n_roadlines}."
    assert n_junctions >= min_junctions, f"Expected >= {min_junctions} junctions, got {n_junctions}."

    logging.info(f"Map name      : {parsed_map.name}")
    logging.info(f"Lanes         : {n_lanes}")
    logging.info(f"RoadLines     : {n_roadlines}")
    logging.info(f"Junctions     : {n_junctions}")
    logging.info(f"Parse time    : {t2 - t1:.3f}s")