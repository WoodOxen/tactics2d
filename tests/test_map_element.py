##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_map_element.py
# @Description: This file defines the test cases for the map element module.
# @Author: Yueyuan Li
# @Version: 1.0.0

import logging

import pytest
from shapely.geometry import Polygon

import tactics2d.map.element as map_element
from tactics2d.map.generator import ParkingLotGenerator


@pytest.mark.map_element
def test_node():
    node1 = map_element.Node(0, 1, 2)
    node2 = map_element.Node(1, 3, 4)
    assert node1.x == 1
    assert node1.y == 2

    node3 = node1 + node2
    assert node3.x == 4
    assert node3.y == 6

    node4 = node1 - node2
    assert node4.x == -2
    assert node4.y == -2


@pytest.mark.map_element
def test_area():
    area = map_element.Area(
        id_="1",
        geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        line_ids=None,
        regulatory_ids=None,
        type_="multipolygon",
        subtype=None,
        color=None,
        location=None,
        inferred_participants=None,
        speed_limit=20,
        speed_limit_unit="km/h",
        speed_limit_mandatory=True,
        custom_tags=None,
    )

    logging.info(f"shape: {area.shape}")


@pytest.mark.map_element
def test_map():
    map_ = map_element.Map(name="test_map", scenario_type="test_scenario", country="test_country")
    generator = ParkingLotGenerator()
    generator.generate(map_)

    logging.info(f"map boundary: {map_.boundary}")
    _ = map_.get_by_id(0)
    map_.reset()
