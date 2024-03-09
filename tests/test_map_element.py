##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_map_element.py
# @Description: This file defines the test cases for the map element module.
# @Author: Yueyuan Li
# @Version: 1.0.0

import logging

import pytest
from shapely.geometry import LineString, Polygon

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
def test_lane():
    lane1 = map_element.Lane(
        id_="1",
        left_side=LineString([(0, 0), (0, 1), (0, 2)]),
        right_side=LineString([(1, 0), (1, 1), (1, 2)]),
        line_ids=None,
        regulatory_ids=None,
        type_="lanelet",
        subtype=None,
        color=None,
        location=None,
        inferred_participants=None,
        speed_limit=20,
        speed_limit_unit="km/h",
        speed_limit_mandatory=True,
        custom_tags=None,
    )
    logging.info(f"start: {lane1.starts}")
    logging.info(f"end: {lane1.ends}")
    logging.info(f"shape: {lane1.shape}")

    lane1.add_related_lane(id_="2", relationship=1)
    lane1.add_related_lane(id_="3", relationship=map_element.LaneRelationship.LEFT_NEIGHBOR)

    logging.info(f"related lanes: {lane1.predecessors}")
    logging.info(f"related lanes: {lane1.successors}")
    logging.info(f"related lanes: {lane1.left_neighbors}")
    logging.info(f"related lanes: {lane1.right_neighbors}")

    assert lane1.is_related("2") == map_element.LaneRelationship.PREDECESSOR
    assert lane1.is_related("3") == 3
    assert not lane1.is_related("4")


@pytest.mark.map_element
def test_junction():
    junction = map_element.Junction(
        id_="1", incoming_road="2", connecting_road="3", contact_point="start", lane_links=[]
    )
    junction.add_lane_link(("1", "2"))
    junction.add_lane_link(("3", "4"))
    logging.info(f"lane links: {junction.lane_links}")


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
