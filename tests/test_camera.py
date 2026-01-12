# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for BEVCamera."""


import sys

sys.path.append(".")
sys.path.append("..")

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon

from tactics2d.map.element import Area, Lane, RoadLine
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.sensor.camera import BEVCamera


class TestBEVCamera:
    """Test suite for BEVCamera class."""

    @pytest.fixture
    def mock_map(self):
        """Create a mock map with basic elements."""
        mock_map = Mock()
        mock_map.boundary = (0, 0, 100, 100)  # Required by sensor_base.py

        # Mock areas
        mock_area = Mock(spec=Area)
        mock_area.id_ = 1
        mock_area.color = "area_color"
        mock_area.subtype = None
        mock_area.type_ = None
        mock_area.geometry = Mock()
        mock_area.geometry.exterior.coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        mock_area.geometry.interiors = []
        mock_area.geometry.distance.return_value = 5.0

        # Mock lanes
        mock_lane = Mock(spec=Lane)
        mock_lane.id_ = 2
        mock_lane.color = "lane_color"
        mock_lane.subtype = None
        mock_lane.type_ = None
        mock_lane.geometry = Mock()
        mock_lane.geometry.coords = [(0, 0), (5, 0), (5, 5), (0, 5)]
        mock_lane.geometry.distance.return_value = 3.0

        # Mock roadlines
        mock_roadline = Mock(spec=RoadLine)
        mock_roadline.id_ = 3
        mock_roadline.color = "roadline_color"
        mock_roadline.type_ = "solid"
        mock_roadline.subtype = "solid"
        mock_roadline.geometry = Mock()
        mock_roadline.geometry.coords = [(0, 0), (10, 0)]
        mock_roadline.geometry.distance.return_value = 2.0

        mock_map.areas = {1: mock_area}
        mock_map.lanes = {2: mock_lane}
        mock_map.roadlines = {3: mock_roadline}

        return mock_map

    @pytest.fixture
    def mock_participants(self):
        """Create mock participants."""
        participants = {}

        # Mock vehicle
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id_ = 101
        mock_vehicle.color = "vehicle_color"
        mock_vehicle.subtype = None
        mock_vehicle.type_ = None
        mock_vehicle.geometry = Mock()
        mock_vehicle.geometry.coords = [(0, 0), (2, 0), (2, 1), (0, 1)]
        mock_vehicle.trajectory = Mock()
        mock_vehicle.trajectory.get_state.return_value = Mock(location=(5.0, 5.0), heading=0.0)
        mock_pose = Mock()
        mock_pose.distance.return_value = 0.0
        mock_vehicle.get_pose.return_value = mock_pose  # Shapely polygon with distance method

        # Mock pedestrian
        mock_pedestrian = Mock(spec=Pedestrian)
        mock_pedestrian.id_ = 102
        mock_pedestrian.color = "pedestrian_color"
        mock_pedestrian.subtype = None
        mock_pedestrian.type_ = None
        mock_pedestrian.get_pose.return_value = (Point(3.0, 3.0), 0.5)

        participants[101] = mock_vehicle
        participants[102] = mock_pedestrian

        return participants

    def test_initialization(self, mock_map):
        """Test camera initialization with different parameters."""
        # Test with default perception_range
        camera = BEVCamera(id_=1, map_=mock_map)
        assert camera.id_ == 1
        assert camera.map_ == mock_map
        # When perception_range is None, sensor_base calculates it from map boundary
        assert camera.perception_range is not None
        assert isinstance(camera.perception_range, tuple)
        assert len(camera.perception_range) == 4

        # Test with float perception_range
        camera = BEVCamera(id_=2, map_=mock_map, perception_range=50.0)
        assert camera.id_ == 2
        # Float perception_range is converted to tuple of four values
        assert camera.perception_range == (50.0, 50.0, 50.0, 50.0)

        # Test with tuple perception_range
        camera = BEVCamera(id_=3, map_=mock_map, perception_range=(30.0, 30.0, 45.0, 15.0))
        assert camera.id_ == 3
        assert camera.perception_range == (30.0, 30.0, 45.0, 15.0)

    def test_get_type_method(self, mock_map):
        """Test _get_type method for different element types."""
        camera = BEVCamera(id_=1, map_=mock_map)

        # Test with element having subtype
        element_with_subtype = Mock()
        element_with_subtype.subtype = "test_subtype"
        element_with_subtype.type_ = "test_type"
        assert camera._get_type(element_with_subtype) == "test_subtype"

        # Test with element having only type_
        element_with_type = Mock()
        element_with_type.subtype = None
        element_with_type.type_ = "test_type"
        assert camera._get_type(element_with_type) == "test_type"

        # Test with Area instance
        mock_area = Mock(spec=Area)
        mock_area.subtype = None
        mock_area.type_ = None
        assert camera._get_type(mock_area) == "area"

        # Test with Lane instance
        mock_lane = Mock(spec=Lane)
        mock_lane.subtype = None
        mock_lane.type_ = None
        assert camera._get_type(mock_lane) == "lane"

        # Test with RoadLine instance
        mock_roadline = Mock(spec=RoadLine)
        mock_roadline.subtype = None
        mock_roadline.type_ = None
        assert camera._get_type(mock_roadline) == "roadline"

        # Test with Vehicle instance
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.subtype = None
        mock_vehicle.type_ = None
        assert camera._get_type(mock_vehicle) == "vehicle"

        # Test with Pedestrian instance
        mock_pedestrian = Mock(spec=Pedestrian)
        mock_pedestrian.subtype = None
        mock_pedestrian.type_ = None
        assert camera._get_type(mock_pedestrian) == "pedestrian"

        # Test default fallback
        unknown_element = Mock()
        unknown_element.subtype = None
        unknown_element.type_ = None
        assert camera._get_type(unknown_element) == "default"

    def test_in_perception_range(self, mock_map):
        """Test _in_perception_range method."""
        camera = BEVCamera(id_=1, map_=mock_map)

        # Test when position is None (should return True)
        camera._position = None
        mock_geometry = Mock()
        assert camera._in_perception_range(mock_geometry) is True

        # Test when position is set and geometry is within range
        camera._position = Point(0, 0)
        camera._perception_range = (10.0, 10.0, 10.0, 10.0)  # Sets max_perception_distance to 10.0
        assert camera.max_perception_distance == 10.0
        mock_geometry.distance.return_value = 15.0  # Within 2*10=20
        result = camera._in_perception_range(mock_geometry)
        # Verify distance was called with the position
        assert mock_geometry.distance.called
        assert mock_geometry.distance.call_args[0][0] == camera._position
        assert result, f"_in_perception_range returned {result}, expected True"

        # Test when geometry is outside range
        mock_geometry.distance.return_value = 25.0  # Outside 2*10=20
        assert not camera._in_perception_range(mock_geometry)

    def test_get_map_elements_output_format(self, mock_map):
        """Test _get_map_elements output format."""
        camera = BEVCamera(id_=1, map_=mock_map)
        camera._position = None  # Ensure all elements are in range

        map_data, road_id_set = camera._get_map_elements(prev_road_id_set=set())

        # Check output structure
        assert "road_id_to_remove" in map_data
        assert "road_elements" in map_data
        assert isinstance(map_data["road_id_to_remove"], list)
        assert isinstance(map_data["road_elements"], list)
        assert isinstance(road_id_set, set)

        # Check road elements format
        for element in map_data["road_elements"]:
            assert "id" in element
            assert "shape" in element
            assert "type" in element
            assert "color" in element
            assert "geometry" in element
            assert "line_width" in element

            if element["shape"] == "line":
                assert "line_style" in element

    def test_get_participants_output_format(self, mock_map, mock_participants):
        """Test _get_participants output format."""
        camera = BEVCamera(id_=1, map_=mock_map)
        camera._position = None  # Ensure all participants are in range

        frame = 0
        participant_ids = list(mock_participants.keys())

        participant_data, participant_id_set = camera._get_participants(
            frame=frame,
            participants=mock_participants,
            participant_ids=participant_ids,
            prev_participant_id_set=set(),
        )

        # Check output structure
        assert "participant_id_to_create" in participant_data
        assert "participant_id_to_remove" in participant_data
        assert "participants" in participant_data
        assert isinstance(participant_data["participant_id_to_create"], list)
        assert isinstance(participant_data["participant_id_to_remove"], list)
        assert isinstance(participant_data["participants"], list)
        assert isinstance(participant_id_set, set)

        # Check participants format
        for participant in participant_data["participants"]:
            assert "id" in participant
            assert "shape" in participant
            assert "type" in participant
            assert "color" in participant
            assert "line_width" in participant

            if participant["shape"] == "polygon":
                assert "geometry" in participant
                assert "position" in participant
                assert "rotation" in participant
            elif participant["shape"] == "circle":
                assert "position" in participant
                assert "radius" in participant

    def test_update_method_integration(self, mock_map, mock_participants):
        """Test update method integration."""
        camera = BEVCamera(id_=1, map_=mock_map)

        frame = 0
        participant_ids = list(mock_participants.keys())

        geometry_data, road_id_set, participant_id_set = camera.update(
            frame=frame,
            participants=mock_participants,
            participant_ids=participant_ids,
            prev_road_id_set=set(),
            prev_participant_id_set=set(),
            position=None,
        )

        # Check output structure
        assert "frame" in geometry_data
        assert "map_data" in geometry_data
        assert "participant_data" in geometry_data
        assert geometry_data["frame"] == frame

        # Check map_data structure
        map_data = geometry_data["map_data"]
        assert "road_id_to_remove" in map_data
        assert "road_elements" in map_data

        # Check participant_data structure
        participant_data = geometry_data["participant_data"]
        assert "participant_id_to_create" in participant_data
        assert "participant_id_to_remove" in participant_data
        assert "participants" in participant_data

        # Check return sets
        assert isinstance(road_id_set, set)
        assert isinstance(participant_id_set, set)

    def test_update_with_position(self, mock_map, mock_participants):
        """Test update method with position parameter."""
        camera = BEVCamera(id_=1, map_=mock_map)

        frame = 0
        participant_ids = list(mock_participants.keys())
        position = Point(5.0, 5.0)

        geometry_data, road_id_set, participant_id_set = camera.update(
            frame=frame,
            participants=mock_participants,
            participant_ids=participant_ids,
            prev_road_id_set=set(),
            prev_participant_id_set=set(),
            position=position,
        )

        assert camera._position == position

        # Should still return valid data structure
        assert "frame" in geometry_data
        assert "map_data" in geometry_data
        assert "participant_data" in geometry_data

    def test_none_parameters_handling(self, mock_map, mock_participants):
        """Test handling of None parameters in update method."""
        camera = BEVCamera(id_=1, map_=mock_map)

        frame = 0

        # Test with None for participant_ids
        geometry_data, road_id_set, participant_id_set = camera.update(
            frame=frame,
            participants=mock_participants,
            participant_ids=None,
            prev_road_id_set=None,
            prev_participant_id_set=None,
            position=None,
        )

        # Should still return valid structure
        assert "frame" in geometry_data
        assert "map_data" in geometry_data
        assert "participant_data" in geometry_data

        # Check that empty sets are created for None parameters
        assert isinstance(road_id_set, set)
        assert isinstance(participant_id_set, set)
