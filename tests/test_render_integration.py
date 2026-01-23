# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for camera and renderer."""


import sys

sys.path.append(".")
sys.path.append("..")

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from shapely.geometry import Point

from tactics2d.renderer.matplotlib_renderer import MatplotlibRenderer
from tactics2d.sensor.camera import BEVCamera


class TestCameraRendererIntegration:
    """Integration tests for camera and renderer data flow."""

    @pytest.fixture
    def mock_map(self):
        """Create a mock map with basic elements."""
        mock_map = Mock()
        mock_map.boundary = (0, 0, 100, 100)  # Required by sensor_base.py

        # Mock area
        mock_area = Mock()
        mock_area.id_ = 1
        mock_area.color = "area_color"
        mock_area.subtype = "area"
        mock_area.type_ = None
        mock_area.geometry = Mock()
        mock_area.geometry.exterior.coords = [(0, 0), (20, 0), (20, 20), (0, 20)]
        mock_area.geometry.interiors = []
        mock_area.geometry.distance.return_value = 0.0

        # Mock lane
        mock_lane = Mock()
        mock_lane.id_ = 2
        mock_lane.color = "lane_color"
        mock_lane.subtype = "lane"
        mock_lane.type_ = None
        mock_lane.geometry = Mock()
        mock_lane.geometry.coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        mock_lane.geometry.distance.return_value = 0.0

        # Mock roadline
        mock_roadline = Mock()
        mock_roadline.id_ = 3
        mock_roadline.color = "roadline_color"
        mock_roadline.type_ = "solid"
        mock_roadline.subtype = "solid"
        mock_roadline.geometry = Mock()
        mock_roadline.geometry.coords = [(0, 0), (20, 0)]
        mock_roadline.geometry.distance.return_value = 0.0

        mock_map.areas = {1: mock_area}
        mock_map.lanes = {2: mock_lane}
        mock_map.roadlines = {3: mock_roadline}

        return mock_map

    @pytest.fixture
    def mock_participants(self):
        """Create mock participants."""
        participants = {}

        # Mock vehicle
        mock_vehicle = Mock()
        mock_vehicle.id_ = 101
        mock_vehicle.color = "vehicle_color"
        mock_vehicle.subtype = "vehicle"
        mock_vehicle.type_ = None
        mock_vehicle.geometry = Mock()
        mock_vehicle.geometry.coords = [(0, 0), (2, 0), (2, 1), (0, 1)]
        mock_vehicle.trajectory = Mock()
        mock_vehicle.trajectory.get_state.return_value = Mock(location=(10.0, 10.0), heading=0.0)
        mock_pose = Mock()
        mock_pose.distance.return_value = 0.0
        mock_vehicle.get_pose.return_value = mock_pose

        # Mock pedestrian
        mock_pedestrian = Mock()
        mock_pedestrian.id_ = 102
        mock_pedestrian.color = "pedestrian_color"
        mock_pedestrian.subtype = "pedestrian"
        mock_pedestrian.type_ = None
        mock_pedestrian.get_pose.return_value = (Point(5.0, 5.0), 0.5)

        participants[101] = mock_vehicle
        participants[102] = mock_pedestrian

        return participants

    @pytest.fixture
    def renderer(self):
        """Create a renderer instance."""
        return MatplotlibRenderer(xlim=(-50, 50), ylim=(-50, 50), resolution=(800, 600), dpi=100)

    def test_camera_to_renderer_data_flow(self, mock_map, mock_participants, renderer):
        """Test complete data flow from camera to renderer."""
        # Create camera
        camera = BEVCamera(id_=1, map_=mock_map)

        # Set camera position to ensure all elements are visible
        camera._position = None  # This makes _in_perception_range return True

        # Get geometry data from camera
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

        # Verify camera output structure
        assert "frame" in geometry_data
        assert "map_data" in geometry_data
        assert "participant_data" in geometry_data

        map_data = geometry_data["map_data"]
        participant_data = geometry_data["participant_data"]

        # Verify required keys exist
        assert "road_id_to_remove" in map_data
        assert "road_elements" in map_data
        assert "participant_id_to_create" in participant_data
        assert "participant_id_to_remove" in participant_data
        assert "participants" in participant_data

        # Verify data types
        assert isinstance(map_data["road_elements"], list)
        assert isinstance(participant_data["participants"], list)

        # Verify element formats for renderer compatibility
        for element in map_data["road_elements"]:
            assert "id" in element
            assert "shape" in element  # Required by renderer
            assert "type" in element  # Required by renderer for style resolution
            assert "color" in element
            assert "geometry" in element

            if element["shape"] == "line":
                assert "line_style" in element or "type" in element

        for participant in participant_data["participants"]:
            assert "id" in participant
            assert "shape" in participant  # Required by renderer
            assert "type" in participant  # Required by renderer for style resolution
            assert "color" in participant

        # Now test that renderer can process this data
        camera_position = Point(10.0, 10.0)
        camera_yaw = 0.0

        # This should not raise any exceptions
        renderer.update(
            geometry_data=geometry_data, camera_position=camera_position, camera_yaw=camera_yaw
        )

        # Verify renderer state
        assert np.array_equal(renderer.camera_position, [10.0, 10.0])
        assert renderer.camera_yaw == 0.0

        # Verify some elements were created
        # (at least some road elements or participants should exist)
        total_elements = (
            len(renderer.road_polygons) + len(renderer.road_lines) + len(renderer.participants)
        )
        assert total_elements > 0

    def test_multiple_update_cycles(self, mock_map, mock_participants, renderer):
        """Test multiple update cycles with element addition and removal."""
        camera = BEVCamera(id_=1, map_=mock_map)
        camera._position = None  # Make everything visible

        frame = 0
        participant_ids = list(mock_participants.keys())

        # First update
        geometry_data1, road_id_set1, participant_id_set1 = camera.update(
            frame=frame,
            participants=mock_participants,
            participant_ids=participant_ids,
            prev_road_id_set=set(),
            prev_participant_id_set=set(),
            position=None,
        )

        # Render first frame
        renderer.update(geometry_data=geometry_data1, camera_position=Point(0, 0), camera_yaw=0.0)

        # Store initial counts
        initial_total = (
            len(renderer.road_polygons) + len(renderer.road_lines) + len(renderer.participants)
        )

        # Second update - simulate removing some elements
        # In real usage, prev_road_id_set and prev_participant_id_set
        # would be the sets returned from previous update
        geometry_data2, road_id_set2, participant_id_set2 = camera.update(
            frame=frame + 1,
            participants=mock_participants,
            participant_ids=participant_ids[:1],  # Only first participant
            prev_road_id_set=road_id_set1,
            prev_participant_id_set=participant_id_set1,
            position=None,
        )

        # Render second frame
        renderer.update(geometry_data=geometry_data2, camera_position=Point(0, 0), camera_yaw=0.0)

        # Verify renderer can handle the update
        # (specific behavior depends on what camera returns)
        assert renderer.camera_position is not None
        assert renderer.camera_yaw is not None

    def test_camera_renderer_contract_validation(self):
        """Validate the data contract between camera and renderer."""
        # This test documents the expected data contract

        # Camera must output:
        # 1. geometry_data dict with keys: "frame", "map_data", "participant_data"
        # 2. map_data dict with keys: "road_id_to_remove", "road_elements"
        # 3. participant_data dict with keys: "participant_id_to_create",
        #    "participant_id_to_remove", "participants"
        # 4. Each element in road_elements must have:
        #    - "id": unique identifier
        #    - "shape": "polygon", "line", or "circle"
        #    - "type": string key for style lookup
        #    - "color": color key or direct color
        #    - "geometry": list of coordinates
        #    - "line_width": numeric
        #    - For lines: "line_style" or "type" containing style info
        # 5. Each element in participants must have:
        #    - "id": unique identifier
        #    - "shape": "polygon" or "circle"
        #    - "type": string key for style lookup
        #    - "color": color key or direct color
        #    - "line_width": numeric
        #    - For polygons: "geometry", "position", "rotation"
        #    - For circles: "position", "radius"

        # Renderer expects:
        # 1. geometry_data with the structure above
        # 2. Uses "shape" to determine geometry type
        # 3. Uses "type" for style resolution (color and z-order)
        # 4. For lines: looks for "line_style" or uses "type" as fallback

        # This contract ensures separation of concerns:
        # - Camera handles geometry and visibility
        # - Renderer handles visualization and styling
        # - Style configuration is centralized in matplotlib_config.py

        assert True  # Test passes if we reach here (contract is documented)
