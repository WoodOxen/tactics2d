# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for MatplotlibRenderer."""


import sys

sys.path.append(".")
sys.path.append("..")

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import matplotlib.colors
import numpy as np
import pytest
from shapely.geometry import Point

from tactics2d.renderer.matplotlib_config import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER
from tactics2d.renderer.matplotlib_renderer import MatplotlibRenderer


class TestMatplotlibRenderer:
    """Test suite for MatplotlibRenderer class."""

    @pytest.fixture
    def sample_geometry_data(self):
        """Create sample geometry data for testing."""
        return {
            "map_data": {
                "road_id_to_remove": [],
                "road_elements": [
                    {
                        "id": 1000001,
                        "shape": "polygon",
                        "geometry": [(0, 0), (10, 0), (10, 10), (0, 10)],
                        "color": "area",
                        "type": "area",
                        "line_width": 0,
                    },
                    {
                        "id": 1000002,
                        "shape": "line",
                        "geometry": [(0, 0), (10, 0)],
                        "color": "roadline",
                        "type": "roadline",
                        "line_style": "solid",
                        "line_width": 1.0,
                    },
                ],
            },
            "participant_data": {
                "participant_id_to_create": [101],
                "participant_id_to_remove": [],
                "participants": [
                    {
                        "id": 101,
                        "shape": "polygon",
                        "geometry": [(0, 0), (2, 0), (2, 1), (0, 1)],
                        "position": [5.0, 5.0],
                        "rotation": 0.0,
                        "color": "vehicle",
                        "type": "vehicle",
                        "line_width": 1,
                    },
                    {
                        "id": 102,
                        "shape": "circle",
                        "position": [3.0, 3.0],
                        "radius": 0.5,
                        "color": "pedestrian",
                        "type": "pedestrian",
                        "line_width": 1,
                    },
                ],
            },
        }

    @pytest.fixture
    def renderer(self):
        """Create a renderer instance for testing."""
        return MatplotlibRenderer(xlim=(-50, 50), ylim=(-50, 50), resolution=(800, 600), dpi=100)

    def test_initialization(self):
        """Test renderer initialization with different parameters."""
        # Test basic initialization
        renderer = MatplotlibRenderer(
            xlim=(-100, 100), ylim=(-50, 50), resolution=(1024, 768), dpi=200
        )

        assert renderer.xlim == (-100, 100)
        assert renderer.ylim == (-50, 50)
        assert renderer.resolution == (1024, 768)
        assert renderer.dpi == 200
        assert renderer.width > 0
        assert renderer.height > 0
        assert renderer.camera_position is None
        assert renderer.camera_yaw is None

        # Test with small resolution
        renderer = MatplotlibRenderer(xlim=(-10, 10), ylim=(-5, 5), resolution=(100, 100), dpi=50)

        assert renderer.width >= 1.0
        assert renderer.height >= 1.0

    def test_create_polygon(self, renderer):
        """Test _create_polygon method."""
        element = {
            "id": 1,
            "geometry": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "color": "area",
            "type": "area",
            "line_width": 0.5,
        }

        polygon = renderer._create_polygon(element)
        assert polygon is not None
        assert (
            matplotlib.colors.to_hex(polygon.get_facecolor())
            == COLOR_PALETTE[DEFAULT_COLOR["area"]]
        )
        assert polygon.get_linewidth() == 0.5
        assert polygon.get_zorder() == DEFAULT_ORDER["area"]

        # Test with insufficient points
        element_invalid = {
            "id": 2,
            "geometry": [(0, 0), (10, 0)],  # Only 2 points
            "color": "area",
            "type": "area",
            "line_width": 0.5,
        }

        polygon_invalid = renderer._create_polygon(element_invalid)
        assert polygon_invalid is None

    def test_create_circle(self, renderer):
        """Test _create_circle method."""
        element = {"radius": 5.0, "color": "pedestrian", "type": "pedestrian", "line_width": 1.0}

        circle = renderer._create_circle(element)
        assert circle is not None
        assert (
            matplotlib.colors.to_hex(circle.get_facecolor())
            == COLOR_PALETTE[DEFAULT_COLOR["pedestrian"]]
        )
        assert circle.get_linewidth() == 1.0
        assert circle.get_zorder() == DEFAULT_ORDER["pedestrian"]
        assert circle.get_radius() == 5.0

    def test_create_line(self, renderer):
        """Test _create_line method."""
        # Test solid line
        element_solid = {
            "geometry": [(0, 0), (10, 0)],
            "color": "roadline",
            "type": "roadline",
            "line_style": "solid",
            "line_width": 1.0,
        }

        lines = renderer._create_line(element_solid)
        assert len(lines) == 1
        line = lines[0]
        assert line.get_color() == COLOR_PALETTE[DEFAULT_COLOR["roadline"]]
        assert line.get_linewidth() == 1.0
        assert line.get_zorder() == DEFAULT_ORDER["roadline"]

        # Test dashed line
        element_dashed = {
            "geometry": [(0, 0), (10, 0)],
            "color": "roadline",
            "type": "roadline",
            "line_style": "dashed",
            "line_width": 0.5,
        }

        lines = renderer._create_line(element_dashed)
        assert len(lines) == 1
        line = lines[0]
        # Matplotlib may return either the tuple format or string representation
        linestyle = line.get_linestyle()
        assert linestyle == (0, (5, 5)) or linestyle == "--"  # Default dash pattern

        # Test with line_style in type field as fallback
        element_fallback = {
            "geometry": [(0, 0), (10, 0)],
            "color": "roadline",
            "type": "dashed",
            "line_width": 1.0,
        }

        lines = renderer._create_line(element_fallback)
        assert len(lines) == 1

    def test_transform_to_camera_view(self, renderer):
        """Test _transform_to_camera_view method."""
        # Set camera position and yaw
        renderer.camera_position = np.array([5.0, 5.0])
        renderer.camera_yaw = np.pi / 2  # 90 degrees

        points = np.array([[0, 0], [10, 0], [10, 10]])

        transformed = renderer._transform_to_camera_view(points)

        # Check shape preservation
        assert transformed.shape == points.shape

        # Check translation and rotation
        # With camera at (5,5) and yaw=90°, point (0,0) should become (5,5) after translation
        # and then rotated -90° (clockwise 90°)
        expected = points - np.array([5.0, 5.0])  # Translate
        # Rotate by -90° (clockwise 90°)
        cos_theta = np.cos(-np.pi / 2)
        sin_theta = np.sin(-np.pi / 2)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        expected = expected @ rotation_matrix

        np.testing.assert_array_almost_equal(transformed, expected)

        # Test with camera not set
        renderer.camera_position = None
        renderer.camera_yaw = None

        with pytest.raises(RuntimeError, match="Camera position and yaw must be set"):
            renderer._transform_to_camera_view(points)

    def test_update_method(self, renderer, sample_geometry_data):
        """Test update method with valid geometry data."""
        # Set camera position
        camera_position = Point(0, 0)
        camera_yaw = 0.0

        renderer.update(
            geometry_data=sample_geometry_data,
            camera_position=camera_position,
            camera_yaw=camera_yaw,
        )

        # Check camera state
        assert np.array_equal(renderer.camera_position, [0.0, 0.0])
        assert renderer.camera_yaw == 0.0

        # Check that elements were added
        assert len(renderer.road_polygons) > 0 or len(renderer.road_lines) > 0
        assert len(renderer.participants) > 0

    def test_update_method_exceptions(self, renderer):
        """Test update method with invalid data."""
        # Test missing map_data
        invalid_data = {
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
            }
        }

        with pytest.raises(KeyError, match="must contain 'map_data'"):
            renderer.update(geometry_data=invalid_data, camera_position=[0, 0], camera_yaw=0.0)

        # Test missing participant_data
        invalid_data = {"map_data": {"road_id_to_remove": [], "road_elements": []}}

        with pytest.raises(KeyError, match="must contain 'participant_data'"):
            renderer.update(geometry_data=invalid_data, camera_position=[0, 0], camera_yaw=0.0)

        # Test invalid camera position dimension
        # Create simple geometry data for this test
        simple_geometry_data = {
            "map_data": {"road_id_to_remove": [], "road_elements": []},
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
            },
        }

        with pytest.raises(ValueError, match="must be 2D"):
            renderer.update(
                geometry_data=simple_geometry_data,
                camera_position=[0, 0, 0],  # 3D position
                camera_yaw=0.0,
            )

    def test_update_with_removal(self, renderer, sample_geometry_data):
        """Test update method with element removal."""
        # First update to create elements
        renderer.update(geometry_data=sample_geometry_data, camera_position=[0, 0], camera_yaw=0.0)

        # Store initial counts
        initial_polygon_count = len(renderer.road_polygons)
        initial_line_count = len(renderer.road_lines)
        initial_participant_count = len(renderer.participants)

        # Create update data with removal
        removal_data = {
            "map_data": {
                "road_id_to_remove": [1000001, 1000002],  # Remove all road elements
                "road_elements": [],  # No new elements
            },
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [101, 102],  # Remove all participants
                "participants": [],
            },
        }

        # Second update to remove elements
        renderer.update(geometry_data=removal_data, camera_position=[0, 0], camera_yaw=0.0)

        # Check that elements were removed
        assert len(renderer.road_polygons) == 0
        assert len(renderer.road_lines) == 0
        assert len(renderer.participants) == 0

    def test_save_single_frame(self, renderer, sample_geometry_data):
        """Test save_single_frame method."""
        # First update to create some content
        renderer.update(geometry_data=sample_geometry_data, camera_position=[0, 0], camera_yaw=0.0)

        # Test saving to file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save without returning array
            result = renderer.save_single_frame(save_to=tmp_path, dpi=150)
            assert result is None
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            # Save with returning array
            array_result = renderer.save_single_frame(return_array=True)
            assert isinstance(array_result, np.ndarray)
            assert len(array_result.shape) == 3  # Height, width, channels
            assert array_result.shape[2] == 3  # RGB

            # Save with default dpi
            result = renderer.save_single_frame(save_to=tmp_path)
            assert result is None

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_reset_method(self, renderer, sample_geometry_data):
        """Test reset method."""
        # First update to create some content
        renderer.update(geometry_data=sample_geometry_data, camera_position=[0, 0], camera_yaw=0.0)

        # Verify content exists
        assert len(renderer.road_polygons) > 0 or len(renderer.road_lines) > 0
        assert len(renderer.participants) > 0
        assert renderer.camera_position is not None
        assert renderer.camera_yaw is not None

        # Reset
        renderer.reset()

        # Verify everything is cleared
        assert len(renderer.road_polygons) == 0
        assert len(renderer.road_lines) == 0
        assert len(renderer.participants) == 0
        assert renderer.camera_position is None
        assert renderer.camera_yaw is None

        # Verify axes limits are preserved
        assert renderer.ax.get_xlim() == (-50, 50)
        assert renderer.ax.get_ylim() == (-50, 50)

    def test_destroy_method(self, renderer, sample_geometry_data):
        """Test destroy method."""
        # Create some content
        renderer.update(geometry_data=sample_geometry_data, camera_position=[0, 0], camera_yaw=0.0)

        # Destroy
        renderer.destroy()

        # Verify references are cleared
        assert renderer.fig is None
        assert renderer.ax is None
        assert renderer.camera_position is None
        assert renderer.camera_yaw is None
        assert len(renderer.road_polygons) == 0
        assert len(renderer.road_lines) == 0
        assert len(renderer.participants) == 0

    def test_update_polygon_method(self, renderer):
        """Test _update_polygon method."""
        # Create a mock polygon
        mock_polygon = Mock()
        renderer.camera_position = np.array([0, 0])
        renderer.camera_yaw = 0.0

        geometry = [(0, 0), (10, 0), (10, 10)]
        position = [5.0, 5.0]
        rotation = np.pi / 4  # 45 degrees

        renderer._update_polygon(mock_polygon, geometry, position, rotation)

        # Verify set_xy was called
        assert mock_polygon.set_xy.called

    def test_update_circle_method(self, renderer):
        """Test _update_circle method."""
        # Create a mock circle
        mock_circle = Mock()
        renderer.camera_position = np.array([0, 0])
        renderer.camera_yaw = 0.0

        position = [5.0, 5.0]

        renderer._update_circle(mock_circle, position)

        # Verify set_center was called
        assert mock_circle.set_center.called

    def test_update_line_method(self, renderer):
        """Test _update_line method."""
        # Create mock lines
        mock_line = Mock()
        lines = [mock_line]
        renderer.camera_position = np.array([0, 0])
        renderer.camera_yaw = 0.0

        geometry = [(0, 0), (10, 0)]
        position = [5.0, 5.0]
        rotation = np.pi / 2  # 90 degrees

        renderer._update_line(lines, geometry, position, rotation)

        # Verify set_data was called
        assert mock_line.set_data.called

    def test_edge_cases(self, renderer):
        """Test various edge cases."""
        # Test update with empty geometry data
        empty_data = {
            "map_data": {"road_id_to_remove": [], "road_elements": []},
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
            },
        }

        # Should not raise any exceptions
        renderer.update(geometry_data=empty_data, camera_position=[0, 0], camera_yaw=0.0)

        # Test with invalid shape type in element
        invalid_shape_data = {
            "map_data": {
                "road_id_to_remove": [],
                "road_elements": [
                    {
                        "id": 1,
                        "shape": "invalid_shape",  # Invalid shape type
                        "geometry": [(0, 0), (10, 0)],
                        "color": "area",
                        "type": "area",
                        "line_width": 0,
                    }
                ],
            },
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
            },
        }

        # Should skip invalid shape without crashing
        renderer.update(geometry_data=invalid_shape_data, camera_position=[0, 0], camera_yaw=0.0)
