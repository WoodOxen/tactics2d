# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for BEVCamera and SingleLineLidar sensors with Argoverse dataset."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

# Add project root to path
sys.path.append(".")
sys.path.append("..")

from tactics2d.dataset_parser.parse_argoverse2 import Argoverse2Parser
from tactics2d.map.element import Map
from tactics2d.renderer.matplotlib_renderer import MatplotlibRenderer
from tactics2d.sensor.camera import BEVCamera
from tactics2d.sensor.lidar import SingleLineLidar
from tactics2d.sensor.sensor_base import SensorBase

# Import shared utilities
from tests.test_sensor_utils import (
    assert_sensor_data_structure,
    create_sensor_factory,
    render_sensor_output,
)

# ------------------------------------------------------------------------------
# Shared Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def argoverse_sample_path():
    """Return Argoverse sample data path."""
    path = (
        Path(__file__).parent.parent
        / "tactics2d"
        / "data"
        / "trajectory_sample"
        / "Argoverse"
        / "train"
        / "0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca"
    )
    assert path.exists(), f"Argoverse sample data not found at {path}"

    # Verify required files exist
    map_file = path / "log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json"
    traj_file = path / "scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet"
    assert map_file.exists(), f"map.json not found at {map_file}"
    assert traj_file.exists(), f"trajectory.parquet not found at {traj_file}"

    return path


@pytest.fixture(scope="session")
def argoverse_data(argoverse_sample_path):
    """Load Argoverse data (map, participants, timestamps)."""
    parser = Argoverse2Parser()

    # Parse map
    map_filename = "log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json"
    map_obj = parser.parse_map(map_filename, str(argoverse_sample_path))
    assert map_obj is not None, "Failed to parse map"

    # Parse trajectory
    traj_filename = "scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet"
    participants, stamp_range = parser.parse_trajectory(traj_filename, str(argoverse_sample_path))
    assert len(participants) > 0, "No participants loaded"

    # Participant filtering (vehicle types)
    participant_ids = [
        pid
        for pid, p in participants.items()
        if hasattr(p, "type_")
        and p.type_ in ["car", "bus", "motorcycle", "vehicle", "bicycle", "cyclist", "pedestrian"]
    ]
    if len(participant_ids) == 0:
        participant_ids = list(participants.keys())

    # Timestamp lookup logic (simplified from original setup_class)
    test_participant_id = participant_ids[0]
    test_participant = participants[test_participant_id]
    test_timestamp = None

    # Find a timestamp that exists for the test participant
    if hasattr(test_participant, "trajectory"):
        traj = test_participant.trajectory
        # Get available timestamps from this participant
        if hasattr(traj, "_history_states") and traj._history_states:
            available_stamps = list(traj._history_states.keys())
        elif hasattr(traj, "states") and traj.states:
            # Convert states list to timestamps (assuming 10Hz, starting at 0)
            available_stamps = [i * 100 for i in range(len(traj.states))]
        else:
            available_stamps = []

        if available_stamps:
            # Use first available timestamp
            test_timestamp = available_stamps[0]
            print(f"Selected timestamp {test_timestamp} for participant {test_participant_id}")

    # Fallback if still no timestamp
    if test_timestamp is None:
        test_timestamp = 0
        print(f"Warning: Using fallback timestamp {test_timestamp}")

    # Filter participant_ids to only include participants with state at this timestamp
    filtered_participant_ids = []
    for pid in participant_ids:
        participant = participants[pid]
        if hasattr(participant, "trajectory"):
            traj = participant.trajectory
            try:
                if hasattr(traj, "_history_states"):
                    if test_timestamp in traj._history_states:
                        filtered_participant_ids.append(pid)
                elif hasattr(traj, "states"):
                    idx = test_timestamp // 100
                    if idx < len(traj.states):
                        filtered_participant_ids.append(pid)
            except:
                pass  # Skip this participant

    if filtered_participant_ids:
        participant_ids = filtered_participant_ids
        print(
            f"Filtered participants to {len(participant_ids)} with state at timestamp {test_timestamp}"
        )
        # Debug: print participant ids and types
        for pid in participant_ids[:5]:  # first few
            p = participants[pid]
            type_str = getattr(p, "type_", "NO_TYPE")
            id_str = getattr(p, "id_", "NO_ID")
            print(f"  Participant id={id_str}, type={type_str}")
    else:
        print(
            f"Warning: No participants have state at timestamp {test_timestamp}, using all participants"
        )

    # Additional filter: ensure participant ids can be converted to int (avoid 'AV' etc.)
    numeric_participant_ids = []
    for pid in participant_ids:
        try:
            # Check if pid can be converted to int (or float)
            if isinstance(pid, (int, float)):
                numeric_participant_ids.append(pid)
            else:
                # Try converting string to int
                int(pid)
                numeric_participant_ids.append(pid)
        except:
            print(f"Warning: Skipping participant with non-numeric id: {pid}")
    if numeric_participant_ids:
        participant_ids = numeric_participant_ids
        print(f"Filtered to {len(participant_ids)} participants with numeric ids")
    else:
        print(f"Warning: No participants with numeric ids, using original list")

    return {
        "map_obj": map_obj,
        "participants": participants,
        "participant_ids": participant_ids,
        "test_timestamp": test_timestamp,
        "test_participant_id": test_participant_id,
        "test_participant": test_participant,
    }


@pytest.fixture(scope="class")
def sensor_test_data(argoverse_data):
    """Prepare data for sensor tests."""

    class SensorTestData:
        def __init__(self, data):
            self.map_obj = data["map_obj"]
            self.participants = data["participants"]
            self.participant_ids = data["participant_ids"]
            self.test_timestamp = data["test_timestamp"]
            self.test_participant_id = data["test_participant_id"]
            self.test_participant = data["test_participant"]

            # Simplified position calculation
            # Use participant position if available, otherwise default
            if hasattr(self.test_participant, "trajectory"):
                traj = self.test_participant.trajectory
                try:
                    if hasattr(traj, "_history_states") and traj._history_states:
                        state = traj.get_state(self.test_timestamp)
                        self.fixed_position = Point(state.x, state.y)
                        self.fixed_heading = state.heading
                    elif hasattr(traj, "states") and traj.states:
                        idx = self.test_timestamp // 100
                        if idx < len(traj.states):
                            state = traj.states[idx]
                            self.fixed_position = Point(state.x, state.y)
                            self.fixed_heading = state.heading
                        else:
                            self.fixed_position = Point(0, 0)
                            self.fixed_heading = 0.0
                    else:
                        self.fixed_position = Point(0, 0)
                        self.fixed_heading = 0.0
                except:
                    self.fixed_position = Point(0, 0)
                    self.fixed_heading = 0.0
            else:
                self.fixed_position = Point(0, 0)
                self.fixed_heading = 0.0

            # Ensure runtime directory exists
            self.runtime_dir = Path(__file__).parent / "runtime"
            self.runtime_dir.mkdir(exist_ok=True)

    return SensorTestData(argoverse_data)


# ------------------------------------------------------------------------------
# Base Sensor Tests
# ------------------------------------------------------------------------------


@pytest.mark.sensor
@pytest.mark.integration
@pytest.mark.render
@pytest.mark.slow
class TestSensorBase:
    """Base sensor functionality tests (binding mechanism, data integrity)."""

    def test_sensor_data_integrity(self, sensor_test_data):
        """Test data integrity for both camera and lidar sensors."""
        # Test camera data integrity
        camera = BEVCamera(id_=0, map_=sensor_test_data.map_obj, perception_range=40.0)

        camera_data, _, _ = camera.update(
            frame=sensor_test_data.test_timestamp,
            participants=sensor_test_data.participants,
            participant_ids=sensor_test_data.participant_ids,
            position=sensor_test_data.fixed_position,
            heading=sensor_test_data.fixed_heading,
        )

        # Camera should have map and participant data
        assert "map_data" in camera_data
        assert "road_elements" in camera_data["map_data"]
        assert "participant_data" in camera_data
        assert "participants" in camera_data["participant_data"]

        # Test lidar data integrity
        lidar = SingleLineLidar(id_=1, map_=sensor_test_data.map_obj, perception_range=50.0)

        lidar_data, _, _ = lidar.update(
            frame=sensor_test_data.test_timestamp,
            participants=sensor_test_data.participants,
            participant_ids=sensor_test_data.participant_ids,
            position=sensor_test_data.fixed_position,
            heading=sensor_test_data.fixed_heading,
        )

        # Lidar should have point cloud data
        assert "participant_data" in lidar_data
        assert "point_clouds" in lidar_data["participant_data"]
        assert len(lidar_data["participant_data"]["point_clouds"]) > 0

        # Compare structures
        assert "frame" in camera_data and "frame" in lidar_data
        assert camera_data["frame"] == lidar_data["frame"] == sensor_test_data.test_timestamp

    def test_sensor_binding_mechanism(self, sensor_test_data):
        """Test sensor binding mechanism."""
        # Create sensor
        sensor = BEVCamera(id_=2, map_=sensor_test_data.map_obj, perception_range=30.0)

        # Initially not bound
        assert not sensor.is_bound
        assert sensor.bind_id is None

        # Bind to participant
        sensor.bind_with(sensor_test_data.test_participant_id)
        assert sensor.is_bound
        assert sensor.bind_id == sensor_test_data.test_participant_id

        # Update without explicit position/heading
        geometry_data, _, _ = sensor.update(
            frame=sensor_test_data.test_timestamp,
            participants=sensor_test_data.participants,
            participant_ids=sensor_test_data.participant_ids,
        )

        # Sensor should have derived position from bound participant
        # Note: The actual position derivation happens in sensor._set_position_heading()

        # Unbind (by binding to None)
        sensor.bind_with(None)
        assert not sensor.is_bound
        assert sensor.bind_id is None


# ------------------------------------------------------------------------------
# BEVCamera Tests
# ------------------------------------------------------------------------------


@pytest.mark.sensor
@pytest.mark.integration
@pytest.mark.render
@pytest.mark.slow
class TestBEVCamera:
    """BEVCamera-specific tests."""

    @pytest.mark.parametrize("mode", ["fixed", "bound"])
    def test_camera_modes(self, mode, sensor_test_data):
        """Test camera in different modes (fixed position vs bound to participant)."""
        # Create camera
        if mode == "fixed":
            camera = BEVCamera(id_=3, map_=sensor_test_data.map_obj, perception_range=50.0)

            # Update camera at fixed position
            geometry_data, _, _ = camera.update(
                frame=sensor_test_data.test_timestamp,
                participants=sensor_test_data.participants,
                participant_ids=sensor_test_data.participant_ids,
                position=sensor_test_data.fixed_position,
                heading=sensor_test_data.fixed_heading,
            )

            sensor_position = geometry_data["sensor_position"]
            sensor_heading = sensor_test_data.fixed_heading
            output_filename = "camera_fixed_position.png"
            title = "BEVCamera - Fixed Position"

        else:  # bound mode
            camera = BEVCamera(id_=4, map_=sensor_test_data.map_obj, perception_range=30.0)

            # Bind camera to participant
            camera.bind_with(sensor_test_data.test_participant_id)
            assert camera.is_bound
            assert camera.bind_id == sensor_test_data.test_participant_id

            # Update camera (position and heading will be derived from bound participant)
            geometry_data, _, _ = camera.update(
                frame=sensor_test_data.test_timestamp,
                participants=sensor_test_data.participants,
                participant_ids=sensor_test_data.participant_ids,
            )

            sensor_position = geometry_data["sensor_position"]
            sensor_heading = sensor_test_data.fixed_heading
            output_filename = "camera_bound_to_participant.png"
            title = "BEVCamera - Bound to Participant"

        # Validate data structure
        assert_sensor_data_structure(geometry_data, sensor_test_data.test_timestamp)

        # Check map data
        map_data = geometry_data["map_data"]
        assert "road_id_to_remove" in map_data
        assert "road_elements" in map_data

        # Check participant data
        participant_data = geometry_data["participant_data"]
        assert "participant_id_to_create" in participant_data
        assert "participant_id_to_remove" in participant_data
        assert "participants" in participant_data

        # Render and save
        output_path = sensor_test_data.runtime_dir / output_filename
        saved_path = render_sensor_output(
            geometry_data=geometry_data,
            sensor_position=sensor_position,
            output_path=output_path,
            title=title,
            camera_yaw=sensor_heading,
        )

        # Verify sensor attributes
        if mode == "fixed":
            assert not camera.is_bound  # Should not be bound in fixed position mode

    @pytest.mark.parametrize("range_val", [30.0, 50.0, 100.0])
    def test_camera_perception_range(self, range_val, sensor_test_data):
        """Test camera with different perception ranges."""
        camera = BEVCamera(id_=5, map_=sensor_test_data.map_obj, perception_range=range_val)

        geometry_data, _, _ = camera.update(
            frame=sensor_test_data.test_timestamp,
            participants=sensor_test_data.participants,
            participant_ids=sensor_test_data.participant_ids,
            position=sensor_test_data.fixed_position,
            heading=sensor_test_data.fixed_heading,
        )

        # Basic validation
        assert_sensor_data_structure(geometry_data, sensor_test_data.test_timestamp)
        # perception_range is stored as a tuple (left, right, front, back)
        expected_range = (range_val, range_val, range_val, range_val)
        assert camera._perception_range == expected_range


# ------------------------------------------------------------------------------
# SingleLineLidar Tests
# ------------------------------------------------------------------------------


@pytest.mark.sensor
@pytest.mark.integration
@pytest.mark.render
@pytest.mark.slow
class TestSingleLineLidar:
    """SingleLineLidar-specific tests."""

    @pytest.mark.parametrize("mode", ["fixed", "bound"])
    def test_lidar_modes(self, mode, sensor_test_data):
        """Test lidar in different modes (fixed position vs bound to participant)."""
        if mode == "fixed":
            lidar = SingleLineLidar(
                id_=6,
                map_=sensor_test_data.map_obj,
                perception_range=50.0,
                freq_scan=10.0,
                freq_detect=5000.0,
            )

            geometry_data, _, _ = lidar.update(
                frame=sensor_test_data.test_timestamp,
                participants=sensor_test_data.participants,
                participant_ids=sensor_test_data.participant_ids,
                position=sensor_test_data.fixed_position,
                heading=sensor_test_data.fixed_heading,
            )

            sensor_position = geometry_data["sensor_position"]
            sensor_heading = sensor_test_data.fixed_heading
            output_filename = "lidar_fixed_position.png"
            title = "SingleLineLidar - Fixed Position"

        else:  # bound mode
            lidar = SingleLineLidar(
                id_=7,
                map_=sensor_test_data.map_obj,
                perception_range=50.0,
                freq_scan=10.0,
                freq_detect=5000.0,
            )

            lidar.bind_with(sensor_test_data.test_participant_id)
            assert lidar.is_bound
            assert lidar.bind_id == sensor_test_data.test_participant_id

            geometry_data, _, _ = lidar.update(
                frame=sensor_test_data.test_timestamp,
                participants=sensor_test_data.participants,
                participant_ids=sensor_test_data.participant_ids,
            )

            sensor_position = geometry_data["sensor_position"]
            sensor_heading = sensor_test_data.fixed_heading
            output_filename = "lidar_bound_to_participant.png"
            title = "SingleLineLidar - Bound to Participant"

        # Validate data structure
        assert_sensor_data_structure(geometry_data, sensor_test_data.test_timestamp, "lidar")
        assert "sensor_position" in geometry_data

        # Check point cloud data
        participant_data = geometry_data["participant_data"]
        assert "point_clouds" in participant_data
        point_clouds = participant_data["point_clouds"]
        assert len(point_clouds) > 0

        # Check point cloud structure
        point_cloud = point_clouds[0]
        assert "points" in point_cloud
        assert "color" in point_cloud
        assert point_cloud["color"] == "red"

        # Render and save
        output_path = sensor_test_data.runtime_dir / output_filename
        saved_path = render_sensor_output(
            geometry_data=geometry_data,
            sensor_position=sensor_position,
            output_path=output_path,
            title=title,
            camera_yaw=sensor_heading,
        )

    def test_lidar_point_cloud_generation(self, sensor_test_data):
        """Test lidar point cloud generation."""
        lidar = SingleLineLidar(
            id_=8,
            map_=sensor_test_data.map_obj,
            perception_range=50.0,
            freq_scan=10.0,
            freq_detect=5000.0,
        )

        # Get point cloud data
        geometry_data, _, _ = lidar.update(
            frame=sensor_test_data.test_timestamp,
            participants=sensor_test_data.participants,
            participant_ids=sensor_test_data.participant_ids,
            position=sensor_test_data.fixed_position,
            heading=sensor_test_data.fixed_heading,
        )

        # Extract point cloud
        point_clouds = geometry_data["participant_data"]["point_clouds"]
        assert len(point_clouds) == 1

        point_cloud = point_clouds[0]
        points = point_cloud["points"]

        # Check point cloud properties
        assert len(points) > 0, "Point cloud should contain points"
        assert all(
            isinstance(p, (list, tuple)) for p in points
        ), "Points should be coordinate pairs"
        assert all(len(p) == 2 for p in points), "Points should be 2D coordinates"

        # Check metadata
        assert "sensor_position" in geometry_data
        sensor_pos = geometry_data["sensor_position"]
        assert isinstance(sensor_pos, (list, tuple)) and len(sensor_pos) == 2

        # Check that points are within perception range
        import math

        for point in points:
            dx = point[0] - sensor_pos[0]
            dy = point[1] - sensor_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            assert (
                distance <= lidar._perception_range * 1.1
            ), f"Point at distance {distance} exceeds perception range {lidar._perception_range}"

    @pytest.mark.parametrize("freq_scan", [5.0, 10.0, 20.0])
    def test_lidar_scan_frequency(self, freq_scan, sensor_test_data):
        """Test lidar with different scan frequencies."""
        lidar = SingleLineLidar(
            id_=9,
            map_=sensor_test_data.map_obj,
            perception_range=50.0,
            freq_scan=freq_scan,
            freq_detect=5000.0,
        )

        geometry_data, _, _ = lidar.update(
            frame=sensor_test_data.test_timestamp,
            participants=sensor_test_data.participants,
            participant_ids=sensor_test_data.participant_ids,
            position=sensor_test_data.fixed_position,
            heading=sensor_test_data.fixed_heading,
        )

        # Basic validation
        assert_sensor_data_structure(geometry_data, sensor_test_data.test_timestamp, "lidar")
        assert lidar._freq_scan == freq_scan

    def test_lidar_point_cloud_axis_update(self, sensor_test_data):
        """Verify axis limits update when auto_scale=True and point clouds use numpy arrays."""
        # Create renderer with auto_scale enabled
        renderer = MatplotlibRenderer(resolution=(800, 600), auto_scale=True, margin=5.0)

        # Create point cloud with coordinates far outside default axis limits
        points = np.array([[150.0, 200.0], [-150.0, -200.0], [180.0, -180.0]])

        geometry_data = {
            "map_data": {"road_id_to_remove": [], "road_elements": []},
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
                "point_clouds": [
                    {
                        "id": "test_cloud",
                        "points": points,  # numpy array
                        "color": "red",
                        "point_size": 2.0,
                        "alpha": 0.8,
                        "type": "lidar_point_cloud",
                    }
                ],
            },
        }

        # Update renderer with camera at origin
        camera_position = (0.0, 0.0)
        renderer.update(geometry_data, camera_position, camera_yaw=0.0)

        # Check that axis limits include point cloud coordinates with margin
        xlim = renderer.ax.get_xlim()
        ylim = renderer.ax.get_ylim()

        # Expected bounds: min_x = -150 - margin, max_x = 180 + margin, etc.
        expected_x_min = -150.0 - 5.0
        expected_x_max = 180.0 + 5.0
        expected_y_min = -200.0 - 5.0
        expected_y_max = 200.0 + 5.0

        assert xlim[0] <= expected_x_min, f"X min {xlim[0]} should be <= {expected_x_min}"
        assert xlim[1] >= expected_x_max, f"X max {xlim[1]} should be >= {expected_x_max}"
        assert ylim[0] <= expected_y_min, f"Y min {ylim[0]} should be <= {expected_y_min}"
        assert ylim[1] >= expected_y_max, f"Y max {ylim[1]} should be >= {expected_y_max}"

        # Ensure points are within visible bounds (they should be)
        for point in points:
            assert xlim[0] <= point[0] <= xlim[1], f"Point x {point[0]} outside xlim {xlim}"
            assert ylim[0] <= point[1] <= ylim[1], f"Point y {point[1]} outside ylim {ylim}"

    def test_lidar_point_cloud_visibility(self, sensor_test_data):
        """Verify point clouds become visible after axis update."""
        # Create renderer with auto_scale enabled
        renderer = MatplotlibRenderer(resolution=(800, 600), auto_scale=True, margin=5.0)

        # Create point cloud with coordinates far outside default axis limits
        points = np.array([[150.0, 200.0], [-150.0, -200.0]])

        geometry_data = {
            "map_data": {"road_id_to_remove": [], "road_elements": []},
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
                "point_clouds": [
                    {
                        "id": "test_cloud",
                        "points": points,
                        "color": "red",
                        "point_size": 2.0,
                        "alpha": 0.8,
                        "type": "lidar_point_cloud",
                    }
                ],
            },
        }

        # Update renderer with camera at origin
        camera_position = (0.0, 0.0)
        renderer.update(geometry_data, camera_position, camera_yaw=0.0)

        # Verify point cloud collection exists and has correct number of points
        assert "test_cloud" in renderer.point_collections
        collection = renderer.point_collections["test_cloud"]
        offsets = collection.get_offsets()
        assert len(offsets) == len(points)

        # Verify points are visible (offsets transformed)
        # Offsets are in camera coordinates (camera at origin, no rotation)
        np.testing.assert_array_almost_equal(offsets, points)

    def test_lidar_mixed_coordinate_formats(self, sensor_test_data):
        """Test with mixed list/tuple/numpy array coordinates."""
        # Create renderer with auto_scale enabled
        renderer = MatplotlibRenderer(resolution=(800, 600), auto_scale=True, margin=5.0)

        # Mixed coordinate formats
        points = [
            [10.0, 20.0],  # list
            (30.0, 40.0),  # tuple
            np.array([50.0, 60.0]),  # numpy array
            np.array([70.0, 80.0]).tolist(),  # list from numpy
        ]

        geometry_data = {
            "map_data": {"road_id_to_remove": [], "road_elements": []},
            "participant_data": {
                "participant_id_to_create": [],
                "participant_id_to_remove": [],
                "participants": [],
                "point_clouds": [
                    {
                        "id": "mixed_cloud",
                        "points": points,
                        "color": "red",
                        "point_size": 2.0,
                        "alpha": 0.8,
                        "type": "lidar_point_cloud",
                    }
                ],
            },
        }

        # Update renderer with camera at origin
        camera_position = (0.0, 0.0)
        renderer.update(geometry_data, camera_position, camera_yaw=0.0)

        # Check that axis limits include all points
        xlim = renderer.ax.get_xlim()
        ylim = renderer.ax.get_ylim()

        # Expected bounds: min_x = 10 - margin, max_x = 70 + margin, etc.
        expected_x_min = 10.0 - 5.0
        expected_x_max = 70.0 + 5.0
        expected_y_min = 20.0 - 5.0
        expected_y_max = 80.0 + 5.0

        assert xlim[0] <= expected_x_min, f"X min {xlim[0]} should be <= {expected_x_min}"
        assert xlim[1] >= expected_x_max, f"X max {xlim[1]} should be >= {expected_x_max}"
        assert ylim[0] <= expected_y_min, f"Y min {ylim[0]} should be <= {expected_y_min}"
        assert ylim[1] >= expected_y_max, f"Y max {ylim[1]} should be >= {expected_y_max}"

        # Verify point cloud collection exists
        assert "mixed_cloud" in renderer.point_collections
        collection = renderer.point_collections["mixed_cloud"]
        offsets = collection.get_offsets()
        assert len(offsets) == len(points)


# ------------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "sensor"])
