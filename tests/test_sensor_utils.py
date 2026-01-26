# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared utilities for sensor tests."""

from pathlib import Path

import numpy as np

from tactics2d.renderer.matplotlib_renderer import MatplotlibRenderer


def render_sensor_output(
    geometry_data, sensor_position, output_path, title="Sensor Output", camera_yaw=0.0
):
    """Render sensor output and save image.

    Simplified version of the original _render_and_save_sensor_output method.
    """
    # Initialize renderer with auto-scaling enabled
    renderer = MatplotlibRenderer(
        xlim=None,  # Will be auto-scaled
        ylim=None,  # Will be auto-scaled
        resolution=(1600, 1200),
        dpi=150,
        auto_scale=True,
        margin=10.0,
    )

    # Set background color for better visibility
    renderer.ax.set_facecolor("lightgray")

    # Update renderer
    print(f"[DEBUG Test] Geometry data keys: {list(geometry_data.keys())}")
    if "map_data" in geometry_data:
        map_data = geometry_data["map_data"]
        print(f"[DEBUG Test] Map data keys: {list(map_data.keys())}")
        if "road_elements" in map_data:
            print(f"[DEBUG Test] Road elements count: {len(map_data['road_elements'])}")
    if "participant_data" in geometry_data:
        participant_data = geometry_data["participant_data"]
        print(f"[DEBUG Test] Participant data keys: {list(participant_data.keys())}")
        if "point_clouds" in participant_data:
            point_clouds = participant_data["point_clouds"]
            print(f"[DEBUG Test] Number of point clouds: {len(point_clouds)}")
            for i, pc in enumerate(point_clouds):
                points = pc.get("points", [])
                print(f"[DEBUG Test] Point cloud {i}: {len(points)} points")
        if "participants" in participant_data:
            print(f"[DEBUG Test] Participants count: {len(participant_data['participants'])}")
    renderer.update(
        geometry_data=geometry_data, camera_position=sensor_position, camera_yaw=camera_yaw
    )
    print(f"[DEBUG Test] Final axis limits: x={renderer.ax.get_xlim()}, y={renderer.ax.get_ylim()}")
    print(f"[DEBUG Test] Point collections in renderer: {list(renderer.point_collections.keys())}")

    # Save image
    output_path = Path(output_path)
    renderer.save_single_frame(save_to=str(output_path))

    # Verify file was created
    if not output_path.exists():
        raise RuntimeError(f"Failed to save image to {output_path}")

    # Cleanup
    renderer.destroy()

    return str(output_path)


def create_sensor_factory(sensor_type, sensor_id, map_obj, **kwargs):
    """Create sensor instance factory function."""
    if sensor_type == "camera":
        from tactics2d.sensor.camera import BEVCamera

        return BEVCamera(id_=sensor_id, map_=map_obj, **kwargs)
    elif sensor_type == "lidar":
        from tactics2d.sensor.lidar import SingleLineLidar

        return SingleLineLidar(id_=sensor_id, map_=map_obj, **kwargs)
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


def assert_sensor_data_structure(geometry_data, expected_frame, sensor_type=None):
    """Validate sensor data structure."""
    assert isinstance(geometry_data, dict), "Geometry data should be a dict"
    assert (
        geometry_data["frame"] == expected_frame
    ), f"Frame mismatch: {geometry_data['frame']} != {expected_frame}"
    if sensor_type:
        assert (
            geometry_data.get("sensor_type") == sensor_type
        ), f"Sensor type mismatch: {geometry_data.get('sensor_type')} != {sensor_type}"
