# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""matplotlib renderer implementation."""


import logging
import os
from typing import Any, Dict, List, Optional, Tuple

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from numpy.typing import ArrayLike
from shapely.geometry import Point

from .matplotlib_config import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER


class MatplotlibRenderer:
    """Matplotlib-based renderer for 2D traffic simulation visualization."""

    def __init__(
        self,
        resolution: Tuple[float, float],
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        dpi: int = 200,
        auto_scale: bool = True,
    ):
        """Initialize the MatplotlibRenderer.

        Args:
            resolution: Output resolution in pixels.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            dpi: Dots per inch for rendering.
            auto_scale: Whether to auto-scale axes based on geometry data.

        Raises:
            ValueError: If resolution contains non-positive values.
        """
        if xlim is None:
            xlim = (-100.0, 100.0)
        if ylim is None:
            ylim = (-100.0, 100.0)

        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("resolution must contain positive values.")

        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution
        self.dpi = dpi
        self.width = max(self.resolution[0] / self.dpi, 1)
        self.height = max(self.resolution[1] / self.dpi, 1)
        self._auto_scale_enabled = auto_scale

        self.sensor_position = None
        self.camera_yaw = None
        self.road_lines = dict()
        self.road_line_geometry = dict()
        self.road_polygons = dict()
        self.road_polygon_geometry = dict()
        self.participants = dict()
        self.point_collections = dict()
        self.point_collection_geometry = dict()

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(self.width, self.height)
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)

        self.ax.set_aspect("equal")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_axis_off()

    def _extract_coordinates(self, point_data):
        """Extract x, y coordinates from various point-like data types."""
        try:
            if hasattr(point_data, "__len__") and len(point_data) >= 2:
                x = float(point_data[0])
                y = float(point_data[1])
                return x, y
        except (TypeError, IndexError, ValueError):
            pass
        return None

    def _calculate_bounds(
        self, geometry_data: dict, perception_range
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate bounds from geometry data in world coordinates."""
        if self.sensor_position is not None:
            if isinstance(perception_range, (list, tuple)) and len(perception_range) == 4:
                min_x = self.sensor_position[0] - perception_range[0]
                max_x = self.sensor_position[0] + perception_range[1]
                min_y = self.sensor_position[1] - perception_range[3]
                max_y = self.sensor_position[1] + perception_range[2]
            elif perception_range is not None:
                min_x = self.sensor_position[0] - perception_range
                max_x = self.sensor_position[0] + perception_range
                min_y = self.sensor_position[1] - perception_range
                max_y = self.sensor_position[1] + perception_range
            else:
                min_x, max_x = self.xlim
                min_y, max_y = self.ylim
        else:
            min_x, max_x = self.xlim
            min_y, max_y = self.ylim

        return (min_x, max_x), (min_y, max_y)

    def auto_scale(self, geometry_data: dict, perception_range) -> None:
        """Automatically calculate and set axis limits."""
        if not self._auto_scale_enabled:
            return

        world_bounds = self._calculate_bounds(geometry_data, perception_range)
        (x_min, x_max), (y_min, y_max) = world_bounds

        world_width = x_max - x_min
        world_height = y_max - y_min

        if world_width <= 0:
            world_width = 1.0
        if world_height <= 0:
            world_height = 1.0

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        resolution_aspect = self.resolution[1] / self.resolution[0]
        current_aspect = world_height / world_width

        if current_aspect > resolution_aspect:
            new_width = world_height / resolution_aspect
            new_height = world_height
        else:
            new_width = world_width
            new_height = world_width * resolution_aspect

        new_x_min = center_x - new_width / 2
        new_x_max = center_x + new_width / 2
        new_y_min = center_y - new_height / 2
        new_y_max = center_y + new_height / 2

        self.ax.set_xlim(new_x_min, new_x_max)
        self.ax.set_ylim(new_y_min, new_y_max)

        self.xlim = (new_x_min, new_x_max)
        self.ylim = (new_y_min, new_y_max)

    def _resolve_style(self, color_key: str, type_key) -> tuple:
        """Resolve style keys to concrete color value and z-order."""
        if color_key in COLOR_PALETTE:
            color = COLOR_PALETTE[color_key]
        elif color_key in DEFAULT_COLOR:
            color_value = DEFAULT_COLOR[color_key]
            color = COLOR_PALETTE.get(color_value, color_value)
        elif isinstance(color_key, str) and color_key.startswith("#"):
            color = color_key
        elif type_key in DEFAULT_COLOR:
            color_value = DEFAULT_COLOR[type_key]
            color = COLOR_PALETTE.get(color_value, color_value)
        else:
            color = COLOR_PALETTE["black"]

        if isinstance(type_key, (int, float)):
            z_order = type_key
        elif isinstance(type_key, str) and type_key in DEFAULT_ORDER:
            z_order = DEFAULT_ORDER[type_key]
        elif isinstance(color_key, str) and color_key in DEFAULT_ORDER:
            z_order = DEFAULT_ORDER[color_key]
        else:
            z_order = 1

        return color, z_order

    def _default_dash_pattern(self) -> tuple[float, float]:
        """Return the default dashed-line pattern in Matplotlib points."""
        return 5.0, 5.0

    def _aligned_dash_offset(
        self, line_shape: np.ndarray, dash_pattern: tuple[float, float]
    ) -> float:
        """Compute display-space dash phase for visually aligned dashed lines."""
        if len(line_shape) < 2:
            return 0.0

        p0 = line_shape[0]
        p1 = None

        for point in line_shape[1:]:
            if np.linalg.norm(point - p0) > 1e-9:
                p1 = point
                break

        if p1 is None:
            return 0.0

        display_points = self.ax.transData.transform(np.vstack([p0, p1]))
        direction = display_points[1] - display_points[0]
        norm = np.linalg.norm(direction)

        if norm < 1e-9:
            return 0.0

        direction = direction / norm
        dpi = self.fig.dpi if self.fig is not None else self.dpi
        start_in_points = display_points[0] * 72.0 / dpi

        scalar = float(np.dot(start_in_points, direction))
        period = float(sum(dash_pattern))

        if period <= 0.0:
            return 0.0

        return scalar % period

    def _create_polygon(self, element: Dict[str, Any]) -> Optional[Polygon]:
        """Create a matplotlib Polygon from element data."""
        if len(element["geometry"]) < 3:
            logging.warning(f"Polygon with id {element['id']} has less than 3 points, skipping.")
            return None

        color, z_order = self._resolve_style(element["color"], element.get("type"))

        return Polygon(
            xy=element["geometry"],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            antialiased=False,
            zorder=z_order,
        )

    def _create_circle(self, element: Dict[str, Any]) -> Circle:
        """Create a matplotlib Circle from element data."""
        color, z_order = self._resolve_style(element["color"], element.get("type"))

        return Circle(
            xy=(0, 0),
            radius=element["radius"],
            facecolor=color,
            linewidth=element["line_width"],
            zorder=z_order,
        )

    def _create_line(self, element: Dict[str, Any]) -> List[Line2D]:
        """Create matplotlib Line2D objects from element data."""
        line_shape = np.asarray(element["geometry"], dtype=float)
        lines = []

        if len(line_shape) < 2:
            return lines

        color, z_order = self._resolve_style(element["color"], element.get("type"))

        line_style = element.get("line_style", element.get("type", "solid"))
        line_style = str(line_style).lower()

        if "dashed" in line_style:
            dash_pattern = self._default_dash_pattern()
            global_s = element.get("custom_tags", {}).get("dash_offset", None)
            if global_s is not None:
                pts_tf = self.ax.transData.transform([[0, 0], [1, 0]])
                meter_to_display = float(np.linalg.norm(pts_tf[1] - pts_tf[0]))
                dpi = self.fig.dpi if self.fig is not None else self.dpi
                meter_to_pts = meter_to_display * 72.0 / dpi
                period = float(sum(dash_pattern))
                dash_offset = (float(global_s) * meter_to_pts) % period if period > 0 else 0.0
            else:
                dash_offset = self._aligned_dash_offset(line_shape, dash_pattern)
            line = Line2D(
                line_shape[:, 0],
                line_shape[:, 1],
                linewidth=element["line_width"],
                linestyle=(dash_offset, dash_pattern),
                color=color,
                zorder=z_order,
            )
            line._tactics2d_auto_dash_align = True
            line._tactics2d_dash_pattern = dash_pattern
            lines.append(line)

        elif "solid" in line_style:
            lines.append(
                Line2D(
                    line_shape[:, 0],
                    line_shape[:, 1],
                    linewidth=element["line_width"],
                    color=color,
                    zorder=z_order,
                )
            )

        return lines

    def _create_points(self, element: Dict[str, Any]) -> Optional[PathCollection]:
        """Create a matplotlib PathCollection from point cloud data."""
        points = np.array(element.get("points", []))

        color, z_order = self._resolve_style(
            element.get("color", "red"), element.get("type", "lidar_point_cloud")
        )

        unit_circle = Path.unit_circle()
        collection = PathCollection(
            (unit_circle,),
            sizes=[element.get("point_size", 1.0)],
            facecolors=color,
            alpha=element.get("alpha", 0.8),
            edgecolors="none",
            zorder=z_order,
            pickradius=0,
            offsets=points,
        )

        return collection

    def _transform_to_camera_view(self, points: ArrayLike) -> np.ndarray:
        """Transform points from world coordinates to camera view coordinates."""
        if self.sensor_position is None or self.camera_yaw is None:
            raise RuntimeError("Camera position and yaw must be set before transformation.")

        points = np.array(points)

        dx, dy = -self.sensor_position
        cos_theta = np.cos(-self.camera_yaw)
        sin_theta = np.sin(-self.camera_yaw)

        translated = points + np.array([dx, dy])
        rotated = np.dot(translated, np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]))
        transformed = rotated - np.array([dx, dy])

        return transformed

    def _update_polygon(
        self,
        polygon: Polygon,
        geometry: ArrayLike,
        position: ArrayLike = (0, 0),
        rotation: float = 0,
    ) -> None:
        """Update polygon geometry with position, rotation, and camera transformation."""
        center = np.array(position)
        yaw = rotation
        shape = np.array(geometry)

        cos_theta = np.cos(yaw)
        sin_theta = np.sin(yaw)
        rotation_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        transformed = (shape @ rotation_mat.T) + center
        transformed = self._transform_to_camera_view(transformed)
        polygon.set_xy(transformed)

    def _update_circle(self, circle: Circle, position: ArrayLike) -> None:
        """Update circle position with camera transformation."""
        transformed_center = self._transform_to_camera_view(position)
        circle.set_center(transformed_center)

    def _update_line(
        self,
        lines: List[Line2D],
        geometry: ArrayLike,
        position: ArrayLike = (0, 0),
        rotation: float = 0,
    ) -> None:
        """Update line geometry with position, rotation, and camera transformation."""
        center = np.array(position)
        yaw = rotation
        shape = np.array(geometry)

        if len(shape) < 2:
            return

        cos_theta = np.cos(yaw)
        sin_theta = np.sin(yaw)
        rotation_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        transformed = (shape @ rotation_mat.T) + center
        transformed = self._transform_to_camera_view(transformed)

        for line in lines:
            line.set_data(transformed[:, 0], transformed[:, 1])

            if getattr(line, "_tactics2d_auto_dash_align", False):
                dash_pattern = getattr(
                    line, "_tactics2d_dash_pattern", self._default_dash_pattern()
                )
                dash_offset = self._aligned_dash_offset(transformed, dash_pattern)
                line.set_linestyle((dash_offset, dash_pattern))

    def _update_points(self, point_collection: PathCollection, points: ArrayLike) -> None:
        """Update point collection coordinates and apply camera transformation."""
        if len(points) == 0:
            return

        transformed_points = self._transform_to_camera_view(points)
        point_collection.set_offsets(transformed_points)

    def update(self, geometry_data: dict):
        """Update the renderer with new geometry data and camera view."""
        metadata = geometry_data["metadata"]
        sensor_position = metadata["sensor_position"]
        camera_yaw = metadata["sensor_yaw"]
        perception_range = metadata.get("perception_range", None)

        if isinstance(sensor_position, Point):
            self.sensor_position = np.array([sensor_position.x, sensor_position.y])
        else:
            sensor_position_array = np.asarray(sensor_position)
            if sensor_position_array.size != 2:
                raise ValueError(
                    f"Camera position must be 2D, got shape {sensor_position_array.shape}"
                )
            self.sensor_position = sensor_position_array[:2]

        self.camera_yaw = camera_yaw

        if "map_data" not in geometry_data:
            raise KeyError("geometry_data must contain 'map_data' key")
        if "participant_data" not in geometry_data:
            raise KeyError("geometry_data must contain 'participant_data' key")

        map_data = geometry_data["map_data"]
        participant_data = geometry_data["participant_data"]

        required_map_keys = ["road_id_to_remove", "road_elements"]
        for key in required_map_keys:
            if key not in map_data:
                raise KeyError(f"map_data must contain '{key}' key")

        required_participant_keys = [
            "participant_id_to_create",
            "participant_id_to_remove",
            "participants",
        ]
        for key in required_participant_keys:
            if key not in participant_data:
                raise KeyError(f"participant_data must contain '{key}' key")

        road_id_to_remove = map_data["road_id_to_remove"]
        road_elements = map_data["road_elements"]
        participant_id_to_create = participant_data["participant_id_to_create"]
        participant_id_to_remove = participant_data["participant_id_to_remove"]
        participants = participant_data["participants"]
        point_clouds = participant_data.get("point_clouds", [])

        for element in road_elements:
            element_id = element["id"]
            if element_id in self.road_polygons or element_id in self.road_lines:
                continue

            if element.get("shape") == "polygon":
                polygon = self._create_polygon(element)
                if polygon is None:
                    continue
                self.road_polygons[element_id] = polygon
                self.road_polygon_geometry[element_id] = element["geometry"]
                self.ax.add_patch(polygon)

            elif element.get("shape") == "line":
                lines = self._create_line(element)
                self.road_lines[element_id] = lines
                self.road_line_geometry[element_id] = element["geometry"]
                for line in lines:
                    self.ax.add_line(line)

        for element_id, geometry in self.road_polygon_geometry.items():
            self._update_polygon(self.road_polygons[element_id], geometry)

        for element_id, geometry in self.road_line_geometry.items():
            self._update_line(self.road_lines[element_id], geometry)

        for element_id in road_id_to_remove:
            if element_id in self.road_polygons:
                polygon = self.road_polygons.pop(element_id, None)
                if polygon is not None:
                    polygon.remove()
                self.road_polygon_geometry.pop(element_id, None)

            elif element_id in self.road_lines:
                lines = self.road_lines.pop(element_id, None)
                if lines is not None:
                    for line in lines:
                        line.remove()
                self.road_line_geometry.pop(element_id, None)

        for id_ in participant_id_to_remove:
            participant = self.participants.pop(id_, None)
            if participant is not None:
                participant.remove()

        for participant in participants:
            id_ = participant["id"]

            if id_ in participant_id_to_create and id_ not in self.participants:
                if participant.get("shape") == "polygon":
                    patch = self._create_polygon(participant)
                    if patch is None:
                        continue
                elif participant.get("shape") == "circle":
                    patch = self._create_circle(participant)
                else:
                    continue
                self.participants[id_] = patch
                self.ax.add_patch(patch)

            if id_ in self.participants:
                if participant.get("shape") == "polygon":
                    self._update_polygon(
                        self.participants[id_],
                        participant["geometry"],
                        participant["position"],
                        participant["rotation"],
                    )
                elif participant.get("shape") == "circle":
                    self._update_circle(self.participants[id_], participant["position"])

        for collection in self.point_collections.values():
            collection.remove()
        self.point_collections.clear()
        self.point_collection_geometry.clear()

        for i, point_cloud in enumerate(point_clouds):
            pc_id = point_cloud.get("id", f"point_cloud_{i}")
            collection = self._create_points(point_cloud)

            if collection is None:
                continue

            self.point_collections[pc_id] = collection
            self.point_collection_geometry[pc_id] = point_cloud.get("points", [])
            self.ax.add_collection(collection)

        for pc_id, geometry in self.point_collection_geometry.items():
            if pc_id in self.point_collections:
                self._update_points(self.point_collections[pc_id], geometry)

        if self._auto_scale_enabled:
            self.auto_scale(geometry_data, perception_range)
            for element_id, geometry in self.road_line_geometry.items():
                self._update_line(self.road_lines[element_id], geometry)

    def save_single_frame(
        self, save_to: Optional[str] = None, dpi: Optional[int] = None, return_array: bool = False
    ):
        """Save the current frame to file or return as numpy array."""
        try:
            dpi = int(dpi)
        except (TypeError, ValueError):
            dpi = self.dpi

        self.fig.canvas.draw()
        if save_to is not None:
            dir_name = os.path.dirname(save_to)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self.fig.savefig(save_to, dpi=dpi)

        if return_array:
            try:
                image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return image
            except AttributeError:
                image = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
                image = image[:, :, 1:4]
                return image
            except Exception as e:
                logging.error(f"Failed to convert figure to array: {e}")

        return None

    def reset(self):
        """Reset the renderer to initial state."""
        self.sensor_position = None
        self.camera_yaw = None

        for lines in self.road_lines.values():
            for line in lines:
                line.remove()
        self.road_lines.clear()
        self.road_line_geometry.clear()

        for polygon in self.road_polygons.values():
            polygon.remove()
        self.road_polygons.clear()
        self.road_polygon_geometry.clear()

        for patch in self.participants.values():
            patch.remove()
        self.participants.clear()

        for collection in self.point_collections.values():
            collection.remove()
        self.point_collections.clear()
        self.point_collection_geometry.clear()

        for patch in list(self.ax.patches):
            patch.remove()
        for line in list(self.ax.lines):
            line.remove()
        for text in list(self.ax.texts):
            text.remove()
        for artist in list(self.ax.artists):
            artist.remove()
        for collection in list(self.ax.collections):
            collection.remove()

        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()

    def destroy(self):
        """Destroy the renderer and release all matplotlib resources."""
        self.reset()

        if self.fig is not None:
            plt.close(self.fig)

        self.fig = None
        self.ax = None
        self.sensor_position = None
        self.camera_yaw = None
        self.road_lines.clear()
        self.road_line_geometry.clear()
        self.road_polygons.clear()
        self.road_polygon_geometry.clear()
        self.participants.clear()
