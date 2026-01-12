# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""matplotlib renderer implementation."""


import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Set matplotlib backend to Agg before import
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from numpy.typing import ArrayLike
from shapely.geometry import Point

from .matplotlib_config import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER


class MatplotlibRenderer:
    """Matplotlib-based renderer for 2D traffic simulation visualization.

    This renderer uses matplotlib to create and update visual representations of
    road elements, participants (vehicles, pedestrians, etc.), and camera views.
    It supports dynamic updates of geometry data with camera transformations.

    Attributes:
        xlim (Tuple[float, float]): X-axis limits of the view.
        ylim (Tuple[float, float]): Y-axis limits of the view.
        resolution (Tuple[float, float]): Output resolution in pixels (width, height).
        dpi (int): Dots per inch for rendering.
        width (float): Figure width in inches.
        height (float): Figure height in inches.
        camera_position (np.ndarray): Current camera position [x, y].
        camera_yaw (float): Current camera yaw angle in radians.
        road_lines (Dict[str, List[Line2D]]): Dictionary mapping road line IDs to Line2D objects.
        road_line_geometry (Dict[str, np.ndarray]): Dictionary mapping road line IDs to geometry data.
        road_polygons (Dict[str, Polygon]): Dictionary mapping road polygon IDs to Polygon objects.
        road_polygon_geometry (Dict[str, np.ndarray]): Dictionary mapping road polygon IDs to geometry data.
        participants (Dict[str, Union[Polygon, Circle]]): Dictionary mapping participant IDs to shape objects.
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        ax (matplotlib.axes.Axes): Matplotlib axes object.
    """

    def __init__(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        resolution: Tuple[float, float],
        dpi: int = 200,
    ):
        """Initialize the MatplotlibRenderer.

        Args:
            xlim (Tuple[float, float]): X-axis limits of the view (min, max).
            ylim (Tuple[float, float]): Y-axis limits of the view (min, max).
            resolution (Tuple[float, float]): Output resolution in pixels (width, height).
            dpi (int, optional): Dots per inch for rendering. Defaults to 200.

        Raises:
            ValueError: If resolution contains non-positive values.
        """
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution
        self.dpi = dpi
        self.width = max(self.resolution[0] / self.dpi, 1)
        self.height = max(self.resolution[1] / self.dpi, 1)

        aspect_ratio = (self.ylim[1] - self.ylim[0]) / (self.xlim[1] - self.xlim[0])
        height = self.width * aspect_ratio
        if height < 1:
            self.width = self.height / aspect_ratio
        else:
            self.height = height

        self.camera_position = None
        self.camera_yaw = None
        self.road_lines = dict()
        self.road_line_geometry = dict()
        self.road_polygons = dict()
        self.road_polygon_geometry = dict()
        self.participants = dict()

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(self.width, self.height)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

        self.ax.set_aspect("equal")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_axis_off()

    def _resolve_style(self, color_key: str, type_key) -> tuple:
        """Resolve style keys to concrete color value and z-order.

        Args:
            color_key: Color key (could be color name like 'white' or type key like 'vehicle')
            type_key: Element type key (string) or numeric z-order (int/float)

        Returns:
            tuple: (color_value, z_order)
        """
        # Resolve color
        if color_key in COLOR_PALETTE:
            # Direct color name like 'white', 'red', etc.
            color = COLOR_PALETTE[color_key]
        elif color_key in DEFAULT_COLOR:
            # Type-based key like 'vehicle', 'lane', etc.
            color_value = DEFAULT_COLOR[color_key]
            # The value might be a color name or hex code
            if color_value in COLOR_PALETTE:
                color = COLOR_PALETTE[color_value]
            else:
                color = color_value  # Assume it's already a hex code or valid color
        elif isinstance(color_key, str) and color_key.startswith("#"):
            # Hex color code like "#FF0000"
            color = color_key
        else:
            # Fallback to default color
            color = COLOR_PALETTE["black"]

        # Resolve z-order from type key
        if isinstance(type_key, (int, float)):
            # Already a numeric value
            z_order = type_key
        elif isinstance(type_key, str) and type_key in DEFAULT_ORDER:
            # String key, look up in DEFAULT_ORDER
            z_order = DEFAULT_ORDER[type_key]
        else:
            # Fallback: try to use color_key as type key
            if isinstance(color_key, str) and color_key in DEFAULT_ORDER:
                z_order = DEFAULT_ORDER[color_key]
            else:
                # Fallback to default z-order
                z_order = 1

        return color, z_order

    def _create_polygon(self, element: Dict[str, Any]) -> Optional[Polygon]:
        """Create a matplotlib Polygon from element data.

        Args:
            element (Dict[str, Any]): Polygon element data with keys:
                'id' (str): Element identifier.
                'geometry' (List[Tuple[float, float]]): Polygon vertices.
                'color' (str): Color key (color name or type key).
                'line_width' (float): Border line width.
                'order' (str): Z-order key (type key).

        Returns:
            Optional[Polygon]: Polygon object if geometry has at least 3 points,
                otherwise None.
        """
        if len(element["geometry"]) < 3:
            logging.warning(f"Polygon with id {element['id']} has less than 3 points, skipping.")
            return None

        # Resolve color and z-order from style keys
        color, z_order = self._resolve_style(element["color"], element.get("type"))

        return Polygon(
            xy=element["geometry"],
            closed=True,
            facecolor=color,
            linewidth=element["line_width"],
            zorder=z_order,
        )

    def _create_circle(self, element: Dict[str, Any]) -> Circle:
        """Create a matplotlib Circle from element data.

        Args:
            element (Dict[str, Any]): Circle element data with keys:
                'radius' (float): Circle radius.
                'color' (str): Color key (color name or type key).
                'line_width' (float): Border line width.
                'order' (str): Z-order key (type key).

        Returns:
            Circle: Circle object centered at (0, 0).
        """
        # Resolve color and z-order from style keys
        color, z_order = self._resolve_style(element["color"], element.get("type"))

        return Circle(
            xy=(0, 0),
            radius=element["radius"],
            facecolor=color,
            linewidth=element["line_width"],
            zorder=z_order,
        )

    def _create_line(self, element: Dict[str, Any]) -> List[Line2D]:
        """Create matplotlib Line2D objects from element data.

        Args:
            element (Dict[str, Any]): Line element data with keys:
                'geometry' (List[Tuple[float, float]]): Line vertices.
                'type' (str): Line type ('solid', 'dashed', or combinations).
                'color' (str): Color key (color name or type key).
                'line_width' (float): Line width.
                'order' (str): Z-order key (type key).

        Returns:
            List[Line2D]: List of Line2D objects (currently 1, but could be multiple
                for complex line styles like "solid_dashed").

        Note:
            Currently only handles "solid" and "dashed" line types.
            TODO: Handle "solid_solid", "solid_dashed", "dashed_solid", and "dashed_dashed".
        """
        line_shape = np.array(element["geometry"])
        lines = []

        # Resolve color and z-order from style keys
        color, z_order = self._resolve_style(element["color"], element.get("type"))

        # TODO: Handle "solid_solid", "solid_dashed", "dashed_solid", and "dashed_dashed"
        line_style = element.get("line_style", element["type"])
        if "dashed" in line_style:
            lines.append(
                Line2D(
                    line_shape[:, 0],
                    line_shape[:, 1],
                    linewidth=element["line_width"],
                    linestyle=(0, (5, 5)),
                    color=color,
                    zorder=z_order,
                )
            )

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

    def _transform_to_camera_view(self, points: ArrayLike) -> np.ndarray:
        """Transform points from world coordinates to camera view coordinates.

        Args:
            points (ArrayLike): Array of points with shape (N, 2) to transform.

        Returns:
            np.ndarray: Transformed points with shape (N, 2).

        Raises:
            RuntimeError: If camera position or yaw is not set.
        """
        if self.camera_position is None or self.camera_yaw is None:
            raise RuntimeError("Camera position and yaw must be set before transformation.")

        points = np.array(points)
        dx, dy = -self.camera_position
        cos_theta = np.cos(-self.camera_yaw)
        sin_theta = np.sin(-self.camera_yaw)

        translated = points + np.array([dx, dy])
        rotated = np.dot(translated, np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]))

        return rotated

    def _update_polygon(
        self,
        polygon: Polygon,
        geometry: ArrayLike,
        position: ArrayLike = (0, 0),
        rotation: float = 0,
    ) -> None:
        """Update polygon geometry with position, rotation, and camera transformation.

        Args:
            polygon (Polygon): Matplotlib Polygon object to update.
            geometry (ArrayLike): Original polygon vertices with shape (N, 2).
            position (ArrayLike, optional): Position offset [x, y]. Defaults to (0, 0).
            rotation (float, optional): Rotation angle in radians. Defaults to 0.
        """
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
        """Update circle position with camera transformation.

        Args:
            circle (Circle): Matplotlib Circle object to update.
            position (ArrayLike): Circle center position [x, y].
        """
        transformed_center = self._transform_to_camera_view(position)
        circle.set_center(transformed_center)

    def _update_line(
        self,
        lines: List[Line2D],
        geometry: ArrayLike,
        position: ArrayLike = (0, 0),
        rotation: float = 0,
    ) -> None:
        """Update line geometry with position, rotation, and camera transformation.

        Args:
            lines (List[Line2D]): List of Line2D objects to update.
            geometry (ArrayLike): Original line vertices with shape (N, 2).
            position (ArrayLike, optional): Position offset [x, y]. Defaults to (0, 0).
            rotation (float, optional): Rotation angle in radians. Defaults to 0.
        """
        for line in lines:
            # TODO: Handle "solid_solid", "solid_dashed", "dashed_solid", and "dashed_dashed"
            center = np.array(position)
            yaw = rotation
            shape = np.array(geometry)

            cos_theta = np.cos(yaw)
            sin_theta = np.sin(yaw)
            rotation_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            transformed = (shape @ rotation_mat.T) + center
            transformed = self._transform_to_camera_view(transformed)
            line.set_data(transformed[:, 0], transformed[:, 1])

    def update(
        self, geometry_data: dict, camera_position: Union[Point, ArrayLike], camera_yaw: float = 0
    ):
        """Update the renderer with new geometry data and camera view.

        This method processes geometry data to create, update, and remove
        road elements and participants. It applies camera transformations
        to all visible elements.

        Args:
            geometry_data (dict): Geometry data containing map and participant information.
                Expected structure:
                {
                    "map_data": {
                        "road_id_to_remove": List[str],
                        "road_elements": List[Dict[str, Any]]
                    },
                    "participant_data": {
                        "participant_id_to_create": List[str],
                        "participant_id_to_remove": List[str],
                        "participants": List[Dict[str, Any]]
                    }
                }
            camera_position (Union[Point, ArrayLike]): Camera position as shapely Point
                or 2D array-like [x, y].
            camera_yaw (float, optional): Camera yaw angle in radians. Defaults to 0.

        Raises:
            KeyError: If required keys are missing in geometry_data.
            ValueError: If camera_position is not 2D.
        """
        if isinstance(camera_position, Point):
            self.camera_position = np.array([camera_position.x, camera_position.y])
        else:
            camera_position_array = np.asarray(camera_position)
            if camera_position_array.size != 2:
                raise ValueError(
                    f"Camera position must be 2D, got shape {camera_position_array.shape}"
                )
            self.camera_position = camera_position_array[:2]  # Ensure only first 2 elements

        self.camera_yaw = camera_yaw

        # Validate geometry_data structure
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

        # Add new road elements
        for element in road_elements:
            element_id = element["id"]
            if element_id in self.road_polygons or element_id in self.road_lines:
                continue  # Skip if already exists

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

        # Update road elements
        for element_id, geometry in self.road_polygon_geometry.items():
            self._update_polygon(self.road_polygons[element_id], geometry)

        for element_id, geometry in self.road_line_geometry.items():
            self._update_line(self.road_lines[element_id], geometry)

        # Remove road elements
        for element_id in road_id_to_remove:
            if element_id in self.road_polygons:
                polygon = self.road_polygons.pop(element_id, None)
                if polygon is not None:
                    polygon.remove()
                self.road_polygon_geometry.pop(element_id, None)

            elif element_id in self.road_lines:
                lines = self.road_lines.pop(element_id, None)
                for line in lines:
                    line.remove()
                self.road_line_geometry.pop(element_id, None)

        # Remove dead participants
        for id_ in participant_id_to_remove:
            participant = self.participants.pop(id_, None)
            if participant is not None:
                participant.remove()

        for participant in participants:
            id_ = participant["id"]

            # Create
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

            # Update
            if id_ in self.participants:
                if participant.get("shape") == "polygon":
                    self._update_polygon(
                        self.participants[id_],
                        participant["geometry"],
                        participant["position"],
                        participant["rotation"],
                    )
                    # add triangle indicator for direction
                    points = np.array(participant.geometry.coords)
                    triangle = [
                        ((points[0] + points[1]) / 2).tolist(),
                        ((points[1] + points[2]) / 2).tolist(),
                        ((points[3] + points[0]) / 2).tolist(),
                    ]
                    self._update_polygon(
                        self.participants[id_] + 0.5,
                        triangle,
                        participant["position"],
                        participant["rotation"],
                    )

                elif participant.get("shape") == "circle":
                    self._update_circle(self.participants[id_], participant["position"])

    def save_single_frame(
        self, save_to: Optional[str] = None, dpi: Optional[int] = None, return_array: bool = False
    ):
        """Save the current frame to file or return as numpy array.

        Args:
            save_to (str, optional): File path to save the image. If None, image is not saved.
                Defaults to None.
            dpi (int, optional): Dots per inch for saved image. If None, uses instance dpi.
                Defaults to None.
            return_array (bool, optional): If True, return image as numpy array.
                Defaults to False.

        Returns:
            Optional[np.ndarray]: RGB image array with shape (height, width, 3) if return_array=True,
                otherwise None.

        Raises:
            ValueError: If save_to path is invalid or cannot write.
            RuntimeError: If matplotlib figure is not properly initialized.
        """
        try:
            dpi = int(dpi)
        except (TypeError, ValueError):
            dpi = self.dpi

        self.fig.canvas.draw()
        if save_to is not None:
            self.fig.savefig(save_to, dpi=dpi)

        if return_array:
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

        return None

    def reset(self):
        """Reset the renderer to initial state.

        This method removes all road elements, participants, and resets camera state.
        The figure and axes are preserved with their original limits and settings.
        """
        self.camera_position = None
        self.camera_yaw = None

        # Remove road lines
        for lines in self.road_lines.values():
            for line in lines:
                line.remove()
        self.road_lines.clear()
        self.road_line_geometry.clear()

        # Remove road polygons
        for polygon in self.road_polygons.values():
            polygon.remove()
        self.road_polygons.clear()
        self.road_polygon_geometry.clear()

        # Remove participants
        for patch in self.participants.values():
            patch.remove()
        self.participants.clear()

        # Remove any remaining artists (patches, lines, texts, etc.)
        for patch in list(self.ax.patches):
            patch.remove()
        for line in list(self.ax.lines):
            line.remove()
        for text in list(self.ax.texts):
            text.remove()
        for artist in list(self.ax.artists):
            artist.remove()

        # Do NOT use ax.clear(), instead reapply axis limits and settings
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()

    def destroy(self):
        """Destroy the renderer and release all matplotlib resources.

        This method resets the renderer, closes the figure, and clears
        all references to allow proper cleanup by matplotlib.
        After calling destroy(), the renderer cannot be used further.

        Note:
            It's recommended to call destroy() when the renderer is no longer needed
            to prevent memory leaks.
        """
        # First reset to remove all artists
        self.reset()

        # Close the figure
        if self.fig is not None:
            plt.close(self.fig)

        # Clear all references
        self.fig = None
        self.ax = None
        self.camera_position = None
        self.camera_yaw = None
        self.road_lines.clear()
        self.road_line_geometry.clear()
        self.road_polygons.clear()
        self.road_polygon_geometry.clear()
        self.participants.clear()
