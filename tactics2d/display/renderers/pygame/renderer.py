# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pygame renderer implementation.

Consumes geometry_data from sensor computation and renders it onto a
pygame surface using pygame.draw calls.  Separating rendering from
computation avoids duplicating the scan / map-traversal logic.
"""


from typing import Optional, Tuple

import numpy as np
import pygame

from ..config import COLOR_PALETTE, DEFAULT_COLOR


class PygameRenderer:
    """Renders *geometry_data* (produced by a sensor) onto a pygame surface.

    Parameters
    ----------
    window_size : Tuple[int, int]
        (width, height) of the pygame surface in pixels.
    scale : float
        Pixels-per-meter factor for the global-to-pixel transform.
    perception_range : Tuple[float, float, float, float]
        (left, right, front, back) perception range in meters.
    map_boundary : Tuple[float, float, float, float] | None
        (xmin, xmax, ymin, ymax) of the map, needed for the static
        (non-tracking) transform.  *None* is safe but may misplace
        geometry when no position is available.
    """

    def __init__(
        self,
        window_size: Tuple[int, int],
        scale: float,
        perception_range: Tuple[float, float, float, float],
        map_boundary: Optional[Tuple[float, float, float, float]] = None,
    ):
        self._surface = pygame.Surface(window_size)
        self.window_size = window_size
        self.scale = scale
        self.perception_range = perception_range
        self.map_boundary = map_boundary

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        geometry_data: dict,
        position: Optional["Point"] = None,
        heading: Optional[float] = None,
        background_color: Optional[Tuple[int, int, int]] = None,
    ):
        """Draw *geometry_data* onto the internal pygame surface.

        Parameters
        ----------
        geometry_data : dict
            The structured dict produced by a sensor ``update()`` call.
        position : Point | None
            Sensor position in global coordinates.  *None* triggers the
            static (full-map) transform.
        heading : float | None
            Sensor yaw angle in radians.
        background_color : Tuple[int, int, int] | None
            Optional (R, G, B) fill colour.  Defaults to
            ``COLOR_PALETTE["white"]``.
        """
        if background_color is not None:
            self._surface.fill(background_color)
        else:
            self._surface.fill(pygame.Color(COLOR_PALETTE["white"]))

        transform = self._build_transform(position, heading)

        map_data = geometry_data.get("map_data", {})
        for element in map_data.get("road_elements", []):
            self._draw_road_element(element, transform)

        participant_data = geometry_data.get("participant_data", {})
        for participant in participant_data.get("participants", []):
            self._draw_participant(participant, transform)

        for point_cloud in participant_data.get("point_clouds", []):
            self._draw_point_cloud(point_cloud, transform)

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def get_observation(self) -> np.ndarray:
        """Return the rendered surface as a (H, W, 3) numpy array."""
        return pygame.surfarray.array3d(self._surface)

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------

    def _build_transform(self, position: Optional["Point"], heading: Optional[float]):
        """Build the affine transform matrix ``(a, b, d, e, x_off, y_off)``.

        Pixel coordinates are obtained via::

            px = a * world_x + b * world_y + x_off
            py = d * world_x + e * world_y + y_off
        """
        if position is None or heading is None:
            # Static mode -- centre the whole map in the window.
            x_center = 0.0
            y_center = 0.0
            if self.map_boundary is not None:
                x_center = 0.5 * (self.map_boundary[0] + self.map_boundary[1])
                y_center = 0.5 * (self.map_boundary[2] + self.map_boundary[3])

            return (
                self.scale,
                0.0,
                0.0,
                -self.scale,
                0.5 * self.window_size[0] - self.scale * x_center,
                0.5 * self.window_size[1] + self.scale * y_center,
            )

        theta = heading - np.pi / 2
        a = self.scale * np.cos(theta)
        b = self.scale * np.sin(theta)
        d = self.scale * np.sin(theta)
        e = -self.scale * np.cos(theta)
        x_off = self.scale * (
            self.perception_range[0] - position.x * np.cos(theta) - position.y * np.sin(theta)
        )
        y_off = self.scale * (
            self.perception_range[2] - position.x * np.sin(theta) + position.y * np.cos(theta)
        )
        return (a, b, d, e, x_off, y_off)

    @staticmethod
    def _transform_coords(coords, transform):
        a, b, d, e, x_off, y_off = transform
        return [(a * x + b * y + x_off, d * x + e * y + y_off) for x, y in coords]

    @staticmethod
    def _apply_transform(point, transform):
        a, b, d, e, x_off, y_off = transform
        x, y = point
        return (a * x + b * y + x_off, d * x + e * y + y_off)

    # ------------------------------------------------------------------
    # Colour resolution
    # ------------------------------------------------------------------

    def _resolve_color(self, element: dict):
        colour_str = element.get("color")
        if colour_str in COLOR_PALETTE:
            return pygame.Color(COLOR_PALETTE[colour_str])

        # Fallback: look up the element *type* in DEFAULT_COLOR.
        element_type = element.get("type", "")
        if element_type in DEFAULT_COLOR:
            fallback = DEFAULT_COLOR[element_type]
            if fallback in COLOR_PALETTE:
                return pygame.Color(COLOR_PALETTE[fallback])

        return pygame.Color(COLOR_PALETTE.get("white", "#f1f2f6"))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_road_element(self, element: dict, transform):
        shape = element.get("shape")
        colour = self._resolve_color(element)

        if shape == "polygon":
            pts = self._transform_coords(element.get("geometry", []), transform)
            pygame.draw.polygon(self._surface, colour, pts)

        elif shape == "line":
            pts = self._transform_coords(element.get("geometry", []), transform)
            line_width = max(1, int(element.get("line_width", 1) * self.scale))
            pygame.draw.lines(self._surface, colour, False, pts, line_width)

    def _draw_participant(self, participant: dict, transform):
        shape = participant.get("shape")
        colour = self._resolve_color(participant)
        position = participant.get("position", [0, 0])
        rotation = participant.get("rotation", 0.0)

        if shape == "polygon":
            # The geometry is in local coordinates (centered at origin).
            # Convert to world coordinates using position + rotation,
            # then apply the camera (world→pixel) transform.
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            px, py = position
            pts = []
            for lx, ly in participant.get("geometry", []):
                wx = lx * cos_r - ly * sin_r + px
                wy = lx * sin_r + ly * cos_r + py
                pts.append(self._apply_transform((wx, wy), transform))
            pygame.draw.polygon(self._surface, colour, pts)

            # Draw a thin outline for heading arrows so they remain
            # visible even when the vehicle colour is dark.
            if participant.get("type") == "heading_arrow":
                pygame.draw.polygon(self._surface, (0, 0, 0), pts, width=1)

        elif shape == "circle":
            a, b, d, e, x_off, y_off = transform
            cx = a * position[0] + b * position[1] + x_off
            cy = d * position[0] + e * position[1] + y_off
            radius = max(1, int(participant.get("radius", 0.5) * self.scale))
            pygame.draw.circle(self._surface, colour, (cx, cy), radius)

    def _draw_point_cloud(self, point_cloud: dict, transform):
        colour_str = point_cloud.get("color", "red")
        colour = (
            pygame.Color(COLOR_PALETTE[colour_str])
            if colour_str in COLOR_PALETTE
            else pygame.Color(COLOR_PALETTE["red"])
        )
        point_size = max(1, int(point_cloud.get("point_size", 1)))
        a, b, d, e, x_off, y_off = transform

        points = point_cloud.get("points", [])
        if not points:
            return
        arr = np.asarray(points, dtype=float)
        px = (arr[:, 0] * a + arr[:, 1] * b + x_off).astype(int)
        py = (arr[:, 0] * d + arr[:, 1] * e + y_off).astype(int)

        # Clip to the surface bounds before drawing.
        height, width = self._surface.get_height(), self._surface.get_width()
        in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px, py = px[in_bounds], py[in_bounds]
        if len(px) == 0:
            return

        if point_size <= 1:
            # Fast path: write 1 px dots directly through the surface pixel view
            # (one vectorized assignment instead of one draw call per point).
            try:
                pixels = pygame.surfarray.pixels3d(self._surface)
                pixels[px, py] = (colour.r, colour.g, colour.b)
                del pixels  # release the surface lock
            except (ValueError, pygame.error):
                for x, y in zip(px.tolist(), py.tolist()):
                    pygame.draw.circle(self._surface, colour, (x, y), point_size)
        else:
            for x, y in zip(px.tolist(), py.tolist()):
                pygame.draw.circle(self._surface, colour, (x, y), point_size)
