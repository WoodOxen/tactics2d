# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Map rasterization utilities for BITS-style behavior imitation."""

from typing import Iterable, Optional, Sequence

import numpy as np
from matplotlib.path import Path
from shapely.geometry import LinearRing, LineString, Polygon

from tactics2d.map.element import Map

from .config import BitsConfig
from .schema import BitsRaster


class BitsRasterizer:
    """Rasterize Tactics2D maps in an agent-centric frame.

    The returned image has three static semantic channels:

    1. drivable lane and area polygons;
    2. lane centerlines and road lines;
    3. pedestrian-oriented areas such as crosswalks and walkways.
    """

    DRIVABLE_AREA_SUBTYPES = {"drivable_area", "parking", "road_segment"}
    PEDESTRIAN_AREA_SUBTYPES = {"crosswalk", "walkway"}

    def __init__(self, config: Optional[BitsConfig] = None):
        self.config = config or BitsConfig()

    def rasterize(self, map_: Map, agent_from_world: np.ndarray) -> BitsRaster:
        """Rasterize ``map_`` around the agent frame defined by ``agent_from_world``."""

        raster_from_agent = self.raster_from_agent()
        agent_from_raster = np.linalg.inv(raster_from_agent)
        raster_from_world = raster_from_agent @ agent_from_world

        size = self.config.raster_size
        image = np.zeros((3, size, size), dtype=np.float32)
        drivable_map = np.zeros((size, size), dtype=bool)

        for lane in map_.lanes.values():
            lane_mask = np.zeros_like(drivable_map)
            self._fill_geometry(lane_mask, getattr(lane, "geometry", None), raster_from_world)
            if lane_mask.any():
                drivable_map |= lane_mask
                image[0, lane_mask] = 1.0

            centerline = lane.centerline()
            if centerline is not None:
                self._draw_line(image[1], centerline.coords, raster_from_world)

        for area in map_.areas.values():
            subtype = getattr(area, "subtype", None)
            if subtype in self.DRIVABLE_AREA_SUBTYPES:
                area_mask = np.zeros_like(drivable_map)
                self._fill_geometry(area_mask, getattr(area, "geometry", None), raster_from_world)
                if area_mask.any():
                    drivable_map |= area_mask
                    image[0, area_mask] = 1.0
            elif subtype in self.PEDESTRIAN_AREA_SUBTYPES:
                self._fill_geometry(image[2], getattr(area, "geometry", None), raster_from_world)

        for roadline in map_.roadlines.values():
            width = self._line_width_pixels(getattr(roadline, "width", None))
            self._draw_line(image[1], roadline.geometry.coords, raster_from_world, width)

        return BitsRaster(
            image=image,
            drivable_map=drivable_map,
            raster_from_agent=raster_from_agent,
            agent_from_raster=agent_from_raster,
            static_image=image,
        )

    def rasterize_agents(
        self,
        ego_history_positions: np.ndarray,
        ego_history_yaws: np.ndarray,
        ego_history_availabilities: np.ndarray,
        ego_extent: np.ndarray,
        other_history_positions: np.ndarray,
        other_history_yaws: np.ndarray,
        other_history_availabilities: np.ndarray,
        other_extents: np.ndarray,
        raster_from_agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Rasterize ego and neighboring agents' history in the current ego frame."""

        transform = raster_from_agent if raster_from_agent is not None else self.raster_from_agent()
        time_steps = ego_history_positions.shape[0]
        size = self.config.raster_size
        dynamic_image = np.zeros((time_steps, size, size), dtype=np.float32)

        for step in range(time_steps):
            for agent_index in range(other_history_positions.shape[0]):
                if not bool(other_history_availabilities[agent_index, step]):
                    continue
                extent = other_extents[agent_index]
                if not np.any(extent):
                    continue
                self._paint_agent_box(
                    dynamic_image[step],
                    other_history_positions[agent_index, step],
                    self._yaw_scalar(other_history_yaws[agent_index, step]),
                    extent,
                    transform,
                    value=-1.0,
                )

            if bool(ego_history_availabilities[step]):
                self._paint_agent_box(
                    dynamic_image[step],
                    ego_history_positions[step],
                    self._yaw_scalar(ego_history_yaws[step]),
                    ego_extent,
                    transform,
                    value=1.0,
                )

        return dynamic_image

    def attach_agent_history(
        self,
        raster: BitsRaster,
        ego_history_positions: np.ndarray,
        ego_history_yaws: np.ndarray,
        ego_history_availabilities: np.ndarray,
        ego_extent: np.ndarray,
        other_history_positions: np.ndarray,
        other_history_yaws: np.ndarray,
        other_history_availabilities: np.ndarray,
        other_extents: np.ndarray,
    ) -> BitsRaster:
        """Prepend dynamic history channels to a static map raster."""

        dynamic_image = self.rasterize_agents(
            ego_history_positions=ego_history_positions,
            ego_history_yaws=ego_history_yaws,
            ego_history_availabilities=ego_history_availabilities,
            ego_extent=ego_extent,
            other_history_positions=other_history_positions,
            other_history_yaws=other_history_yaws,
            other_history_availabilities=other_history_availabilities,
            other_extents=other_extents,
            raster_from_agent=raster.raster_from_agent,
        )
        static_image = raster.static_image if raster.static_image is not None else raster.image
        image = np.concatenate([dynamic_image, static_image], axis=0)
        return BitsRaster(
            image=image,
            drivable_map=raster.drivable_map,
            raster_from_agent=raster.raster_from_agent,
            agent_from_raster=raster.agent_from_raster,
            static_image=static_image,
            dynamic_image=dynamic_image,
        )

    def raster_from_agent(self) -> np.ndarray:
        """Return the transform from BITS agent coordinates to raster pixels.

        Agent coordinates use x forward and y left. Raster coordinates use
        column right and row down, with the ego position placed at one quarter
        of the image width and halfway down the image.
        """

        size = float(self.config.raster_size)
        pixels_per_meter = 1.0 / float(self.config.pixel_size)
        return np.asarray(
            [
                [pixels_per_meter, 0.0, 0.25 * size],
                [0.0, -pixels_per_meter, 0.5 * size],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _fill_geometry(self, mask: np.ndarray, geometry, transform: np.ndarray) -> None:
        if geometry is None:
            return
        if isinstance(geometry, Polygon):
            exterior = self._transform_coords(geometry.exterior.coords, transform)
            holes = [
                self._transform_coords(interior.coords, transform)
                for interior in geometry.interiors
            ]
            self._fill_polygon(mask, exterior, holes)
        elif isinstance(geometry, LinearRing):
            self._fill_polygon(mask, self._transform_coords(geometry.coords, transform))
        elif isinstance(geometry, LineString):
            self._draw_line(mask, geometry.coords, transform)
        elif hasattr(geometry, "exterior"):
            self._fill_polygon(mask, self._transform_coords(geometry.exterior.coords, transform))
        elif hasattr(geometry, "coords"):
            self._draw_line(mask, geometry.coords, transform)

    def _fill_polygon(
        self, mask: np.ndarray, exterior: np.ndarray, holes: Optional[Iterable[np.ndarray]] = None
    ) -> None:
        if exterior.shape[0] < 3:
            return

        polygon_mask = np.zeros_like(mask, dtype=bool)
        self._fill_ring(polygon_mask, exterior, True)
        for hole in holes or ():
            self._fill_ring(polygon_mask, hole, False)
        mask[...] = np.maximum(mask, polygon_mask).astype(mask.dtype, copy=False)

    def _fill_ring(self, mask: np.ndarray, coords: np.ndarray, value: bool) -> None:
        if coords.shape[0] < 3:
            return

        height, width = mask.shape[-2:]
        min_col = max(0, int(np.floor(np.min(coords[:, 0]))))
        max_col = min(width, int(np.ceil(np.max(coords[:, 0]))) + 1)
        min_row = max(0, int(np.floor(np.min(coords[:, 1]))))
        max_row = min(height, int(np.ceil(np.max(coords[:, 1]))) + 1)
        if min_col >= max_col or min_row >= max_row:
            return

        cols = np.arange(min_col, max_col, dtype=float) + 0.5
        rows = np.arange(min_row, max_row, dtype=float) + 0.5
        grid_cols, grid_rows = np.meshgrid(cols, rows)
        points = np.column_stack([grid_cols.ravel(), grid_rows.ravel()])
        inside = (
            Path(coords).contains_points(points).reshape((max_row - min_row, max_col - min_col))
        )
        window = mask[min_row:max_row, min_col:max_col]
        window[inside] = value

    def _draw_line(
        self,
        mask: np.ndarray,
        coords: Sequence[Sequence[float]],
        transform: np.ndarray,
        width_pixels: int = 1,
    ) -> None:
        points = self._transform_coords(coords, transform)
        if points.shape[0] < 2:
            return

        radius = max(0, int(width_pixels) // 2)
        for start, end in zip(points[:-1], points[1:]):
            delta = end - start
            length = float(np.hypot(delta[0], delta[1]))
            sample_count = max(2, int(np.ceil(length * 2.0)) + 1)
            samples = np.linspace(start, end, sample_count)
            pixel_cols = np.rint(samples[:, 0]).astype(int)
            pixel_rows = np.rint(samples[:, 1]).astype(int)
            self._paint_pixels(mask, pixel_cols, pixel_rows, radius)

    def _paint_agent_box(
        self,
        mask: np.ndarray,
        position: Sequence[float],
        yaw: float,
        extent: Sequence[float],
        raster_from_agent: np.ndarray,
        value: float,
    ) -> None:
        box = self._agent_box_corners(position, yaw, extent)
        raster_box = self._transform_coords(box, raster_from_agent)
        box_mask = np.zeros_like(mask, dtype=bool)
        self._fill_polygon(box_mask, raster_box)
        mask[box_mask] = value

    @staticmethod
    def _agent_box_corners(
        position: Sequence[float], yaw: float, extent: Sequence[float]
    ) -> np.ndarray:
        length, width = float(extent[0]), float(extent[1])
        half_length = max(0.0, length) * 0.5
        half_width = max(0.0, width) * 0.5
        corners = np.asarray(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ],
            dtype=float,
        )
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
        return corners @ rotation.T + np.asarray(position, dtype=float)[:2]

    @staticmethod
    def _paint_pixels(mask: np.ndarray, cols: np.ndarray, rows: np.ndarray, radius: int) -> None:
        height, width = mask.shape[-2:]
        for col, row in zip(cols, rows):
            col_min = max(0, col - radius)
            col_max = min(width, col + radius + 1)
            row_min = max(0, row - radius)
            row_max = min(height, row + radius + 1)
            if col_min < col_max and row_min < row_max:
                mask[row_min:row_max, col_min:col_max] = 1

    @staticmethod
    def _transform_coords(coords: Sequence[Sequence[float]], transform: np.ndarray) -> np.ndarray:
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        points = points[:, :2]
        homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=float)])
        transformed = (transform @ homogeneous.T).T
        return transformed[:, :2]

    def _line_width_pixels(self, width_meters: Optional[float]) -> int:
        if width_meters is None:
            return 1
        return max(1, int(round(float(width_meters) / self.config.pixel_size)))

    @staticmethod
    def _yaw_scalar(yaw) -> float:
        return float(np.asarray(yaw, dtype=float).reshape(-1)[0])
