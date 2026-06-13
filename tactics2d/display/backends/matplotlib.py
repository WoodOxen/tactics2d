# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Matplotlib display backend — wraps MatplotlibRenderer as a DisplayBackend."""

from __future__ import annotations

from typing import Any

import numpy as np

from tactics2d.display.renderers import MatplotlibRenderer

from ..backend import DisplayBackend
from ..snapshot import SceneSnapshot


class MatplotlibBackend(DisplayBackend):
    """Matplotlib display backend.

    Wraps :class:`~tactics2d.display.renderers.MatplotlibRenderer` to produce RGB
    arrays or saved images from a :class:`SceneSnapshot`.

    This backend is **headless** by default (uses the Agg backend internally
    from ``MatplotlibRenderer``).

    Example::

        backend = MatplotlibBackend(resolution=(800, 600))
        rgb = backend.render(snapshot)   # (600, 800, 3) ndarray
        backend.save_frame("frame.png")
        backend.close()
    """

    backend_name = "matplotlib"
    supports_rgb_array = True
    supports_interactive = True
    is_headless = True

    def __init__(
        self,
        resolution: tuple[float, float] = (800, 600),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        dpi: int = 200,
        auto_scale: bool = True,
    ):
        self._resolution = resolution
        self._xlim = xlim
        self._ylim = ylim
        self._dpi = dpi
        self._auto_scale = auto_scale

        self._renderer: MatplotlibRenderer | None = None

    # ------------------------------------------------------------------
    # DisplayBackend interface
    # ------------------------------------------------------------------

    def reset(self, snapshot: SceneSnapshot | None = None) -> None:
        """Reset the matplotlib renderer.

        Args:
            snapshot: Optional snapshot to render as initial frame.
        """
        self.close()
        self._renderer = MatplotlibRenderer(
            resolution=self._resolution,
            xlim=self._xlim,
            ylim=self._ylim,
            dpi=self._dpi,
            auto_scale=self._auto_scale,
        )
        if snapshot is not None:
            self.render(snapshot)

    def render(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        """Render the snapshot and return an RGB array.

        Args:
            snapshot: The scene snapshot to render.

        Returns:
            (H, W, 3) RGB array.
        """
        if self._renderer is None:
            return None

        geometry_data = self._snapshot_to_geometry_data(snapshot)
        self._renderer.update(geometry_data)
        return self._renderer.save_single_frame(return_array=True)

    def close(self) -> None:
        """Destroy the matplotlib figure and release resources."""
        if self._renderer is not None:
            self._renderer.destroy()
            self._renderer = None

    def save_frame(self, path: str) -> None:
        """Save the current frame to an image file.

        Args:
            path: File path for the output image.
        """
        if self._renderer is not None:
            self._renderer.save_single_frame(save_to=path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_to_geometry_data(snapshot: SceneSnapshot) -> dict[str, Any]:
        """Convert SceneSnapshot to the geometry_data dict expected by MatplotlibRenderer."""
        # Build road_elements
        road_elements = []
        for road_id, re in snapshot.road_elements.items():
            element = {
                "id": road_id,
                "shape": re.shape,
                "type": re.type_,
                "geometry": re.geometry,
                "color": re.color or re.type_,
                "line_width": re.line_width,
            }
            if re.line_style:
                element["line_style"] = re.line_style
            road_elements.append(element)

        # Build participants
        participants = []
        for pid, pe in snapshot.participants.items():
            participant = {
                "id": pid,
                "shape": pe.shape,
                "type": pe.type_,
                "geometry": pe.geometry,
                "position": list(pe.position),
                "rotation": pe.rotation,
                "color": pe.color or pe.type_,
                "line_width": 1.0,
            }
            participants.append(participant)

        # Build point clouds
        point_clouds = []
        for pc in snapshot.point_clouds:
            point_clouds.append(
                {
                    "id": pc.id_,
                    "points": pc.points,
                    "color": pc.color,
                    "point_size": pc.point_size,
                    "alpha": pc.alpha,
                    "type": "lidar_point_cloud",
                }
            )

        # Camera position
        sensor_position = [0.0, 0.0]
        sensor_yaw = 0.0
        perception_range = None
        if snapshot.cameras:
            cam = snapshot.cameras[0]
            sensor_position = list(cam.position)
            sensor_yaw = cam.yaw
            perception_range = cam.perception_range

        return {
            "metadata": {
                "sensor_position": sensor_position,
                "sensor_yaw": sensor_yaw,
                "perception_range": perception_range,
            },
            "map_data": {
                "road_id_to_remove": snapshot.road_ids_to_remove,
                "road_elements": road_elements,
            },
            "participant_data": {
                "participant_id_to_create": snapshot.participant_ids_to_create,
                "participant_id_to_remove": snapshot.participant_ids_to_remove,
                "participants": participants,
                "point_clouds": point_clouds,
            },
        }
