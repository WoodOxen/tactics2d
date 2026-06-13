# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser display backend — sends SceneSnapshot to a browser frontend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..backend import DisplayBackend
from ..renderers.web.renderer import FrontendRenderer, start_server_process, stop_server_process
from ..snapshot import SceneSnapshot

LOGGER = logging.getLogger(__name__)


class BrowserBackend(DisplayBackend):
    """Browser display backend.

    Sends :class:`SceneSnapshot` data to a running browser frontend server
    via HTTP.  If the server is not running and ``auto_start_server`` is
    ``True``, it will be started automatically in a background process.

    The browser frontend can also be started independently via ``tactics2d start``.

    Example::

        backend = BrowserBackend(auto_start_server=True)
        backend.reset()
        backend.render(snapshot)   # sends snapshot via HTTP
        backend.close()
    """

    backend_name = "browser"
    supports_rgb_array = False
    supports_interactive = True
    is_headless = True

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_fps: int = 30,
        auto_start_server: bool = True,
        open_browser: bool = False,
        pid_file: Path | None = None,
        timeout: float = 5.0,
    ):
        self._host = host
        self._port = port
        self._max_fps = max_fps
        self._auto_start_server = auto_start_server
        self._open_browser = open_browser
        self._pid_file = pid_file
        self._timeout = timeout

        self._renderer: FrontendRenderer | None = None
        self._server_process = None
        self._did_start_server = False

    # ------------------------------------------------------------------
    # DisplayBackend interface
    # ------------------------------------------------------------------

    def reset(self, snapshot: SceneSnapshot | None = None) -> None:
        """Ensure the server is running and optionally send an initial frame.

        Args:
            snapshot: Optional initial snapshot to render immediately.
        """
        self._ensure_server()
        if snapshot is not None:
            self.render(snapshot)

    def render(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        """Send the snapshot to the frontend server.

        Args:
            snapshot: The scene snapshot to send.

        Returns:
            Always ``None`` (the browser handles the actual rendering).
        """
        if self._renderer is None:
            LOGGER.warning("BrowserBackend not connected; call reset() first.")
            return None

        sensor_data = self._snapshot_to_sensors(snapshot)
        self._renderer.send_frame(sensor_data, frame=snapshot.frame)
        return None

    def close(self) -> None:
        """Disconnect from the server.

        If this backend started the server, it will be stopped as well.
        """
        self._renderer = None
        if self._did_start_server and self._pid_file:
            try:
                stop_server_process(self._pid_file)
                LOGGER.info("Stopped frontend server process.")
            except Exception:
                LOGGER.warning("Failed to stop frontend server process.")
            finally:
                self._did_start_server = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_server(self) -> None:
        """Connect to an existing server or start a new one."""
        if self._renderer is not None:
            return

        renderer = FrontendRenderer(
            self._host, self._port, max_fps=self._max_fps, timeout=self._timeout
        )

        # Try to connect to an already running server
        if renderer.wait_until_ready(timeout=0.5):
            self._renderer = renderer
            LOGGER.info("Connected to existing frontend server at %s:%s.", self._host, self._port)
            return

        # Start a new server if allowed
        if self._auto_start_server:
            LOGGER.info("Starting frontend server at %s:%s...", self._host, self._port)
            self._server_process = start_server_process(
                self._host,
                self._port,
                demo=False,
                max_fps=self._max_fps,
                open_browser=self._open_browser,
                pid_file=self._pid_file,
            )
            if renderer.wait_until_ready(timeout=self._timeout):
                self._renderer = renderer
                self._did_start_server = True
                LOGGER.info("Frontend server started at %s:%s.", self._host, self._port)
                return

            raise RuntimeError(
                f"Frontend server did not start on {self._host}:{self._port} "
                f"within {self._timeout}s."
            )

        raise RuntimeError(
            f"No frontend server found on {self._host}:{self._port} "
            f"and auto_start_server is False."
        )

    @staticmethod
    def _snapshot_to_sensors(snapshot: SceneSnapshot) -> list[dict[str, Any]]:
        """Convert a SceneSnapshot to the sensor data list expected by FrontendRenderer.send_frame()."""
        sensors = []

        # Build map_data
        road_elements = []
        for road_id, re in snapshot.road_elements.items():
            element = {"id": road_id, "shape": re.shape, "type": re.type_, "geometry": re.geometry}
            if re.color:
                element["color"] = re.color
            if re.line_width != 1.0:
                element["line_width"] = re.line_width
            if re.line_style:
                element["line_style"] = re.line_style
            road_elements.append(element)

        map_data = {
            "road_id_to_remove": snapshot.road_ids_to_remove,
            "road_elements": road_elements,
        }

        # Build participant_data
        participants_list = []
        for pid, pe in snapshot.participants.items():
            participant = {
                "id": pid,
                "shape": pe.shape,
                "type": pe.type_,
                "geometry": pe.geometry,
                "position": list(pe.position),
                "rotation": pe.rotation,
            }
            if pe.color:
                participant["color"] = pe.color
            if pe.velocity:
                participant["velocity"] = list(pe.velocity)
            participants_list.append(participant)

        participant_data = {
            "participant_id_to_create": snapshot.participant_ids_to_create,
            "participant_id_to_remove": snapshot.participant_ids_to_remove,
            "participants": participants_list,
        }

        # Add point clouds
        if snapshot.point_clouds:
            point_clouds_list = []
            for pc in snapshot.point_clouds:
                point_clouds_list.append(
                    {
                        "id": pc.id_,
                        "points": pc.points,
                        "color": pc.color,
                        "point_size": pc.point_size,
                        "alpha": pc.alpha,
                    }
                )
            participant_data["point_clouds"] = point_clouds_list

        # Build metadata
        metadata = {"sensor_position": [0.0, 0.0], "sensor_yaw": 0.0, "perception_range": None}
        if snapshot.cameras:
            cam = snapshot.cameras[0]
            metadata["sensor_position"] = list(cam.position)
            metadata["sensor_yaw"] = cam.yaw
            metadata["perception_range"] = cam.perception_range

        geometry_data = {
            "metadata": metadata,
            "map_data": map_data,
            "participant_data": participant_data,
        }

        sensor_entry = {
            "id": "camera_0",
            "perception_range": metadata["perception_range"] or 80,
            "position": metadata["sensor_position"],
            "yaw": metadata["sensor_yaw"],
            "frame": snapshot.frame,
            "map_data": map_data,
            "participant_data": participant_data,
        }

        sensors.append(sensor_entry)
        return sensors
