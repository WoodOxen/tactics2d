# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pygame display backend — wraps RenderManager as a DisplayBackend."""

from __future__ import annotations

from typing import Any

import numpy as np

from tactics2d.display.renderers import RenderManager
from tactics2d.display.sensor import BEVCamera, SingleLineLidar

from ..backend import DisplayBackend
from ..snapshot import ParticipantElement, SceneSnapshot


class PygameBackend(DisplayBackend):
    """Pygame display backend.

    Wraps :class:`~tactics2d.display.renderers.RenderManager` with
    :class:`~tactics2d.display.sensor.BEVCamera`  (and optionally
    :class:`~tactics2d.display.sensor.SingleLineLidar`) to produce
    either an on-screen window (``render_mode="human"``) or an off-screen
    RGB array (``render_mode="rgb_array"``).

    Example::

        backend = PygameBackend(off_screen=True, window_size=(200, 200))
        backend.reset()
        rgb = backend.render(snapshot)   # (200, 200, 3) ndarray
        backend.close()
    """

    backend_name = "pygame"
    supports_rgb_array = True
    supports_interactive = True
    is_headless = False

    def __init__(
        self,
        off_screen: bool = False,
        fps: int = 60,
        window_size: tuple[int, int] = (500, 500),
        sensor_window_size: tuple[int, int] = (200, 200),
        lidar_enabled: bool = False,
        lidar_range: float = 20.0,
        lidar_lines: int = 360,
    ):
        self._off_screen = off_screen
        self._fps = fps
        self._window_size = window_size
        self._sensor_window_size = sensor_window_size
        self._lidar_enabled = lidar_enabled
        self._lidar_range = lidar_range
        self._lidar_lines = lidar_lines

        self._manager: RenderManager | None = None
        self._camera: BEVCamera | None = None
        self._lidar: SingleLineLidar | None = None

    # ------------------------------------------------------------------
    # DisplayBackend interface
    # ------------------------------------------------------------------

    def reset(self, snapshot: SceneSnapshot | None = None) -> None:
        """Reset the render manager and re-create sensors.

        Args:
            snapshot: Optional snapshot with camera metadata to configure sensors.
        """
        self.close()

        self._manager = RenderManager(
            fps=self._fps, windows_size=self._window_size, off_screen=self._off_screen
        )

        perception_range = (20, 20, 20, 20)  # default
        if snapshot and snapshot.cameras:
            cam = snapshot.cameras[0]
            perception_range = (
                (cam.perception_range,) * 4
                if isinstance(cam.perception_range, (int, float))
                else cam.perception_range
            )

        self._camera = BEVCamera(
            id_=0,
            map_=None,  # will be set via the first render call
            perception_range=perception_range,
        )
        self._manager.add_sensor(
            self._camera,
            window_size=self._sensor_window_size,
            off_screen=self._off_screen,
            main_sensor=True,
        )
        self._manager.bind(0, 0)

        if self._lidar_enabled:
            self._lidar = SingleLineLidar(
                id_=1,
                map_=None,
                perception_range=self._lidar_range,
                freq_detect=self._lidar_lines * 10,
            )
            self._manager.add_sensor(
                self._lidar, window_size=self._sensor_window_size, off_screen=self._off_screen
            )
            self._manager.bind(1, 0)

    def render(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        """Render the snapshot.

        For ``off_screen=False``, the result is displayed in a pygame window.
        For ``off_screen=True``, the result is returned as an RGB array.

        Args:
            snapshot: The scene snapshot to render.

        Returns:
            (H, W, 3) RGB array if off_screen, otherwise ``None``.
        """
        if self._manager is None:
            return None

        # Build a participants-like dict from the snapshot for the manager
        # (this is only used for bound sensor position tracking)
        participants = self._snapshot_to_participants(snapshot)
        participant_ids = list(participants.keys())

        # We need to work around the fact that the sensor needs a map_ reference.
        # Currently snapshot doesn't include a map_ object, so we render via the
        # manager.get_observation() path instead.
        if self._off_screen:
            # For off-screen mode, just return the camera observation if available
            # The manager.update + manager.render() combo is needed for pygame
            pass

        # For interactive mode, still blit to screen
        self._manager.update(participants, participant_ids, snapshot.frame)
        self._manager.render()

        if self._off_screen:
            obs_list = self._manager.get_observation()
            if obs_list:
                return obs_list[0]
        return None

    def close(self) -> None:
        """Release pygame resources."""
        if self._manager is not None:
            self._manager.close()
            self._manager = None
        self._camera = None
        self._lidar = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_to_participants(snapshot: SceneSnapshot) -> dict[int, Any]:
        """Build a minimal participants-like dict from SceneSnapshot.

        This creates a duck-typed participant with enough interface to satisfy
        ``BEVCamera.update()`` trajectory lookups.
        """
        participants: dict[int, Any] = {}
        for pid, pe in snapshot.participants.items():
            participants[pid] = _SnapshotParticipant(pe)
        return participants


class _SnapshotParticipant:
    """Minimal participant wrapper for snapshot-based rendering."""

    def __init__(self, element: ParticipantElement):
        self.id_ = element.id_
        self.length = element.length or 4.5
        self.width = element.width or 1.8

        class _State:
            x = element.position[0]
            y = element.position[1]
            heading = element.rotation
            location = (element.position[0], element.position[1])
            frame = 0

        self._state = _State()
        self.current_state = self._state
        self.trajectory = self

    def get_pose(self, frame: int = None):
        """Return footprint as a shapely polygon."""
        from shapely.geometry import Polygon

        geom = self._state_geometry()
        return Polygon(geom) if len(geom) >= 3 else Polygon()

    def _state_geometry(self):
        hw = self.width / 2
        hh = self.length / 2
        cx, cy = self._state.x, self._state.y
        cos_t = np.cos(self._state.heading)
        sin_t = np.sin(self._state.heading)
        return [
            (cx + (-hh * cos_t - hw * sin_t), cy + (-hh * sin_t + hw * cos_t)),
            (cx + (+hh * cos_t - hw * sin_t), cy + (+hh * sin_t + hw * cos_t)),
            (cx + (+hh * cos_t + hw * sin_t), cy + (+hh * sin_t - hw * cos_t)),
            (cx + (-hh * cos_t + hw * sin_t), cy + (-hh * sin_t - hw * cos_t)),
        ]

    def is_active(self, frame: int) -> bool:
        return True

    def get_state(self, frame: int):
        return self._state
