# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pygame render manager implementation."""


import logging
import os
import warnings
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
import pygame
from shapely.geometry import Point

from tactics2d.display.sensor.camera import BEVCamera
from tactics2d.display.sensor.sensor_base import SensorBase

from .renderer import PygameRenderer


class LayoutStyle(Enum):
    HIERARCHY = 1
    BLOCK = 2


class RenderManager:
    """Manage pygame rendering for one or more (sensor, renderer) pairs.

    Sensors  (:class:`~tactics2d.display.sensor.sensor_base.SensorBase`)
    perform pure computation and produce *geometry_data* dicts.  A
    :class:`PygameRenderer` is created automatically for each sensor
    and draws the *geometry_data* onto a ``pygame.Surface``.
    """

    layout_styles = {"hierarchy", "block"}

    def __init__(
        self,
        fps: int = 60,
        windows_size: Tuple[int, int] = (800, 800),
        layout_style: str = "hierarchy",
        off_screen: bool = False,
    ):
        self.fps = fps
        self.windows_size = windows_size
        self.off_screen = off_screen

        if layout_style not in self.layout_styles:
            raise ValueError(f"Layout style must be one of {self.layout_styles}.")
        self.layout_style = (
            LayoutStyle.HIERARCHY if layout_style == "hierarchy" else LayoutStyle.BLOCK
        )

        if off_screen and "SDL_VIDEODRIVER" not in os.environ and "DISPLAY" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        flags = pygame.HIDDEN if self.off_screen else pygame.SHOWN
        pygame.init()
        self._clock = pygame.time.Clock()
        self._screen = pygame.display.set_mode(size=self.windows_size, flags=flags)

        self._sensors: dict = dict()
        self._renderers: dict = dict()
        self._off_screen_flags: dict = dict()
        self._bound_sensors: dict = dict()
        self._layouts: dict = dict()
        self._obs_callbacks: dict = dict()

    @property
    def graphic_driver(self) -> str:
        return pygame.display.get_driver()

    def _rearrange_layout(self):
        sensor_to_display = [
            sid
            for sid, sensor in self._sensors.items()
            if not self._off_screen_flags.get(sid, True)
        ]

        if self.layout_style == LayoutStyle.HIERARCHY:
            if not hasattr(self, "main_sensor"):
                self.main_sensor = list(self._sensors.keys())[0]

            n = 3 if len(sensor_to_display) < 4 else len(sensor_to_display) - 1
            sub_cnt = 0
            for sid in sensor_to_display:
                renderer = self._renderers[sid]
                if sid == self.main_sensor:
                    scale = min(
                        self.windows_size[0] / renderer.window_size[0],
                        self.windows_size[1] / renderer.window_size[1],
                    )
                    coords = (0.5 * (self.windows_size[0] - scale * renderer.window_size[0]), 0)
                else:
                    sub_width = self.windows_size[0] / n - 10
                    sub_height = self.windows_size[1] / n - 10
                    scale = min(
                        sub_width / renderer.window_size[0], sub_height / renderer.window_size[1]
                    )
                    coords = (sub_cnt * (sub_width + 10) + 5, self.windows_size[1] - sub_height + 5)
                    sub_cnt += 1

                self._layouts[sid] = (scale, coords)

        elif self.layout_style == LayoutStyle.BLOCK:
            n_sensors = len(sensor_to_display)
            n = int(np.ceil(np.sqrt(n_sensors)))
            width = self.windows_size[0] / n
            height = self.windows_size[1] / np.ceil(n_sensors / n)
            for i, sid in enumerate(sensor_to_display):
                renderer = self._renderers[sid]
                scale = min(width / renderer.window_size[0], height / renderer.window_size[1])
                coords = (
                    (i % n) * width + (width - renderer.window_size[0] * scale) / 2,
                    (i // n) * height + (height - renderer.window_size[1] * scale) / 2,
                )
                self._layouts[sid] = (scale, coords)

    def add_sensor(
        self,
        sensor: SensorBase,
        window_size: Tuple[int, int] = (200, 200),
        off_screen: bool = True,
        main_sensor: bool = False,
        observation_callback: Optional[Callable[[], np.ndarray]] = None,
    ):
        """Register a sensor with an automatically created PygameRenderer.

        Args:
            sensor: Computation sensor (BEVCamera, SingleLineLidar, …).
            window_size: (width, height) in pixels of the render surface.
            off_screen: Whether to skip displaying this sensor's surface.
            main_sensor: Whether this is the primary (largest) sensor in
                hierarchy layout.
            observation_callback: Callable that returns the observation
                array for this sensor.  Defaults to the renderer's
                ``get_observation`` (the rendered image) for cameras,
                and the sensor's ``get_observation`` (raw scan result)
                for LiDAR sensors.
        """
        sensor_id = sensor.id_
        if sensor_id in self._sensors:
            raise KeyError(f"ID {sensor_id} is already registered with the render manager.")

        # Create the PygameRenderer.
        pr = sensor.perception_range
        perception_width = pr[0] + pr[1]
        perception_height = pr[2] + pr[3]
        scale_w = window_size[0] / perception_width
        scale_h = window_size[1] / perception_height
        scale = max(scale_w, scale_h)

        if scale_w != scale_h:
            logging.warning(
                "The x-y proportion of the perception range and the rendering window "
                "is inconsistent."
            )

        map_boundary = sensor.map_.boundary if sensor.map_ is not None else None
        renderer = PygameRenderer(window_size, scale, pr, map_boundary)

        self._sensors[sensor_id] = sensor
        self._renderers[sensor_id] = renderer
        self._off_screen_flags[sensor_id] = off_screen

        # Determine observation source.
        if observation_callback is not None:
            self._obs_callbacks[sensor_id] = observation_callback
        else:
            # Default: camera → rendered image; LiDAR → raw scan.
            if isinstance(sensor, BEVCamera):
                self._obs_callbacks[sensor_id] = renderer.get_observation
            else:
                self._obs_callbacks[sensor_id] = sensor.get_observation

        if main_sensor:
            self.main_sensor = sensor_id

        if not off_screen:
            self._rearrange_layout()

    def remove_sensor(self, id_: int):
        try:
            self._sensors.pop(id_)
        except KeyError:
            warnings.warn(f"Sensor {id_} does not exist.")
            return

        self._renderers.pop(id_, None)
        self._off_screen_flags.pop(id_, None)
        self._obs_callbacks.pop(id_, None)

        if id_ in self._bound_sensors:
            self.unbind(id_)

        if id_ in self._layouts:
            self._layouts.pop(id_)

    def is_bound(self, id_) -> bool:
        return id_ in self._bound_sensors

    def get_bind_id(self, id_) -> int:
        return self._bound_sensors.get(id_)

    def bind(self, id_: int, participant_id: int):
        if id_ not in self._sensors:
            raise KeyError(f"Sensor {id_} is not registered in the render manager.")

        if id_ in self._bound_sensors:
            warnings.warn(
                f"Sensor {id_} was bound with participant "
                f"{self._bound_sensors[id_]}. Now it is bound with {participant_id}."
            )

        self._sensors[id_].bind_with(participant_id)
        self._bound_sensors[id_] = participant_id

    def unbind(self, id_):
        try:
            self._bound_sensors.pop(id_)
            self._sensors[id_].bind_with(None)
        except KeyError:
            warnings.warn(f"Sensor {id_} is not bound with any participant.")

    def update(self, participants: dict, participant_ids: list, frame: int = None):
        to_remove = []
        for sensor_id, sensor in self._sensors.items():
            renderer = self._renderers[sensor_id]

            if sensor_id in self._bound_sensors:
                participant_id = self._bound_sensors[sensor_id]
                if participant_id not in participants:
                    self.unbind(sensor_id)
                    to_remove.append(sensor_id)
                    continue

                participant = participants[participant_id]
                try:
                    state = participant.trajectory.get_state(frame)
                    position = Point(state.location)
                    heading = state.heading
                except KeyError:
                    self.unbind(sensor_id)
                    to_remove.append(sensor_id)
                    continue
            else:
                position = None
                heading = None

            # Compute geometry_data via the sensor.
            geometry_data, _, _ = sensor.update(
                frame, participants, participant_ids, position=position, heading=heading
            )

            # Render via the PygameRenderer.
            background_color = None
            metadata = geometry_data.get("metadata", {})
            if metadata.get("sensor_type") == "lidar":
                background_color = (0, 0, 0)

            renderer.render(geometry_data, position, heading, background_color)

        for sensor_id in to_remove:
            self.remove_sensor(sensor_id)

    def render(self):
        self._clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        blit_sequence = []
        for sensor_id, layout_info in self._layouts.items():
            renderer = self._renderers[sensor_id]
            surface = pygame.transform.scale_by(renderer.surface, layout_info[0])
            blit_sequence.append((surface, layout_info[1]))

        if self._screen is not None:
            self._screen.blits(blit_sequence)
        pygame.display.flip()

    def get_observation(self) -> list:
        return [cb() for cb in self._obs_callbacks.values()]

    def reset(self):
        sensor_ids = list(self._sensors.keys())
        for sensor_id in sensor_ids:
            self.remove_sensor(sensor_id)

        self._sensors.clear()
        self._renderers.clear()
        self._off_screen_flags.clear()
        self._bound_sensors.clear()
        self._layouts.clear()
        self._obs_callbacks.clear()

    def close(self):
        pygame.quit()
