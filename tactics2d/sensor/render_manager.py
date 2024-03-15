##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: render_manager.py
# @Description: This file defines the class for managing the rendering of the scenario.
# @Author: Yueyuan Li
# @Version: 1.0.0

import warnings
from enum import Enum
from typing import Tuple

import numpy as np
import pygame
from shapely.geometry import Point


class LayoutStyle(Enum):
    HIERARCHY = 1
    BLOCK = 2


class RenderManager:
    """This class manages the rendering of the scenario.

    By RenderManager, the sensors are registered and can be bound to the participants. The rendering is conducted by the pygame library.

    Attributes:
        fps (int): The frame rate of the rendering. Defaults to 60.
        windows_size (Tuple[int, int]): The size of the rendering window. Defaults to (800, 800).
        layout_style (str): The style of the layout of the rendering window. The available choices are ["hierarchy", "block"]. Defaults to "hierarchy".
        off_screen (bool): Whether to render the scenario off screen. Defaults to False.
        graphic_driver (str): The graphic driver used by the pygame library. This attribute is **read-only**.
    """

    layout_styles = {"hierarchy", "block"}

    def __init__(
        self,
        fps: int = 60,
        windows_size: Tuple[int, int] = (800, 800),
        layout_style: str = "hierarchy",
        off_screen: bool = False,
    ):
        """Initialize the render manager.

        Args:
            fps (int, optional): The frame rate of the rendering.
            windows_size (Tuple[int, int], optional): The size of the rendering window.
            layout_style (str, optional): The style of the layout of the rendering window. The available choices are ["hierarchy", "block"].
            off_screen (bool, optional): Whether to render the scenario off screen.
        """
        self.fps = fps
        self.windows_size = windows_size
        self.off_screen = off_screen

        if layout_style not in self.layout_styles:
            raise ValueError(f"Layout style must be one of {self.layout_styles}.")
        self.layout_style = (
            LayoutStyle.HIERARCHY if layout_style == "hierarchy" else LayoutStyle.BLOCK
        )

        flags = pygame.HIDDEN if self.off_screen else pygame.SHOWN
        pygame.init()
        self._clock = pygame.time.Clock()
        self._screen = pygame.display.set_mode(size=self.windows_size, flags=flags)

        self._sensors = dict()
        self._bound_sensors = dict()
        self._layouts = dict()

    @property
    def graphic_driver(self) -> str:
        return pygame.display.get_driver()

    def _rearrange_layout(self):
        sensor_to_display = [sensor for sensor in self._sensors.values() if not sensor.off_screen]

        if self.layout_style == LayoutStyle.HIERARCHY:
            if not hasattr(self, "main_sensor"):
                self.main_sensor = list(self._sensors.keys())[0]

            n = 3 if len(sensor_to_display) < 4 else len(sensor_to_display) - 1
            sub_cnt = 0
            for sensor in sensor_to_display:
                if sensor.id_ == self.main_sensor:
                    scale = min(
                        self.windows_size[0] / sensor.window_size[0],
                        self.windows_size[1] / sensor.window_size[1],
                    )
                    coords = (0.5 * (self.windows_size[0] - scale * sensor.window_size[0]), 0)
                else:
                    sub_width = self.windows_size[0] / n - 10
                    sub_height = self.windows_size[1] / n - 10
                    scale = min(
                        sub_width / sensor.window_size[0], sub_height / sensor.window_size[1]
                    )
                    coords = (sub_cnt * (sub_width + 10) + 5, self.windows_size[1] - sub_height + 5)
                    sub_cnt += 1

                self._layouts[sensor.id_] = (scale, coords)

        elif self.layout_style == LayoutStyle.BLOCK:
            n = int(np.ceil(np.sqrt(len(sensor_to_display))))
            width = self.windows_size[0] / n
            height = self.windows_size[1] / np.ceil(len(sensor_to_display) / n)
            for i, sensor in enumerate(self._sensors.values()):
                scale = min(width / sensor.window_size[0], height / sensor.window_size[1])
                coords = (
                    (i % n) * width + (width - sensor.window_size[0] * scale) / 2,
                    (i // n) * height + (height - sensor.window_size[1] * scale) / 2,
                )
                self._layouts[sensor.id_] = (scale, coords)

    def add_sensor(self, sensor, main_sensor: bool = False):
        """This function adds a sensor to the render manager.

        Args:
            sensor (SensorBase): The sensor instance to be added.
            main_sensor (bool, optional): Whether the sensor is the main sensor for display. This argument only take effect when the the layout style is hierarchical.

        Raises:
            KeyError: If the sensor has conflicted id with registered sensors.
        """

        if sensor.id_ not in self._sensors:
            self._sensors[sensor.id_] = sensor
        else:
            raise KeyError(f"ID {sensor.id_} is used by the other sensor.")

        if main_sensor:
            self.main_sensor = sensor.id_

        if not sensor.off_screen:
            self._rearrange_layout()

    def remove_sensor(self, id_: int):
        """This function removes a sensor from the manager.

        Args:
            id_ (int): The id of the sensor.
        """
        try:
            self._sensors.pop(id_)
        except KeyError:
            warnings.warn(f"Sensor {id_} does not exist.")

        if id_ in self._bound_sensors:
            self.unbind(id_)

        if id_ in self._layouts:
            self._layouts.pop(id_)

    def is_bound(self, id_) -> bool:
        """This function checks whether the sensor is bound to a participant.

        Args:
            id_ (int): The id of the sensor.

        Returns:
            If the sensor is bound to a participant, return True. Otherwise, return False.
        """
        return id_ in self._bound_sensors

    def get_bind_id(self, id_) -> int:
        """This function gets the id of the participant that the sensor is bound to.

        Args:
            id_ (int): The id of the sensor.

        Returns:
            The id of the participant that the sensor is bound to. If the sensor is not bound to any participant, return None.
        """
        return self._bound_sensors.get(id_)

    def bind(self, id_: int, participant_id: int):
        """This function binds a registered sensor with a participant.

        Args:
            id_ (int): The id of the sensor.
            participant_id (int): The id of the participant.

        Raises:
            KeyError: If the sensor is not registered in the manager.
        """

        if id_ not in self._sensors:
            raise KeyError(f"Sensor {id_} is not registered in the render manager.")

        if id_ in self._bound_sensors:
            warnings.warn(
                f"Sensor {id_} was bound with participant \
                {self._bound_sensors[id_]}. Now it is bound with {participant_id}."
            )

        self._sensors[id_].set_bind_id(participant_id)
        self._bound_sensors[id_] = participant_id

    def unbind(self, id_):
        """This function unbinds a registered sensor from its bound participant.

        Args:
            id_ (int): The id of the sensor.
        """

        try:
            self._bound_sensors.pop(id_)
            self._sensors[id_].set_bind_id(None)
        except KeyError:
            warnings.warn(f"Sensor {id_} is not bound with any participant.")

    def update(self, participants: dict, participant_ids: list, frame: int = None):
        """Sync the viewpoint of the sensors with their bound participants. Update the
            observation of all the sensors.

        Args:
            participants (dict): The dictionary of all participants. The render manager will detect which of them is alive.
            frame (int): Update the sensors to the given frame. If None, the sensors will update to the current frame. The default unit is millisecond.
        """
        to_remove = []
        for id_, sensor in self._sensors.items():
            if id_ in self._bound_sensors:
                participant = participants[self._bound_sensors[id_]]
                try:
                    state = participant.trajectory.get_state(frame)
                    sensor.update(
                        participants, participant_ids, frame, Point(state.location), state.heading
                    )
                except KeyError:
                    self.unbind(id_)
                    to_remove.append(id_)
            else:
                sensor.update(participants, participant_ids, frame)

        for id_ in to_remove:
            self.remove_sensor(id_)

    def render(self):
        """Render the observation of all the sensors."""

        self._clock.tick(self.fps)

        blit_sequence = []
        for id_, layout_info in self._layouts.items():
            surface = pygame.transform.scale_by(self._sensors[id_].surface, layout_info[0])
            blit_sequence.append((surface, layout_info[1]))

        if self._screen is not None:
            self._screen.blits(blit_sequence)
        pygame.display.flip()

    def get_observation(self) -> list:
        """Get the observation of all the sensors.

        Returns:
            list: A list of 3d arrays.
        """
        observations = []
        for sensor in self._sensors.values():
            observations.append(sensor.get_observation())

        return observations

    def reset(self):
        """Reset the render manager."""
        sensor_ids = list(self._sensors.keys())
        for id_ in sensor_ids:
            self.remove_sensor(id_)

        self._sensors = dict()
        self._bound_sensors = dict()
        self._layouts = dict()

    def close(self):
        """Close the render manager."""
        pygame.quit()
