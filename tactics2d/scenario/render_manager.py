from typing import Tuple
import warnings

import numpy as np
from shapely.geometry import Point
import pygame

from tactics2d.sensor import SensorBase


class RenderManager:
    """This class manages the rendering of the scenario.

    By RenderManager, the sensors are registered and can be bound to the participants.
        The rendering is done by the pygame library.

    Attributes:
        fps (int): The frame rate of the rendering. Defaults to 60.
        windows_size (Tuple[int, int]): The size of the rendering window. Defaults to (800, 800).
        layout_style (str): The style of the layout of the rendering window. The available
            choices are ["hierarchical", "block"]. Defaults to "hierarchical".
        off_screen (bool): Whether to render the scenario off screen. Defaults to False.
    """

    layout_styles = {"hierarchical", "block"}

    def __init__(
        self,
        fps: int = 60,
        windows_size: Tuple[int, int] = (800, 800),
        layout_style: str = "hierarchical",
        off_screen: bool = False,
    ):
        self.fps = fps
        self.windows_size = windows_size
        self.off_screen = off_screen

        if layout_style not in self.layout_styles:
            raise ValueError(f"Layout style must be one of {self.layout_styles}.")
        self.layout_style = layout_style

        flags = pygame.HIDDEN if self.off_screen else pygame.SHOWN

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size=self.windows_size, flags=flags)

        self.sensors = dict()
        self.bound_sensors = dict()
        self.layouts = dict()

    @property
    def graphic_driver(self) -> str:
        return pygame.display.get_driver()

    def _rearrange_layout(self):
        sensor_to_display = [
            sensor for sensor in self.sensors.values() if not sensor.off_screen
        ]

        if self.layout_style == "hierarchical":
            if not hasattr(self, "main_sensor"):
                self.main_sensor = list(self.sensors.keys())[0]

            n = 3 if len(sensor_to_display) < 4 else len(sensor_to_display) - 1
            sub_cnt = 0
            for sensor in sensor_to_display:
                if sensor.id_ == self.main_sensor:
                    scale = min(
                        self.windows_size[0] / sensor.window_size[0],
                        self.windows_size[1] / sensor.window_size[1],
                    )
                    coords = (
                        0.5 * (self.windows_size[0] - scale * sensor.window_size[0]),
                        0,
                    )
                else:
                    sub_width = self.windows_size[0] / n - 10
                    sub_height = self.windows_size[1] / n - 10
                    scale = min(
                        sub_width / sensor.window_size[0],
                        sub_height / sensor.window_size[1],
                    )
                    coords = (
                        sub_cnt * (sub_width + 10) + 5,
                        self.windows_size[1] - sub_height + 5,
                    )
                    sub_cnt += 1

                self.layouts[sensor.id_] = (scale, coords)

        elif self.layout_style == "block":
            n = int(np.ceil(np.sqrt(len(sensor_to_display))))
            width = self.windows_size[0] / n
            height = self.windows_size[1] / np.ceil(len(sensor_to_display) / n)
            for i, sensor in enumerate(self.sensors.values()):
                scale = min(width / sensor.window_size[0], height / sensor.window_size[1])
                coords = (
                    (i % n) * width + (width - sensor.window_size[0] * scale) / 2,
                    (i // n) * height + (height - sensor.window_size[1] * scale) / 2,
                )
                self.layouts[sensor.id_] = (scale, coords)

    def add_sensor(self, sensor: SensorBase, main_sensor: bool = False):
        """Add a sensor instance to the manager.

        Args:
            sensor (SensorBase): The sensor instance to be added.
            map_ (Map): The map that the sensor belongs to.
            main_sensor (bool, optional): Whether the sensor is the main sensor for display.
                This argument only take effect when the the layout style is hierarchical.
                Defaults to False.

        Raises:
            KeyError: If the sensor has conflicted id with registered sensors.
        """

        if sensor.id_ not in self.sensors:
            self.sensors[sensor.id_] = sensor
        else:
            raise KeyError(f"ID {sensor.id_} is used by the other sensor.")

        if main_sensor:
            self.main_sensor = sensor.id_

        if not sensor.off_screen:
            self._rearrange_layout()

    def remove_sensor(self, sensor_id: int):
        """Remove a registered sensor from the manager.

        Args:
            sensor_id (int): The id of the sensor.
        """
        try:
            self.sensors.pop(sensor_id)
        except KeyError:
            warnings.warn(f"Sensor {sensor_id} does not exist.")

        if sensor_id in self.bound_sensors:
            self.unbind(sensor_id)

        if sensor_id in self.layouts:
            self.layouts.pop(sensor_id)

    def is_bound(self, sensor_id):
        """Check whether the sensor is bound to a participant.

        Args:
            sensor_id (int): The id of the sensor.

        Returns:
            bool: Whether the sensor is bound to a participant.
        """
        if sensor_id not in self.bound_sensors:
            return None
        return self.bound_sensors[sensor_id]

    def bind(self, sensor_id: int, participant_id: int):
        """Bind a registered sensor with a participant.

        Args:
            sensor_id (int): The id of the sensor.
            participant_id (int): The id of the participant.

        Raises:
            KeyError: If the sensor is not registered in the manager.
        """

        if sensor_id not in self.sensors:
            raise KeyError(f"Sensor {sensor_id} is not registered in the render manager.")

        if sensor_id in self.bound_sensors:
            warnings.warn(
                f"Sensor {sensor_id} was bound with participant \
                {self.bound_sensors[sensor_id]}. Now it is bound with {participant_id}."
            )

        self.bound_sensors[sensor_id] = participant_id

    def unbind(self, sensor_id):
        """Unbind a registered sensor from its bound participant.

        Args:
            sensor_id (int): The id of the sensor.
        """

        try:
            self.bound_sensors.pop(sensor_id)
        except KeyError:
            warnings.warn(f"Sensor {sensor_id} is not bound with any participant.")

    def update(self, participants: dict, participant_ids: list, frame: int = None):
        """Sync the viewpoint of the sensors with their bound participants. Update the
            observation of all the sensors.

        Args:
            participants (dict): The dictionary of all participants. The render manager
                will detect which of them is alive.
            frame (int): Update the sensors to the given frame. If None, the sensors
                will update to the current frame. The default unit is millisecond.
                Defaults to None.
        """
        to_remove = []
        for sensor_id, sensor in self.sensors.items():
            if sensor_id in self.bound_sensors:
                participant = participants[self.bound_sensors[sensor_id]]
                try:
                    state = participant.trajectory.get_state(frame)
                    sensor.update(
                        participants,
                        participant_ids,
                        frame,
                        Point(state.location),
                        state.heading,
                    )
                except KeyError:
                    self.unbind(sensor_id)
                    to_remove.append(sensor_id)
            else:
                sensor.update(participants, participant_ids, frame)

        for sensor_id in to_remove:
            self.remove_sensor(sensor_id)

    def render(self):
        """Render the observation of all the sensors."""

        self.clock.tick(self.fps)

        blit_sequence = []
        for sensor_id, layout_info in self.layouts.items():
            surface = pygame.transform.scale_by(
                self.sensors[sensor_id].surface, layout_info[0]
            )
            blit_sequence.append((surface, layout_info[1]))

        if self.screen is not None:
            self.screen.blits(blit_sequence)
        pygame.display.flip()

    def get_observation(self) -> list:
        """Get the observation of all the sensors.

        Returns:
            list: A list of 3d arrays.
        """
        observations = []
        for sensor in self.sensors.values():
            observations.append(sensor.get_observation())

        return observations

    def reset(self):
        """Reset the render manager."""
        for sensor_id in self.sensors:
            self.remove_sensor(sensor_id)

    def close(self):
        """Close the render manager."""
        pygame.quit()
