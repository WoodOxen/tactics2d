from typing import Tuple
import warnings

import numpy as np
from shapely.geometry import Point
import pygame

# from tactics2d.traffic.scenario import Scenario
from tactics2d.render.sensors import SensorBase
# from tactics2d.render.sensors import TopDownCamera, SingleLineLidar


class RenderManager(object):
    layout_styles = {"hierarchical", "modular"}
    def __init__(
        self, map_,
        fps: int = 60, windows_size: Tuple[int, int] = (800, 800), layout_style: str = "hierarchical", off_screen: bool = False
    ):
        self.map_ = map_
        self.fps = fps
        self.windows_size = windows_size
        self.off_screen = off_screen

        if layout_style not in self.layout_styles:
            raise ValueError(f"Layout style must be one of {self.layout_styles}.")
        self.layout_style = layout_style

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = None

        self.sensors = {}
        self.bound_sensors = {}
        self.layouts = {}

    @property
    def driver(self) -> str:
        return pygame.display.get_driver()

    def _rearrange_layout(self):
        sensor_to_display = [sensor for sensor in self.sensors.values() if not sensor.off_screen]

        if self.screen is None and len(sensor_to_display) > 0:
            self.screen = pygame.display.set_mode(
                self.windows_size, flags=pygame.SHOWN)

        if self.layout_style == "hierarchical":
            pass
        elif self.layout_style == "modular":
            n = int(np.ceil(np.sqrt(len(sensor_to_display))))
            width = self.windows_size[0] / n
            height = self.windows_size[1] / np.ceil(len(sensor_to_display) / n)
            for i, sensor in enumerate(self.sensors.values()):
                scale = min(
                    width / sensor.window_size[0], 
                    height / sensor.window_size[1]
                )
                coords = (
                    (i % n) * width + (width - sensor.window_size[0] * scale) / 2,
                    (i // n) * height + (height - sensor.window_size[1] * scale) / 2
                )
                self.layouts[sensor.sensor_id] = (scale, coords)

    def add_sensor(self, sensor: SensorBase):
        if sensor.sensor_id not in self.sensors:
            self.sensors[sensor.sensor_id] = sensor
        else:
            raise KeyError(f"ID {sensor.sensor_id} is used by the other sensor.")

        if not sensor.off_screen:
            self._rearrange_layout()

    def bind(self, sensor_id, participant_id):
        if sensor_id not in self.sensors:
            raise KeyError(f"Sensor {sensor_id} is not registered in the render manager.")
        
        if sensor_id in self.bound_sensors:
            warnings.warn(f"Sensor {sensor_id} was bound with participant {self.bound_sensors[sensor_id]}. Now it is bound with {participant_id}.")

        self.bound_sensors[sensor_id] = participant_id

    def remove_sensor(self, sensor_id):
        if sensor_id in self.sensors:
            self.sensors.pop(sensor_id)

    def update(self, participants: dict):
        for sensor_id, sensor in self.sensors.items():
            if sensor_id in self.bound_sensors:
                participant = participants[self.bound_sensors[sensor_id]]
                sensor.update(participants, Point(participant.location), participant.heading)
            else:
                sensor.update(participants)

    def render(self):
        self.clock.tick(self.fps)

        blit_sequence = []
        for sensor_id, layout_info in self.layouts.items():
            surface = pygame.transform.scale_by(self.sensors[sensor_id].surface, layout_info[0])
        # surface = pygame.transform.scale_by(self.sensors[1].surface, 1)
            blit_sequence.append((surface, layout_info[1]))

        # self.screen.blit(surface, (0,0))
        # print(blit_sequence)
        if self.screen is not None:
            self.screen.blits(blit_sequence)
        pygame.display.flip()

    def get_observation(self):
        return


