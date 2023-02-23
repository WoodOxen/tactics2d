from typing import Tuple

import pygame

# from tactics2d.traffic.scenario import Scenario
from tactics2d.render.sensors import SensorBase
# from tactics2d.render.sensors import TopDownCamera, SingleLineLidar


class RenderManager(object):
    def __init__(self, map_, participants, windows_size: Tuple[int, int] = (800, 800), fps: int = 60, off_screen: bool = False):
        self.map_ = map_
        self.participants = participants
        self.windows_size = windows_size
        self.fps = fps
        self.off_screen = off_screen

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(
            self.windows_size,
            flags=pygame.HIDDEN if self.off_screen else pygame.SHOWN
        )

        self.sensors = dict()
        self.bound_sensors = []
        self.layout = None

    @property
    def driver(self) -> str:
        return pygame.display.get_driver()

    def set_mode(self, **kwargs):
        """The interface of tactics's renderer to set mode by pygame.display.set_mode"""
        pygame.display.set_mode(**kwargs)

    def _rearrange_layout(self):
        pass

    def add_sensor(self, sensor: SensorBase, display=True):
        if issubclass(sensor, SensorBase):
            if sensor.sensor_id not in self.sensors:
                self.sensors[sensor.sensor_id] = sensor
            else:
                raise ValueError(f"ID {sensor.sensor_id} is used by the other sensor.")
        else:
            raise TypeError("The sensor must be a subclass of SensorBase.")

        if display:
            self.layout = self._rearrange_layout()

    def bind(self, sensor_id, participant_id):
        if sensor_id in self.sensors and participant_id in self.participants:
            self.bound_sensors.append((sensor_id, participant_id))
        else:
            raise ValueError()

    def remove_sensor(self, sensor_id):
        if sensor_id in self.sensors:
            self.sensors.pop(sensor_id)

    def get_observation(self):
        return


