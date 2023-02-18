import pygame

# from tactics2d.traffic.scenario import Scenario
from tactics2d.render.sensors import SensorBase
# from tactics2d.render.sensors import TopDownCamera, SingleLineLidar


class RenderManager(object):
    def __init__(self, map_, participants) -> None:
        self.map_ = map_
        self.participants = participants

        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.surfaces = []

        self.sensors = dict()
        self.bound_sensors = []

    @property
    def driver(self) -> str:
        return pygame.display.get_driver()

    def set_mode(self, **kwargs):
        """The interface of tactics's renderer to set mode by pygame.display.set_mode"""
        pygame.display.set_mode(**kwargs)

    def add_sensor(self, sensor: SensorBase):
        if issubclass(sensor, SensorBase):
            if sensor.sensor_id not in self.sensors:
                self.sensors[sensor.sensor_id] = sensor
            else:
                raise ValueError(f"ID {sensor.sensor_id} is used by the other sensor.")
        return True

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


