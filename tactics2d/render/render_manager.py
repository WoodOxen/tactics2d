import pygame


class RenderManager(object):
    def __init__(self, ) -> None:
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.surfaces = []
        self.bound_sensors = []

    @property
    def driver(self) -> str:
        return pygame.display.get_driver()

    def set_mode(self, **kwargs):
        """The interface of tactics's renderer to set mode by pygame.display.set_mode"""
        pygame.display.set_mode(**kwargs)

    def add_sensor(self, sensor, sensor_id):
        return True

    def bind(self, sensor_id, participant_id):
        self.bound_sensors.append((sensor_id, participant_id))

    def remove_sensor(self, sensor_id):
        return

    def get_observation(self):
        return


