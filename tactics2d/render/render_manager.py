import pygame


class RenderManager(object):
    def __init__(self, ) -> None:
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.surfaces = []

    @property
    def driver(self) -> str:
        return pygame.display.get_driver()

    def set_mode(self, **kwargs):
        """The interface of tactics's renderer to set mode by pygame.display.set_mode"""
        pygame.display.set_mode(**kwargs)

    def add_sensor(self, sensor, sensor_id):
        return True

    def remove_sensor(self, sensor_id):
        return

    def get_observation(self):
        return


