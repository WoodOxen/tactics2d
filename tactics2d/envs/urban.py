import gym

from tactics2d.map_base.lane import Lane
from tactics2d.map_base.area import Area
from tactics2d.map_base.map import Map


class UrbanEnv(gym.Env):
    def __init__(self, render_mode: str = "human"):

        self.render_mode = render_mode

