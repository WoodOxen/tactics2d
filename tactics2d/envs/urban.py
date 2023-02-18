import gym

# from tactics2d.map.element import Map


class UrbanEnv(gym.Env):
    def __init__(self, render_mode: str = "human"):

        self.render_mode = render_mode

