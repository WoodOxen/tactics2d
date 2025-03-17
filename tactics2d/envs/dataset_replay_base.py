import gymnasium as gym


class DatasetReplayEnvBase(gym.Env):
    def __init__(self):
        super().__init__()
