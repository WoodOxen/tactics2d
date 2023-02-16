class Cyclist(object):
    def __init__(self, id: int):
        self.id = id
        self.trajectory = None
        self.controller = None

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory