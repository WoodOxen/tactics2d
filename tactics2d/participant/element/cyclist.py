class Cyclist(object):
    def __init__(self, id_: int):
        self.id_ = id_
        self.trajectory = None
        self.controller = None

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory