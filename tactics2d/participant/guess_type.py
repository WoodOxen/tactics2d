from tactics2d.trajectory.element.trajectory import Trajectory

DEFAULT_GUESS_TYPE = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "cyclist": "cyclist",
    "pedestrian": "pedestrian",
}

class GuessType:

    @staticmethod
    def guess_by_size(size_info:tuple, hint_type: str):
        return DEFAULT_GUESS_TYPE[hint_type]

    @staticmethod
    def guess_by_trajectory(trajectory: Trajectory, hint_type: str):
        return DEFAULT_GUESS_TYPE[hint_type]