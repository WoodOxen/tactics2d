import tensorflow as tf

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other


class WOMDParser:
    """This class implements a parser for Waymo Open Motion Dataset.

    Ettinger, Scott, et al. "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    """
    CLASS_MAPPING = {
        "vehicle": Vehicle,
        "cyclist": Cyclist,
        "pedestrian": Pedestrian,
        "other": Other,
    }

    def parse_trajectory(self):
        return