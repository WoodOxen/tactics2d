import numpy as np
from shapely.geometry.base import BaseGeometry

from .default import *


class Lane(object):
    def __init__(
        self, 
        shape: BaseGeometry = None, 
        subtype: str = None, 
        drivable: bool = None,
        participants: list = None,
        speed_limit: float = None,
        color: float = None,
    ):

        self.shape = shape
        self.subtype = subtype

        if self.subtype in DEFAULT_LANE:
            default_attrs = DEFAULT_LANE[self.subtype]
            self.drivable = drivable \
                if drivable is not None else default_attrs["drivable"]
            self.participants = participants \
                if participants is not None else default_attrs["participants"]
            self.speed_limit = speed_limit \
                if speed_limit is not None else default_attrs["speed_limit"]
            self.color = color \
                if color is not None else default_attrs["color"]
        else:
            self.drivable = drivable
            self.participants = participants
            self.speed_limit = speed_limit
            self.color = color
    
    def get_shape(self):
        return np.array(self.shape.coords)


class Area(object):
    def __init__(
        self,
        shape: BaseGeometry = None, 
        subtype: str = None,
        color: float = None,
    ):
        self.shape = shape
        self.subtype = subtype

        if self.subtype in DEFAULT_AREA:
            default_attrs = DEFAULT_AREA[self.subtype]
            self.color = color \
                if color is not None else default_attrs["color"]
        else:
            self.color = color

    def get_shape(self):
        return np.array(self.shape.coords)