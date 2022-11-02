from shapely.geometry import Polygon

from Tacktics2D.elements.base.road.defaults import *


class Area(object):
    def __init__(
        self, id: str, 
        shape: Polygon = None, 
        subtype: str = None,
        participants: list = None
    ):

        self.id = id
        self.shape = shape
        self.subtype = subtype

        if self.subtype in DEFAULT_AREA:
            default_attrs = DEFAULT_AREA[self.subtype]
            self.participants = participants \
                if participants is not None else default_attrs["participants"]
        else:
            self.participants = participants

    def get_shape(self) -> list:
        """Return the shape of the region
        """
        return list(self.shape.coords)

    def get_area(self) -> float:
        return self.shape.area
