from shapely.geometry import Polygon

from Tacktics2D.elements.base.road.defaults import *


class Area(object):
    """_summary_

    Args:
        id (int): region id, which must be unique
        shape (Polygon): 
        subtype (str): region type, which can be from the defaults or customized
        participant (set): the traffic participant types that are allowed in the region.
            Defaults to ALL_PARTICIPANT.
    """
    def __init__(
        self, id: int, shape: Polygon = None, subtype: str = None, 
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
