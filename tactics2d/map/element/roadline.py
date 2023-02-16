from typing import Tuple

import shapely
from shapely.geometry import LineString, Point


class RoadLine:
    """An implementation of the lanelet2-style linestring.

    The detailed definition of the lanelet2-style linestring can be found here:
    https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md

    Attributes:
        id_ (str): _description_
        linestring (LineString): _description_
        type (str, optional): _description_. Defaults to "virtual".
        subtype (str, optional): _description_. Defaults to None.
        color (tuple, optional): color of the lane marking. Defaults to None.
        lane_change (Tuple[bool, bool], optional): _description_. Defaults to (True, True).
        width (float, optional): width of the line (in m). The linestring then represents the centerline of the object. Defaults to None.
        height (float, optional): height of line (in m). The linestring then represents the lower outline/lowest edge of the object. Defaults to None.
        temporary (bool, optional): _description_. Defaults to False.
    """
    def __init__(
        self, id_: str, linestring: LineString,
        type_: str = "virtual", subtype: str = None, color: tuple = None,
        lane_change: Tuple[bool, bool] = (True, True),
        width: float = None, height: float = None,
        temporary: bool = False, 
        custom_tags: dict = None
    ):

        self.id_ = id_
        self.linestring = linestring
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.lane_change = lane_change
        self.width = width
        self.height = height
        self.temporary = temporary
        self.custom_tags = custom_tags

    @property
    def head(self) -> Point:
        return shapely.get_point(self.linestring, 0)
    
    @property
    def tail(self) -> Point:
        return shapely.get_point(self.linestring, -1)

    @property
    def shape(self) -> list:
        """Get shape of the roadline
        """
        return list(self.linestring.coords)