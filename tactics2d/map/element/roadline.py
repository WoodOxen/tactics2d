from typing import Tuple
import logging

import shapely
from shapely.geometry import LineString, Point


class RoadLine:
    """This class implements the lenelet2-style map element *LineString*.

    Detailed definition of lanelet2-style lane:
        [LaneletPrimitives](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md)

    Attributes:
        id_ (str): The unique identifier of the roadline.
        linestring (LineString): The shape of the line expressed in geometry format.
        type_ (str, optional): The type of the roadline. Defaults to "virtual".
        subtype (str, optional): The subtype of the line. Defaults to None.
        color (tuple, optional): The color of the lane marking. Defaults to None.
        lane_change (Tuple[bool, bool], optional): Whether a vehicle can switch to a left lane
            or a right lane. The first element in the tuple indicates the left. The second
            element in the tuple indicates the right. Defaults to None.
        width (float, optional): The width of the line (in m). The linestring then represents
            the centerline of the object. Defaults to None.
        height (float, optional): The height of line (in m). The linestring then represents the
            lower outline/lowest edge of the object. Defaults to None.
        temporary (bool, optional): Whether the roadline is a temporary lane mark or not.
            Defaults to False.
        custom_tags (dict, optional): The custom tags of the raodline. Defaults to None.
        head (Point): The head point of the roadline. This attribute is automatically calculated
            when requested.
        end (Point): The end point of the roadline. This attribute is automatically calculated
            when requested.
        shape (list): The shape of the roadline. This attribute is automatically calculated when
            requested.
    """

    def __init__(
        self,
        id_: str,
        linestring: LineString,
        type_: str = "virtual",
        subtype: str = None,
        color: tuple = None,
        lane_change: Tuple[bool, bool] = None,
        width: float = None,
        height: float = None,
        temporary: bool = False,
        custom_tags: dict = None,
    ):
        """Initialize the attributes in the class."""
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

        self._check_lane_change()

    @property
    def head(self) -> Point:
        return shapely.get_point(self.linestring, 0)

    @property
    def end(self) -> Point:
        return shapely.get_point(self.linestring, -1)

    @property
    def shape(self) -> list:
        return list(self.linestring.coords)

    def _check_lane_change(self):
        if self.subtype == "solid":
            if self.lane_change is None:
                self.lane_change = (False, False)
            elif self.lane_change != (False, False):
                logging.warning(
                    f"The lane change rule of a solid roadline is supposed to be (False, False). Line {self.id_} has lane change rule {self.lane_change}."
                )

        elif self.subtype == "solid_solid":
            if self.lane_change is None:
                self.lane_change = (False, False)
            elif self.lane_change != (False, False):
                logging.warning(
                    f"The lane change rule of a solid_solid roadline is supposed to be (False, False). Line {self.id_} has lane change rule {self.lane_change}."
                )

        elif self.subtype == "dashed":
            if self.lane_change is None:
                self.lane_change = (True, True)
            elif self.lane_change != (True, True):
                logging.warning(
                    f"The lane change rule of a dashed roadline is supposed to be (True, True). Line {self.id_} has lane change rule {self.lane_change}."
                )

        elif self.subtype == "solid_dashed":
            if self.lane_change is None:
                self.lane_change = (True, False)
            elif self.lane_change != (True, False):
                logging.warning(
                    f"The lane change rule of a solid_dashed roadline is supposed to be (True, False). Line {self.id_} has lane change rule {self.lane_change}."
                )

        elif self.subtype == "dashed_solid":
            if self.lane_change is None:
                self.lane_change = (False, True)
            elif self.lane_change != (False, True):
                logging.warning(
                    f"The lane change rule of a dashed_solid roadline is supposed to be (False, True). Line {self.id_} has lane change rule {self.lane_change}."
                )

        else:
            if self.lane_change is None:
                self.lane_change = (True, True)
