# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Roadline implementation."""


import logging
from typing import Tuple

import shapely
from shapely.geometry import LineString, Point


class RoadLine:
    """This class implements the map element *LineString*

    !!! quote "Reference"
        - [OpenStreetMap's description of a way](https://wiki.openstreetmap.org/wiki/Way)
        - [Lanelet2's description of a line](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md).

    Attributes:
        id_ (str): The unique identifier of the roadline.
        geometry (LineString): The geometry information of the roadline.
        type_ (str, optional): The type of the roadline. Defaults to "virtual".
        subtype (str, optional): The subtype of the line. Defaults to None.
        color (Any): The color of the area. If not specified, the color will be assigned based on the rendering template later. Defaults to None.
        width (float, optional): The width of the line (in m). Used in rendering. Defaults to None.
        height (float, optional): The height of line (in m). The linestring then represents the lower outline/lowest edge of the object. Defaults to None.
        lane_change (Tuple[bool, bool], optional): Whether a vehicle can switch to a left lane or a right lane. The first element in the tuple denotes whether the vehicle can switch from the left to the right of the line, and the second element denotes whether the vehicle can switch from the right to the left of the line. Defaults to None.
        temporary (bool, optional): Whether the roadline is a temporary lane mark or not. Defaults to False.
        custom_tags (dict, optional): The custom tags of the raodline. Defaults to None.
        head (Point): The head point of the roadline. This attribute is **read-only**.
        end (Point): The end point of the roadline. This attribute is **read-only**.
        shape (list): The shape of the roadline. This attribute is **read-only**.
    """

    __slots__ = (
        "id_",
        "geometry",
        "type_",
        "subtype",
        "color",
        "width",
        "height",
        "lane_change",
        "temporary",
        "custom_tags",
    )

    def __init__(
        self,
        id_: str,
        geometry: LineString,
        type_: str = "virtual",
        subtype: str = None,
        color: tuple = None,
        width: float = None,
        height: float = None,
        lane_change: Tuple[bool, bool] = None,
        temporary: bool = False,
        custom_tags: dict = None,
    ):
        """Initialize an instance of the class.

        Args:
            id_ (str): The unique identifier of the roadline.
            geometry (LineString): The shape of the line expressed in geometry format.
            type_ (str, optional): The type of the roadline.
            subtype (str, optional): The subtype of the line.
            color (tuple, optional): The color of the area. If not specified, the color will be assigned based on the rendering template later.
            width (float, optional): The width of the line (in m). Used in rendering.
            height (float, optional):
            lane_change (Tuple[bool, bool], optional): Whether a vehicle can switch to a left lane or a right lane. The first element in the tuple denotes whether the vehicle can switch from the left to the right of the line, and the second element denotes whether the vehicle can switch from the right to the left of the line.
            temporary (bool, optional): Whether the roadline is a temporary lane mark or not.
            custom_tags (dict, optional): The custom tags of the raodline.
        """
        self.id_ = id_
        self.geometry = geometry
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.width = width
        self.height = height
        self.lane_change = lane_change
        self.temporary = temporary
        self.custom_tags = custom_tags

        self._set_lane_change()

    def _set_lane_change(self):
        def set_by_type(type_: str, lane_change: Tuple[bool, bool]):
            if self.lane_change is None:
                self.lane_change = lane_change
            elif self.lane_change != lane_change:
                logging.warning(
                    f"The lane change rule of a {type_} roadline is supposed to be {lane_change}. Line {self.id_} has lane change rule {self.lane_change}."
                )

        if self.subtype == "solid":
            set_by_type(self.subtype, (False, False))
        elif self.subtype == "solid_solid":
            set_by_type(self.subtype, (False, False))
        elif self.subtype == "dashed":
            set_by_type(self.subtype, (True, True))
        elif self.subtype == "solid_dashed":
            set_by_type(self.subtype, (False, True))
        elif self.subtype == "dashed_solid":
            set_by_type(self.subtype, (True, False))
        elif self.type_ in ["curbstone", "road_border"]:
            set_by_type(self.type_, (False, False))
        elif self.subtype in [
            "guard_rail",
            "wall",
            "fence",
            "zebra_marking",
            "pedestrian_marking",
            "bike_marking",
            "keepout",
            "jersey_barrier",
            "gate",
            "door",
            "rail",
        ]:
            set_by_type(self.subtype, (False, False))

        if self.lane_change is None:
            self.lane_change = (True, True)

    @property
    def head(self) -> Point:
        return shapely.get_point(self.geometry, 0)

    @property
    def end(self) -> Point:
        return shapely.get_point(self.geometry, -1)

    @property
    def shape(self) -> list:
        return list(self.geometry.coords)
