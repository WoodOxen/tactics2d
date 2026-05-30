# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Roadline implementation."""


import logging
from typing import Tuple

import shapely
from shapely.geometry import LineString, Point


class RoadLine:
    """This class implements the map element *LineString*.

    !!! quote "Reference"
        - [OpenStreetMap's description of a way](https://wiki.openstreetmap.org/wiki/Way)
        - [Lanelet2's description of a line](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md).

    Attributes:
        id_ (str): The unique identifier of the roadline.
        geometry (LineString): The geometry information of the roadline.
        type_ (str, optional): The type of the roadline. Defaults to "virtual".
        subtype (str, optional): The subtype of the line. Defaults to None.
        color (Any): The color of the line. If not specified, the color will be
            assigned based on the rendering template later. Defaults to None.
        width (float, optional): The width of the line in metres. Defaults to None.
        height (float, optional): The height of line in metres. Defaults to None.
        lane_change (Tuple[bool, bool], optional): Whether a vehicle can cross
            the line from left to right and from right to left. Defaults to None.
        temporary (bool, optional): Whether the roadline is temporary. Defaults to False.
        custom_tags (dict, optional): Custom tags of the roadline. Defaults to None.
        head (Point): The head point of the roadline. Read-only.
        end (Point): The end point of the roadline. Read-only.
        shape (list): The shape of the roadline. Read-only.
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
        """Initialize an instance of RoadLine.

        Args:
            id_: The unique identifier of the roadline.
            geometry: The shape of the line expressed in geometry format.
            type_: The type of the roadline.
            subtype: The subtype of the line.
            color: The color of the line. If not specified, the color will be
                assigned based on the rendering template later.
            width: The width of the line in metres.
            height: The height of the line in metres.
            lane_change: Whether a vehicle can cross from left to right and
                from right to left.
            temporary: Whether the roadline is temporary.
            custom_tags: Custom tags of the roadline.
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
        self.custom_tags = custom_tags or {}

        self._set_lane_change()

    def _set_lane_change(self):
        """Infer lane-change permission from native roadline subtype."""

        def set_by_type(type_: str, lane_change: Tuple[bool, bool]):
            if self.lane_change is None:
                self.lane_change = lane_change
            elif self.lane_change != lane_change:
                logging.warning(
                    f"The lane change rule of a {type_} roadline is supposed to be "
                    f"{lane_change}. Line {self.id_} has lane change rule "
                    f"{self.lane_change}."
                )

        if self.subtype in ["solid", "solid_solid"]:
            set_by_type(self.subtype, (False, False))
        elif self.subtype in ["dashed", "dashed_dashed"]:
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
