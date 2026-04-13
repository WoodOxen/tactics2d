# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Area implementation."""


import logging
from typing import Any, Tuple

from shapely.geometry import Polygon


class Area:
    """This class implements the *Area*

    !!! quote "Reference"
        [Lanelet2's description of an area](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md

    Attributes:
        id_ (str): The unique identifier of the area.
        geometry (Polygon): The geometry information of the area.
        line_ids (dict): The ids of the lines that circle this area. Defaults to dict(inner=[], outer=[]).
        regulatory_ids (set): The ids of the regulations that apply to the area. Defaults to set().
        type_ (str): The type of the area. The default value is "multipolygon".
        subtype (str): The subtype of the area. Defaults to None.
        color (Any): The color of the area. If not specified, the color will be assigned based on the rendering template later. Defaults to None.
        location (str): The location of the area (urban, nonurban, etc.). Defaults to None.
        inferred_participants (list): The allowing type of traffic participants that can pass the area. If not specified, the area is not restricted to any type of traffic participants. Defaults to None.
        speed_limit (float): The speed limit in this area The unit is `m/s`. Defaults to None.
        speed_limit_mandatory (bool): Whether the speed limit is mandatory or not. Defaults to True.
        custom_tags (dict): The custom tags of the lane. Defaults to None.
        shape (List[list, list]): The shape of the area. The first list contains the outer shape, and the second list contains the inner shapes. Defaults to None. This attribute is **read-only**.
    """

    __slots__ = (
        "id_",
        "geometry",
        "line_ids",
        "regulatory_ids",
        "type_",
        "subtype",
        "color",
        "location",
        "inferred_participants",
        "speed_limit_mandatory",
        "custom_tags",
        "speed_limit",
    )

    _speed_units = ["km/h", "mi/h", "m/s", "mph"]

    def __init__(
        self,
        id_: str,
        geometry: Polygon,
        line_ids: dict = dict(inner=[], outer=[]),
        regulatory_ids: set = set(),
        type_: str = "multipolygon",
        subtype: str = None,
        color: Any = None,
        location: str = None,
        inferred_participants: list = None,
        speed_limit: float = None,
        speed_limit_unit: str = "km/h",
        speed_limit_mandatory: bool = True,
        custom_tags: dict = None,
    ):
        """Initialize an instance of this class.

        Args:
            id_ (str): The unique identifier of the area.
            geometry (Polygon): The geometry shape of the area.
            line_ids (dict, optional): The ids of the lines that circle this area.
            regulatory_ids (set, optional): The ids of the regulations that apply to the area.
            type_ (str, optional): The type of the area.
            subtype (str, optional): The subtype of the area.
            color (Any, optional): The color of the area. If not specified, the color will be assigned based on the rendering template later.
            location (str, optional): The location of the area (urban, nonurban, etc.).
            inferred_participants (list, optional): The allowing type of traffic participants that can pass the area. If not specified, the area is not restricted to any type of traffic participants.
            speed_limit (float, optional): The speed limit in this area.
            speed_limit_unit (str, optional): The unit of speed limit in this area. The valid units are `km/h`, `mi/h`, and `m/s`. Defaults to "km/h". The speed limit will be automatically converted to `m/s` when initializing the instance. If the unit is invalid, the speed limit will be set to None.
            speed_limit_mandatory (bool, optional): Whether the speed limit is mandatory or not.
            custom_tags (dict, optional): The custom tags of the area.
        """
        self.id_ = id_
        self.geometry = geometry
        self.line_ids = line_ids
        self.regulatory_ids = regulatory_ids
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.location = location
        self.inferred_participants = inferred_participants
        self.speed_limit_mandatory = speed_limit_mandatory
        self.custom_tags = custom_tags

        self._set_speed_limit_unit(speed_limit, speed_limit_unit)

    def _set_speed_limit_unit(self, speed_limit: float, speed_limit_unit: str):
        if not speed_limit_unit in self._speed_units:
            logging.warning(
                "Invalid speed limit unit %s. The legal units types are %s"
                % (speed_limit_unit, ", ".join(self._speed_units))
            )
            self.speed_limit = None

        if speed_limit is None:
            self.speed_limit = None
        elif speed_limit_unit == "m/s":
            pass
        elif speed_limit_unit == "km/h":
            self.speed_limit = round(speed_limit / 3.6, 3)
        elif speed_limit_unit == "mi/h" or speed_limit_unit == "mph":
            self.speed_limit = round(speed_limit / 2.237, 3)

    @property
    def shape(self, outer_only: bool = False) -> Tuple[list, list]:
        outer_shape = list(self.geometry.exterior.coords)
        if outer_only:
            return outer_shape
        inner_shapes = []
        for interior in self.geometry.interiors:
            inner_shapes.append(list(interior.coords))

        return outer_shape, inner_shapes
