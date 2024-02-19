##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: area.py
# @Description: This file defines a class for a map area.
# @Author: Yueyuan Li
# @Version: 1.0.0


import warnings

from shapely.geometry import Polygon


class Area:
    """This class implements the lenelet2-style map element *area*.

    !!! info "Definition of a lanelet2-style area"
        [LaneletPrimitives.md](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md)

    Attributes:
        id_ (str): The unique identifier of the area.
        geometry (Polygon): The shape of the area expressed in geometry format.
        line_ids (dict): The ids of the lines that circle this area. Defaults to None.
        type_ (str): The type of the area. The default value is "multipolygon".
        subtype (str): The subtype of the area. Defaults to None.
        color (tuple): The color of the area. Defaults to None.
        location (str): The location of the area (urban, nonurban, etc.). Defaults to None.
        inferred_participants (list): The allowing type of traffic participants that can pass the area. Defaults to None.
        speed_limit (float): The speed limit in this area. Defaults to None.
        speed_limit_unit (str): The unit of speed limit in this area. The valid units are `"km/h"`, `"mi/h"`, and `"m/s"`. Defaults to `"km/h"`.
        speed_limit_mandatory (bool): Whether the speed limit is mandatory or not. Defaults to True.
        custom_tags (dict): The custom tags of the area. Defaults to None.
    """

    _SPEED_UNIT = ["km/h", "mi/h", "m/s"]

    def __init__(
        self,
        id_: str,
        geometry: Polygon,
        line_ids: dict = None,
        type_: str = "multipolygon",
        subtype: str = None,
        color: tuple = None,
        location: str = None,
        inferred_participants: list = None,
        speed_limit: float = None,
        speed_limit_unit: str = "km/h",
        speed_limit_mandatory: bool = True,
        custom_tags: dict = None,
    ):
        self.id_ = id_
        self.geometry = geometry
        self.line_ids = line_ids
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.location = location
        self.inferred_participants = inferred_participants
        self.speed_limit = speed_limit
        self.speed_limit_unit = speed_limit_unit
        self.speed_limit_mandatory = speed_limit_mandatory
        self.custom_tags = custom_tags

        if self.speed_limit_unit not in self._SPEED_UNIT:
            warnings.warn(
                "Invalid speed limit unit %s. The legal units types are %s"
                % (self.speed_limit_unit, ", ".join(self._SPEED_UNIT))
            )

    def shape(self, outer_only: bool = False):
        outer_shape = list(self.polygon.exterior.coords)
        if outer_only:
            return outer_shape
        inners_shape = list(self.polygon.interiors.coords)
        return outer_shape, inners_shape
