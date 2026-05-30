# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regulatory implementation."""

from typing import Optional

from shapely.geometry import Point


class RegulatoryMember:
    """This class implements the subelement of the Regulatory class."""

    __slots__ = ("ref", "type_", "role")

    def __init__(self, ref: str, type_: str, role: str):
        self.ref = ref
        self.type_ = type_
        self.role = role


class Regulatory:
    """This class implements the [lanelet2-style map element *RegulatoryElement*](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/RegulatoryElementTagging.md).

    !!! note
        This class is still under development. It is supposed to support the detection of traffic events in the future.

    Attributes:
        id_ (str): The id of the regulatory element.
        relations (dict): A dictionary of the relations that the regulatory element belongs to. The key is the id of the relation, and the value is the role of the relation. Defaults to an empty dictionary.
        ways (dict): A dictionary of the ways that the regulatory element belongs to. The key is the id of the way, and the value is the role of the way. Defaults to an empty dictionary.
        nodes (dict): A dictionary of the nodes that the regulatory element belongs to. The key is the id of the node, and the value is the role of the node. Defaults to an empty dictionary.
        type_ (str): The type of the regulatory element. Defaults to "regulatory_element".
        subtype (str): The subtype of the regulatory element.
        position (str): The position of the regulatory element. Defaults to None.
        location (str): The location of the regulatory element. Defaults to None.
        dynamic (bool): Whether the regulatory element is dynamic. Defaults to False.
        fallback (bool): Whether the regulatory element is a fallback. Defaults to False.
        custom_tags (dict): The custom tags of the regulatory element. Defaults to None.
    """

    __slots__ = (
        "id_",
        "relations",
        "ways",
        "nodes",
        "type_",
        "subtype",
        "position",
        "location",
        "dynamic",
        "fallback",
        "custom_tags",
    )

    def __init__(
        self,
        id_: str,
        relations: dict = dict(),
        ways: dict = dict(),
        nodes: dict = dict(),
        type_: str = "regulatory_element",
        subtype: str = None,
        position: str = None,
        location: str = None,
        dynamic: bool = False,
        fallback: bool = False,
        custom_tags: dict = None,
    ):
        """Initialize the attributes in the class.

        Args:
            id_ (str): The id of the regulatory element.
            relations (dict, optional): A dictionary of the relations that the regulatory element belongs to. The key is the id of the relation, and the value is the role of the relation.
            ways (dict, optional): A dictionary of the ways that the regulatory element belongs to. The key is the id of the way, and the value is the role of the way.
            nodes (dict, optional): A dictionary of the nodes that the regulatory element belongs to. The key is the id of the node, and the value is the role of the node.
            type_ (str, optional): The type of the regulatory element.
            subtype (str, optional): The subtype of the regulatory element.
            position (str, optional): The position of the regulatory element.
            location (str, optional): The location of the regulatory element.
            dynamic (bool, optional): Whether the regulatory element is dynamic.
            fallback (bool, optional): Whether the regulatory element is a fallback.
            custom_tags (dict, optional): The custom tags of the regulatory element.
        """

        if subtype is None:
            raise ValueError("The subtype of Regulatory %s is not defined!" % id_)

        self.id_ = id_
        self.relations = relations
        self.ways = ways
        self.nodes = nodes
        self.type_ = type_
        self.subtype = subtype
        self.position = position
        self.location = location
        self.dynamic = dynamic
        self.fallback = fallback
        self.custom_tags = custom_tags

    def is_stop_sign(self) -> bool:
        """Return whether this regulatory element is a stop sign."""

        return self.subtype == "stop_sign"

    def is_traffic_light(self) -> bool:
        """Return whether this regulatory element is a traffic light."""

        return self.subtype == "traffic_light"

    def applies_to_lane(self, lane_id: str) -> bool:
        """Return whether this regulatory element is bound to a lane."""

        tags = self.custom_tags or {}
        tagged_lane_id = tags.get("lane_id")
        if tagged_lane_id == lane_id:
            return True
        lane_ids = tags.get("lane_ids", [])
        return lane_id in self.ways or lane_id in lane_ids

    def state_at(self, time_ms: Optional[int] = None) -> Optional[dict]:
        """Return the nearest dynamic state record, if one is available."""

        states = (self.custom_tags or {}).get("states", [])
        if not states:
            return None
        if time_ms is None:
            return states[-1]
        return min(states, key=lambda state: abs(int(state.get("time_ms", 0)) - int(time_ms)))

    def stop_point_at(self, time_ms: Optional[int] = None) -> Optional[Point]:
        """Return the stop point from the dynamic state or static position."""

        state_record = self.state_at(time_ms)
        if state_record is not None and "stop_point" in state_record:
            return self._point_from_value(state_record["stop_point"])
        return self._point_from_value(self.position)

    @staticmethod
    def _point_from_value(value) -> Optional[Point]:
        if value is None:
            return None
        if isinstance(value, Point):
            return value
        return Point(float(value[0]), float(value[1]))
