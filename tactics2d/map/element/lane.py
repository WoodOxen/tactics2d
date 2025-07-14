##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: lane.py
# @Description: This file defines a class for a map lane.
# @Author: Yueyuan Li
# @Version: 1.0.0


import logging
from enum import IntEnum
from typing import Any, Union

import numpy as np
from shapely.geometry import LinearRing, LineString


class LaneRelationship(IntEnum):
    PREDECESSOR = 1
    SUCCESSOR = 2
    LEFT_NEIGHBOR = 3
    RIGHT_NEIGHBOR = 4


class Lane:
    """This class implements the map element *Lane*

    !!! quote "Reference"
        [Lanelet2's description of a lane](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md).

    Attributes:
        id_ (str): The unique identifier of the lane.
        left_side (LineString): The left side of the lane.
        right_side (LineString): The right side of the lane.
        line_ids (dict): The ids of the roadline components. The dictionary has two keys: `left` and `right`. Defaults to dict(left=[], right=[]).
        regulatory_ids (set): The ids of the regulations that apply to the lane. Defaults to set().
        type_ (str): The type of the lane. The default value is `"lanelet"`.
        subtype (str): The subtype of the lane. Defaults to None.
        color (Any): The color of the area. If not specified, the color will be assigned based on the rendering template later. Defaults to None.
        location (str): The location of the lane (urban, nonurban, etc.). Defaults to None.
        inferred_participants (list): he allowing type of traffic participants that can pass the area. If not specified, the area is not restricted to any type of traffic participants. Defaults to None.
        speed_limit (float): The speed limit in this area The unit is `m/s`. Defaults to None.
        speed_limit_mandatory (bool): Whether the speed limit is mandatory or not. Defaults to True.
        custom_tags (dict): The custom tags of the area. Defaults to None.
        custom_tags (dict): The custom tags of the lane. Defaults to None.
        predecessors (set): The ids of the available lanes before entering the current lane.
        successors (set): The ids of the available lanes after exiting the current lane.
        left_neighbors (set): The ids of the lanes that is adjacent to the left side of the current lane and in the same direction.
        right_neighbors (set): The ids of the lanes that is adjacent to the right side of the current lane and in the same direction.
        starts (list): The start points of the lane.
        ends (list): The end points of the lane.
        geometry (LinearRing): The geometry representation of the lane. This attribute will be automatically obtained during the initialization if there is no None in left_side and right_side.
        shape (list): The shape of the lane. This attribute is **read-only**.
    """

    _speed_units = ["km/h", "mi/h", "m/s", "mph"]

    def __init__(
        self,
        id_: str,
        left_side: LineString = None,
        right_side: LineString = None,
        geometry: LinearRing = None,
        line_ids: set = dict(left=[], right=[]),
        regulatory_ids: set = set(),
        type_: str = "lanelet",
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
            id_ (str): The unique identifier of the lane.
            left_side (LineString, optional): The left side of the lane. Defaults to None.
            right_side (LineString, optional): The right side of the lane. Defaults to None.
            geometry (LinearRing, optional): The geometry of the lane. This parameter only takes effect when the `left_side` or `right_side` is None. Defaults to None.
            line_ids (set, optional): The ids of the lines that make up the lane.
            regulatory_ids (set, optional): The ids of the regulations that apply to the lane.
            type_ (str, optional): The type of the lane.
            subtype (str, optional): The subtype of the lane. Defaults to None.
            color (Any, optional): The color of the lane. If not specified, the color will be assigned based on the rendering template later. Defaults to None.
            location (str, optional): The location of the lane (urban, nonurban, etc.). Defaults to None.
            inferred_participants (list, optional): The allowing type of traffic participants that can pass the lane. If not specified, the lane is not restricted to any type of traffic participants. Defaults to None.
            speed_limit (float, optional): The speed limit in this lane. Defaults to None.
            speed_limit_unit (str, optional): The unit of speed limit in this lane. The valid units are `km/h`, `mi/h`, and `m/s`. Defaults to "km/h". The speed limit will be automatically converted to `m/s` when initializing the instance. If the unit is invalid, the speed limit will be set to None.
            speed_limit_mandatory (bool, optional): Whether the speed limit is mandatory or not.
            custom_tags (dict, optional): The custom tags of the lane. Defaults to None.
        """
        self.id_ = id_
        self.left_side = left_side
        self.right_side = right_side
        self.line_ids = line_ids
        self.regulatory_ids = regulatory_ids
        self.type_ = type_
        self.subtype = subtype
        self.color = color
        self.location = location
        self.inferred_participants = inferred_participants
        self.speed_limit_mandatory = speed_limit_mandatory
        self.custom_tags = custom_tags

        self._check_boundary()
        self.centerline = self._get_centerline()

        if not None in [left_side, right_side]:
            self.geometry = LinearRing(
                list(left_side.coords) + list(reversed(list(right_side.coords)))
            )
        else:
            self.geometry = geometry

        self._set_speed_limit_unit(speed_limit, speed_limit_unit)

        self.predecessors = set()
        self.successors = set()
        self.left_neighbors = set()
        self.right_neighbors = set()

    def _check_boundary(self):
        if self.left_side is None or self.right_side is None:
            return

        left_coords = np.array(self.left_side.coords)
        right_coords = np.array(self.right_side.coords)

        # 1. Check left and right side are correctly defined
        n = min(len(left_coords), len(right_coords), 10)
        indices = np.linspace(0, min(len(left_coords), len(right_coords)) - 1, n, dtype=int)
        cross_sum = 0
        for idx in indices:
            if idx < len(left_coords) - 1 and idx < len(right_coords) - 1:
                lane_vec = 0.5 * (
                    (left_coords[idx + 1] - left_coords[idx])
                    + (right_coords[idx + 1] - right_coords[idx])
                )
            else:
                lane_vec = 0.5 * (
                    (left_coords[-1] - left_coords[-2]) + (right_coords[-1] - right_coords[-2])
                )
            left_to_right = right_coords[idx] - left_coords[idx]
            cross_z = np.cross(lane_vec[:2], left_to_right[:2])
            cross_sum += cross_z
        if cross_sum < 0:
            self.left_side, self.right_side = self.right_side, self.left_side
            left_coords = np.array(self.left_side.coords)
            right_coords = np.array(self.right_side.coords)
            logging.debug(
                f"Swapped left_side and right_side for lane {self.id_} to ensure correct orientation."
            )

        # 2. Ensuring that the start points are on the entrance of the lane
        left_start = left_coords[0]
        left_end = left_coords[-1]
        right_start = right_coords[0]
        right_end = right_coords[-1]
        dist_starts = np.linalg.norm(left_start - right_start)
        dist_ends = np.linalg.norm(left_end - right_end)
        if dist_starts > dist_ends:
            self.left_side = LineString(left_coords[::-1])
            self.right_side = LineString(right_coords[::-1])
            logging.debug(
                f"Reversed left_side and right_side for lane {self.id_} to ensure entrance direction."
            )

    def _get_centerline(self):
        """Interpolate the side with fewer points to match the other, then compute the centerline."""
        if self.left_side is None or self.right_side is None:
            return None

        left_coords = np.array(self.left_side.coords)
        right_coords = np.array(self.right_side.coords)

        n_left = len(left_coords)
        n_right = len(right_coords)

        if n_left > n_right:
            base = left_coords
            ref = right_coords
            ref_line = LineString(ref)
            ref_dist = np.linspace(0, ref_line.length, n_left)
            ref_points = [ref_line.interpolate(d) for d in ref_dist]
            ref_coords = np.array([[p.x, p.y] for p in ref_points])
            left_coords_interp = base
            right_coords_interp = ref_coords
        elif n_right > n_left:
            base = right_coords
            ref = left_coords
            ref_line = LineString(ref)
            ref_dist = np.linspace(0, ref_line.length, n_right)
            ref_points = [ref_line.interpolate(d) for d in ref_dist]
            ref_coords = np.array([[p.x, p.y] for p in ref_points])
            left_coords_interp = ref_coords
            right_coords_interp = base
        else:
            left_coords_interp = left_coords
            right_coords_interp = right_coords

        centerline = 0.5 * (left_coords_interp + right_coords_interp)
        return LineString(centerline)

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
    def starts(self) -> list:
        return [self.left_side.coords[0], self.right_side.coords[0]]

    @property
    def ends(self) -> list:
        return [self.left_side.coords[-1], self.right_side.coords[-1]]

    @property
    def shape(self) -> list:
        return list(self.geometry.coords)

    def is_related(self, id_: str) -> LaneRelationship:
        """Check if a given lane is related to the lane

        Args:
            id_ (str): The given lane's id.
        """
        if id_ in self.predecessors:
            return LaneRelationship.PREDECESSOR
        elif id_ in self.successors:
            return LaneRelationship.SUCCESSOR
        elif id_ in self.left_neighbors:
            return LaneRelationship.LEFT_NEIGHBOR
        elif id_ in self.right_neighbors:
            return LaneRelationship.RIGHT_NEIGHBOR

        return False

    def add_related_lane(self, id_: Union[str, list], relationship: LaneRelationship):
        """Add a related lane's id to the corresponding list

        Args:
            id_ (str): The related lane's id
            relationship (LaneRelationship): The relationship of the lanes. The possible values are 1/`PREDECESSOR`, 2/`SUCCESSOR`, 3/`LEFT_NEIGHBOR`, and 4/`RIGHT_NEIGHBOR`.
        """
        if id_ is None:
            return

        if isinstance(id_, str):
            if id_ == self.id_:
                logging.warning(f"Lane {self.id_} cannot be a related lane to itself.")
                return
            if relationship == LaneRelationship.PREDECESSOR or relationship == int(
                LaneRelationship.PREDECESSOR
            ):
                self.predecessors.add(id_)
            elif relationship == LaneRelationship.SUCCESSOR or relationship == int(
                LaneRelationship.SUCCESSOR
            ):
                self.successors.add(id_)
            elif relationship == LaneRelationship.LEFT_NEIGHBOR or relationship == int(
                LaneRelationship.LEFT_NEIGHBOR
            ):
                self.left_neighbors.add(id_)
            elif relationship == LaneRelationship.RIGHT_NEIGHBOR or relationship == int(
                LaneRelationship.RIGHT_NEIGHBOR
            ):
                self.right_neighbors.add(id_)

        elif isinstance(id_, list):
            if self.id_ in id_:
                id_ = [i for i in id_ if i != self.id_]
                logging.warning(f"Lane {self.id_} cannot be a related lane to itself.")

            if relationship == LaneRelationship.PREDECESSOR:
                self.predecessors.update(id_)
            elif relationship == LaneRelationship.SUCCESSOR:
                self.successors.update(id_)
            elif relationship == LaneRelationship.LEFT_NEIGHBOR:
                self.left_neighbors.update(id_)
            elif relationship == LaneRelationship.RIGHT_NEIGHBOR:
                self.right_neighbors.update(id_)
