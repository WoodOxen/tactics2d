# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base implementation."""


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from shapely.geometry import Point

from tactics2d.map.element import Map


class SensorBase(ABC):
    """This class defines a base interface for sensors.

    Attributes:
        id_ (int): The unique identifier of the sensor. This attribute is **read-only** once the instance is initialized.
        map_ (Map): The map that the sensor is attached to. This attribute is **read-only** once the instance is initialized.
        perception_range (Union[float, Tuple[float]]): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None. This attribute is **read-only** once the instance is initialized.
        position (Point): The position of the sensor in the global 2D coordinate system.
        bind_id (Any): The unique identifier of object that the sensor is bound to. This attribute is **read-only** and can only be set using the `bind_with` method.
        is_bound (bool): Whether the sensor is bound to an object. This attribute is **read-only** once the instance is initialized.
    """

    _id = None
    _map = None
    _perception_range = None
    _bind_id = None
    _position = None

    def __init__(self, id_: int, map_: Map, perception_range: Union[float, Tuple[float]] = None):
        """Initialize the sensor.

        Args:
            id_ (int): The unique identifier of the sensor.
            map_ (Map): The map that the sensor is attached to.
            perception_range (Union[float, Tuple[float]], optional): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. This can be a single float value or a tuple of four values representing the perception range in each direction (left, right, front, back). Defaults to None.
        """

        self._id = id_
        self._map = map_
        self.transform_matrix = None

        if perception_range is None:
            width = (map_.boundary[1] - map_.boundary[0]) / 2
            height = (map_.boundary[3] - map_.boundary[2]) / 2
            self._perception_range = (width, width, height, height)
        elif isinstance(perception_range, float) or isinstance(perception_range, int):
            self._perception_range = (
                perception_range,
                perception_range,
                perception_range,
                perception_range,
            )
        else:
            self._perception_range = perception_range

    @property
    def id_(self) -> int:
        return self._id

    @property
    def map_(self) -> Map:
        return self._map

    @property
    def perception_range(self) -> Union[float, Tuple[float, float, float, float]]:
        return self._perception_range

    @property
    def max_perception_distance(self) -> float:
        return np.max(self.perception_range)

    @property
    def position(self) -> Point:
        return self._position

    @property
    def bind_id(self) -> Optional[int]:
        return self._bind_id

    @property
    def is_bound(self) -> bool:
        return self._bind_id is not None

    def bind_with(self, bind_id: int) -> None:
        """Bind the sensor to a participant with the given ID.

        Args:
            bind_id: ID of the participant to bind to.
        """
        self._bind_id = bind_id

    def _update_transform_matrix(self) -> None:
        """Update the transformation matrix for sensor coordinate conversion.

        The transformation matrix converts points from global coordinate system to sensor-local
        coordinate system, adjusting for sensor position and heading (theta = heading - π/2).
        """
        theta = self._heading - np.pi / 2

        self.transform_matrix = np.array(
            [
                np.cos(theta),
                np.sin(theta),
                np.sin(theta),
                -np.cos(theta),
                -self._position.x * np.cos(theta) - self._position.y * np.sin(theta),
                -self._position.x * np.sin(theta) + self._position.y * np.cos(theta),
            ]
        )

    @abstractmethod
    def update(
        self,
        frame: int,
        participants: Dict,
        participant_ids: List,
        prev_road_id_set: Set = None,
        prev_participant_id_set: Set = None,
        position: Point = None,
        heading: float = None,
    ) -> Tuple[Dict, Set, Set]:
        """This function is used to update the sensor's position and observation.

        Args:
            frame (int): The frame number of the observation.
            participants (Dict[int, Participant]): The participants in the scenario.
            participant_ids (List[int]): The list of participant IDs to be considered.
            prev_road_id_set (Set, optional): The set of road IDs that were rendered in the previous frame.
                Used for incremental updates. Defaults to None.
            prev_participant_id_set (Set, optional): The set of participant IDs that were rendered in the previous frame.
                Used for incremental updates. Defaults to None.
            position (Point, optional): The position of the sensor. Defaults to None.
            heading (float, optional): The heading of the sensor in radians. Defaults to None.

        Returns:
            Tuple[Dict, Set, Set]: (geometry_data, road_id_set, participant_id_set)
                - geometry_data: A dictionary containing the geometry data for rendering
                - road_id_set: The set of road IDs that were rendered in the current frame
                - participant_id_set: The set of participant IDs that were rendered in the current frame
        """

    def _setup_update_parameters(
        self,
        participant_ids: Optional[List[int]],
        prev_road_id_set: Optional[Set[int]],
        prev_participant_id_set: Optional[Set[int]],
    ) -> Tuple[List[int], Set[int], Set[int]]:
        """Setup default values for update parameters.

        Args:
            participant_ids: List of participant IDs or None
            prev_road_id_set: Set of road IDs from previous frame or None
            prev_participant_id_set: Set of participant IDs from previous frame or None

        Returns:
            Tuple of (participant_ids, prev_road_id_set, prev_participant_id_set) with defaults applied
        """
        if participant_ids is None:
            participant_ids = []
        if prev_road_id_set is None:
            prev_road_id_set = set()
        if prev_participant_id_set is None:
            prev_participant_id_set = set()

        return participant_ids, prev_road_id_set, prev_participant_id_set

    def _set_position_heading(self, position: Optional[Point], heading: Optional[float]) -> None:
        """Set sensor position and heading with default values.

        If position or heading is None, sets default position (map center) and heading (π/2).
        Calls _update_transform_matrix() after setting values.

        Args:
            position: Point or None
            heading: float or None
        """
        self._position = position
        self._heading = heading

        if None in [self._position, self._heading]:
            # Set default position to map center
            self._position = Point(
                0.5 * (self.map_.boundary[0] + self.map_.boundary[1]),
                0.5 * (self.map_.boundary[2] + self.map_.boundary[3]),
            )
            self._heading = 0

        self._update_transform_matrix()
