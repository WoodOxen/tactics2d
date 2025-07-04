##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: sensor_base.py
# @Description: This file defines a base interface for sensors.
# @Author: Tactics2D Team
# @Version: 0.1.9

from abc import ABC, abstractmethod
from typing import Tuple, Union

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
    def id_(self):
        return self._id

    @property
    def map_(self):
        return self._map

    @property
    def perception_range(self):
        return self._perception_range

    @property
    def max_perception_distance(self):
        return np.max(self.perception_range)

    @property
    def position(self):
        return self._position

    @property
    def bind_id(self):
        return self._bind_id

    @property
    def is_bound(self):
        return self._bind_id is not None

    def bind_with(self, bind_id):
        self._bind_id = bind_id

    @abstractmethod
    def update(self, participants, position: Point = None):
        """This function is used to update the sensor's position and observation.

        Args:
            participants (Dict[int, Participant]): The participants in the scenario.
            position (Point, optional): The position of the sensor. Defaults to None.
        """
