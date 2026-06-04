# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base classes for pygame-backed renderer sensors."""


import logging
from typing import Tuple, Union

import numpy as np
import pygame
from shapely.geometry import Point

from tactics2d.map.element import Map


class PygameSensorBase:
    """Base interface for sensors rendered with pygame surfaces."""

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: Union[float, Tuple[float]] = None,
        window_size: Tuple[int, int] = (200, 200),
        off_screen: bool = True,
    ):
        self.id_ = id_
        self.map_ = map_
        self.off_screen = off_screen
        self.window_size = window_size
        self._surface = pygame.Surface(self.window_size)
        self._bind_id = None
        self._heading = None
        self._position = None

        if perception_range is None:
            width = (map_.boundary[1] - map_.boundary[0]) / 2
            height = (map_.boundary[3] - map_.boundary[2]) / 2
            self.perception_range = (width, width, height, height)
        elif isinstance(perception_range, (float, int)):
            self.perception_range = (
                perception_range,
                perception_range,
                perception_range,
                perception_range,
            )
        else:
            self.perception_range = perception_range

        perception_width = self.perception_range[0] + self.perception_range[1]
        perception_height = self.perception_range[2] + self.perception_range[3]

        scale_width = window_size[0] / perception_width
        scale_height = window_size[1] / perception_height
        self.scale = max(scale_width, scale_height)

        if scale_width != scale_height:
            logging.warning(
                "The x-y proportion of the perception range and the rendering window "
                "is inconsistent."
            )

    @property
    def heading(self):
        return self._heading

    @property
    def position(self):
        return self._position

    @property
    def max_perception_distance(self):
        return np.max(self.perception_range)

    @property
    def surface(self):
        return self._surface

    @property
    def bind_id(self):
        return self._bind_id

    def set_bind_id(self, bind_id):
        self._bind_id = bind_id

    def update(self, participants, participant_ids: list, frame: int = None):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
