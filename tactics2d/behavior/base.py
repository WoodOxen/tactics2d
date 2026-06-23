# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared behavior model interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

from tactics2d.map.element import Map
from tactics2d.participant.trajectory import Trajectory


class BehaviorModelBase(ABC):
    """Base interface for Tactics2D behavior models."""

    @abstractmethod
    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Plan future trajectories for the specified traffic participants."""
