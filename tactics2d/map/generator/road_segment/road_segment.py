# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""RoadSegment abstract base class implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..rules.module_types import RoadModuleResult, RoadPort


class RoadSegment(ABC):
    """Abstract base for all road segment generators.

    Each concrete subclass encapsulates structural configuration in ``__init__``
    (lane side, taper geometry, sampling interval, etc.) and exposes a single
    ``build`` method that receives the runtime connection sockets and returns a
    :class:`~tactics2d.map.generator.rules.module_types.RoadModuleResult`.

    Typical usage::

        seg = Fork(fork_side="right", taper_length=65.0)
        result = seg.build(main_in, main_out, branch_out, id_offset=0)
        next_result = seg.build(p1, p2, p3, id_offset=result.id_counter)
    """

    @abstractmethod
    def build(self, *args: RoadPort, id_offset: int = 0, **kwargs) -> RoadModuleResult:
        """Build the road segment from the given port sockets.

        Args:
            *args: Positional :class:`RoadPort` sockets. The exact sequence and
                semantics depend on the concrete subclass.
            id_offset: First element id assigned by the generator. Pass the
                previous result's ``id_counter`` to chain calls without id
                collisions.
            **kwargs: Per-call overrides such as lane counts or speed limits;
                each subclass documents which keys are accepted.

        Returns:
            A :class:`RoadModuleResult` containing lanes, roadlines, ports,
            interfaces, quality statistics, and the next available id counter.
        """
