# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Common data types for socket-driven road element generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tactics2d.map.element import Area, Junction, Lane, RoadLine

PortKind = Literal[
    "generic",
    "entry",
    "exit",
    "main_in",
    "main_out",
    "forward_in",
    "forward_out",
    "backward_in",
    "backward_out",
    "ramp",
    "ramp_in",
    "ramp_out",
    "fork_main_in",
    "fork_main_out",
    "fork_branch_out",
    "merge_main_in",
    "merge_branch_in",
    "merge_main_out",
    "intersection_in",
    "intersection_out",
    "roundabout_in",
    "roundabout_out",
    "adapter_in",
    "adapter_out",
]

RampKind = Literal["exit", "entrance"]
MainRoadType = Literal["freeway", "urban"]
RampSide = Literal["right", "left"]


@dataclass
class RoadPort:
    """Connection socket of a road module.

    Ports describe where one road module connects to another.  Generators
    consume input ports and produce output ports; assemblers match output ports
    to input ports by comparing ``point`` and ``heading``.

    Attributes:
        point: Socket position in world coordinates, shape ``(2,)``.
        heading: Travel direction at the socket in radians, measured from the
            positive x-axis counter-clockwise.
        lane_num: Number of lanes exposed by this port.
        lane_width: Width of each lane in metres.
        speed_limit: Speed limit at the socket in km/h.
        kind: Semantic port type used by assemblers to validate connections.
        name: Human-readable label for logging and debugging.
        lane_ids: Ids of the lanes associated with this socket, ordered left to
            right relative to the travel direction.
        metadata: Arbitrary key-value payload for generator-specific
            bookkeeping (e.g. module name, branch side).  Not interpreted by
            the assembler.
    """

    point: np.ndarray
    heading: float
    lane_num: int
    lane_width: float = 3.5
    speed_limit: float = 50.0
    kind: PortKind = "generic"
    name: str = ""
    lane_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise scalar and collection fields to canonical Python types."""
        self.point = np.asarray(self.point, dtype=float)
        self.heading = float(self.heading)
        self.lane_num = int(self.lane_num)
        self.lane_width = float(self.lane_width)
        self.speed_limit = float(self.speed_limit)
        self.lane_ids = tuple(str(lane_id) for lane_id in self.lane_ids)


@dataclass
class RoadModuleResult:
    """Generated road module result.

    Attributes:
        lanes: Generated Lane objects.
        roadlines: Generated RoadLine objects.
        ports: Named RoadPort dictionary, keyed by logical port name.
        id_counter: Next available element id.  Pass as ``id_offset`` to the
            next generator to prevent id collisions across modules.
        junctions: Generated Junction objects, e.g. intersection pavement polygons
            and roundabout rings.
        areas: Generated Area objects, e.g. gore islands and surface fill polygons.
    """

    lanes: list[Lane]
    roadlines: list[RoadLine]
    ports: dict[str, RoadPort]
    id_counter: int
    junctions: list[Junction] = field(default_factory=list)
    areas: list[Area] = field(default_factory=list)


def make_port(
    base_port: RoadPort,
    *,
    kind: str | None = None,
    name: str | None = None,
    lane_ids: tuple[str, ...] | list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RoadPort:
    """Create a derived ``RoadPort`` from a base port with updated semantic fields.

    Geometry (``point``, ``heading``, ``lane_num``, ``lane_width``,
    ``speed_limit``) is copied verbatim.  ``metadata`` is merged: entries from
    ``base_port.metadata`` are overlaid with entries from ``metadata``, with
    the latter taking priority.

    Args:
        base_port: Source port whose geometry and lane attributes are copied.
        kind: Override for ``RoadPort.kind``. Defaults to ``base_port.kind``.
        name: Override for ``RoadPort.name``. Defaults to ``base_port.name``.
        lane_ids: Override for ``RoadPort.lane_ids``. Defaults to
            ``base_port.lane_ids``.
        metadata: Additional metadata merged on top of ``base_port.metadata``.
            Keys present in both take the value from this argument.

    Returns:
        A new ``RoadPort`` with the same geometry as ``base_port`` and the
        updated semantic fields.
    """
    merged = dict(base_port.metadata)
    if metadata is not None:
        merged.update(metadata)

    return RoadPort(
        point=np.asarray(base_port.point, dtype=float),
        heading=float(base_port.heading),
        lane_num=int(base_port.lane_num),
        lane_width=float(base_port.lane_width),
        speed_limit=float(base_port.speed_limit),
        kind=kind if kind is not None else base_port.kind,
        name=name if name is not None else base_port.name,
        lane_ids=tuple(lane_ids) if lane_ids is not None else tuple(base_port.lane_ids),
        metadata=merged,
    )
