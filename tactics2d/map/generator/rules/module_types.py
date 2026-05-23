# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Common data types for socket-driven road element generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tactics2d.map.element import Junction, Lane, RoadLine

PortKind = Literal[
    "generic",
    "entry",
    "exit",
    "main_in",
    "main_out",
    "backward_in",
    "backward_out",
    "ramp",
    "ramp_in",
    "ramp_out",
    "intersection_in",
    "intersection_out",
    "roundabout_in",
    "roundabout_out",
    "adapter_in",
    "adapter_out",
]


@dataclass
class RoadPort:
    """Connection port of a road module.

    Args:
        point: Port point in world coordinates.
        heading: Port heading in radians.
        lane_num: Number of lanes exposed by this port.
        lane_width: Lane width in metres.
        speed_limit: Speed limit in km/h.
        kind: Semantic port kind.
        name: Human-readable port name.
        lane_ids: Lane ids associated with this port.
        metadata: Additional metadata used by generators and assemblers.
        custom_tags: Compatibility alias for metadata.
    """

    point: np.ndarray
    heading: float
    lane_num: int
    lane_width: float = 3.5
    speed_limit: float = 50.0
    kind: str = "generic"
    name: str = ""
    lane_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    custom_tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize fields and keep metadata/custom_tags compatible."""
        self.point = np.asarray(self.point, dtype=float)
        self.heading = float(self.heading)
        self.lane_num = int(self.lane_num)
        self.lane_width = float(self.lane_width)
        self.speed_limit = float(self.speed_limit)
        self.lane_ids = tuple(str(lane_id) for lane_id in self.lane_ids)

        if self.metadata and not self.custom_tags:
            self.custom_tags = dict(self.metadata)
        elif self.custom_tags and not self.metadata:
            self.metadata = dict(self.custom_tags)
        elif self.metadata and self.custom_tags:
            merged = dict(self.metadata)
            merged.update(self.custom_tags)
            self.metadata = merged
            self.custom_tags = dict(merged)


@dataclass
class RoadModuleResult:
    """Generated road module result.

    Args:
        lanes: Generated Lane objects.
        roadlines: Generated RoadLine objects.
        ports: Named RoadPort dictionary.
        interfaces: Legacy interface dictionaries for tests and old callers.
        quality: Geometry/topology quality statistics.
        id_counter: Next available id.
        junctions: Generated Junction objects.
    """

    lanes: list[Lane]
    roadlines: list[RoadLine]
    ports: dict[str, RoadPort]
    interfaces: list[dict[str, Any]]
    quality: dict[str, Any]
    id_counter: int
    junctions: list[Junction] = field(default_factory=list)


def make_port(
    base_port: RoadPort,
    *,
    kind: str | None = None,
    name: str | None = None,
    lane_ids: tuple[str, ...] | list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> RoadPort:
    """Create a new RoadPort from a base port with updated semantic fields."""
    merged_metadata = dict(getattr(base_port, "metadata", {}))

    if getattr(base_port, "custom_tags", None):
        merged_metadata.update(base_port.custom_tags)

    if metadata is not None:
        merged_metadata.update(metadata)

    if custom_tags is not None:
        merged_metadata.update(custom_tags)

    return RoadPort(
        point=np.asarray(base_port.point, dtype=float),
        heading=float(base_port.heading),
        lane_num=int(base_port.lane_num),
        lane_width=float(base_port.lane_width),
        speed_limit=float(base_port.speed_limit),
        kind=kind if kind is not None else getattr(base_port, "kind", "generic"),
        name=name if name is not None else getattr(base_port, "name", ""),
        lane_ids=tuple(lane_ids) if lane_ids is not None else tuple(base_port.lane_ids),
        metadata=merged_metadata,
        custom_tags=dict(merged_metadata),
    )


def ports_to_interfaces(ports: dict[str, RoadPort]) -> list[dict[str, Any]]:
    """Convert RoadPort objects into legacy interface dictionaries."""
    interfaces = []

    for key, port in ports.items():
        metadata = dict(getattr(port, "metadata", {}))

        if getattr(port, "custom_tags", None):
            metadata.update(port.custom_tags)

        interfaces.append(
            {
                "name": port.name or key,
                "kind": port.kind,
                "point": np.asarray(port.point, dtype=float),
                "heading": float(port.heading),
                "lane_num": int(port.lane_num),
                "lane_width": float(port.lane_width),
                "speed_limit": float(port.speed_limit),
                "lane_ids": list(port.lane_ids),
                "metadata": metadata,
                "custom_tags": dict(metadata),
            }
        )

    return interfaces
