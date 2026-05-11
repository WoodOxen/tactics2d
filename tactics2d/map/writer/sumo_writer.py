# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml writer.

Writes a Tactics2D ``Map`` object to a SUMO-compatible net.xml XML tree.
Each ``Lane`` becomes a SUMO ``<lane>`` element. Lanes sharing the same
SUMO edge id (recovered from ``lane.custom_tags["sumo_id"]``) are grouped
under a single ``<edge>`` element. Lanes without a ``sumo_id`` each become
their own single-lane edge.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString


class SumoWriter:
    """Writes a Tactics2D Map to a SUMO net.xml XML tree.

    Lanes that carry a ``sumo_id`` in their ``custom_tags`` are grouped by
    the edge portion of that id (everything before the last ``_``). Lanes
    without a ``sumo_id`` are each written as a single-lane edge using their
    Tactics2D id as the edge id.

    Lane centre-lines are taken from ``custom_tags["centerline"]`` when
    available, and fall back to interpolation from the left and right boundary
    geometries otherwise.

    Example:
    ```python
    from tactics2d.map.parser import NetXMLParser
    from tactics2d.map.writer import SumoWriter

    map_ = NetXMLParser().parse("map.net.xml")
    root = SumoWriter().build(map_)
    ```
    """

    _DEFAULT_SPEED: float = 50.0 / 3.6
    _DEFAULT_WIDTH: float = 3.2
    _WIDTH_SAMPLES: int = 10

    def build(self, map_: Map) -> ET.Element:
        root = ET.Element(
            "net",
            {
                "version": "1.9",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": ("http://sumo.dlr.de/xsd/net_file.xsd"),
            },
        )

        self.write_location(root, map_.boundary)

        for edge_id, lanes in self._group_lanes(map_).items():
            self.write_edge(root, edge_id, lanes)

        for junction in map_.junctions.values():
            self.write_junction(root, junction)
            for conn in junction.connections.values():
                self.write_connection(root, conn)

        return root

    def write_location(self, root: ET.Element, boundary: tuple) -> None:
        min_x, max_x, min_y, max_y = boundary
        conv = f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}"
        ET.SubElement(
            root,
            "location",
            {
                "netOffset": "0.00,0.00",
                "convBoundary": conv,
                "origBoundary": conv,
                "projParameter": "!",
            },
        )

    def write_edge(self, root: ET.Element, edge_id: str, lanes: list) -> None:
        tags = lanes[0].custom_tags or {}
        edge = ET.SubElement(
            root,
            "edge",
            {
                "id": edge_id,
                "from": tags.get("from_node", ""),
                "to": tags.get("to_node", ""),
                "priority": "1",
            },
        )

        sorted_lanes = sorted(
            lanes,
            key=lambda ln: (
                int((ln.custom_tags or {}).get("sumo_id", "").rsplit("_", 1)[-1])
                if (ln.custom_tags or {}).get("sumo_id", "").rsplit("_", 1)[-1].isdigit()
                else 0
            ),
        )

        for index, lane in enumerate(sorted_lanes):
            self.write_lane(edge, lane, index)

    def write_lane(self, edge: ET.Element, lane: Lane, index: int) -> None:
        sumo_id = (lane.custom_tags or {}).get("sumo_id") or f"{edge.get('id')}_{index}"
        centerline = self._get_centerline(lane)
        speed = lane.speed_limit if lane.speed_limit else self._DEFAULT_SPEED
        width = self._get_width(lane)
        length = float(LineString(centerline).length) if len(centerline) >= 2 else 0.0

        ET.SubElement(
            edge,
            "lane",
            {
                "id": sumo_id,
                "index": str(index),
                "speed": f"{speed:.2f}",
                "length": f"{length:.2f}",
                "width": f"{width:.2f}",
                "shape": self._shape_to_str(centerline),
            },
        )

    def write_junction(self, root: ET.Element, junction: Junction) -> None:
        tags = junction.custom_tags or {}
        shape_pts = tags.get("shape", [])
        ET.SubElement(
            root,
            "junction",
            {
                "id": tags.get("sumo_id", str(junction.id_)),
                "type": tags.get("type", "priority"),
                "x": str(tags.get("x", "0")),
                "y": str(tags.get("y", "0")),
                "incLanes": "",
                "intLanes": "",
                "shape": self._shape_to_str(shape_pts),
            },
        )

    def write_connection(self, root: ET.Element, conn: Junction) -> None:
        tags = conn.custom_tags or {}
        from_edge = tags.get("from_edge", "")
        to_edge = tags.get("to_edge", "")
        if not from_edge or not to_edge:
            return
        ET.SubElement(
            root,
            "connection",
            {
                "from": from_edge,
                "to": to_edge,
                "fromLane": tags.get("from_lane", "0"),
                "toLane": tags.get("to_lane", "0"),
                "dir": tags.get("dir", "s"),
                "state": tags.get("state", "M"),
            },
        )

    @staticmethod
    def _group_lanes(map_: Map) -> dict[str, list]:
        groups: dict[str, list] = defaultdict(list)
        for lane in map_.lanes.values():
            sumo_id = (lane.custom_tags or {}).get("sumo_id", "")
            if sumo_id and "_" in sumo_id:
                edge_id = sumo_id.rsplit("_", 1)[0]
            else:
                edge_id = sumo_id or str(lane.id_)
            groups[edge_id].append(lane)
        return dict(groups)

    @staticmethod
    def _get_centerline(lane: Lane) -> list:
        tags = getattr(lane, "custom_tags", {}) or {}
        centerline = tags.get("centerline")
        if centerline and len(centerline) >= 2:
            return centerline

        left, right = lane.left_side, lane.right_side
        if left is None or right is None:
            return []
        length = min(left.length, right.length)
        if length < 1e-6:
            return []
        n = max(2, int(length / 0.5))
        s_vals = np.linspace(0, 1, n)
        return [
            (
                (left.interpolate(s, normalized=True).x + right.interpolate(s, normalized=True).x)
                / 2.0,
                (left.interpolate(s, normalized=True).y + right.interpolate(s, normalized=True).y)
                / 2.0,
            )
            for s in s_vals
        ]

    @staticmethod
    def _get_width(lane: Lane) -> float:
        left, right = lane.left_side, lane.right_side
        if left is None or right is None:
            return SumoWriter._DEFAULT_WIDTH
        s_vals = np.linspace(0, 1, SumoWriter._WIDTH_SAMPLES)
        return float(
            np.mean(
                [
                    left.interpolate(s, normalized=True).distance(
                        right.interpolate(s, normalized=True)
                    )
                    for s in s_vals
                ]
            )
        )

    @staticmethod
    def _shape_to_str(coords: list) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
