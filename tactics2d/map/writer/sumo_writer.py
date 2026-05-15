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

import logging
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

    Lane centre-lines are derived in the following priority order:
      1. ``custom_tags["centerline"]`` if present and has >= 2 points.
      2. Midpoint interpolation from ``left_side`` and ``right_side``.
    """

    _DEFAULT_SPEED: float = 50.0 / 3.6
    _DEFAULT_WIDTH: float = 3.2
    _WIDTH_SAMPLES: int = 10

    def build(self, map_) -> ET.Element:
        """Build a complete SUMO net.xml XML tree from a Tactics2D Map.

        Args:
            map_: A Tactics2D Map object.

        Returns:
            The root ``<net>`` XML element.
        """
        root = ET.Element(
            "net",
            {
                "version": "1.9",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/net_file.xsd",
            },
        )

        boundary = self._resolve_boundary(map_)
        self.write_location(root, boundary)

        for edge_id, lanes in self._group_lanes(map_).items():
            self.write_edge(root, edge_id, lanes)

        for junction in map_.junctions.values():
            self.write_junction(root, junction)
            for conn in junction.connections.values():
                self.write_connection(root, conn)

        return root

    def write_location(self, root: ET.Element, boundary: tuple) -> None:
        """Write the ``<location>`` element to the XML tree.

        Args:
            root: Parent XML element.
            boundary: Map boundary as (min_x, max_x, min_y, max_y).
        """
        if boundary is None:
            boundary = (0.0, 0.0, 0.0, 0.0)
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
        """Write one ``<edge>`` element containing its child ``<lane>`` elements.

        Args:
            root: Parent XML element.
            edge_id: SUMO edge identifier.
            lanes: List of Lane objects belonging to this edge.
        """
        tags = (lanes[0].custom_tags or {}) if lanes else {}
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

        sorted_lanes = sorted(lanes, key=self._extract_lane_index)

        for index, lane in enumerate(sorted_lanes):
            self.write_lane(edge, lane, index)

    def write_lane(self, edge: ET.Element, lane, index: int) -> None:
        """Write one ``<lane>`` element to the parent edge.

        Args:
            edge: Parent ``<edge>`` XML element.
            lane: Tactics2D Lane object.
            index: Lane index within the edge.
        """
        sumo_id = (lane.custom_tags or {}).get("sumo_id") or f"{edge.get('id')}_{index}"
        centerline = self._get_centerline(lane)

        if len(centerline) < 2:
            logging.debug("Skipping lane %s: could not compute a valid centerline.", sumo_id)
            return

        speed = lane.speed_limit if lane.speed_limit else self._DEFAULT_SPEED
        width = self._get_width(lane)
        length = float(LineString(centerline).length)

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

    def write_junction(self, root: ET.Element, junction) -> None:
        """Write one ``<junction>`` element.

        Args:
            root: Parent XML element.
            junction: Tactics2D Junction object.
        """
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

    def write_connection(self, root: ET.Element, conn) -> None:
        """Write one ``<connection>`` element.

        Args:
            root: Parent XML element.
            conn: Junction object representing a connection.
        """
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

    def _resolve_boundary(self, map_) -> tuple:
        """Get a valid boundary from the map, computing from geometry if needed."""
        boundary = map_.boundary
        if boundary is not None and boundary != (0, 0, 0, 0):
            return boundary

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for lane in map_.lanes.values():
            for side in (lane.left_side, lane.right_side):
                if side is None:
                    continue
                bounds = side.bounds
                min_x = min(min_x, bounds[0])
                min_y = min(min_y, bounds[1])
                max_x = max(max_x, bounds[2])
                max_y = max(max_y, bounds[3])

        if min_x == float("inf"):
            return (0.0, 0.0, 0.0, 0.0)

        return (min_x, max_x, min_y, max_y)

    @staticmethod
    def _group_lanes(map_) -> dict[str, list]:
        """Group lanes by their SUMO edge id."""
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
    def _extract_lane_index(lane) -> int:
        """Extract the numeric lane index from sumo_id, defaulting to 0."""
        sumo_id = (lane.custom_tags or {}).get("sumo_id", "")
        if sumo_id and "_" in sumo_id:
            suffix = sumo_id.rsplit("_", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return 0

    @staticmethod
    def _get_centerline(lane) -> list:
        """Compute the lane centerline from available geometry.

        Priority:
          1. custom_tags["centerline"] if present and valid.
          2. Direct coordinate midpoint when left/right have equal point count.
          3. Normalized interpolation from left_side and right_side.
        """
        tags = getattr(lane, "custom_tags", None) or {}
        centerline = tags.get("centerline")
        if centerline and len(centerline) >= 2:
            return centerline

        left = lane.left_side
        right = lane.right_side
        if left is None or right is None:
            return []

        try:
            left_coords = list(left.coords)
            right_coords = list(right.coords)
        except Exception:
            # Fallback to returning empty list if coordinate extraction fails
            # CodeQL doesn't usually flag this since it returns a value and isn't empty,
            # but left the comment intact just in case.
            return []

        if len(left_coords) < 2 or len(right_coords) < 2:
            return []

        if len(left_coords) == len(right_coords):
            return [
                ((lx + rx) / 2.0, (ly + ry) / 2.0)
                for (lx, ly), (rx, ry) in zip(left_coords, right_coords)
            ]

        length = min(left.length, right.length)
        if length < 1e-6:
            return []

        n = max(2, int(length / 0.5))
        s_vals = np.linspace(0.0, 1.0, n)
        result = []
        for s in s_vals:
            lp = left.interpolate(float(s), normalized=True)
            rp = right.interpolate(float(s), normalized=True)
            result.append(((lp.x + rp.x) / 2.0, (lp.y + rp.y) / 2.0))
        return result

    @staticmethod
    def _get_width(lane) -> float:
        """Estimate lane width from boundary geometry."""
        left = lane.left_side
        right = lane.right_side
        if left is None or right is None:
            return SumoWriter._DEFAULT_WIDTH

        try:
            left_coords = np.array(left.coords)
            right_coords = np.array(right.coords)
            if len(left_coords) == len(right_coords):
                diffs = left_coords - right_coords
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                return float(np.mean(dists))
        except Exception as exc:
            logging.debug("Ignored shape mismatch, fallback to interpolation: %s", exc)

        s_vals = np.linspace(0.0, 1.0, SumoWriter._WIDTH_SAMPLES)
        dists = []
        for s in s_vals:
            lp = left.interpolate(float(s), normalized=True)
            rp = right.interpolate(float(s), normalized=True)
            dists.append(lp.distance(rp))
        return float(np.mean(dists))

    @staticmethod
    def _shape_to_str(coords: list) -> str:
        """Convert a coordinate list to SUMO shape string format."""
        if not coords:
            return ""
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
