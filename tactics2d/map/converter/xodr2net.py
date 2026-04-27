# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE xodr to SUMO net.xml converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.parser import XODRParser

_SKIP_SUBTYPES = {"roadmark", "virtual", "none"}
_MIN_LANE_WIDTH = 0.5  # excludes degenerate lanes and narrow boundary elements (e.g. border ~0.34 m); explicit subtype filtering may be added in future


class Xodr2NetConverter:
    """This class implements a converter from OpenDRIVE (.xodr) to SUMO (.net.xml).

    The converter reads an OpenDRIVE file using XODRParser, then writes the
    parsed map into SUMO net.xml format. Lane centre-lines are derived from
    the mean of left and right boundary polylines. Lane widths are estimated
    as the mean point-to-point distance between boundary samples. Non-drivable
    elements such as virtual boundaries and road markings are filtered out by
    subtype and minimum width threshold. Connections are resolved using the
    xodr_road_id stored in each lane's custom_tags, matched against the
    incoming_road and connecting_road fields of each junction connection.

    Example:
        >>> from tactics2d.map.converter import Xodr2NetConverter
        >>> converter = Xodr2NetConverter()
        >>> converter.convert("path/to/map.xodr", "path/to/output.net.xml")
    """

    def _boundary_to_centerline(self, left: LineString, right: LineString, n: int = 50) -> list:
        """Derive a centre-line from left and right lane boundaries.

        Args:
            left (LineString): Left boundary of the lane.
            right (LineString): Right boundary of the lane.
            n (int): Number of sample points. Defaults to 50.

        Returns:
            list: List of (x, y) tuples representing the centre-line.
                Returns an empty list if either boundary is degenerate.
        """
        if min(left.length, right.length) < 1e-6:
            return []
        s_vals = np.linspace(0, 1, n)
        return [
            (
                (left.interpolate(s, normalized=True).x + right.interpolate(s, normalized=True).x)
                / 2,
                (left.interpolate(s, normalized=True).y + right.interpolate(s, normalized=True).y)
                / 2,
            )
            for s in s_vals
        ]

    def _boundary_to_width(self, left: LineString, right: LineString, n: int = 10) -> float:
        """Estimate lane width as mean point-to-point distance between boundaries.

        Args:
            left (LineString): Left boundary of the lane.
            right (LineString): Right boundary of the lane.
            n (int): Number of sample points. Defaults to 10.

        Returns:
            float: Estimated lane width in metres.
        """
        s_vals = np.linspace(0, 1, n)
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

    def _shape_to_str(self, coords: list) -> str:
        """Convert a list of (x, y) tuples to a SUMO shape string.

        Args:
            coords (list): List of (x, y) coordinate tuples.

        Returns:
            str: Space-separated coordinate pairs, e.g. '0.00,1.00 2.00,3.00'.
        """
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenDRIVE file to a SUMO net.xml file.

        Reads the OpenDRIVE file with XODRParser, maps the resulting
        Tactics2D Map to SUMO net.xml elements, and writes the output file.
        Each drivable Tactics2D Lane becomes a SUMO edge with a single lane
        child. Connections are resolved by matching xodr road ids stored in
        lane custom_tags against junction connection incoming_road and
        connecting_road fields, using lane_links for precise lane-level mapping.

        Args:
            input_path (str): Path to the input .xodr file.
            output_path (str): Path to the output .net.xml file.

        Returns:
            str: The output file path.

        Example:
            >>> from tactics2d.map.converter import Xodr2NetConverter
            >>> converter = Xodr2NetConverter()
            >>> converter.convert("map.xodr", "map.net.xml")
        """
        map_ = XODRParser().parse(input_path)
        boundary = map_.boundary

        root = ET.Element(
            "net",
            {
                "version": "1.9",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/net_file.xsd",
            },
        )

        ET.SubElement(
            root,
            "location",
            {
                "netOffset": "0.00,0.00",
                "convBoundary": f"{boundary[0]:.2f},{boundary[2]:.2f},{boundary[1]:.2f},{boundary[3]:.2f}",
                "origBoundary": f"{boundary[0]:.2f},{boundary[2]:.2f},{boundary[1]:.2f},{boundary[3]:.2f}",
                "projParameter": "!",
            },
        )

        # write edges
        written_lane_ids = set()
        for lane_id, lane in map_.lanes.items():
            if lane.subtype in _SKIP_SUBTYPES:
                continue
            if lane.left_side is None or lane.right_side is None:
                continue

            width = self._boundary_to_width(lane.left_side, lane.right_side)
            if width < _MIN_LANE_WIDTH:
                continue

            center = self._boundary_to_centerline(lane.left_side, lane.right_side)
            if len(center) < 2:
                continue

            speed = (lane.speed_limit / 3.6) if lane.speed_limit else 13.89

            edge = ET.SubElement(root, "edge", {"id": str(lane_id), "priority": "1"})
            ET.SubElement(
                edge,
                "lane",
                {
                    "id": f"{lane_id}_0",
                    "index": "0",
                    "speed": f"{speed:.2f}",
                    "length": f"{lane.left_side.length:.2f}",
                    "width": f"{width:.2f}",
                    "shape": self._shape_to_str(center),
                },
            )
            written_lane_ids.add(lane_id)

        # build xodr road id -> tactics2d lane id list mapping
        road_to_lanes: dict[str, list] = {}
        for lane_id, lane in map_.lanes.items():
            if lane_id not in written_lane_ids:
                continue
            tags = lane.custom_tags or {}
            rid = tags.get("xodr_road_id", "")
            if not rid:
                continue
            road_to_lanes.setdefault(rid, []).append(lane_id)

        # write junctions and connections
        for junc_id, junction in map_.junctions.items():
            tags = junction.custom_tags or {}
            ET.SubElement(
                root,
                "junction",
                {
                    "id": str(junc_id),
                    "type": tags.get("type", "priority"),
                    "x": tags.get("x", "0"),
                    "y": tags.get("y", "0"),
                    "incLanes": "",
                    "intLanes": "",
                    "shape": "",
                },
            )

            for conn in junction.connections.values():
                incoming_road = str(conn.incoming_road) if conn.incoming_road else ""
                connecting_road = str(conn.connecting_road) if conn.connecting_road else ""
                if not incoming_road or not connecting_road:
                    continue

                from_lanes = road_to_lanes.get(incoming_road, [])
                to_lanes = road_to_lanes.get(connecting_road, [])
                if not from_lanes or not to_lanes:
                    continue

                if conn.lane_links:
                    for from_idx, to_idx in conn.lane_links:
                        try:
                            from_idx = int(from_idx)
                            to_idx = int(to_idx)
                        except (ValueError, TypeError):
                            continue

                        fi = max(min(abs(from_idx), len(from_lanes)) - 1, 0)
                        ti = max(min(abs(to_idx), len(to_lanes)) - 1, 0)

                        ET.SubElement(
                            root,
                            "connection",
                            {
                                "from": str(from_lanes[fi]),
                                "to": str(to_lanes[ti]),
                                "fromLane": "0",
                                "toLane": "0",
                                "dir": "s",
                                "state": "M",
                            },
                        )
                else:
                    ET.SubElement(
                        root,
                        "connection",
                        {
                            "from": str(from_lanes[0]),
                            "to": str(to_lanes[0]),
                            "fromLane": "0",
                            "toLane": "0",
                            "dir": "s",
                            "state": "M",
                        },
                    )

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )

        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
