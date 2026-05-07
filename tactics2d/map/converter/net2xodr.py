# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml to OpenDRIVE xodr converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from shapely.geometry import LineString

from tactics2d.interpolator.param_poly3 import ParamPoly3, split_polyline
from tactics2d.map.parser import NetXMLParser


class Net2XodrConverter:
    """This class implements a converter from SUMO (.net.xml) to OpenDRIVE (.xodr). The converter reads a SUMO net.xml file using NetXMLParser, then writes the parsed map into OpenDRIVE xodr format. Each Tactics2D Lane becomes an OpenDRIVE road. The lane centre-line is derived from left and right boundaries and fitted to a paramPoly3 geometry per segment for compact, accurate output. Lane width is estimated as the mean point-to-point distance between boundaries.

    Example:
    ```python
    from tactics2d.map.converter import Net2XodrConverter

    converter = Net2XodrConverter()
    converter.convert("path/to/map.net.xml", "path/to/output.xodr")
    ```
    """

    _MAX_SEG_LENGTH = 20.0

    def _get_centerline(self, lane, n: int = 50) -> list:
        """Derive centre-line from left and right lane boundaries.

        Args:
            lane: Tactics2D Lane object.
            n (int): Number of sample points. Defaults to 50.

        Returns:
            list: List of (x, y) tuples. Empty if boundaries are degenerate.
        """
        left = lane.left_side
        right = lane.right_side
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
                / 2,
                (left.interpolate(s, normalized=True).y + right.interpolate(s, normalized=True).y)
                / 2,
            )
            for s in s_vals
        ]

    def _get_width(self, lane, n: int = 10) -> float:
        """Estimate lane width as mean point-to-point distance between boundaries.

        Args:
            lane: Tactics2D Lane object.
            n (int): Number of sample points. Defaults to 10.

        Returns:
            float: Estimated lane width in metres.
        """
        left = lane.left_side
        right = lane.right_side
        if left is None or right is None:
            return 3.2
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

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a SUMO net.xml file to an OpenDRIVE xodr file.

        Reads the SUMO network file with NetXMLParser, maps the resulting
        Tactics2D Map to OpenDRIVE elements, and writes the output file.
        Each Tactics2D Lane becomes an OpenDRIVE road with a single driving
        lane. Centre-lines are fitted to paramPoly3 geometries for accuracy.
        Junctions and connections are preserved.

        Args:
            input_path (str): Path to the input .net.xml file.
            output_path (str): Path to the output .xodr file.

        Returns:
            str: The output file path.

        Example:
        ```python
        from tactics2d.map.converter import Net2XodrConverter

        converter = Net2XodrConverter()
        converter.convert("map.net.xml", "map.xodr")
        ```
        """
        map_ = NetXMLParser().parse(input_path)
        root = ET.Element("OpenDRIVE")

        ET.SubElement(
            root,
            "header",
            {
                "revMajor": "1",
                "revMinor": "6",
                "name": "",
                "version": "1.00",
                "date": "",
                "north": "0",
                "south": "0",
                "east": "0",
                "west": "0",
            },
        )

        for lane_id, lane in map_.lanes.items():
            pts = self._get_centerline(lane)
            if len(pts) < 2:
                continue

            width = self._get_width(lane)
            speed = lane.speed_limit if lane.speed_limit else 50.0 / 3.6
            sumo_id = (lane.custom_tags or {}).get("sumo_id", str(lane_id))
            total_length = float(LineString(pts).length)

            road = ET.SubElement(
                root,
                "road",
                {
                    "name": sumo_id,
                    "length": f"{total_length:.4f}",
                    "id": str(lane_id),
                    "junction": "-1",
                },
            )

            ET.SubElement(road, "type", {"s": "0.0", "type": "town"})

            plan_view = ET.SubElement(road, "planView")
            s_offset = 0.0
            for seg in split_polyline(pts, self._MAX_SEG_LENGTH):
                fit = ParamPoly3.fit(seg)
                if fit is None:
                    continue
                x, y, hdg, length, aU, bU, cU, dU, aV, bV, cV, dV = fit
                geom = ET.SubElement(
                    plan_view,
                    "geometry",
                    {
                        "s": f"{s_offset:.4f}",
                        "x": f"{x:.4f}",
                        "y": f"{y:.4f}",
                        "hdg": f"{hdg:.6f}",
                        "length": f"{length:.4f}",
                    },
                )
                pp3 = ET.SubElement(geom, "paramPoly3", {"pRange": "normalized"})
                for k, v in zip(
                    ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV"),
                    (aU, bU, cU, dU, aV, bV, cV, dV),
                ):
                    pp3.set(k, f"{v:.6f}")
                s_offset += length

            ET.SubElement(road, "elevationProfile")
            ET.SubElement(road, "lateralProfile")

            lanes_elem = ET.SubElement(road, "lanes")
            lane_section = ET.SubElement(lanes_elem, "laneSection", {"s": "0.0"})
            ET.SubElement(lane_section, "left")

            center = ET.SubElement(lane_section, "center")
            center_lane = ET.SubElement(
                center, "lane", {"id": "0", "type": "none", "level": "false"}
            )
            ET.SubElement(
                center_lane,
                "roadMark",
                {
                    "sOffset": "0",
                    "type": "solid",
                    "weight": "standard",
                    "color": "standard",
                    "width": "0.13",
                },
            )

            right_elem = ET.SubElement(lane_section, "right")
            right_lane = ET.SubElement(
                right_elem,
                "lane",
                {"id": "-1", "type": lane.subtype if lane.subtype else "driving", "level": "false"},
            )
            ET.SubElement(
                right_lane,
                "width",
                {"sOffset": "0", "a": f"{width:.4f}", "b": "0", "c": "0", "d": "0"},
            )
            ET.SubElement(
                right_lane,
                "roadMark",
                {
                    "sOffset": "0",
                    "type": "solid",
                    "weight": "standard",
                    "color": "standard",
                    "width": "0.13",
                },
            )
            ET.SubElement(
                right_lane, "speed", {"sOffset": "0", "max": f"{speed * 3.6:.3f}", "unit": "km/h"}
            )

        for junc_id, junction in map_.junctions.items():
            junc_elem = ET.SubElement(root, "junction", {"name": "", "id": str(junc_id)})
            for conn_id, conn in junction.connections.items():
                tags = conn.custom_tags or {}
                from_edge = tags.get("from_edge", "")
                to_edge = tags.get("to_edge", "")
                from_lane = tags.get("from_lane", "0")
                to_lane = tags.get("to_lane", "0")
                if not from_edge or not to_edge:
                    continue
                conn_elem = ET.SubElement(
                    junc_elem,
                    "connection",
                    {
                        "id": str(conn_id),
                        "incomingRoad": from_edge,
                        "connectingRoad": to_edge,
                        "contactPoint": "start",
                    },
                )
                ET.SubElement(
                    conn_elem,
                    "laneLink",
                    {
                        "from": f"-{from_lane}" if from_lane != "0" else "-1",
                        "to": f"-{to_lane}" if to_lane != "0" else "-1",
                    },
                )

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
