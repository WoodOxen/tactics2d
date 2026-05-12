# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml to OpenDRIVE xodr converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from shapely.geometry import LineString

from tactics2d.map.parser import NetXMLParser
from tactics2d.map.writer import XodrWriter


class Net2XodrConverter:
    """Converts a SUMO (.net.xml) map to OpenDRIVE (.xodr) format.

    Reads a SUMO net.xml file using NetXMLParser, then writes the parsed
    map into OpenDRIVE xodr format via XodrWriter. Each Tactics2D Lane
    becomes an OpenDRIVE road with a single driving lane. The road name
    is taken from the SUMO edge id stored in ``lane.custom_tags["sumo_id"]``.

    Example:
    ```python
    from tactics2d.map.converter import Net2XodrConverter

    converter = Net2XodrConverter()
    converter.convert("path/to/map.net.xml", "path/to/output.xodr")
    ```
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a SUMO net.xml file to an OpenDRIVE xodr file.

        Args:
            input_path (str): Path to the input .net.xml file.
            output_path (str): Path to the output .xodr file.

        Returns:
            str: The output file path.
        """
        map_ = NetXMLParser().parse(input_path)
        writer = XodrWriter()

        root = ET.Element("OpenDRIVE")
        writer.write_header(root)

        for lane_id, lane in map_.lanes.items():
            pts = writer._get_centerline(lane)
            if len(pts) < 2:
                continue

            sumo_id = (lane.custom_tags or {}).get("sumo_id", str(lane_id))
            total_length = float(LineString(pts).length)
            speed = lane.speed_limit if lane.speed_limit else 50.0 / 3.6

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
            writer.write_plan_view(road, pts)
            ET.SubElement(road, "elevationProfile")
            ET.SubElement(road, "lateralProfile")
            writer.write_lanes(road, lane, map_)

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
