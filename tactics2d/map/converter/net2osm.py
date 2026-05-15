# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml to OpenStreetMap Lanelet2 converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from tactics2d.map.parser import NetXMLParser
from tactics2d.map.writer import OsmWriter


class Net2OsmConverter:
    """Converts a SUMO (.net.xml) file to a Lanelet2-annotated OSM (.osm) file.

    Each SUMO lane becomes one Lanelet2 lanelet relation. Boundaries are taken
    directly from the parsed ``Lane.left_side`` and ``Lane.right_side`` so
    geometry is consistent with the rest of Tactics2D.

    Example:
    ```python
    from tactics2d.map.converter import Net2OsmConverter

    converter = Net2OsmConverter()
    converter.convert("map.net.xml", "map.osm")
    ```
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a SUMO net.xml file to a Lanelet2 OSM file.

        Args:
            input_path (str): Path to the input .net.xml file.
            output_path (str): Path to the output .osm file.

        Returns:
            str: The output file path.
        """
        map_ = NetXMLParser().parse(input_path)
        osm_root = OsmWriter().build(map_)

        xml_str = minidom.parseString(ET.tostring(osm_root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
