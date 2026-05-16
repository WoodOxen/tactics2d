# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenStreetMap Lanelet2 to SUMO net.xml converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from tactics2d.map.parser import OSMParser
from tactics2d.map.writer import SumoWriter


class Osm2NetConverter:
    """Converts a Lanelet2-annotated OSM (.osm) file to a SUMO (.net.xml) file.

        Example:
    ```python
        from tactics2d.map.converter import Osm2NetConverter

        converter = Osm2NetConverter()
        converter.convert("map.osm", "map.net.xml")
    ```
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a Lanelet2 OSM file to a SUMO net.xml file.

        Args:
            input_path (str): Path to the input .osm file.
            output_path (str): Path to the output .net.xml file.

        Returns:
            str: The output file path.
        """
        map_ = OSMParser(lanelet2=True).parse(input_path)
        root = SumoWriter().build(map_)

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
