# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenStreetMap Lanelet2 to OpenDRIVE converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from tactics2d.map.parser import OSMParser
from tactics2d.map.writer import XodrWriter


class Osm2XodrConverter:
    """Converts a Lanelet2-annotated OSM file to OpenDRIVE (.xodr) format.

    Each Tactics2D Lane becomes an OpenDRIVE road. Predecessor/successor
    topology and junction areas are derived from lane endpoint proximity.
    Lane boundary subtypes are mapped to xodr roadMark types. Speed limits
    are converted from m/s to km/h.

    Example:
    ```python
    from tactics2d.map.converter import Osm2XodrConverter

    converter = Osm2XodrConverter()
    converter.convert("map.osm", "map.xodr")
    ```
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a Lanelet2 OSM file to an OpenDRIVE xodr file.

        Args:
            input_path (str): Path to the input Lanelet2 .osm file.
            output_path (str): Path to the output .xodr file.

        Returns:
            str: The output file path.
        """
        map_ = OSMParser(lanelet2=True).parse(input_path)
        root = XodrWriter().build(map_)

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
