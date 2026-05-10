# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE to OpenStreetMap Lanelet2 converter implementation."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom

from tactics2d.map.parser import XODRParser
from tactics2d.map.writer import OsmWriter


class Xodr2OsmConverter:
    """Converts an OpenDRIVE (.xodr) file to a Lanelet2-annotated OSM (.osm) file.

    The converter first parses the xodr input into a Tactics2D ``Map`` via
    ``XODRParser``, then serialises the Map to OSM XML via ``OsmWriter``.
    Every xodr lane becomes one Lanelet2 lanelet relation; boundaries are
    taken directly from the parsed ``Lane.left_side`` / ``Lane.right_side``
    so geometry is consistent with the rest of Tactics2D.

    Example:
    ```python
    from tactics2d.map.converter import Xodr2OsmConverter

    converter = Xodr2OsmConverter()
    converter.convert("map.xodr", "map.osm")
    ```
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenDRIVE xodr file to a Lanelet2 OSM file.

        Args:
            input_path: Path to the input .xodr file.
            output_path: Path to the output .osm file.

        Returns:
            The output file path.
        """
        map_ = XODRParser().parse(input_path)
        logging.info("Parsed %d lanes from %s.", len(map_.lanes), input_path)

        osm_root = OsmWriter().build(map_)

        xml_str = minidom.parseString(ET.tostring(osm_root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logging.info("Written %s.", output_path)
        return output_path
