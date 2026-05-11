# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE xodr to SUMO net.xml converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from tactics2d.map.parser import XODRParser
from tactics2d.map.writer import SumoWriter


class Xodr2NetConverter:
    """Converts an OpenDRIVE (.xodr) file to a SUMO (.net.xml) file.

    Reads the OpenDRIVE file with XODRParser, then serialises the resulting
    Tactics2D Map to SUMO net.xml format via SumoWriter.

    Example:
    ```python
    from tactics2d.map.converter import Xodr2NetConverter

    converter = Xodr2NetConverter()
    converter.convert("path/to/map.xodr", "path/to/output.net.xml")
    ```
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenDRIVE xodr file to a SUMO net.xml file.

        Args:
            input_path (str): Path to the input .xodr file.
            output_path (str): Path to the output .net.xml file.

        Returns:
            str: The output file path.
        """
        map_ = XODRParser().parse(input_path)
        root = SumoWriter().build(map_)

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
