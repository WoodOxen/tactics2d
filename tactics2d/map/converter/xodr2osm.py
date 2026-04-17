# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE xodr to OSM converter implementation."""

from __future__ import annotations

import logging
import os
import subprocess
import xml.etree.ElementTree as ET

import sumolib


class Xodr2OsmConverter:
    """This class implements a converter from OpenDRIVE format (.xodr) to
    OpenStreetMap format (.osm).

    The conversion is a two-step process: first converting ``.xodr`` to
    ``.net.xml`` using SUMO's ``netconvert`` tool, then converting the
    ``.net.xml`` to ``.osm`` using ``sumolib``.

    !!! quote "Reference"
        [SUMO netconvert documentation](https://sumo.dlr.de/docs/netconvert.html)

    Example:
```python
        from tactics2d.map.converter import Xodr2OsmConverter

        converter = Xodr2OsmConverter()
        converter.convert("/path/to/map.xodr", "/path/to/output.osm")
```
    """

    def _check_netconvert(self):
        result = subprocess.run(["which", "netconvert"], capture_output=True, text=True)
        if result.returncode != 0:
            raise EnvironmentError(
                "netconvert not found. Please install SUMO: "
                "https://sumo.dlr.de/docs/Installing/index.html"
            )

    def _get_env(self) -> dict:
        env = os.environ.copy()
        if "SUMO_HOME" not in env:
            for candidate in ["/usr/share/sumo", "/opt/sumo", "/usr/local/share/sumo"]:
                if os.path.isdir(candidate):
                    env["SUMO_HOME"] = candidate
                    break
        return env

    def _net2osm(self, net_path: str, output_path: str):
        """Convert a SUMO net.xml to OSM format using sumolib.

        Args:
            net_path (str): Path to the input .net.xml file.
            output_path (str): Path to the output .osm file.
        """
        net = sumolib.net.readNet(net_path, withInternal=False)

        root = ET.Element("osm", version="0.6")
        node_id = -1

        for node in net.getNodes():
            x, y = node.getCoord()
            lon, lat = x, y
            ET.SubElement(root, "node", {
                "id": str(node_id),
                "lat": f"{lat:.7f}",
                "lon": f"{lon:.7f}",
                "version": "1",
            })
            node._osm_id = node_id
            node_id -= 1

        way_id = -1
        for edge in net.getEdges():
            way = ET.SubElement(root, "way", {
                "id": str(way_id),
                "version": "1",
            })
            ET.SubElement(way, "nd", ref=str(edge.getFromNode()._osm_id))
            ET.SubElement(way, "nd", ref=str(edge.getToNode()._osm_id))
            ET.SubElement(way, "tag", k="highway", v="road")
            lanes = len(edge.getLanes())
            ET.SubElement(way, "tag", k="lanes", v=str(lanes))
            way_id -= 1

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenDRIVE file to OpenStreetMap format.

        The conversion is performed in two steps:
        1. ``.xodr`` → ``.net.xml`` using netconvert
        2. ``.net.xml`` → ``.osm`` using sumolib

        Args:
            input_path (str): The absolute path to the input ``.xodr`` file.
            output_path (str): The absolute path to the output ``.osm`` file.

        Returns:
            str: The absolute path to the generated ``.osm`` file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            EnvironmentError: If netconvert is not installed.
            RuntimeError: If the conversion fails.

        Example:
```python
            from tactics2d.map.converter import Xodr2OsmConverter

            converter = Xodr2OsmConverter()
            output = converter.convert(
                "/path/to/map.xodr",
                "/path/to/map.osm"
            )
            print(f"Converted map saved to: {output}")
```
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self._check_netconvert()

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        tmp_net = os.path.join(output_dir, "_tmp_xodr2osm.net.xml")

        cmd = [
            "netconvert",
            "--opendrive-files", input_path,
            "--output-file", tmp_net,
        ]
        logging.info("Step 1 - Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=self._get_env())
        if result.returncode != 0:
            raise RuntimeError(f"netconvert (xodr->net) failed:\n{result.stderr}")

        logging.info("Step 2 - Converting net.xml to osm")
        try:
            self._net2osm(tmp_net, output_path)
        finally:
            if os.path.isfile(tmp_net):
                os.remove(tmp_net)

        logging.info("Conversion successful: %s", output_path)
        return output_path
