# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OSM to OpenDRIVE xodr converter implementation."""

from __future__ import annotations

import logging
import os
import subprocess


class Osm2XodrConverter:
    """This class implements a converter from OpenStreetMap format (.osm) to
    OpenDRIVE format (.xodr).

    The conversion is performed by calling SUMO's ``netconvert`` command-line tool,
    which must be installed and available in the system PATH.

    !!! quote "Reference"
        [SUMO netconvert OSM import](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html)

    Example:
        ```python
        from tactics2d.map.converter import Osm2XodrConverter

        converter = Osm2XodrConverter()
        converter.convert("/path/to/map.osm", "/path/to/output.xodr")
        ```
    """

    def _check_netconvert(self):
        """Check whether netconvert is available in the system PATH.

        Raises:
            EnvironmentError: If netconvert is not found.
        """
        result = subprocess.run(
            ["which", "netconvert"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise EnvironmentError(
                "netconvert not found. Please install SUMO: "
                "https://sumo.dlr.de/docs/Installing/index.html"
            )

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenStreetMap file to OpenDRIVE format.

        This function calls SUMO's ``netconvert`` tool to perform the conversion.
        The input file must be a valid OpenStreetMap ``.osm`` file.

        Args:
            input_path (str): The absolute path to the input ``.osm`` file.
            output_path (str): The absolute path to the output ``.xodr`` file.

        Returns:
            str: The absolute path to the generated ``.xodr`` file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            EnvironmentError: If netconvert is not installed.
            RuntimeError: If the conversion fails.

        Example:
            ```python
            from tactics2d.map.converter import Osm2XodrConverter

            converter = Osm2XodrConverter()
            output = converter.convert(
                "/path/to/map.osm",
                "/path/to/map.xodr"
            )
            print(f"Converted map saved to: {output}")
            ```
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self._check_netconvert()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cmd = [
            "netconvert",
            "--osm-files", input_path,
            "--opendrive-output", output_path,
        ]

        logging.info("Running: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"netconvert failed with error:\n{result.stderr}"
            )

        logging.info("Conversion successful: %s", output_path)
        return output_path
