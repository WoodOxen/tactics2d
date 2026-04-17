# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml to OpenDRIVE xodr converter implementation."""

from __future__ import annotations

import logging
import os
import subprocess


class Net2XodrConverter:
    """This class implements a converter from SUMO network format (.net.xml) to
    OpenDRIVE format (.xodr).

    The conversion is performed by calling SUMO's ``netconvert`` command-line tool,
    which must be installed and available in the system PATH.

    !!! quote "Reference"
        [SUMO netconvert documentation](https://sumo.dlr.de/docs/netconvert.html)

    Example:
        ```python
        from tactics2d.map.converter import Net2XodrConverter

        converter = Net2XodrConverter()
        converter.convert("/path/to/map.net.xml", "/path/to/output.xodr")
        ```
    """

    def _get_env(self) -> dict:
        env = os.environ.copy()
        if "SUMO_HOME" not in env:
            for candidate in ["/usr/share/sumo", "/opt/sumo", "/usr/local/share/sumo"]:
                if os.path.isdir(candidate):
                    env["SUMO_HOME"] = candidate
                    break
        return env

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
        """Convert a SUMO network file to OpenDRIVE format.

        This function calls SUMO's ``netconvert`` tool to perform the conversion.
        The input file must be a valid SUMO ``.net.xml`` file.

        Args:
            input_path (str): The absolute path to the input ``.net.xml`` file.
            output_path (str): The absolute path to the output ``.xodr`` file.

        Returns:
            str: The absolute path to the generated ``.xodr`` file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            EnvironmentError: If netconvert is not installed.
            RuntimeError: If the conversion fails.

        Example:
            ```python
            from tactics2d.map.converter import Net2XodrConverter

            converter = Net2XodrConverter()
            output = converter.convert(
                "/path/to/cologne.net.xml",
                "/path/to/cologne.xodr"
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
            "--sumo-net-file", input_path,
            "--opendrive-output", output_path,
        ]

        logging.info("Running: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, env=self._get_env())

        if result.returncode != 0:
            raise RuntimeError(
                f"netconvert failed with error:\n{result.stderr}"
            )

        logging.info("Conversion successful: %s", output_path)
        return output_path
