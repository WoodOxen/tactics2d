# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE xodr to SUMO net.xml converter implementation."""

from __future__ import annotations

import logging
import os
import subprocess


class Xodr2NetConverter:
    """This class implements a converter from OpenDRIVE format (.xodr) to
    SUMO network format (.net.xml).

    The conversion is performed by calling SUMO's ``netconvert`` command-line tool,
    which must be installed and available in the system PATH.

    !!! quote "Reference"
        [SUMO netconvert OpenDRIVE import](https://sumo.dlr.de/docs/Networks/Import/OpenDRIVE.html)

    Example:
```python
        from tactics2d.map.converter import Xodr2NetConverter

        converter = Xodr2NetConverter()
        converter.convert("/path/to/map.xodr", "/path/to/output.net.xml")
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

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenDRIVE file to SUMO network format.

        Args:
            input_path (str): The absolute path to the input ``.xodr`` file.
            output_path (str): The absolute path to the output ``.net.xml`` file.

        Returns:
            str: The absolute path to the generated ``.net.xml`` file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            EnvironmentError: If netconvert is not installed.
            RuntimeError: If the conversion fails.

        Example:
```python
            from tactics2d.map.converter import Xodr2NetConverter

            converter = Xodr2NetConverter()
            output = converter.convert(
                "/path/to/map.xodr",
                "/path/to/map.net.xml"
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
            "--opendrive-files", input_path,
            "--output-file", output_path,
        ]

        logging.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=self._get_env())

        if result.returncode != 0:
            raise RuntimeError(f"netconvert failed with error:\n{result.stderr}")

        logging.info("Conversion successful: %s", output_path)
        return output_path
