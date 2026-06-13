# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Common utilities implementation."""


import sys
from pathlib import Path

sys.path.append(".")
sys.path.append("..")


def get_absolute_path(file_path: str) -> str:
    """This function resolves the absolute path of a given file by searching through all directories in `sys.path`.

    Args:
        file_path (str): The relative file path of the target file.

    Returns:
        file_path (str): The absolute file path if found within the system paths; otherwise, returns the original file path.
    """
    for folder_path in sys.path:
        abs_path = Path(folder_path) / file_path
        if abs_path.exists():
            return str(abs_path.resolve())
    return file_path
