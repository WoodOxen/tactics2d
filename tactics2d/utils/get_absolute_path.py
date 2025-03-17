##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: get_absolute_path.py
# @Description: This file implements a function to get the absolute path of a file.
# @Author: Tactics2D Team
# @Version:

import os
import sys

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
        abs_path = os.path.join(folder_path, file_path)
        if os.path.exists(abs_path):
            return os.path.abspath(abs_path)
    return file_path
