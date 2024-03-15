##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the map parser module.
# @Author: Yueyuan Li
# @Version: 1.0.0


from .parse_osm import OSMParser
from .parse_xodr import XODRParser

__all__ = ["OSMParser", "XODRParser"]
