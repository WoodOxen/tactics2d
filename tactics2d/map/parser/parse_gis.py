##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_gis.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import logging
import os

import geopandas as gpd
from shapely.geometry import LineString, Polygon

# from collections import Union, List


# from tactics2d.map.element import Area, Lane, Map, Node, Regulatory, RoadLine
class GISParser:
    def load_roadmark(self, df):
        return

    def load_area(self, df):
        return

    def parse(self, file_path):
        """_summary_

        Args:
            file_path (str): The absolute path of a `.shp` file or a list of `.shp` files.
        """
        if isinstance(file_path, str):
            gdf = [gpd.read_file(file_path)]
        else:
            gdf = []
            for f in file_path:
                df = gpd.read_file(f)
                gdf.append(df)

        for df in gdf:
            continue


if __name__ == "__main__":
    parser = GISParser()
    parser.parse(
        "/Users/rowena/Documents/Codes/tactics2d/data/NGSIM/US-101-LosAngeles-CA/gis-files/US-101.shp"
    )
