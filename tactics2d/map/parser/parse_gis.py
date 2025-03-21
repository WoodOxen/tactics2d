##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_gis.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import logging
import os

import geopandas as gpd
from pyproj import Proj
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

# from collections import Union, List

# from tactics2d.map.element import Area, Lane, Map, Node, Regulatory, RoadLine


class GISParser:
    def load_roadline(self, gdf_row):
        geometry = LineString(gdf_row.geometry)
        coords = list(geometry.coords)
        # projector = Proj(proj="utm", ellps="WGS84", zone=1, datum="WGS84")
        projector = Proj(
            "+proj=lcc +lat_0=34.139 +lat_1=34.135 +lon_0=-118.359 +lon_1=-118.365 +x_0=0 +y_0=0  +datum=NAD83 +units=m +no_defs"
        )

        print(coords[0], projector(coords[0][0], coords[0][1], inverse=True))

        return

    def load_lane(self, gdf_row):
        return

    def load_area(self, gef_row):
        return

    def parse_gdf(self, gdf: gpd.GeoDataFrame, map_):
        for i, row in gdf.iterrows():
            if isinstance(row.geometry, Point):
                pass
            elif isinstance(row.geometry, LineString):
                self.load_roadline(row)
            elif isinstance(row.geometry, MultiLineString):
                self.load_lane(row)
            break

        print(gdf.head())

    def parse(self, file_path):
        """_summary_

        Args:
            file_path (str): The absolute path of a `.shp` file or a list of `.shp` files.
        """
        gdfs = []
        if isinstance(file_path, str):
            gdf = gpd.read_file(file_path)
            gdfs.append(gdf)
        else:
            for f in file_path:
                gdf = gpd.read_file(f)
                gdfs.append(gdf)

        # map_ = Map()

        print(gdfs[0].head())
        print(gdfs[0].crs)
        print(gdfs[0].geom_type.unique())

        for gdf in gdfs:
            self.parse_gdf(gdf, None)


# EPSG:26945

if __name__ == "__main__":
    parser = GISParser()
    parser.parse(
        "/home/rowena/Documents/Tactics2D/tactics2d/data/NGSIM/US-101-LosAngeles-CA/gis-files/US-101.shp"
    )
