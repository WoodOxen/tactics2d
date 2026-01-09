# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""gis parser implementation."""


import geopandas as gpd
from pyproj import Proj
from shapely.geometry import LineString, MultiLineString, Point

from tactics2d.map.element import Map, RoadLine

# from collections import Union, List


class GISParser:
    """_summary_"""

    def __init__(self):
        self.projector = None

    def _project(self, geometry):
        coords = list(geometry.coords)

        # TODO: Currently the projector is handling the NGSIM dataset with dirty trick, convert feet to meter only
        # if self.projector is not None:
        coords = [[coord[0] * 0.3048, coord[1] * 0.3048] for coord in coords]

        return coords

    def load_roadline(self, gdf_row, id_):
        geometry = LineString(gdf_row.geometry)
        coords = self._project(geometry)

        roadline = RoadLine(
            str(id_),
            geometry=LineString(coords),
            # color=None if gdf_row.LINEWIDTH == 0.0 else "black",
            color="black",
            # width=gdf_row.LINEWIDTH,
            width=1.0,
        )

        id_ += 1
        return roadline, id_

    def load_lane(self, gdf_row, id_):
        geometries = MultiLineString(gdf_row.geometry).geoms
        roadlines = []

        for geometry in geometries:
            coords = self._project(geometry)
            roadline = RoadLine(
                str(id_),
                LineString(coords),
                # color=None if gdf_row.LINEWIDTH == 0.0 else "black",
                color="black",
                # width=gdf_row.LINEWIDTH
                width=1.0,
            )
            roadlines.append(roadline)

            id_ += 1

        return roadlines, None, id_

    def load_area(self, gef_row):
        return

    def parse(self, file_path, projector: Proj = None):
        """_summary_

        Args:
            file_path (str): The absolute path of a `.shp` file or a list of `.shp` files.
        """
        self.projector = projector

        gdfs = []
        if isinstance(file_path, str):
            gdf = gpd.read_file(file_path)
            gdfs.append(gdf)
        else:
            for f in file_path:
                gdf = gpd.read_file(f)
                gdfs.append(gdf)

        map_ = Map()
        id_ = 0

        for gdf in gdfs:
            for i, row in gdf.iterrows():
                # TODO: handle Points in the future
                if isinstance(row.geometry, Point):
                    pass
                elif isinstance(row.geometry, LineString):
                    roadline, id_ = self.load_roadline(row, id_)
                    map_.add_roadline(roadline)
                elif isinstance(row.geometry, MultiLineString):
                    roadlines, lane, id_ = self.load_lane(row, id_)
                    for roadline in roadlines:
                        map_.add_roadline(roadline)

        return map_
