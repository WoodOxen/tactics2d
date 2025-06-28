##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_gis.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import logging
import os

import geopandas as gpd
import pyogrio
from pyproj import Proj
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from tactics2d.map.element import Area, Lane, Map, Node, Regulatory, RoadLine

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

    def _parse_nuplan(self, file: str, folder: str) -> Map:
        """This function parses a map from a single NuPlan map file. The map file is expected to be a geopackage file (.gpkg).

        TODO: the parsing of lane connectors is not implemented yet.

        A NuPlan map includes the following layers:
        - baseline_paths*
        - boundaries
        - carpark_areas
        - crosswalks
        - dubins_nodes*
        - generic_drivable_areas
        - gen_lane_connectors_scaled_width_polygons*
        - intersections
        - lane_connectors
        - lane_group_connectors*
        - lane_groups_polygons*
        - lanes_polygons
        - meta
        - road_segments
        - stop_polygons*
        - traffic_lights
        - walkways

        In this parser, we parse all the layers without *.
        """
        map_file = os.path.join(folder, file)
        map_meta = gpd.read_file(map_file, layer="meta", engine="pyogrio")
        projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

        def load_utm_coords(layer_name):
            gdf_in_pixel_coords = pyogrio.read_dataframe(map_file, layer=layer_name)
            gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
            return gdf_in_utm_coords

        map_ = Map(name="nuplan_" + file.split(".")[0])

        boundaries = load_utm_coords("boundaries")
        for _, row in boundaries.iterrows():
            boundary_ids = [int(s) for s in row["boundary_segment_fids"].split(",") if s.isdigit()]
            boundary_id = boundary_ids[0] - 1
            boundary = RoadLine(id_=str(boundary_id), geometry=LineString(row["geometry"]))
            map_.add_roadline(boundary)

        id_cnt = max(np.array(list(map_.ids.keys()), np.int64)) + 1

        # lane_polygons = gpd.read_file(map_file, layer="lanes_polygons")
        # for _, row in lane_polygons.iterrows():
        #     lane_polygon = Lane(
        #         id_=str(row["lane_fid"]),
        #         left_side=map_.get_by_id(str(row["left_boundary_fid"])).geometry,
        #         right_side=map_.get_by_id(str(row["right_boundary_fid"])).geometry,
        #         line_ids=set([str(row["left_boundary_fid"]), str(row["right_boundary_fid"])]),
        #         subtype=None,
        #         speed_limit=row["speed_limit_mps"],
        #         speed_limit_unit="m/s",
        #     )
        #     lane_polygon.add_related_lane(str(row["from_edge_fid"]), LaneRelationship.PREDECESSOR)
        #     lane_polygon.add_related_lane(str(row["to_edge_fid"]), LaneRelationship.SUCCESSOR)
        #     map_.add_lane(lane_polygon)

        lane_connectors = load_utm_coords("lane_connectors")
        for _, row in lane_connectors.iterrows():
            # TODO: parse the polygon in lane to left and right side
            pass

        area_dict = {"crosswalks": "crosswalk", "carpark_areas": "parking", "walkways": "walkway"}

        for key, value in area_dict.items():
            df_areas = load_utm_coords(key)
            for _, row in df_areas.iterrows():
                area = Area(
                    id_=str(id_cnt),
                    geometry=Polygon(row["geometry"]),
                    subtype=value,
                    custom_tags={"heading": row["heading"]} if key == "carpark_areas" else None,
                )
                id_cnt += 1
                map_.add_area(area)

        traffic_lights = load_utm_coords("traffic_lights")
        for _, row in traffic_lights.iterrows():
            traffic_light = Regulatory(
                id_=str(id_cnt),
                subtype="traffic_light",
                position=Point(row["geometry"].x, row["geometry"].y),
                custom_tags={"heading": row["ori_mean_yaw"]},
            )
            id_cnt += 1
            map_.add_regulatory(traffic_light)

        return map_

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
