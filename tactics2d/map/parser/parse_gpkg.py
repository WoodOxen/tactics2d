###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_gpkg.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

import logging

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.geometry import LineString, LinearRing, Point, Polygon

from tactics2d.map.element import Area, Lane, Map, Node, Regulatory, RoadLine


class GPKGParser:
    """This class implements a parser for maps in the `.gpkg` format.

    Since GPKG is a general data structure and the specific column definitions can vary widely between datasets, you must specify the dataset being used during initialization.

    The parsing details of NuPlan can be found here.
    """

    _NUPLAN_ROADLINE_MAPPING = {0: "dashed", 1: "virtual", 2: "solid", 3: "virtual"}

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.id_cnt = 0

    @staticmethod
    def load_utm_coords(file_path, layer_name, projection_system):
        gdf_in_pixel_coords = pyogrio.read_dataframe(file_path, layer=layer_name)
        gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
        return gdf_in_utm_coords

    def _parse_nuplan(self, file_path) -> Map:
        map_meta = gpd.read_file(file_path, layer="meta", engine="pyogrio")
        projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]
        type_mapping = self._NUPLAN_ROADLINE_MAPPING

        map_ = Map(name="nuplan_" + file_path.split("/")[-1].split(".")[0])

        boundaries = self.load_utm_coords(file_path, "boundaries", projection_system)
        for _, row in boundaries.iterrows():
            roadline = RoadLine(
                id_=int(row["boundary_segment_fids"].split(",")[0]),
                type_=type_mapping[row["boundary_type_fid"]],
                geometry=LineString(row["geometry"]),
            )
            map_.add_roadline(roadline)

        # load lands
        lane_dict = {"lanes_polygons": "lane"}

        for key, value in lane_dict.items():
            df_lanes = self.load_utm_coords(file_path, key, projection_system)
            for _, row in df_lanes.iterrows():
                lane = Lane(
                    id_=row["lane_fid"],
                    geometry=LinearRing(Polygon(row["geometry"]).exterior),
                    subtype=value,
                )
                self.id_cnt += 1
                map_.add_lane(lane)

        # load areas
        area_dict = {"carpark_areas": "parking", "crosswalks": "crosswalk", "intersections": "lane", "walkways": "walkway"}

        for key, value in area_dict.items():
            df_areas = self.load_utm_coords(file_path, key, projection_system)
            for _, row in df_areas.iterrows():
                area = Area(
                    id_=self.id_cnt,
                    geometry=Polygon(row["geometry"]),
                    subtype=value,
                    custom_tags={"heading": row["heading"]} if key == "carpark_areas" else None,
                )
                self.id_cnt += 1
                map_.add_area(area)

        id_cnt = max(np.array(list(map_.ids.keys()), np.int64)) + 1

        traffic_lights = self.load_utm_coords(file_path, "traffic_lights", projection_system)
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

    def parse(self, file_path: str):
        if self.dataset == "nuplan":
            return self._parse_nuplan(file_path)

        return None
