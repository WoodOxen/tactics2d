###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_gpkg.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

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

    def _load_roadline(self, row):
        if self.dataset == "nuplan":
            type_mapping = self._NUPLAN_ROADLINE_MAPPING
            roadline = RoadLine(
                id_=abs(int(row["boundary_segment_fids"])),
                type_=type_mapping[row["boundary_type_fid"]],
                geometry=LineString(row["geometry"]),
            )
            return roadline

        return None

    def _parse_nuplan(self, file_path) -> Map:
        map_meta = gpd.read_file(file_path, layer="meta", engine="pyogrio")
        projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

        map_ = Map(name="nuplan_" + file_path.splilt("/")[-1].split(".")[0])

        boundaries = self.load_utm_coords(file_path, "boundaries", projection_system)
        boundaries = boundaries.set_index(boundaries["boundary_segment_fids"])

        # load roadlines and lanes
        lane_dict = {"lanes_polygons": "lane", "crosswalks": "crosswalk", "walkways": "walkway"}

        for key, value in lane_dict.items():
            df_lanes = self.load_utm_coords(file_path, key, projection_system)
            for _, row in df_lanes.iterrows():
                left_bound = self._load_roadline(boundaries[row["left_boundary_fid"]])
                right_bound = self._load_roadline(boundaries[row["right_boundary_fid"]])
                lane = Lane(
                    id_=abs(int(row["lane_fid"])),
                    left_side=left_bound,
                    right_side=right_bound,
                    line_ids={
                        "left": list(boundaries[row["left_boundary_fid"]]),
                        "right": list(boundaries[row["right_boundary_fid"]]),
                    },
                    type_=value,
                    speed_limit=row["speed_limit_mps"],
                    speed_limit_mandatory=False,
                    speed_limit_unit="m/s",
                )

                map_.add_roadline(left_bound)
                map_.add_roadline(right_bound)
                map_.add_lane(lane)

        # load areas
        area_dict = {"carpark_areas": "parking", "generic_drivable_areas": "driving"}

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

        # lane_connectors = self.load_utm_coords("lane_connectors")
        # for _, row in lane_connectors.iterrows():

        traffic_lights = self.load_utm_coords("traffic_lights")
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
