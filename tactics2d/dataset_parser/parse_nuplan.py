# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""NuPlan dataset parser implementation."""

import datetime
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import LinearRing, LineString, Point, Polygon

from tactics2d.map.element import Area, Junction, Lane, LaneRelationship, Map, Regulatory, RoadLine
from tactics2d.participant.element import Cyclist, Other, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class NuPlanParser:
    """Parser for the NuPlan dataset.

    The parser converts NuPlan database logs and geopackage maps into Tactics2D
    participants and map elements. It keeps dataset-specific details in
    ``custom_tags`` while exposing common map semantics such as lane
    centerlines, boundaries, successors, neighbors, drivable areas, stop
    polygons, and traffic lights.

    !!! quote "Reference"
        Caesar, Holger, et al. "nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles." arXiv preprint arXiv:2106.11810 (2021).
    """

    _CLASS_MAPPING = {
        "vehicle": Vehicle,
        "bicycle": Cyclist,
        "pedestrian": Pedestrian,
        "traffic_cone": Other,
        "barrier": Other,
        "czone_sign": Other,
        "generic_object": Other,
    }

    _TYPE_MAPPING = {
        "vehicle": "vehicle",
        "bicycle": "bicycle",
        "pedestrian": "pedestrian",
        "traffic_cone": "traffic_cone",
        "barrier": "barrier",
        "czone_sign": "czone_sign",
        "generic_object": "generic_object",
    }

    _ROADLINE_MAPPING = {
        0: ("line_thin", "dashed"),
        1: ("virtual", None),
        2: ("line_thin", "solid"),
        3: ("virtual", None),
    }

    _LANE_TYPE_MAPPING = {
        0: "road",
        1: "bicycle_lane",
    }

    _STOP_POLYGON_MAPPING = {
        0: "crosswalk_stop_line",
        1: "stop_sign",
        2: "traffic_light",
        3: "turn_stop",
        4: "yield",
    }

    # Millisecond-level timestamp at 2021-01-01 00:00:00.
    _DATETIME = datetime.datetime(2021, 1, 1, 0, 0, 0).timestamp() * 1000

    def __init__(self):
        self.transform_matrix = np.zeros((6, 1))

    def get_location(self, file: str, folder: str) -> str:
        """Get the NuPlan location of a single trajectory database."""

        file_path = self._resolve_path(file, folder)
        with sqlite3.connect(file_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT location FROM log;")
            location = cursor.fetchone()[0]

        return location

    def parse_trajectory(
        self, file: str, folder: str, time_range: Tuple[int, int] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """Parse trajectories from a single NuPlan database file.

        Args:
            file: The name or path of the NuPlan sqlite database file.
            folder: The folder containing ``file``.
            time_range: Optional millisecond range relative to 2021-01-01
                00:00:00. When omitted, the whole database is parsed.

        Returns:
            ``(participants, actual_time_range)`` where participants are
            Tactics2D participant objects and the range is in milliseconds.
        """

        participants = {}
        time_stamps = set()
        file_path = self._resolve_path(file, folder)

        if time_range is None:
            time_range = (-float("inf"), float("inf"))

        with sqlite3.connect(file_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT
                    track.token AS track_token,
                    category.name AS category_name,
                    track.width AS width,
                    track.length AS length,
                    track.height AS height
                FROM track
                INNER JOIN category ON category.token = track.category_token
                """
            )
            for row in cursor.fetchall():
                category_name = row["category_name"]
                participant_cls = self._CLASS_MAPPING.get(category_name, Other)
                id_ = int.from_bytes(row["track_token"], byteorder="big")
                participants[row["track_token"]] = participant_cls(
                    id_=id_,
                    type_=self._TYPE_MAPPING.get(category_name, category_name),
                    trajectory=Trajectory(id_=id_, fps=20, stable_freq=False),
                    length=row["length"],
                    width=row["width"],
                    height=row["height"],
                )

            cursor.execute(
                """
                SELECT
                    lidar_box.track_token AS track_token,
                    lidar_box.x AS x,
                    lidar_box.y AS y,
                    lidar_box.z AS z,
                    lidar_box.yaw AS yaw,
                    lidar_box.vx AS vx,
                    lidar_box.vy AS vy,
                    lidar_box.vz AS vz,
                    lidar_pc.timestamp AS timestamp
                FROM lidar_box
                INNER JOIN lidar_pc ON lidar_pc.token = lidar_box.lidar_pc_token
                ORDER BY lidar_pc.timestamp
                """
            )
            for row in cursor.fetchall():
                time_stamp = int(row["timestamp"] / 1000 - self._DATETIME)
                if time_stamp < time_range[0] or time_stamp > time_range[1]:
                    continue

                participant = participants.get(row["track_token"])
                if participant is None:
                    continue

                time_stamps.add(time_stamp)
                participant.trajectory.add_state(
                    State(
                        frame=time_stamp,
                        x=row["x"],
                        y=row["y"],
                        heading=row["yaw"],
                        vx=row["vx"],
                        vy=row["vy"],
                    )
                )

        participants = {
            participant.id_: participant
            for participant in participants.values()
            if len(participant.trajectory) > 0
        }
        actual_time_range = (
            (min(time_stamps), max(time_stamps)) if time_stamps else (np.inf, -np.inf)
        )
        return participants, actual_time_range

    def parse_map(self, file: str, folder: Optional[str] = None) -> Map:
        """Parse a NuPlan geopackage map into a Tactics2D ``Map``.

        Args:
            file: The path or name of a NuPlan ``map.gpkg`` file.
            folder: Optional folder containing ``file``.

        Returns:
            A Tactics2D map populated with common lane-level semantics.
        """

        file_path = self._resolve_path(file, folder)
        projection_system = self._load_projection_system(file_path)
        map_ = Map(name="nuplan_" + Path(file_path).parent.parent.name)

        boundaries = self._load_layer(file_path, "boundaries", projection_system, fid_as_index=True)
        baseline_paths = self._load_layer(
            file_path, "baseline_paths", projection_system, fid_as_index=True
        )
        lanes = self._load_layer(file_path, "lanes_polygons", projection_system, fid_as_index=True)
        lane_connectors = self._load_layer(
            file_path, "lane_connectors", projection_system, fid_as_index=True
        )
        connector_polygons = self._load_layer(
            file_path,
            "gen_lane_connectors_scaled_width_polygons",
            projection_system,
            fid_as_index=True,
        )

        boundary_geometries = self._load_roadlines(map_, boundaries)
        self._load_lanes(map_, lanes, baseline_paths, boundary_geometries)
        self._load_lane_connectors(
            map_, lane_connectors, connector_polygons, baseline_paths, boundary_geometries
        )
        self._load_successor_relationships(map_, lane_connectors)
        self._load_neighbor_relationships(map_)
        self._load_areas(file_path, projection_system, map_)
        self._load_junctions(file_path, projection_system, map_)
        self._load_stop_polygons(file_path, projection_system, map_)
        self._load_traffic_lights(file_path, projection_system, map_)

        return map_

    @staticmethod
    def _resolve_path(file: str, folder: Optional[str] = None) -> str:
        path = Path(file)
        if folder is not None:
            path = Path(folder) / path
        return str(path)

    @staticmethod
    def _split_ids(value) -> list:
        if value is None or pd.isna(value):
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def _as_int_str(value) -> str:
        return str(int(value))

    @staticmethod
    def _load_projection_system(file_path: str) -> str:
        map_meta = gpd.read_file(file_path, layer="meta", engine="pyogrio")
        return map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

    @staticmethod
    def _load_layer(file_path: str, layer_name: str, projection_system: str, fid_as_index=False):
        gdf = pyogrio.read_dataframe(file_path, layer=layer_name, fid_as_index=fid_as_index)
        return gdf if gdf.empty else gdf.to_crs(projection_system)

    def _load_roadlines(self, map_: Map, boundaries) -> Dict[str, LineString]:
        boundary_geometries = {}
        for fid, row in boundaries.iterrows():
            boundary_id = self._as_int_str(fid)
            type_, subtype = self._ROADLINE_MAPPING.get(
                int(row["boundary_type_fid"]), ("virtual", None)
            )
            geometry = LineString(row["geometry"])
            boundary_geometries[boundary_id] = geometry
            map_.add_roadline(
                RoadLine(
                    id_=boundary_id,
                    type_=type_,
                    subtype=subtype,
                    geometry=geometry,
                    custom_tags={
                        "nuplan_layer": "boundaries",
                        "boundary_segment_fids": self._split_ids(
                            row.get("boundary_segment_fids")
                        ),
                        "has_reflectors": bool(row.get("has_reflectors", False)),
                    },
                )
            )
        return boundary_geometries

    def _centerline_by_lane_id(self, baseline_paths, lane_id: str) -> Optional[np.ndarray]:
        if baseline_paths.empty:
            return None
        lane_fids = baseline_paths["lane_fid"].dropna().astype(int).astype(str)
        matches = baseline_paths.loc[lane_fids[lane_fids == lane_id].index]
        if matches.empty:
            return None
        return np.asarray(matches.iloc[0]["geometry"].coords, dtype=float)

    def _centerline_by_connector_id(
        self, baseline_paths, connector_id: str
    ) -> Optional[np.ndarray]:
        if baseline_paths.empty:
            return None
        connector_fids = baseline_paths["lane_connector_fid"].dropna().astype(int).astype(str)
        matches = baseline_paths.loc[connector_fids[connector_fids == connector_id].index]
        if matches.empty:
            return None
        return np.asarray(matches.iloc[0]["geometry"].coords, dtype=float)

    def _load_lanes(self, map_: Map, lanes, baseline_paths, boundary_geometries) -> None:
        for fid, row in lanes.iterrows():
            lane_id = self._as_int_str(row.get("lane_fid", fid))
            left_id = self._as_int_str(row["left_boundary_fid"])
            right_id = self._as_int_str(row["right_boundary_fid"])
            centerline = self._centerline_by_lane_id(baseline_paths, lane_id)

            custom_tags = {
                "nuplan_layer": "lanes_polygons",
                "lane_group_fid": self._as_int_str(row["lane_group_fid"]),
                "lane_index": int(row["lane_index"]),
                "from_edge_fid": self._as_int_str(row["from_edge_fid"]),
                "to_edge_fid": self._as_int_str(row["to_edge_fid"]),
            }
            if centerline is not None:
                custom_tags["centerline"] = centerline

            lane = Lane(
                id_=lane_id,
                left_side=boundary_geometries.get(left_id),
                right_side=boundary_geometries.get(right_id),
                geometry=LinearRing(row["geometry"].exterior.coords),
                line_ids={"left": [left_id], "right": [right_id]},
                subtype=self._LANE_TYPE_MAPPING.get(int(row.get("lane_type_fid", 0)), "road"),
                location="urban",
                speed_limit=row.get("speed_limit_mps"),
                speed_limit_unit="m/s",
                custom_tags=custom_tags,
            )
            map_.add_lane(lane)

    def _load_lane_connectors(
        self, map_: Map, lane_connectors, connector_polygons, baseline_paths, boundary_geometries
    ) -> None:
        connector_polygon_by_id = {
            self._as_int_str(row["lane_connector_fid"]): row
            for _, row in connector_polygons.iterrows()
        }
        for fid, row in lane_connectors.iterrows():
            connector_id = self._as_int_str(fid)
            polygon_row = connector_polygon_by_id.get(connector_id)
            if polygon_row is None:
                continue

            left_id = self._as_int_str(polygon_row["left_boundary_fid"])
            right_id = self._as_int_str(polygon_row["right_boundary_fid"])
            centerline = self._centerline_by_connector_id(baseline_paths, connector_id)
            custom_tags = {
                "nuplan_layer": "lane_connectors",
                "entry_lane_fid": self._as_int_str(row["entry_lane_fid"]),
                "exit_lane_fid": self._as_int_str(row["exit_lane_fid"]),
                "intersection_fid": self._as_int_str(row["intersection_fid"]),
                "turn_type_fid": int(row["turn_type_fid"]),
                "from_edge_fid": self._as_int_str(polygon_row["from_edge_fid"]),
                "to_edge_fid": self._as_int_str(polygon_row["to_edge_fid"]),
                "traffic_light_stop_line_fids": self._split_ids(
                    row.get("traffic_light_stop_line_fids")
                ),
            }
            if centerline is not None:
                custom_tags["centerline"] = centerline

            connector_lane = Lane(
                id_=connector_id,
                left_side=boundary_geometries.get(left_id),
                right_side=boundary_geometries.get(right_id),
                geometry=LinearRing(polygon_row["geometry"].exterior.coords),
                line_ids={"left": [left_id], "right": [right_id]},
                subtype="lane_connector",
                location="urban",
                speed_limit=row.get("speed_limit_mps"),
                speed_limit_unit="m/s",
                custom_tags=custom_tags,
            )
            map_.add_lane(connector_lane)

    def _load_successor_relationships(self, map_: Map, lane_connectors) -> None:
        for fid, row in lane_connectors.iterrows():
            connector_id = self._as_int_str(fid)
            entry_id = self._as_int_str(row["entry_lane_fid"])
            exit_id = self._as_int_str(row["exit_lane_fid"])
            if entry_id in map_.lanes and connector_id in map_.lanes:
                map_.lanes[entry_id].add_related_lane(connector_id, LaneRelationship.SUCCESSOR)
                map_.lanes[connector_id].add_related_lane(entry_id, LaneRelationship.PREDECESSOR)
            if connector_id in map_.lanes and exit_id in map_.lanes:
                map_.lanes[connector_id].add_related_lane(exit_id, LaneRelationship.SUCCESSOR)
                map_.lanes[exit_id].add_related_lane(connector_id, LaneRelationship.PREDECESSOR)

    def _load_neighbor_relationships(self, map_: Map) -> None:
        lanes_by_group = {}
        for lane in map_.lanes.values():
            if (lane.custom_tags or {}).get("nuplan_layer") != "lanes_polygons":
                continue
            lanes_by_group.setdefault(lane.custom_tags["lane_group_fid"], []).append(lane)

        for group_lanes in lanes_by_group.values():
            group_lanes.sort(key=lambda lane: int(lane.custom_tags["lane_index"]))
            for index, lane in enumerate(group_lanes):
                if index > 0:
                    lane.add_related_lane(
                        group_lanes[index - 1].id_, LaneRelationship.LEFT_NEIGHBOR
                    )
                if index < len(group_lanes) - 1:
                    lane.add_related_lane(
                        group_lanes[index + 1].id_, LaneRelationship.RIGHT_NEIGHBOR
                    )

    def _load_areas(self, file_path: str, projection_system: str, map_: Map) -> None:
        layer_specs = {
            "generic_drivable_areas": "drivable_area",
            "carpark_areas": "parking",
            "crosswalks": "crosswalk",
            "walkways": "walkway",
            "road_segments": "road_segment",
        }
        for layer_name, subtype in layer_specs.items():
            areas = self._load_layer(file_path, layer_name, projection_system, fid_as_index=True)
            for fid, row in areas.iterrows():
                custom_tags = {"nuplan_layer": layer_name}
                if layer_name == "carpark_areas":
                    custom_tags["heading"] = row.get("heading")
                if layer_name == "crosswalks":
                    custom_tags["lane_ids"] = self._split_ids(row.get("lane_fids"))
                    custom_tags["intersection_ids"] = self._split_ids(row.get("intersection_fids"))
                    custom_tags["is_marked"] = bool(row.get("is_marked", False))
                if layer_name == "road_segments":
                    custom_tags["lane_group_fids"] = self._split_ids(row.get("lane_group_fids"))
                map_.add_area(
                    Area(
                        id_=self._as_int_str(fid),
                        geometry=Polygon(row["geometry"].exterior),
                        subtype=subtype,
                        custom_tags=custom_tags,
                    )
                )

    def _load_junctions(self, file_path: str, projection_system: str, map_: Map) -> None:
        intersections = self._load_layer(
            file_path, "intersections", projection_system, fid_as_index=True
        )
        for fid, row in intersections.iterrows():
            geometry = Polygon(row["geometry"].exterior)
            map_.add_junction(
                Junction(
                    id_=self._as_int_str(fid),
                    custom_tags={
                        "nuplan_layer": "intersections",
                        "shape": list(geometry.exterior.coords),
                        "intersection_type_fid": int(row.get("intersection_type_fid", -1)),
                        "is_mini": bool(row.get("is_mini", False)),
                    },
                )
            )

    def _load_stop_polygons(self, file_path: str, projection_system: str, map_: Map) -> None:
        stop_polygons = self._load_layer(
            file_path, "stop_polygons", projection_system, fid_as_index=True
        )
        for fid, row in stop_polygons.iterrows():
            subtype = self._STOP_POLYGON_MAPPING.get(
                int(row.get("stop_polygon_type_fid", -1)), "stop_line"
            )
            lane_ids = self._split_ids(row.get("lane_fids"))
            lane_connector_ids = self._split_ids(row.get("lane_connector_fids"))
            applies_to = {lane_id: "refers" for lane_id in lane_ids + lane_connector_ids}
            centroid = row["geometry"].centroid
            map_.add_regulatory(
                Regulatory(
                    id_=self._as_int_str(fid),
                    ways=applies_to,
                    subtype=subtype,
                    position=Point(centroid.x, centroid.y),
                    custom_tags={
                        "nuplan_layer": "stop_polygons",
                        "lane_ids": lane_ids,
                        "lane_connector_ids": lane_connector_ids,
                        "traffic_light_ids": self._split_ids(row.get("traffic_light_fids")),
                        "crosswalk_ids": self._split_ids(row.get("crosswalk_fids")),
                        "geometry": row["geometry"],
                    },
                )
            )

    def _load_traffic_lights(self, file_path: str, projection_system: str, map_: Map) -> None:
        traffic_lights = self._load_layer(
            file_path, "traffic_lights", projection_system, fid_as_index=True
        )
        for fid, row in traffic_lights.iterrows():
            map_.add_regulatory(
                Regulatory(
                    id_=self._as_int_str(fid),
                    subtype="traffic_light",
                    dynamic=True,
                    position=Point(row["geometry"].x, row["geometry"].y),
                    custom_tags={
                        "nuplan_layer": "traffic_lights",
                        "heading": row.get("ori_mean_yaw"),
                        "light_face_type_fid": int(row.get("light_face_type_fid", -1)),
                    },
                )
            )
