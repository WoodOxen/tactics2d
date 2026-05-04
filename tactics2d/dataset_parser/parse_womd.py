# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Waymo Open Motion Dataset parser implementation."""


import os
from typing import List, Tuple, Union

import numpy as np
import tfrecord
from shapely.geometry import LineString, Point, Polygon

from tactics2d.dataset_parser.womd_proto import scenario_pb
from tactics2d.map.element import Area, Lane, LaneRelationship, Map, Regulatory, RoadLine
from tactics2d.participant.element import Cyclist, Other, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class WOMDParser:
    """This class implements a parser for Waymo Open Motion Dataset (WOMD).

    !!! quote "Reference"
        Ettinger, Scott, et al. "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

    Because loading the tfrecord file is time consuming, the trajectory and the map parsers provide two ways to load the file. The first way is to load the file directly from the given file path. The second way is to load the file from a tfrecord.tfrecord_iterator object. If the tfrecord.tfrecord_iterator object is given, the parser will ignore the file path.
    """

    _TYPE_MAPPING = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}

    _CLASS_MAPPING = {0: Other, 1: Vehicle, 2: Pedestrian, 3: Cyclist, 4: Other}

    _ROADLINE_TYPE_MAPPING = {
        0: ["virtual", None, None],
        1: ["line_thin", "dashed", "white"],
        2: ["line_thin", "solid", "white"],
        3: ["line_thin", "solid_solid", "white"],
        4: ["line_thin", "dashed", "yellow"],
        5: ["line_thin", "dashed_dashed", "yellow"],
        6: ["line_thin", "solid", "yellow"],
        7: ["line_thin", "solid_solid", "yellow"],
        8: [None, "dashed", "yellow"],
    }

    _LANE_TYPE_MAPPING = {0: "road", 1: "highway", 2: "road", 3: "bicycle_lane"}

    _DEFAULT_LANE_WIDTH = {
        "bicycle_lane": 1.8,
        "road": 3.6,
        "highway": 3.8,
    }

    _TRAFFIC_SIGNAL_STATE_MAPPING = {
        0: "unknown",
        1: "arrow_stop",
        2: "arrow_caution",
        3: "arrow_go",
        4: "stop",
        5: "caution",
        6: "go",
        7: "flashing_stop",
        8: "flashing_caution",
    }

    def _get_dataset(self, **kwargs):
        if "dataset" in kwargs:
            return kwargs["dataset"]
        if "file" in kwargs and "folder" in kwargs:
            file_path = os.path.join(kwargs["folder"], kwargs["file"])
            return tfrecord.tfrecord_iterator(file_path, compression_type=None)

        raise KeyError("Either dataset or file and folder should be given as keyword arguments.")

    def _resolve_scenario_data(
        self, scenario_id: Union[str, int], dataset
    ) -> Tuple[Union[scenario_pb.Scenario, None], List[str]]:
        scenario_ids, cached_data = self.get_scenario_ids(dataset, cache_data=True)
        if len(cached_data) == 0:
            return None, scenario_ids

        data_id = 0
        if isinstance(scenario_id, str):
            if scenario_id in scenario_ids:
                data_id = scenario_ids.index(scenario_id)
        elif isinstance(scenario_id, int):
            data_id = scenario_id % len(scenario_ids)

        scenario = scenario_pb.Scenario()
        scenario.ParseFromString(cached_data[data_id])
        return scenario, scenario_ids

    def get_scenario_ids(
        self, dataset, cache_data=False
    ) -> Union[List[str], Tuple[List[str], List[bytes]]]:
        """This function get the list of scenario ids from the given tfrecord file.

        Args:
            dataset (tfrecord.tfrecord_iterator): The dataset to parse.
            cache_data (bool): If True, also cache the raw data bytes for each scenario.

        Returns:
            id_list (List[str]): A list of scenario ids looking like ["637f20cafde22ff8", ...].
            If cache_data is True, returns a tuple (id_list, data_list) where data_list contains
            the raw bytes for each scenario.
        """

        id_list = []
        data_list = [] if cache_data else None

        for data in dataset:
            proto_bytes = data.tobytes()
            if cache_data and data_list is not None:
                data_list.append(proto_bytes)
            scenario = scenario_pb.Scenario()
            scenario.ParseFromString(proto_bytes)
            id_list.append(scenario.scenario_id)

        if cache_data:
            return id_list, data_list
        return id_list

    def _interpolate_heading(self, heading_0: float, heading_1: float, ratio: float) -> float:
        """Interpolate heading on the shortest angular arc."""
        delta = np.arctan2(np.sin(heading_1 - heading_0), np.cos(heading_1 - heading_0))
        return float(heading_0 + ratio * delta)

    def _fill_short_invalid_gaps(
        self,
        track,
        timestamps: List[float],
        max_gap_frames: int,
    ) -> List[State]:
        """Fill short internal invalid gaps for visualization-friendly continuity.

        This only interpolates gaps bracketed by valid states on both sides.
        It never extrapolates before the first valid state or after the last valid
        state, so objects that truly enter late or leave early are left unchanged.
        """
        filled_states = []
        if max_gap_frames <= 0:
            return filled_states

        valid_indices = [i for i, state_ in enumerate(track.states) if state_.valid]
        if len(valid_indices) < 2:
            return filled_states

        for start_idx, end_idx in zip(valid_indices, valid_indices[1:]):
            missing = end_idx - start_idx - 1
            if missing <= 0 or missing > max_gap_frames:
                continue

            start_state = track.states[start_idx]
            end_state = track.states[end_idx]
            for gap_idx in range(start_idx + 1, end_idx):
                ratio = (gap_idx - start_idx) / (end_idx - start_idx)
                frame = int(round(timestamps[gap_idx] * 1000))
                filled_states.append(
                    State(
                    frame=frame,
                    x=(1 - ratio) * start_state.center_x + ratio * end_state.center_x,
                    y=(1 - ratio) * start_state.center_y + ratio * end_state.center_y,
                    heading=self._interpolate_heading(start_state.heading, end_state.heading, ratio),
                    vx=(1 - ratio) * start_state.velocity_x + ratio * end_state.velocity_x,
                    vy=(1 - ratio) * start_state.velocity_y + ratio * end_state.velocity_y,
                )
                )

        return filled_states

    def parse_trajectory(
        self, scenario_id: Union[str, int] = None, **kwargs
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses trajectories from a single WOMD file. Because the duration of the scenario has been well articulated, the parser will not provide an option to select time range within a single scenario. The states were collected at 10Hz.

        Args:
            scenario_id (Union[str, int], optional): The id of the scenario to parse. If the scenario id is a string, the parser will search for the scenario id in the file. If the scenario id is an integer, the parser will parse `scenario_id`-th scenario in the file. If the scenario id is None or is not found, the first scenario in the file will be parsed.

        Keyword Args:
            dataset (tfrecord.tfrecord_iterator, optional): The dataset to parse.
            file (str, optional): The name of the trajectory file. The file is expected to be a tfrecord file (.tfrecord).
            folder (str, optional): The path to the folder containing the tfrecord file.

        Returns:
            participants (dict): A dictionary of participants. If the scenario id is not found, return None.
            actual_time_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.

        Raises:
            KeyError: Either dataset or file and folder should be given as keyword arguments.
        """
        participants = dict()
        time_stamps = set()

        dataset = self._get_dataset(**kwargs)
        scenario, _ = self._resolve_scenario_data(scenario_id, dataset)
        fill_invalid_gaps = kwargs.get("fill_invalid_gaps", False)
        max_gap_frames = kwargs.get("max_gap_frames", 1)

        if scenario is None:
            if time_stamps:
                actual_time_range = (min(time_stamps), max(time_stamps))
            else:
                actual_time_range = (np.inf, -np.inf)
            return participants, actual_time_range

        timestamps = scenario.timestamps_seconds
        for track in scenario.tracks:
            parsed_states = []
            width = 0
            length = 0
            height = 0
            cnt = 0
            for i, state_ in enumerate(track.states):
                if not state_.valid:
                    continue
                parsed_states.append(
                    State(
                    frame=int(round(timestamps[i] * 1000)),
                    x=state_.center_x,
                    y=state_.center_y,
                    heading=state_.heading,
                    vx=state_.velocity_x,
                    vy=state_.velocity_y,
                )
                )

                width += state_.width
                length += state_.length
                height += state_.height
                cnt += 1

            if cnt == 0:
                continue

            if fill_invalid_gaps:
                parsed_states.extend(
                    self._fill_short_invalid_gaps(track, timestamps, max_gap_frames)
                )

            trajectory = Trajectory(id_=track.id, fps=10, stable_freq=False)
            parsed_states.sort(key=lambda state: state.frame)
            for state in parsed_states:
                time_stamps.add(state.frame)
                trajectory.add_state(state)

            participant = self._CLASS_MAPPING[track.object_type](
                id_=track.id,
                type_=self._TYPE_MAPPING[track.object_type],
                trajectory=trajectory,
                length=length / cnt,
                width=width / cnt,
                height=height / cnt,
            )
            participants[track.id] = participant

        if time_stamps:
            actual_time_range = (min(time_stamps), max(time_stamps))
        else:
            actual_time_range = (np.inf, -np.inf)

        return participants, actual_time_range

    def _compute_centerline_left_unit_normals(self, points: np.ndarray) -> np.ndarray:
        """Compute left-pointing unit normal vectors along a lane centerline.

        The input is treated as an ordered lane centerline polyline. Tangent vectors
        are estimated from neighbouring samples, normalized, and then rotated by
        +90 degrees to obtain left-facing unit normals. These normals are later used
        to reconstruct lane sides from centerline samples and per-side lateral
        offsets derived from WOMD boundary metadata.

        Args:
            points (np.ndarray): A polyline of shape ``(N, 2)`` containing centerline
                points in world coordinates.

        Returns:
            np.ndarray: An array of shape ``(N, 2)`` containing one left-pointing unit
            normal vector for each centerline sample. Empty and single-point inputs
            are handled gracefully.
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return np.empty((0, 2))
        if len(pts) == 1:
            return np.zeros((1, 2))

        tangents = np.empty_like(pts)
        tangents[1:-1] = pts[2:] - pts[:-2]
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]

        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        tangents /= norms

        return np.column_stack([-tangents[:, 1], tangents[:, 0]])

    def _interpolate_nan_values(self, values: np.ndarray, fallback: float) -> np.ndarray:
        """Fill missing offset values by interpolation and nearest-value extension."""
        values = np.asarray(values, dtype=np.float64).copy()
        if len(values) == 0:
            return values

        valid = np.isfinite(values)
        if not np.any(valid):
            values[:] = fallback
            return values

        valid_idx = np.flatnonzero(valid)
        values = np.interp(np.arange(len(values)), valid_idx, values[valid_idx])
        return values

    def _estimate_side_offsets(self, boundaries, lane_points: np.ndarray, map_: Map) -> Tuple[np.ndarray, List[str]]:
        """Estimate lateral offsets from centerline to one lane side.

        WOMD stores lane boundaries as segments referenced by lane polyline index ranges.
        The previous implementation stitched full boundary polylines together and ignored
        these ranges, which often distorted lanes inside intersections. Here we instead
        project each centerline sample in the covered index range to the referenced
        boundary feature and use the measured distance as the side offset.
        """
        n_points = len(lane_points)
        offsets = np.full(n_points, np.nan, dtype=np.float64)
        line_ids = []
        if n_points == 0:
            return offsets, line_ids

        for boundary in boundaries:
            id_ = "%05d" % boundary.boundary_feature_id
            line_ids.append(id_)
            roadline = map_.roadlines.get(id_)
            if roadline is None or roadline.geometry is None:
                continue

            start_idx = int(boundary.lane_start_index)
            end_idx = int(boundary.lane_end_index)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            start_idx = max(0, min(start_idx, n_points - 1))
            end_idx = max(0, min(end_idx, n_points - 1))
            if start_idx > end_idx:
                continue

            for idx in range(start_idx, end_idx + 1):
                dist = roadline.geometry.distance(Point(lane_points[idx]))
                if not np.isfinite(dist):
                    continue
                if np.isnan(offsets[idx]) or dist < offsets[idx]:
                    offsets[idx] = dist

        return offsets, line_ids

    def _build_lane_sides(
        self, lane_feature, lane_subtype: str, map_: Map
    ) -> Tuple[LineString, LineString, dict]:
        """Construct lane sides from the centerline and segmented boundary metadata."""
        centerline = np.array([[point.x, point.y] for point in lane_feature.polyline], dtype=np.float64)
        if len(centerline) < 2:
            return None, None, {"left": [], "right": []}

        left_offsets, left_ids = self._estimate_side_offsets(
            lane_feature.left_boundaries, centerline, map_
        )
        right_offsets, right_ids = self._estimate_side_offsets(
            lane_feature.right_boundaries, centerline, map_
        )

        default_width = self._DEFAULT_LANE_WIDTH.get(lane_subtype, 3.6)
        default_half_width = default_width / 2

        if np.any(np.isfinite(left_offsets)):
            left_offsets = self._interpolate_nan_values(left_offsets, default_half_width)
        if np.any(np.isfinite(right_offsets)):
            right_offsets = self._interpolate_nan_values(right_offsets, default_half_width)

        if not np.any(np.isfinite(left_offsets)) and np.any(np.isfinite(right_offsets)):
            left_offsets = right_offsets.copy()
        if not np.any(np.isfinite(right_offsets)) and np.any(np.isfinite(left_offsets)):
            right_offsets = left_offsets.copy()

        left_fill = np.nanmedian(left_offsets) if np.any(np.isfinite(left_offsets)) else default_half_width
        right_fill = (
            np.nanmedian(right_offsets) if np.any(np.isfinite(right_offsets)) else default_half_width
        )

        left_offsets = self._interpolate_nan_values(left_offsets, left_fill)
        right_offsets = self._interpolate_nan_values(right_offsets, right_fill)

        normals = self._compute_centerline_left_unit_normals(centerline)
        left_side = centerline + normals * left_offsets[:, np.newaxis]
        right_side = centerline - normals * right_offsets[:, np.newaxis]

        return (
            LineString(left_side.tolist()),
            LineString(right_side.tolist()),
            {"left": left_ids, "right": right_ids},
        )

    def _parse_map_features(self, map_feature, map_: Map):
        if map_feature.HasField("lane"):
            lane_feature = map_feature.lane
            lane_subtype = self._LANE_TYPE_MAPPING[lane_feature.type]
            left_side, right_side, line_ids = self._build_lane_sides(
                lane_feature, lane_subtype, map_
            )

            if left_side is None or right_side is None:
                return

            lane = Lane(
                id_="%05d" % map_feature.id,
                left_side=left_side,
                right_side=right_side,
                line_ids=line_ids,
                subtype=lane_subtype,
                speed_limit=lane_feature.speed_limit_mph,
                speed_limit_unit="mi/h",
                custom_tags={
                    "interpolating": lane_feature.interpolating,
                    "centerline": [[point.x, point.y] for point in lane_feature.polyline],
                },
            )

            for entry_lane in lane_feature.entry_lanes:
                lane.add_related_lane("%05d" % entry_lane, LaneRelationship.PREDECESSOR)
            for exit_lane in lane_feature.exit_lanes:
                lane.add_related_lane("%05d" % exit_lane, LaneRelationship.SUCCESSOR)
            for left_neighbor in lane_feature.left_neighbors:
                lane.add_related_lane(
                    "%05d" % left_neighbor.feature_id, LaneRelationship.LEFT_NEIGHBOR
                )
            for right_neighbor in lane_feature.right_neighbors:
                lane.add_related_lane(
                    "%05d" % right_neighbor.feature_id, LaneRelationship.RIGHT_NEIGHBOR
                )

            map_.add_lane(lane)

        elif map_feature.HasField("road_line"):
            type_, subtype, color = self._ROADLINE_TYPE_MAPPING[map_feature.road_line.type]
            points = [[point.x, point.y] for point in map_feature.road_line.polyline]
            if len(points) >= 2:
                roadline = RoadLine(
                    id_="%05d" % map_feature.id,
                    geometry=LineString(points),
                    type_=type_,
                    subtype=subtype,
                    color=color,
                )
                map_.add_roadline(roadline)

        elif map_feature.HasField("road_edge"):
            points = [[point.x, point.y] for point in map_feature.road_edge.polyline]
            if len(points) < 2:
                return
            roadline = RoadLine(
                id_="%05d" % map_feature.id,
                geometry=LineString(points),
                type_="road_border",
            )
            map_.add_roadline(roadline)

        elif map_feature.HasField("stop_sign"):
            stop_sign = Regulatory(
                id_="%05d" % map_feature.id,
                ways={"%05d" % way_id: "refers" for way_id in map_feature.stop_sign.lane},
                subtype="stop_sign",
                position=Point(map_feature.stop_sign.position.x, map_feature.stop_sign.position.y),
            )
            map_.add_regulatory(stop_sign)

        elif map_feature.HasField("crosswalk"):
            crosswalk = Area(
                id_="%05d" % map_feature.id,
                geometry=Polygon([[point.x, point.y] for point in map_feature.crosswalk.polygon]),
                subtype="crosswalk",
            )
            map_.add_area(crosswalk)

        elif map_feature.HasField("speed_bump"):
            speed_bump = Area(
                id_="%05d" % map_feature.id,
                geometry=Polygon([[point.x, point.y] for point in map_feature.speed_bump.polygon]),
                subtype="speed_bump",
            )
            map_.add_area(speed_bump)

        elif map_feature.HasField("driveway"):
            area = Area(
                id_="%05d" % map_feature.id,
                geometry=Polygon([[point.x, point.y] for point in map_feature.driveway.polygon]),
                subtype="drivable_area",
            )
            map_.add_area(area)

    def _parse_dynamic_map_features(self, dynamic_map_state, map_: Map, timestamp_ms: int):
        for lane_state in dynamic_map_state.lane_states:
            lane_id = "%05d" % lane_state.lane
            regulatory_id = f"traffic_light_{lane_id}"
            state_name = self._TRAFFIC_SIGNAL_STATE_MAPPING.get(lane_state.state, "unknown")
            stop_point = None
            if lane_state.HasField("stop_point"):
                stop_point = Point(lane_state.stop_point.x, lane_state.stop_point.y)

            if regulatory_id not in map_.regulations:
                regulation = Regulatory(
                    id_=regulatory_id,
                    ways={lane_id: "refers"},
                    subtype="traffic_light",
                    position=stop_point,
                    dynamic=True,
                    custom_tags={
                        "states": [],
                        "lane_id": lane_id,
                    },
                )
                map_.add_regulatory(regulation)
            else:
                regulation = map_.regulations[regulatory_id]
                if regulation.position is None and stop_point is not None:
                    regulation.position = stop_point

            state_record = {
                "time_ms": timestamp_ms,
                "state": state_name,
            }
            if stop_point is not None:
                state_record["stop_point"] = [stop_point.x, stop_point.y]
            regulation.custom_tags["states"].append(state_record)

        return

    def parse_map(self, scenario_id=None, **kwargs) -> Map:
        """This function parses the map from a single WOMD file.

        Args:
            scenario_id (str, optional): The id of the scenario to parse. If the scenario id is not given, the first scenario in the file will be parsed.

        Keyword Args:
            dataset (tfrecord.tfrecord_iterator, optional): The dataset to parse.
            file (str, optional): The name of the trajectory file. The file is expected to be a tfrecord file (.tfrecord).
            folder (str, optional): The path to the folder containing the tfrecord file.

        Returns:
            map_ (Map): A map object.

        Raises:
            KeyError: Either dataset or file and folder should be given as keyword arguments.
        """

        dataset = self._get_dataset(**kwargs)
        scenario, _ = self._resolve_scenario_data(scenario_id, dataset)
        if scenario is None:
            return None

        map_ = Map(name="womd_" + scenario.scenario_id)
        for map_feature in scenario.map_features:
            if map_feature.HasField("road_line") or map_feature.HasField("road_edge"):
                self._parse_map_features(map_feature, map_)
        for map_feature in scenario.map_features:
            if not map_feature.HasField("road_line") and not map_feature.HasField("road_edge"):
                self._parse_map_features(map_feature, map_)
        for i, dynamic_map_state in enumerate(scenario.dynamic_map_states):
            timestamp_ms = int(round(scenario.timestamps_seconds[i] * 1000))
            self._parse_dynamic_map_features(dynamic_map_state, map_, timestamp_ms)

        return map_
