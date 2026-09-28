# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""PINNS dataset parser implementation."""


import json
import os
import re
from typing import Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Polygon

from tactics2d.map.element import Area, Lane, Map, RoadLine
from tactics2d.participant.element import Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class PINNSParser:
    """This class implements a parser for PINNS dataset.

    PINNS (Pedestrian-vehicle interaction dataset) provides pedestrian-vehicle
    interaction trajectories collected from uncalibrated surveillance cameras.
    The raw label files contain frame-level object records, and the accompanying
    ``calib.json`` files provide scene metadata and BEV calibration information.
    """

    _LABEL_COLUMNS = [
        "frame",
        "class",
        "track_id",
        "image_x",
        "image_y",
        "confidence",
        "image_width",
        "image_height",
        "x",
        "y",
        "interaction_id",
        "heading",
    ]

    _TYPE_MAPPING = {
        "Car": "car",
        "Person": "pedestrian",
    }

    _CLASS_MAPPING = {
        "Car": Vehicle,
        "Person": Pedestrian,
    }

    # PINNS 标签没有稳定的物理长宽字段，这里只给参与者设置常用默认尺寸。
    # 原始图像框尺寸会完整保存在每一帧 State 的附加属性中。
    _DEFAULT_SIZE = {
        "Car": (4.5, 1.8),
        "Person": (0.5, 0.5),
    }

    _DEFAULT_LANE_WIDTH = 3.5

    def _get_dataset_root(self, folder: str) -> str:
        folder = os.path.abspath(folder)
        if os.path.basename(folder) == "labels":
            return os.path.dirname(folder)
        return folder

    def _resolve_label_file(self, file: Union[int, str], folder: str) -> str:
        dataset_root = self._get_dataset_root(folder)
        label_folder = os.path.join(dataset_root, "labels")

        # PINNS 没有全局数字编号；若传入整数，按标签文件名排序后取对应下标。
        if isinstance(file, int):
            label_files = sorted(
                f for f in os.listdir(label_folder) if f.lower().endswith(".txt")
            )
            if file < 0 or file >= len(label_files):
                raise FileNotFoundError(f"PINNS label index {file} is out of range.")
            return os.path.join(label_folder, label_files[file])

        if not isinstance(file, str):
            raise TypeError("The input file must be an integer or a string.")

        candidates = []
        if os.path.isabs(file):
            candidates.append(file)
        else:
            candidates.extend(
                [
                    os.path.join(folder, file),
                    os.path.join(label_folder, file),
                ]
            )

        if not file.lower().endswith(".txt"):
            candidates.extend([f"{candidate}.txt" for candidate in candidates])

        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(f"Cannot find PINNS label file {file} in {folder}.")

    def _get_label_stem(self, file: Union[int, str], folder: str) -> str:
        return os.path.splitext(os.path.basename(self._resolve_label_file(file, folder)))[0]

    def _get_scene_name(self, file: Union[int, str], folder: str) -> str:
        # 标签名形如 america_crossroad_summer_daytime_sunny_0.txt，
        # 末尾数字是同一场景下的视频片段编号，去掉后即可定位 calib.json。
        return re.sub(r"_\d+$", "", self._get_label_stem(file, folder))

    def get_scene_info(self, file: Union[int, str], folder: str) -> dict:
        """Get scene metadata and BEV calibration information for a PINNS label file.

        Args:
            file (Union[int, str]): The label file name, stem, absolute path, or sorted label index.
            folder (str): The PINNS dataset root folder or its ``labels`` folder.

        Returns:
            scene_info (dict): The content of the corresponding ``calib.json`` file.
        """
        dataset_root = self._get_dataset_root(folder)
        scene_name = self._get_scene_name(file, folder)
        calib_path = os.path.join(dataset_root, "videos", scene_name, "calib.json")

        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Cannot find PINNS calibration file {calib_path}.")

        with open(calib_path, "r", encoding="utf-8") as calib_file:
            scene_info = json.load(calib_file)

        return scene_info

    def get_time_range(self, file: Union[int, str], folder: str) -> Tuple[int, int]:
        """This function gets the time range of a single PINNS trajectory file.

        Args:
            file (Union[int, str]): The label file name, stem, absolute path, or sorted label index.
            folder (str): The PINNS dataset root folder or its ``labels`` folder.

        Returns:
            actual_time_range (Tuple[int, int]): The time range of the trajectory data.
            The first element is the start time. The second element is the end time.
            The unit of time stamp is millisecond (ms).
        """
        label_path = self._resolve_label_file(file, folder)
        scene_info = self.get_scene_info(file, folder)
        fps = float(scene_info.get("fps", 30.0))
        df = pd.read_csv(
            label_path,
            sep=r"\s+",
            header=None,
            names=self._LABEL_COLUMNS,
            usecols=["frame"],
            engine="python",
        )

        if df.empty:
            return (np.inf, -np.inf)

        start_stamp = int(round(int(df["frame"].min()) * 1000 / fps))
        end_stamp = int(round(int(df["frame"].max()) * 1000 / fps))

        return (start_stamp, end_stamp)

    def _read_label(self, label_path: str) -> pd.DataFrame:
        df = pd.read_csv(
            label_path,
            sep=r"\s+",
            header=None,
            names=self._LABEL_COLUMNS,
            engine="python",
        )

        if len(df.columns) != len(self._LABEL_COLUMNS):
            raise ValueError("PINNS label file must contain 12 whitespace-separated columns.")
        if df.isnull().any(axis=None):
            raise ValueError("PINNS label file contains incomplete rows.")

        unknown_types = set(df["class"].unique()) - set(self._TYPE_MAPPING.keys())
        if unknown_types:
            raise KeyError(f"Unsupported PINNS object classes: {sorted(unknown_types)}.")

        return df

    def _build_state_sequence(self, group: pd.DataFrame, fps: float) -> list:
        group = group.sort_values("time_stamp")
        states = []
        last_row = None

        for _, row in group.iterrows():
            vx, vy = None, None
            if last_row is not None and row["time_stamp"] > last_row["time_stamp"]:
                dt = (row["time_stamp"] - last_row["time_stamp"]) / 1000.0
                vx = (row["x"] - last_row["x"]) / dt
                vy = (row["y"] - last_row["y"]) / dt

            state = State(
                frame=int(row["time_stamp"]),
                x=row["x"],
                y=row["y"],
                heading=row["heading"],
                vx=vx,
                vy=vy,
            )

            # 以下字段是 PINNS 原始标签中的附加信息，保留给后续交互分析和地图绘制使用。
            state.source_frame = int(row["frame"])
            state.image_x = float(row["image_x"])
            state.image_y = float(row["image_y"])
            state.image_width = float(row["image_width"])
            state.image_height = float(row["image_height"])
            state.confidence = float(row["confidence"])
            state.interaction_id = int(row["interaction_id"])
            state.source_class = row["class"]
            states.append(state)
            last_row = row

        # 第一帧没有前序点，使用后一帧速度补齐，避免轨迹起点速度为空。
        if len(states) > 1 and states[0].vx is None:
            states[0].vx = states[1].vx
            states[0].vy = states[1].vy

        return states

    def parse_trajectory(
        self,
        file: Union[int, str],
        folder: str,
        time_range: Tuple[int, int] = None,
        ids: list = None,
    ) -> Tuple[dict, Tuple[int, int]]:
        """Parse trajectories from a PINNS label file.

        Args:
            file (Union[int, str]): The label file name, stem, absolute path, or sorted label index.
            folder (str): The PINNS dataset root folder or its ``labels`` folder.
            time_range (Tuple[int, int], optional): The time range to parse. The unit of
                time stamp is millisecond (ms). If not given, the whole file is parsed.
            ids (list, optional): The list of original PINNS track ids to parse.

        Returns:
            participants (dict): A dictionary of participants. The keys are original PINNS
            track ids, and the values are Tactics2D participant objects.
            actual_stamp_range (Tuple[int, int]): The actual parsed time range in millisecond.
        """
        if time_range is None:
            time_range = (-np.inf, np.inf)
        if ids is not None:
            ids = {int(id_) for id_ in ids}

        label_path = self._resolve_label_file(file, folder)
        scene_info = self.get_scene_info(file, folder)
        fps = float(scene_info.get("fps", 30.0))
        df = self._read_label(label_path)

        # Tactics2D 的 State.frame 使用毫秒时间戳；PINNS 原始 frame 是视频帧序号。
        df["time_stamp"] = (df["frame"] * 1000 / fps).round().astype(int)
        df = df[(df["time_stamp"] >= time_range[0]) & (df["time_stamp"] <= time_range[1])]
        if ids is not None:
            df = df[df["track_id"].isin(ids)]

        if df.empty:
            return {}, (np.inf, -np.inf)

        actual_stamp_range = (int(df["time_stamp"].min()), int(df["time_stamp"].max()))
        participants = {}

        for track_id, group in df.groupby("track_id", sort=False):
            track_id = int(track_id)
            first_row = group.iloc[0]
            class_name = first_row["class"]
            class_ = self._CLASS_MAPPING[class_name]
            type_ = self._TYPE_MAPPING[class_name]
            length, width = self._DEFAULT_SIZE[class_name]

            trajectory = Trajectory(track_id, fps=fps, stable_freq=False)
            for state in self._build_state_sequence(group, fps):
                trajectory.add_state(state)

            participant = class_(
                id_=track_id,
                type_=type_,
                length=length,
                width=width,
                trajectory=trajectory,
            )
            # ParticipantBase 会根据 kwargs 初始化通用字段；这里显式恢复标准化类型，
            # 保证下游测试、渲染和统计逻辑可以直接读取 participant.type_。
            participant.type_ = type_
            participant.source_id = track_id
            participant.source_class = class_name
            participant.scene_name = self._get_scene_name(file, folder)
            participants[track_id] = participant

        return participants, actual_stamp_range

    @staticmethod
    def _polygon_from_calib(scene_info: dict) -> Polygon:
        x_min = float(scene_info.get("x_min", 0.0))
        x_max = float(scene_info.get("x_max", 0.0))
        y_min = float(scene_info.get("y_min", 0.0))
        y_max = float(scene_info.get("y_max", 0.0))
        if x_max <= x_min or y_max <= y_min:
            pts_3d = scene_info.get("pts_3d", [])
            if len(pts_3d) >= 3:
                polygon = Polygon(pts_3d)
                if polygon.is_valid and not polygon.is_empty:
                    return polygon
            raise ValueError("PINNS calibration must provide a valid BEV boundary.")

        return Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

    @staticmethod
    def _trajectory_points(participants: dict, class_name: str) -> list:
        points = []
        for participant in participants.values():
            if getattr(participant, "source_class", None) != class_name:
                continue
            points.extend(state.location for state in participant.trajectory.history_states.values())
        return points

    @staticmethod
    def _area_from_points(points: list, buffer_width: float, clip_polygon: Polygon) -> Polygon:
        if len(points) == 0:
            return None

        # 原始轨迹是离散中心点，使用 convex hull + buffer 得到保守的活动区域估计。
        geometry = MultiPoint(points).convex_hull.buffer(buffer_width)
        if clip_polygon is not None and not clip_polygon.is_empty:
            geometry = geometry.intersection(clip_polygon)

        if geometry.is_empty:
            return None
        if geometry.geom_type == "Polygon":
            return geometry

        polygons = [geom for geom in getattr(geometry, "geoms", []) if geom.geom_type == "Polygon"]
        return max(polygons, key=lambda geom: geom.area) if polygons else None

    @staticmethod
    def _longest_linestring(geometry):
        if geometry.is_empty:
            return None
        if geometry.geom_type == "LineString":
            return geometry
        lines = [geom for geom in getattr(geometry, "geoms", []) if geom.geom_type == "LineString"]
        return max(lines, key=lambda geom: geom.length) if lines else None

    def _lane_from_participant(self, participant, lane_id: str, scene_polygon: Polygon):
        coords = [state.location for state in participant.trajectory.history_states.values()]
        if len(coords) < 2:
            return None

        # 删除连续重复点，避免 LineString 或 offset 计算失败。
        filtered_coords = [coords[0]]
        for coord in coords[1:]:
            if np.linalg.norm(np.asarray(coord) - np.asarray(filtered_coords[-1])) > 1e-3:
                filtered_coords.append(coord)

        if len(filtered_coords) < 2:
            return None

        centerline = LineString(filtered_coords).simplify(0.3, preserve_topology=False)
        if centerline.length < 5.0:
            return None

        half_width = self._DEFAULT_LANE_WIDTH / 2
        left_side = self._longest_linestring(centerline.parallel_offset(half_width, "left"))
        right_side = self._longest_linestring(centerline.parallel_offset(half_width, "right"))
        if left_side is None or right_side is None:
            return None

        lane = Lane(
            id_=lane_id,
            left_side=left_side,
            right_side=right_side,
            subtype="driving",
            location="urban",
            inferred_participants=["vehicle"],
            custom_tags={
                "source": "PINNS trajectory inference",
                "source_track_id": participant.source_id,
                "centerline": list(centerline.coords),
                "inferred": True,
            },
        )

        if scene_polygon is not None and lane.geometry is not None:
            lane.custom_tags["within_scene_boundary"] = scene_polygon.intersects(lane.geometry)

        return lane

    def _add_polygon_area(
        self,
        map_: Map,
        area_id: str,
        polygon: Polygon,
        subtype: str,
        inferred_participants: list,
        custom_tags: dict,
    ) -> None:
        if polygon is None or polygon.is_empty:
            return

        area = Area(
            id_=area_id,
            geometry=polygon,
            subtype=subtype,
            inferred_participants=inferred_participants,
            custom_tags=custom_tags,
        )
        map_.add_area(area)

        coords = list(polygon.exterior.coords)
        for idx in range(len(coords) - 1):
            line = RoadLine(
                id_=f"{area_id}_boundary_{idx}",
                geometry=LineString([coords[idx], coords[idx + 1]]),
                type_="virtual",
                subtype="inferred_boundary",
                custom_tags={"source_area": area_id, "inferred": True},
            )
            map_.add_roadline(line)

    def parse_map(
        self,
        file: Union[int, str],
        folder: str,
        time_range: Tuple[int, int] = None,
        max_lanes: int = 20,
    ) -> Map:
        """Infer a lightweight map from parsed PINNS trajectories and scene metadata.

        PINNS does not provide surveyed HD maps. This function therefore creates an
        inferred map for visualization and downstream analysis: scene boundary from
        ``calib.json``, activity areas from trajectory envelopes, and lane corridors
        from representative vehicle tracks.

        Args:
            file (Union[int, str]): The label file name, stem, absolute path, or sorted label index.
            folder (str): The PINNS dataset root folder or its ``labels`` folder.
            time_range (Tuple[int, int], optional): The time range used for trajectory parsing.
            max_lanes (int, optional): Maximum number of representative vehicle trajectories
                converted into inferred lane corridors.

        Returns:
            map_ (Map): A Tactics2D map object containing inferred PINNS map elements.
        """
        scene_info = self.get_scene_info(file, folder)
        scene_name = self._get_scene_name(file, folder)
        participants, _ = self.parse_trajectory(file, folder, time_range=time_range)

        scene_polygon = self._polygon_from_calib(scene_info)
        map_ = Map(
            name=f"PINNS_{scene_name}",
            scenario_type=scene_info.get("scene"),
            country=scene_info.get("location"),
        )

        map_.customs["source"] = "PINNS"
        map_.customs["generation"] = "inferred_from_trajectories"
        map_.customs["calibration"] = scene_info

        self._add_polygon_area(
            map_,
            "scene_boundary",
            scene_polygon,
            "scene_boundary",
            ["vehicle", "pedestrian"],
            {"source": "PINNS calib.json", "inferred": False},
        )

        vehicle_points = self._trajectory_points(participants, "Car")
        pedestrian_points = self._trajectory_points(participants, "Person")
        all_points = vehicle_points + pedestrian_points

        drivable_area = self._area_from_points(vehicle_points, 2.5, scene_polygon)
        pedestrian_area = self._area_from_points(pedestrian_points, 1.2, scene_polygon)
        interaction_area = self._area_from_points(all_points, 1.5, scene_polygon)

        self._add_polygon_area(
            map_,
            "inferred_drivable_area",
            drivable_area,
            "drivable_area",
            ["vehicle"],
            {"source": "PINNS vehicle trajectories", "inferred": True},
        )
        self._add_polygon_area(
            map_,
            "inferred_pedestrian_area",
            pedestrian_area,
            "walkway",
            ["pedestrian"],
            {"source": "PINNS pedestrian trajectories", "inferred": True},
        )
        self._add_polygon_area(
            map_,
            "inferred_interaction_area",
            interaction_area,
            "interaction_area",
            ["vehicle", "pedestrian"],
            {"source": "PINNS all trajectories", "inferred": True},
        )

        vehicles = [
            participant
            for participant in participants.values()
            if getattr(participant, "source_class", None) == "Car"
        ]
        vehicles.sort(key=lambda item: item.trajectory.last_state.speed or 0.0, reverse=True)
        vehicles.sort(key=lambda item: len(item.trajectory), reverse=True)

        lane_count = 0
        for participant in vehicles:
            if lane_count >= max_lanes:
                break
            lane = self._lane_from_participant(
                participant, f"inferred_lane_{lane_count}", scene_polygon
            )
            if lane is None:
                continue
            map_.add_lane(lane)
            lane_count += 1

        return map_

    def draw_map(
        self,
        file: Union[int, str],
        folder: str,
        output_path: str,
        time_range: Tuple[int, int] = None,
        max_lanes: int = 20,
    ) -> str:
        """Draw the inferred PINNS map and save it as an image.

        Args:
            file (Union[int, str]): The label file name, stem, absolute path, or sorted label index.
            folder (str): The PINNS dataset root folder or its ``labels`` folder.
            output_path (str): The target image path.
            time_range (Tuple[int, int], optional): The time range used for map inference.
            max_lanes (int, optional): Maximum number of inferred lane corridors to draw.

        Returns:
            output_path (str): The saved image path.
        """
        import matplotlib.pyplot as plt

        map_ = self.parse_map(file, folder, time_range=time_range, max_lanes=max_lanes)

        fig, ax = plt.subplots(figsize=(8, 8))
        """
        黑色边框：场景边界
        浅蓝色：车辆可行使区域
        浅橙色：行人活动区域
        浅绿色：人车交互区域
        深色细线：只有车的推断 lane centerline
        """
        area_styles = {
            "scene_boundary": {"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            "drivable_area": {"facecolor": "#d8e8f8", "edgecolor": "#2f6f9f", "alpha": 0.35},
            "walkway": {"facecolor": "#f8dfc7", "edgecolor": "#b76e2b", "alpha": 0.35},
            "interaction_area": {"facecolor": "#d9ead3", "edgecolor": "#4f8f45", "alpha": 0.25},
        }

        for area in map_.areas.values():
            style = area_styles.get(area.subtype, {"facecolor": "none", "edgecolor": "gray"})
            x, y = area.geometry.exterior.xy
            ax.fill(x, y, **style)

        for lane in map_.lanes.values():
            centerline = lane.centerline()
            if centerline is not None:
                x, y = centerline.xy
                ax.plot(x, y, color="#1f1f1f", linewidth=1.0, alpha=0.75)

        for roadline in map_.roadlines.values():
            x, y = roadline.geometry.xy
            ax.plot(x, y, color="#555555", linewidth=0.5, alpha=0.4)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(map_.name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        return output_path
