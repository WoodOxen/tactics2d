#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_driveinsightd.py
# @Description: This file implements a parser of the DriveInsight Dataset.
# @Author: Zexi Chen
# @Version: 1.0.0

"""DriveInsightD 数据集解析器。

数据集格式说明
--------------
每个场景由两个文件组成：
  * {scenario_id}_scenario.xosc  —— OpenSCENARIO 格式，包含参与者定义和轨迹
  * {map_name}.xodr               —— OpenDRIVE 格式，包含静态路网

主要接口
--------
  parser = DriveInsightDParser()

  # 仅解析轨迹（不加载地图）
  participants, time_range = parser.parse_trajectory(file="106", folder="/path/to/data")

  # 一键加载完整场景（轨迹 + 地图 + 环境元数据）
  scenario = parser.parse(scenario_id="106", folder="/path/to/data")
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from tactics2d.map.parser.parse_xodr import XODRParser
from tactics2d.participant.element import Cyclist, Other, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


# ---------------------------------------------------------------------------
# XML 命名空间无关的查找工具
# ---------------------------------------------------------------------------
# OpenSCENARIO 文件有时携带命名空间前缀（如 {http://...}Vehicle），
# 直接用 find/findall 会失效。以下工具函数通过截断前缀来规避该问题。

def _tag(element: ET.Element) -> str:
    """返回去除命名空间前缀后的标签名。"""
    return element.tag.split("}")[-1]


def _find_first(node: ET.Element, tag_name: str) -> Optional[ET.Element]:
    """在 node 的子树中查找第一个匹配 tag_name 的元素（忽略命名空间）。"""
    for element in node.iter():
        if _tag(element) == tag_name:
            return element
    return None


def _find_all(node: ET.Element, tag_name: str) -> list:
    """在 node 的子树中查找所有匹配 tag_name 的元素（忽略命名空间）。"""
    return [element for element in node.iter() if _tag(element) == tag_name]


# ---------------------------------------------------------------------------
# DriveInsightDParser
# ---------------------------------------------------------------------------

class DriveInsightDParser:
    """DriveInsightD 数据集解析器。

    该解析器将 OpenSCENARIO (.xosc) 轨迹文件和 OpenDRIVE (.xodr) 路网文件
    统一解析为 Tactics2D 的内部数据结构。
    """

    # 将 xosc 中的 vehicleCategory 映射到 Tactics2D 的类型字符串
    _TYPE_MAPPING: Dict[str, str] = {
        "car":         "car",
        "truck":       "truck",
        "bus":         "bus",
        "motorcycle":  "motorcycle",
        "bicycle":     "bicycle",
        "pedestrian":  "pedestrian",
        "other":       "other",
    }

    # 将类型字符串映射到对应的 Tactics2D 参与者类
    _CLASS_MAPPING = {
        "car":        Vehicle,
        "truck":      Vehicle,
        "bus":        Vehicle,
        "motorcycle": Cyclist,
        "bicycle":    Cyclist,
        "pedestrian": Pedestrian,
        "other":      Other,
    }

    # 各类型的默认尺寸（长 × 宽，单位：米）
    _DEFAULT_DIMENSIONS: Dict[str, Tuple[float, float]] = {
        "car":        (4.5, 2.0),
        "truck":      (8.0, 2.5),
        "bus":        (12.0, 2.5),
        "motorcycle": (2.2, 0.8),
        "bicycle":    (1.8, 0.6),
        "pedestrian": (0.5, 0.5),
        "other":      (2.0, 1.0),
    }

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _extract_entity_info(self, entity: ET.Element) -> dict:
        """从 <ScenarioObject> 节点中提取参与者类型和尺寸。"""
        vehicle_node    = _find_first(entity, "Vehicle")
        pedestrian_node = _find_first(entity, "Pedestrian")

        if vehicle_node is not None:
            category = vehicle_node.get("vehicleCategory", "car").lower()
            type_    = category if category in self._TYPE_MAPPING else "car"

            default_l, default_w = self._DEFAULT_DIMENSIONS.get(type_, (4.5, 2.0))
            dim_node = _find_first(vehicle_node, "Dimensions")
            length   = float(dim_node.get("length", default_l)) if dim_node is not None else default_l
            width    = float(dim_node.get("width",  default_w)) if dim_node is not None else default_w

        elif pedestrian_node is not None:
            type_  = "pedestrian"
            length, width = self._DEFAULT_DIMENSIONS["pedestrian"]

        else:
            type_  = "other"
            length, width = self._DEFAULT_DIMENSIONS["other"]

        return {"type": type_, "length": length, "width": width}

    def _make_participant(self, info: dict, id_: str):
        """根据实体信息构造 Tactics2D 参与者对象。"""
        type_  = self._TYPE_MAPPING.get(info["type"], "other")
        class_ = self._CLASS_MAPPING.get(info["type"], Other)

        return class_(
            id_=id_,
            type_=type_,
            length=info["length"],
            width=info["width"],
            trajectory=Trajectory(id_=id_, fps=None, stable_freq=False),
        )

    def _extract_metadata(self, root: ET.Element) -> dict:
        """从 xosc 根节点中提取环境元数据（天气、时间、路面状况）。"""
        metadata = {
            "time":          "未知",
            "weather":       "未知",
            "precipitation": "无",
            "friction":      1.0,
        }

        env_node = _find_first(root, "Environment")
        if env_node is None:
            return metadata

        time_node = _find_first(env_node, "TimeOfDay")
        if time_node is not None:
            metadata["time"] = time_node.get("dateTime", "未知")

        weather_node = _find_first(env_node, "Weather")
        if weather_node is not None:
            metadata["weather"] = weather_node.get("cloudState", "未知")
            precip_node = _find_first(weather_node, "Precipitation")
            if precip_node is not None:
                metadata["precipitation"] = precip_node.get("precipitationType", "无")

        road_node = _find_first(env_node, "RoadCondition")
        if road_node is not None:
            metadata["friction"] = float(road_node.get("frictionScaleFactor", 1.0))

        return metadata

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def parse_trajectory(
        self,
        file:        Union[int, str],
        folder:      str,
        stamp_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[dict, Tuple[int, int]]:
        """解析 OpenSCENARIO 文件，返回参与者字典和实际时间范围。

        Parameters
        ----------
        file : int | str
            场景编号，用于定位 {file}_scenario.xosc。
        folder : str
            数据集场景文件夹路径。
        stamp_range : (float, float), optional
            时间戳过滤范围（毫秒）。None 表示加载全部数据。

        Returns
        -------
        participants : dict[str, Vehicle | Cyclist | Pedestrian | Other]
            以参与者名称为键的字典。
        time_range : (int, int)
            实际数据的起止时间戳（毫秒）。
        """
        t_min = stamp_range[0] if stamp_range is not None else -np.inf
        t_max = stamp_range[1] if stamp_range is not None else  np.inf

        scenario_id = str(file)
        xosc_path   = Path(folder) / f"{scenario_id}_scenario.xosc"

        if not xosc_path.exists():
            raise FileNotFoundError(f"找不到场景文件：{xosc_path}")

        root = ET.parse(xosc_path).getroot()

        # 1. 构建参与者对象（不含轨迹）
        participants: dict = {}
        for entity in _find_all(root, "ScenarioObject"):
            name = entity.get("name")
            if not name:
                continue
            info = self._extract_entity_info(entity)
            participants[name] = self._make_participant(info, name)

        # 2. 填充轨迹状态
        actual_t_min =  np.inf
        actual_t_max = -np.inf

        for mg in _find_all(root, "ManeuverGroup"):
            ref_node = _find_first(mg, "EntityRef")
            if ref_node is None:
                continue

            name = ref_node.get("entityRef")
            if not name or name not in participants:
                continue

            for vertex in _find_all(mg, "Vertex"):
                t_ms = int(float(vertex.get("time", 0.0)) * 1000)

                if t_ms < t_min or t_ms > t_max:
                    continue

                actual_t_min = min(actual_t_min, t_ms)
                actual_t_max = max(actual_t_max, t_ms)

                pos = _find_first(vertex, "WorldPosition")
                if pos is None:
                    continue

                state = State(
                    frame=t_ms,
                    x=float(pos.get("x", 0.0)),
                    y=float(pos.get("y", 0.0)),
                    heading=float(pos.get("h", 0.0)),
                    vx=0.0, vy=0.0, ax=0.0, ay=0.0,
                )
                participants[name].trajectory.add_state(state)

        # 过滤掉没有任何轨迹数据的参与者（实体定义存在但无对应 ManeuverGroup）
        participants = {
            k: v for k, v in participants.items()
            if len(v.trajectory.frames) > 0
        }

        if actual_t_min == np.inf:
            logging.warning("场景 %s 中未找到任何轨迹数据。", scenario_id)
            actual_t_min = actual_t_max = 0

        return participants, (int(actual_t_min), int(actual_t_max))

    def parse(
        self,
        scenario_id: str,
        folder:      str,
        map_name:    str = "cz_zlin.xodr",
    ) -> dict:
        """一键解析完整场景，返回包含地图、参与者和元数据的字典。

        Parameters
        ----------
        scenario_id : str
            场景编号。
        folder : str
            数据集场景文件夹路径。
        map_name : str
            路网文件名，默认为 "cz_zlin.xodr"。

        Returns
        -------
        scenario : dict，包含以下键：
            "scenario_id"  —— str
            "metadata"     —— dict（天气、时间、路面摩擦系数）
            "time_range"   —— (int, int)，起止时间戳（毫秒）
            "map"          —— Map 对象
            "participants" —— dict[str, participant]
        """
        base_path = Path(folder)

        # 1. 解析轨迹（同时拿到 root 供后续提取元数据，避免二次读文件）
        xosc_path = base_path / f"{scenario_id}_scenario.xosc"
        if not xosc_path.exists():
            raise FileNotFoundError(f"找不到场景文件：{xosc_path}")

        root         = ET.parse(xosc_path).getroot()
        participants, time_range = self.parse_trajectory(
            file=scenario_id, folder=folder
        )
        metadata = self._extract_metadata(root)

        # 2. 解析静态路网
        map_path = base_path / map_name
        if not map_path.exists():
            raise FileNotFoundError(f"找不到地图文件：{map_path}")

        parsed_map = XODRParser().parse(str(map_path))

        return {
            "scenario_id":  scenario_id,
            "metadata":     metadata,
            "time_range":   time_range,
            "map":          parsed_map,
            "participants": participants,
        }