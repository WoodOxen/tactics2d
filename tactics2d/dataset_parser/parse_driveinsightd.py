#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_driveinsightd.py
# @Description: This file implements a parser of the DriveInsight Dataset.
# @Author: Zexi Chen
# @Version: 1.0.0

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from tactics2d.map.parser.parse_xodr import XODRParser
from tactics2d.participant.element import Cyclist, Other, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _tag(element: ET.Element) -> str:
    return element.tag.split("}")[-1]


def _find_first(node: ET.Element, tag_name: str) -> Optional[ET.Element]:
    for element in node.iter():
        if _tag(element) == tag_name:
            return element
    return None


def _find_all(node: ET.Element, tag_name: str) -> list:
    return [element for element in node.iter() if _tag(element) == tag_name]


class DriveInsightDParser:
    """This class implements a parser of the DriveInsight Dataset.

    !!! quote "Reference"
        Zhdanov, Pavlo, et al. "DriveInsight: CCTV-based dataset capturing real-world scenarios." IEEE Transactions on Circuits and Systems for Video Technology, 2025 (under review).
    """

    _TYPE_MAPPING: Dict[str, str] = {
        "car":        "car",
        "truck":      "truck",
        "bus":        "bus",
        "motorcycle": "motorcycle",
        "bicycle":    "bicycle",
        "pedestrian": "pedestrian",
        "other":      "other",
    }

    _CLASS_MAPPING = {
        "car":        Vehicle,
        "truck":      Vehicle,
        "bus":        Vehicle,
        "motorcycle": Cyclist,
        "bicycle":    Cyclist,
        "pedestrian": Pedestrian,
        "other":      Other,
    }

    _DEFAULT_DIMENSIONS: Dict[str, Tuple[float, float]] = {
        "car":        (4.5, 2.0),
        "truck":      (8.0, 2.5),
        "bus":        (12.0, 2.5),
        "motorcycle": (2.2, 0.8),
        "bicycle":    (1.8, 0.6),
        "pedestrian": (0.5, 0.5),
        "other":      (2.0, 1.0),
    }

    def _extract_entity_info(self, entity: ET.Element) -> dict:
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
            type_           = "pedestrian"
            length, width   = self._DEFAULT_DIMENSIONS["pedestrian"]
        else:
            type_           = "other"
            length, width   = self._DEFAULT_DIMENSIONS["other"]

        return {"type": type_, "length": length, "width": width}

    def _make_participant(self, info: dict, id_: str):
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
        metadata = {
            "time":          "unknown",
            "weather":       "unknown",
            "precipitation": "none",
            "friction":      1.0,
        }

        env_node = _find_first(root, "Environment")
        if env_node is None:
            return metadata

        time_node = _find_first(env_node, "TimeOfDay")
        if time_node is not None:
            metadata["time"] = time_node.get("dateTime", "unknown")

        weather_node = _find_first(env_node, "Weather")
        if weather_node is not None:
            metadata["weather"] = weather_node.get("cloudState", "unknown")
            precip_node = _find_first(weather_node, "Precipitation")
            if precip_node is not None:
                metadata["precipitation"] = precip_node.get("precipitationType", "none")

        road_node = _find_first(env_node, "RoadCondition")
        if road_node is not None:
            metadata["friction"] = float(road_node.get("frictionScaleFactor", 1.0))

        return metadata

    def parse_trajectory(
        self,
        file:        Union[int, str],
        folder:      str,
        stamp_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses trajectories from a DriveInsightD OpenSCENARIO file.

        Args:
            file (Union[int, str]): The scenario identifier used to locate the file
                ``{file}_scenario.xosc``.
            folder (str): The path to the folder containing the scenario file.
            stamp_range (Optional[Tuple[float, float]]): The time range of the
                trajectory data to parse. The unit of time stamp is millisecond (ms).
                If the stamp range is not given, the parser will parse the whole
                trajectory data. Defaults to None.

        Returns:
            participants (dict[str, Union[Vehicle, Cyclist, Pedestrian, Other]]): A
                dictionary mapping participant names to their objects.
            actual_stamp_range (Tuple[int, int]): The actual time range of the
                trajectory data. The first element is the start time and the second
                is the end time. The unit of time stamp is millisecond (ms).
        """
        t_min = stamp_range[0] if stamp_range is not None else -np.inf
        t_max = stamp_range[1] if stamp_range is not None else  np.inf

        scenario_id = str(file)
        xosc_path   = Path(folder) / f"{scenario_id}_scenario.xosc"

        if not xosc_path.exists():
            raise FileNotFoundError(f"Cannot find scenario file: {xosc_path}")

        root = ET.parse(xosc_path).getroot()

        participants: dict = {}
        for entity in _find_all(root, "ScenarioObject"):
            name = entity.get("name")
            if not name:
                continue
            info = self._extract_entity_info(entity)
            participants[name] = self._make_participant(info, name)

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

        participants = {
            k: v for k, v in participants.items()
            if len(v.trajectory.frames) > 0
        }

        if actual_t_min == np.inf:
            logging.warning("No trajectory data found in scenario %s.", scenario_id)
            actual_t_min = actual_t_max = 0

        return participants, (int(actual_t_min), int(actual_t_max))

    def parse(
        self,
        scenario_id: str,
        folder:      str,
        map_name:    str = "cz_zlin.xodr",
    ) -> dict:
        """This function parses a complete DriveInsightD scenario including
        trajectory data, road network, and environment metadata.

        Args:
            scenario_id (str): The scenario identifier.
            folder (str): The path to the folder containing the scenario files.
            map_name (str): The filename of the OpenDRIVE road network.
                Defaults to ``"cz_zlin.xodr"``.

        Returns:
            scenario (dict): A dictionary with the following keys:

                - ``scenario_id`` (str): The scenario identifier.
                - ``metadata`` (dict): Environment metadata with keys ``time``,
                  ``weather``, ``precipitation``, and ``friction``.
                - ``time_range`` (Tuple[int, int]): Start and end timestamps in
                  milliseconds.
                - ``map`` (Map): The parsed Tactics2D Map object.
                - ``participants`` (dict): The parsed participant objects.
        """
        base_path = Path(folder)

        xosc_path = base_path / f"{scenario_id}_scenario.xosc"
        if not xosc_path.exists():
            raise FileNotFoundError(f"Cannot find scenario file: {xosc_path}")

        root = ET.parse(xosc_path).getroot()
        participants, time_range = self.parse_trajectory(
            file=scenario_id, folder=folder
        )
        metadata = self._extract_metadata(root)

        map_path = base_path / map_name
        if not map_path.exists():
            raise FileNotFoundError(f"Cannot find map file: {map_path}")

        parsed_map = XODRParser().parse(str(map_path))

        return {
            "scenario_id":  scenario_id,
            "metadata":     metadata,
            "time_range":   time_range,
            "map":          parsed_map,
            "participants": participants,
        }