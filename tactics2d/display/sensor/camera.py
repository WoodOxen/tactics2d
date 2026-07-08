# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Camera implementation."""


import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon

from tactics2d.map.element import Area, Junction, Lane, Map, RoadLine
from tactics2d.participant.element import Cyclist, Obstacle, Pedestrian, Vehicle

from .sensor_base import SensorBase


class BEVCamera(SensorBase):
    """This class defines the render paradigm of a BEV camera.

    Attributes:
        id_ (int): The unique identifier of the sensor. This attribute is read-only once the instance is initialized.
        map_ (Map): The map that the sensor is attached to. This attribute is read-only once the instance is initialized.
        perception_range (Union[float, Tuple[float]]): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None.
        position (Point): The position of the sensor in the global 2D coordinate system.
        bind_id (Any): The unique identifier of object that the sensor is bound to. This attribute is read-only and can only be set using the bind_with method.
        is_bound (bool): Whether the sensor is bound to an object. This attribute is read-only once the instance is initialized.
    """

    def __init__(self, id_: int, map_: Map, perception_range: Union[float, Tuple[float]] = None):
        """Initialize the BEV camera.

        Args:
            id_ (int): The unique identifier of the camera.
            map_ (Map): The map that the camera is attached to.
            perception_range (Union[float, Tuple[float]], optional): The distance from the camera to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. This can be a single float value or a tuple of four values representing the perception range in each direction (left, right, front, back). Defaults to None.
        """
        super().__init__(id_, map_, perception_range)

    def _in_perception_range(self, geometry: Any) -> bool:
        """Check if geometry is within sensor's perception range.

        Args:
            geometry: Shapely geometry object to check.

        Returns:
            True if geometry is within perception range or sensor has no position, False otherwise.
        """
        if self._position is None:
            return True

        return geometry.distance(self._position) <= self.max_perception_distance * 1.5

    def _is_pavement_junction(self, junction: Junction) -> bool:
        """Return whether a junction is a generated road-surface fill polygon."""
        return junction.custom_tags.get("type") == "road"

    def _get_type(self, element: Any) -> str:
        """Get the type string for a map or participant element.

        Args:
            element: Map element or participant element.

        Returns:
            String type identifier for the element.
        """
        if isinstance(element, Junction):
            return element.custom_tags.get("type") or "junction"
        if hasattr(element, "subtype") and element.subtype:
            return element.subtype
        elif hasattr(element, "type_") and element.type_:
            return element.type_
        elif isinstance(element, Area):
            return "area"
        elif isinstance(element, Lane):
            return "lane"
        elif isinstance(element, RoadLine):
            return "roadline"
        elif isinstance(element, Vehicle):
            return "vehicle"
        elif isinstance(element, Cyclist):
            return "cyclist"
        elif isinstance(element, Pedestrian):
            return "pedestrian"

        return "default"

    def _make_junction_element(
        self, junction: Junction
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Convert a Junction into a renderer element."""
        shape_pts = junction.custom_tags.get("shape", [])
        if len(shape_pts) < 3:
            return None, None

        try:
            junction_polygon = Polygon(shape_pts)
        except Exception:
            return None, None

        if not self._in_perception_range(junction_polygon):
            return None, None

        if self._is_pavement_junction(junction):
            buffered_polygon = junction_polygon.buffer(0.05)
            if buffered_polygon.is_empty or buffered_polygon.geom_type != "Polygon":
                buffered_polygon = junction_polygon

            element_color = "road"
            element_type: str | float = 3.1
            element_geometry = list(buffered_polygon.exterior.coords)
        else:
            element_color = self._get_type(junction)
            element_type = element_color
            element_geometry = list(junction_polygon.exterior.coords)

        element_id = int(2e6 + int(junction.id_))

        return (
            {
                "id": element_id,
                "shape": "polygon",
                "geometry": element_geometry,
                "color": element_color,
                "type": element_type,
                "line_width": 0,
            },
            element_id,
        )

    def _get_map_elements(self, prev_road_id_set: Set[int]) -> Tuple[Dict, Set[int]]:
        """Get map elements within perception range for rendering.

        Render order is areas, lanes, junctions, then roadlines.
        Ordinary junctions stay below lanes through their z-order.
        Pavement-fill junctions use type 3.1 and are inserted after lanes to cover lane end-cap seams.
        """
        road_id_list = []
        road_element_list = []
        white = "white"

        for area in self._map.areas.values():
            if area.geometry is None:
                continue
            if not self._in_perception_range(area.geometry):
                continue

            interiors = list(area.geometry.interiors)
            area_id = int(1e6 + int(area.id_))

            road_element_list.append(
                {
                    "id": area_id,
                    "shape": "polygon",
                    "geometry": list(area.geometry.exterior.coords),
                    "color": area.color,
                    "type": self._get_type(area),
                    "line_width": 0,
                }
            )
            road_id_list.append(area_id)

            for i, interior in enumerate(interiors):
                hole_id = int(1e6 + int(area.id_) + i * 1e5)

                road_element_list.append(
                    {
                        "id": hole_id,
                        "shape": "polygon",
                        "geometry": list(interior.coords),
                        "color": white,
                        "type": "hole",
                        "line_width": 0,
                    }
                )
                road_id_list.append(hole_id)

        for lane in self._map.lanes.values():
            if lane.geometry is None:
                continue
            if not self._in_perception_range(lane.geometry):
                continue

            lane_id = int(1e6 + int(lane.id_))

            road_element_list.append(
                {
                    "id": lane_id,
                    "shape": "polygon",
                    "geometry": list(lane.geometry.coords),
                    "color": lane.color,
                    "type": self._get_type(lane),
                    "line_width": 0,
                }
            )
            road_id_list.append(lane_id)

        for junction in self._map.junctions.values():
            junction_element, junction_id = self._make_junction_element(junction)

            if junction_element is None or junction_id is None:
                continue

            road_element_list.append(junction_element)
            road_id_list.append(junction_id)

        for roadline in self._map.roadlines.values():
            if roadline.geometry is None:
                continue
            if roadline.type_ == "virtual" or not self._in_perception_range(roadline.geometry):
                continue

            line_width = 1
            if roadline.type_ in ["line_thin", "curbstone"]:
                line_width = 0.5
            elif "thick" in roadline.type_:
                line_width = 2

            line_color = roadline.color or self._get_type(roadline)
            roadline_id = int(1e6 + int(roadline.id_))

            road_element_list.append(
                {
                    "id": roadline_id,
                    "shape": "line",
                    "geometry": list(roadline.geometry.coords),
                    "color": line_color,
                    "type": self._get_type(roadline),
                    "line_style": roadline.subtype if roadline.subtype is not None else "solid",
                    "line_width": line_width,
                }
            )
            road_id_list.append(roadline_id)

        road_id_set = set(road_id_list)
        road_id_to_create = road_id_set - prev_road_id_set
        road_id_to_remove = prev_road_id_set - road_id_set

        road_element_to_create = []
        for road_element in road_element_list:
            if road_element["id"] in road_id_to_create:
                road_element_to_create.append(road_element)

        map_data = {
            "road_id_to_remove": list(road_id_to_remove),
            "road_elements": road_element_to_create,
        }

        return map_data, road_id_set

    def _get_participants(
        self,
        frame: int,
        participants: Dict,
        participant_ids: List[int],
        prev_participant_id_set: Set[int],
    ) -> Tuple[Dict, Set[int]]:
        """Get participant elements within perception range for rendering.

        Args:
            frame: Current frame number.
            participants: Dictionary of all participants.
            participant_ids: List of participant IDs to consider.
            prev_participant_id_set: Set of participant IDs from previous frame.

        Returns:
            Tuple of participant_data and participant_id_set.
        """
        participant_id_list = []
        participant_list = []
        black = "black"

        for participant_id in participant_ids:
            participant = participants[participant_id]
            participant_geometry = participant.get_pose(frame)

            if isinstance(participant, Pedestrian):
                participant_radius = participant_geometry[1]
                participant_radius = participant_radius if participant_radius > 0 else 0
                participant_geometry = Point(participant_geometry[0])

            if not self._in_perception_range(participant_geometry):
                continue

            if isinstance(participant, Vehicle) or isinstance(participant, Cyclist):
                points = np.array(participant.geometry.coords)
                triangle = [
                    ((points[0] + points[1]) / 2).tolist(),
                    ((points[1] + points[2]) / 2).tolist(),
                    ((points[3] + points[0]) / 2).tolist(),
                ]
                state = participant.trajectory.get_state(frame)
                position = list(state.location)
                heading = state.heading
                id_ = abs(int(participant.id_))

                participant_list.append(
                    {
                        "id": id_,
                        "shape": "polygon",
                        "geometry": points.tolist(),
                        "position": position,
                        "rotation": heading,
                        "color": participant.color,
                        "type": self._get_type(participant),
                        "line_width": 1,
                    }
                )
                participant_id_list.append(id_)

                participant_list.append(
                    {
                        "id": id_ + 0.5,
                        "shape": "polygon",
                        "geometry": triangle,
                        "position": position,
                        "rotation": heading,
                        "color": black,
                        "type": "heading_arrow",
                        "line_width": 0,
                    }
                )
                participant_id_list.append(id_ + 0.5)

            elif isinstance(participant, Pedestrian):
                id_ = abs(int(participant.id_))
                participant_list.append(
                    {
                        "id": id_,
                        "shape": "circle",
                        "position": [participant_geometry.x, participant_geometry.y],
                        "radius": participant_radius,
                        "color": participant.color,
                        "type": self._get_type(participant),
                        "line_width": 1,
                    }
                )
                participant_id_list.append(id_)

            elif isinstance(participant, Obstacle):
                pass

        participant_id_set = set(participant_id_list)
        participant_id_to_create = participant_id_set - prev_participant_id_set
        participant_id_to_remove = prev_participant_id_set - participant_id_set

        participant_data = {
            "participant_id_to_create": list(participant_id_to_create),
            "participant_id_to_remove": list(participant_id_to_remove),
            "participants": participant_list,
        }

        return participant_data, participant_id_set

    def update(
        self,
        frame: int,
        participants: Dict,
        participant_ids: List,
        prev_road_id_set: Set = None,
        prev_participant_id_set: Set = None,
        position: Point = None,
        heading: float = None,
    ) -> Tuple[Dict, Set, Set]:
        """Update the camera and return geometry data."""
        self._set_position_heading(position, heading)

        participant_ids, prev_road_id_set, prev_participant_id_set = self._setup_update_parameters(
            participant_ids, prev_road_id_set, prev_participant_id_set
        )

        map_data, road_id_set = self._get_map_elements(prev_road_id_set)
        participant_data, participant_id_set = self._get_participants(
            frame, participants, participant_ids, prev_participant_id_set
        )

        geometry_data = {
            "frame": frame,
            "map_data": map_data,
            "participant_data": participant_data,
            "metadata": {
                "timestamp": time.time(),
                "perception_range": self._perception_range,
                "sensor_type": "camera",
                "sensor_id": self.id_,
                "sensor_position": self._position,
                "sensor_yaw": self._heading,
            },
        }

        return geometry_data, road_id_set, participant_id_set
