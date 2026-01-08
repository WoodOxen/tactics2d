##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: camera.py
# @Description: This file implements the render paradigm for camera-type sensors.
# @Author: Tactics2D Team
# @Version: 0.1.8rc1

from typing import Tuple, Union

import numpy as np
from shapely.geometry import Point

from tactics2d.map.element import Area, Lane, Map, RoadLine
from tactics2d.participant.element import Cyclist, Obstacle, Pedestrian, Vehicle

from .render_config import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER
from .sensor_base import SensorBase


class BEVCamera(SensorBase):
    """This class defines the render paradigm of a BEV camera.

    Attributes:
        id_ (int): The unique identifier of the sensor. This attribute is **read-only** once the instance is initialized.
        map_ (Map): The map that the sensor is attached to. This attribute is **read-only** once the instance is initialized.
        perception_range (Union[float, Tuple[float]]): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None.
        position (Point): The position of the sensor in the global 2D coordinate system.
        bind_id (Any): The unique identifier of object that the sensor is bound to. This attribute is **read-only** and can only be set using the `bind_with` method.
        is_bound (bool): Whether the sensor is bound to an object. This attribute is **read-only** once the instance is initialized.
    """

    color_palette = COLOR_PALETTE
    color_mapper = DEFAULT_COLOR
    order_mapper = DEFAULT_ORDER

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: Union[float, Tuple[float]] = None,
        color_palette: dict = None,
        color_mapper: dict = None,
        order_mapper: dict = None,
    ):
        """Initialize the BEV camera.

        Args:
            id_ (int): The unique identifier of the camera.
            map_ (Map): The map that the camera is attached to.
            perception_range (Union[float, Tuple[float]], optional): The distance from the camera to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. This can be a single float value or a tuple of four values representing the perception range in each direction (left, right, front, back). Defaults to None.
            color_palette (dict, optional): _description_. Defaults to None.
            color_mapper (dict, optional): _description_. Defaults to None.
            order_mapper (dict, optional): _description_. Defaults to None.
        """
        super().__init__(id_, map_, perception_range)

        self._map_rendered = False

        if color_palette is not None:
            self.color_palette.update(color_palette)

        if color_mapper is not None:
            self.color_mapper.update(color_mapper)

        if order_mapper is not None:
            self.order_mapper.update(order_mapper)

    def _in_perception_range(self, geometry) -> bool:
        if self._position is None:
            return True

        return geometry.distance(self._position) <= self.max_perception_distance * 2

    def _get_color(self, element):
        if element.color in self.color_palette:
            return self.color_palette[element.color]

        if element.color is None:
            if hasattr(element, "subtype") and element.subtype in self.color_mapper:
                return self.color_mapper[element.subtype]
            if hasattr(element, "type_") and element.type_ in self.color_mapper:
                return self.color_mapper[element.type_]
            elif isinstance(element, Area):
                return self.color_mapper["area"]
            elif isinstance(element, Lane):
                return self.color_mapper["lane"]
            elif isinstance(element, RoadLine):
                return self.color_mapper["roadline"]
            elif isinstance(element, Vehicle):
                return self.color_mapper["vehicle"]
            elif isinstance(element, Cyclist):
                return self.color_mapper["cyclist"]
            elif isinstance(element, Pedestrian):
                return self.color_mapper["pedestrian"]

        return element.color

    def _get_order(self, element):
        if hasattr(element, "subtype") and element.subtype in self.order_mapper:
            return self.order_mapper[element.subtype]
        if hasattr(element, "type_") and element.type_ in self.order_mapper:
            return self.order_mapper[element.type_]
        elif isinstance(element, Area):
            return self.order_mapper["area"]
        elif isinstance(element, Lane):
            return self.order_mapper["lane"]
        elif isinstance(element, RoadLine):
            return self.order_mapper["roadline"]
        elif isinstance(element, Vehicle):
            return self.order_mapper["vehicle"]
        elif isinstance(element, Cyclist):
            return self.order_mapper["cyclist"]
        elif isinstance(element, Pedestrian):
            return self.order_mapper["pedestrian"]

        return 1

    def _get_map_elements(self, prev_road_id_set):
        road_id_list = []
        road_element_list = []
        white = self.color_palette["white"]

        for area in self._map.areas.values():
            if not self._in_perception_range(area.geometry):
                continue

            order = self._get_order(area)
            interiors = list(area.geometry.interiors)

            road_element_list.append(
                {
                    "id": int(1e6 + int(area.id_)),
                    "type": "polygon",
                    "geometry": list(area.geometry.exterior.coords),
                    "color": self._get_color(area),
                    "order": order,
                    "line_width": 0,
                }
            )
            road_id_list.append(int(1e6 + int(area.id_)))

            for i, interior in enumerate(interiors):
                road_element_list.append(
                    {
                        "id": int(1e6 + int(area.id_) + i * 1e5),
                        "type": "polygon",
                        "geometry": list(interior.coords),
                        "color": white,
                        "order": order + 0.1,
                        "line_width": 0,
                    }
                )
                road_id_list.append(int(1e6 + int(area.id_) + i * 1e5))

        for lane in self._map.lanes.values():
            if not self._in_perception_range(lane.geometry):
                continue

            road_element_list.append(
                {
                    "id": int(1e6 + int(lane.id_)),
                    "type": "polygon",
                    "geometry": list(lane.geometry.coords),
                    "color": self._get_color(lane),
                    "order": self._get_order(lane),
                    "line_width": 0,
                }
            )
            road_id_list.append(int(1e6 + int(lane.id_)))

        for roadline in self._map.roadlines.values():
            if roadline.type_ == "virtual" or not self._in_perception_range(roadline.geometry):
                continue

            line_width = 1
            if roadline.type_ in ["line_thin", "curbstone"]:
                line_width = 0.5
            elif "thick" in roadline.type_:
                line_width = 2

            road_element_list.append(
                {
                    "id": int(1e6 + int(roadline.id_)),
                    "type": (
                        "line_" + roadline.subtype if roadline.subtype is not None else "line_solid"
                    ),
                    "geometry": list(roadline.geometry.coords),
                    "color": self._get_color(roadline),
                    "order": self._get_order(roadline),
                    "line_width": line_width,
                }
            )
            road_id_list.append(int(1e6 + int(roadline.id_)))

        # Create the map geometry message flow
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

    def _get_participants(self, frame, participants, participant_ids, prev_participant_id_set):
        participant_id_list = []
        participant_list = []
        black = self.color_palette["black"]

        for participant_id in participant_ids:
            participant = participants[participant_id]
            participant_geometry = participant.get_pose(frame)
            if isinstance(participant, Pedestrian):
                participant_radius = participant_geometry[1]
                participant_radius = participant_radius if participant_radius > 0 else 0
                participant_geometry = Point(participant_geometry[0])

            if not self._in_perception_range(participant_geometry):
                continue

            order = self._get_order(participant)

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
                id_ = abs(participant.id_)

                participant_list.append(
                    {
                        "id": id_,
                        "type": "polygon",
                        "geometry": points.tolist(),
                        "position": position,
                        "rotation": heading,
                        "color": self._get_color(participant),
                        "order": order,
                        "line_width": 1,
                    }
                )
                participant_id_list.append(id_)

                participant_list.append(
                    {
                        "id": id_ + 0.5,
                        "type": "polygon",
                        "geometry": triangle,
                        "position": position,
                        "rotation": heading,
                        "color": black,
                        "order": order + 0.1,
                        "line_width": 0,
                    }
                )
                participant_id_list.append(id_ + 0.5)

            elif isinstance(participant, Pedestrian):
                id_ = abs(participant.id_)
                participant_list.append(
                    {
                        "id": id_,
                        "type": "circle",
                        "position": [participant_geometry.x, participant_geometry.y],
                        "radius": participant_radius,
                        "color": self._get_color(participant),
                        "order": order,
                        "line_width": 1,
                    }
                )
                participant_id_list.append(id_)

            elif isinstance(participant, Obstacle):
                pass

        # Create the participant geometry message flow
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
        participants: dict,
        participant_ids: list,
        prev_road_id_set: set,
        prev_participant_id_set: set,
        position: Point = None,
    ):
        """This function is used to update the camera's position and obtain the geometry data under specific rendering paradigm.

        Args:
            frame (int): The frame of the observation.
            participants (dict): The participants in the scenario.
            participant_ids (list): The list of participant IDs to be rendered.
            prev_road_id_set (set): The set of road IDs that were rendered in the previous frame.
            prev_participant_id_set (set): The set of participant IDs that were rendered in the previous frame.
            position (Point, optional): The position of the camera. Defaults to None.

        Returns:
            geometry_data (dict): The geometry data to be rendered, including a dict with three keys: frame, map_data, and participant_data. The map_data is a dict with keys road_id_to_remove (list) and road_elements (dict), while participant_data is a dict with keys participant_id_to_create (list), participant_id_to_remove (list), and participants (dict).
            road_id_set (set): The set of road IDs that were rendered in the current frame.
            participant_id_set (set): The set of participant IDs that were rendered in the current frame.
        """
        self._position = position

        if participant_ids is None:
            participant_ids = dict()
        if prev_road_id_set is None:
            prev_road_id_set = set()
        if prev_participant_id_set is None:
            prev_participant_id_set = set()

        map_data, road_id_set = self._get_map_elements(prev_road_id_set)
        participant_data, participant_id_set = self._get_participants(
            frame, participants, participant_ids, prev_participant_id_set
        )

        geometry_data = {"frame": frame, "map_data": map_data, "participant_data": participant_data}
        return geometry_data, road_id_set, participant_id_set
