##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: camera.py
# @Description: This file implements the render paradigm for the camera.
# @Author: Tactics2D Team
# @Version: 0.1.9

import logging
from typing import Tuple, Union

import numpy as np
from shapely.geometry import Point

from tactics2d.map.element import Area, Lane, Map, RoadLine
from tactics2d.participant.element import Cyclist, Obstacle, Pedestrian, Vehicle
from tactics2d.renderer.render_template import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER
from tactics2d.renderer.sensor_base import SensorBase


class BEVCamera(SensorBase):
    """This class defines the render paradigm of a BEV camera.

    Attributes:
        id_ (int): The unique identifier of the sensor. This attribute is **read-only** once the instance is initialized.
        map_ (Map): The map that the sensor is attached to. This attribute is **read-only** once the instance is initialized.
        perception_range (Union[float, Tuple[float]]): The distance from the sensor to its maximum detection range in (left, right, front, back). When this value is undefined, the sensor is assumed to detect the whole map. Defaults to None.
        location (Point): The location of the sensor in the global 2D coordinate system.
        bind_id (Any): The unique identifier of object that the sensor is bound to.
    """

    color_palette = COLOR_PALETTE
    color_mapper = DEFAULT_COLOR
    order_mapper = DEFAULT_ORDER

    def __init__(
        self,
        id_: int,
        map_: Map,
        perception_range: Union[float, Tuple[float]],
        resolution: Tuple[int, int] = (800, 800),
        color_palette: dict = None,
        color_mapper: dict = None,
        order_mapper: dict = None,
    ):
        super().__init__(id_, map_, perception_range)

        self._resolution = resolution
        self._map_rendered = False

        if color_palette is not None:
            self.color_palette.update(color_palette)

        if color_mapper is not None:
            self.color_mapper.update(color_mapper)

        if order_mapper is not None:
            self.order_mapper.update(order_mapper)

    @property
    def resolution(self):
        return self._resolution

    def _in_perception_range(self, geometry) -> bool:
        if self._location is None:
            return True
        return geometry.distance(self._location) > self.max_perception_distance * 2

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

    def _get_map_elements(self):
        road_element_list = []
        white = self.color_palette["white"]
        for area in self._map.areas.values():
            if not self._in_perception_range(area.geometry):
                continue

            order = self._get_order(area)
            interiors = list(area.geometry.interiors)

            road_element_list.append(
                {
                    "id": int(area.id_ * 1e6),
                    "type": "polygon",
                    "geometry": list(area.geometry.exterior.coords),
                    "color": self._get_color(area),
                    "order": order,
                    "lineWidth": 0,
                }
            )

            for i, interior in enumerate(interiors):
                road_element_list.append(
                    {
                        "id": int(area.id_ * 1e6 + i),
                        "type": "polygon",
                        "geometry": interior,
                        "color": white,
                        "order": order + 0.1,
                        "lineWidth": 0,
                    }
                )

        for lane in self._map.lanes.values():
            if not self._in_perception_range(lane.geometry):
                continue

            road_element_list.append(
                {
                    "id": int(lane.id_ * 1e6),
                    "type": "polygon",
                    "geometry": list(lane.geometry.coords),
                    "color": self._get_color(lane),
                    "order": self._get_order(lane),
                    "lineWidth": 0,
                }
            )

        for roadline in self._map.roadlines.values():
            if roadline.type_ == "virtual" or not self._in_perception_range(roadline.geometry):
                continue

            road_element_list.append(
                {
                    "id": int(roadline.id_ * 1e6),
                    "type": "dashedline" if "dashed" in roadline.subtype else "line",
                    "geometry": list(roadline.geometry.coords),
                    "color": self._get_color(roadline),
                    "order": self._get_order(roadline),
                    "lineWidth": 2 if "thick" in roadline.type_ else 1,
                }
            )

        return road_element_list

    def _get_participants(self, participants, participant_ids, frame):
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
                points = np.array(participant_geometry.coords)
                triangle = [
                    ((points[0] + points[1]) / 2).tolist(),
                    ((points[1] + points[2]) / 2).tolist(),
                    ((points[3] + points[0]) / 2).tolist(),
                ]

                participant_list.append(
                    {
                        "id": participant.id_,
                        "type": "polygon",
                        "geometry": points.tolist(),
                        "color": self._get_color(participant),
                        "order": order,
                        "lineWidth": 1,
                    }
                )
                participant_list.append(
                    {
                        "id": int(participant.id_ + 1e3),
                        "type": "polygon",
                        "geometry": triangle,
                        "color": black,
                        "order": order,
                        "lineWidth": 0,
                    }
                )
            elif isinstance(participant, Pedestrian):
                participant_list.append(
                    {
                        "id": participant.id_,
                        "type": "circle",
                        "center": participant_geometry,
                        "radius": participant_radius,
                        "color": self._get_color(participant),
                        "order": order,
                        "lineWidth": 1,
                    }
                )
            elif isinstance(participant, Obstacle):
                pass

        return participant_list

    def update(
        self,
        participants,
        participant_ids: list,
        frame: int = None,
        location: Point = None,
    ):
        self._location = location

        element_list = self._get_map_elements()
        element_list.extend(self._get_participants(participants, participant_ids, frame))

        return element_list
