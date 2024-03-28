##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: map.py
# @Description: This file defines a class for a map.
# @Author: Yueyuan Li
# @Version: 1.0.0


import enum
import warnings

import numpy as np

from .area import Area
from .junction import Junction
from .lane import Lane
from .node import Node
from .regulatory import Regulatory
from .roadline import RoadLine


class MapElement(enum.Enum):
    NODE = enum.auto()
    LANE = enum.auto()
    AREA = enum.auto()
    ROADLINE = enum.auto()
    REGULATORY = enum.auto()
    JUNCTION = enum.auto()
    CUSTOM = enum.auto()


class Map:
    """This class implements a map to manage the road elements. The elements in the map include nodes, lanes, areas, roadlines, and regulations. The map is used to store the road elements and provide the interface to access the elements. Every element in the map should have a unique id.

    Attributes:
        name (str): The name of the map. Defaults to None.
        scenario_type (str): The scenario type of the map. Defaults to None.
        country (str): The country that the map is located in. Defaults to None.
        ids (set): The identifier of the elements in the map. All elements in the map should have a unique id. A conflict in ids will raise a KeyError.
        nodes (dict): The nodes in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_node](#tactics2d.map.element.Map.add_node) method.
        lanes (dict): The lanes in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_lane](#tactics2d.map.element.Map.add_lane) method.
        areas (dict): The areas in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_area](#tactics2d.map.element.Map.add_area) method.
        roadlines (dict): The roadlines in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_roadline](#tactics2d.map.element.Map.add_roadline) method.
        regulations (dict): The regulations in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_regulatory](#tactics2d.map.element.Map.add_regulatory) method.
        boundary (tuple): The boundary of the map expressed in the form of (min_x, max_x, min_y, max_y). This attribute is **read-only**.
    """

    def __init__(self, name: str = None, scenario_type: str = None, country: str = None):
        """Initialize the attributes in the class.

        Args:
            name (str, optional): The name of the map. Defaults to None.
            scenario_type (str, optional): The scenario type of the map. Defaults to None.
            country (str, optional): The country that the map is located in. Defaults to None.
        """
        self.name = name
        self.scenario_type = scenario_type
        self.country = country

        self.ids = dict()
        self.nodes = dict()
        self.lanes = dict()
        self.areas = dict()
        self.junctions = dict()
        self.roadlines = dict()
        self.regulations = dict()
        self.customs = dict()
        self._boundary = None

    @property
    def boundary(self):
        if self._boundary is None:
            x_min, x_max, y_min, y_max = (float("inf"), float("-inf"), float("inf"), float("-inf"))

            for node in self.nodes.values():
                x_min = min(x_min, node.x)
                x_max = max(x_max, node.x)
                y_min = min(y_min, node.y)
                y_max = max(y_max, node.y)

            for lane in self.lanes.values():
                lane_coords = np.array(lane.geometry.coords)
                x_min = min(x_min, np.min(lane_coords[:, 0]))
                x_max = max(x_max, np.max(lane_coords[:, 0]))
                y_min = min(y_min, np.min(lane_coords[:, 1]))
                y_max = max(y_max, np.max(lane_coords[:, 1]))

            for area in self.areas.values():
                area_coords = np.array(area.geometry.exterior.coords)
                x_min = min(x_min, np.min(area_coords[:, 0]))
                x_max = max(x_max, np.max(area_coords[:, 0]))
                y_min = min(y_min, np.min(area_coords[:, 1]))
                y_max = max(y_max, np.max(area_coords[:, 1]))

            for roadline in self.roadlines.values():
                roadline_coords = np.array(roadline.geometry.coords)
                x_min = min(x_min, np.min(roadline_coords[:, 0]))
                x_max = max(x_max, np.max(roadline_coords[:, 0]))
                y_min = min(y_min, np.min(roadline_coords[:, 1]))
                y_max = max(y_max, np.max(roadline_coords[:, 1]))

            self._boundary = (x_min - 10, x_max + 10, y_min - 10, y_max + 10)

        return self._boundary

    def add_node(self, node: Node):
        """This function adds a node to the map.

        Args:
            node (Node): The node to be added to the map.

        Raises:
            KeyError: If the id of the node is used by any other road element.
        """
        if node.id_ in self.ids:
            if node.id_ in self.nodes:
                warnings.warn(f"Node {node.id_} already exists! Replaced the node with new data.")
            else:
                raise KeyError(f"The id of Node {node.id_} is used by the other road element.")
        self.nodes[node.id_] = node
        self.ids[node.id_] = MapElement.NODE

    def add_roadline(self, roadline: RoadLine):
        """This function adds a roadline to the map.

        Args:
            roadline (RoadLine): The roadline to be added to the map.

        Raises:
            KeyError: If the id of the roadline is used by any other road element.
        """
        if roadline.id_ in self.ids:
            if roadline.id_ in self.roadlines:
                warnings.warn(
                    f"Roadline {roadline.id_} already exists! Replaced the roadline with new data."
                )
            else:
                raise KeyError(
                    f"The id of Roadline {roadline.id_} is used by the other road element."
                )

        self.roadlines[roadline.id_] = roadline
        self.ids[roadline.id_] = MapElement.ROADLINE

    def add_junction(self, junction: Junction):
        """This function adds a junction to the map.

        Args:
            junction (Junction): The junction to be added to the map.
        """
        if junction.id_ in self.ids:
            if junction.id_ in self.junctions:
                warnings.warn(
                    f"Junction {junction.id_} already exists! Replaced the junction with new data."
                )
            else:
                raise KeyError(
                    f"The id of Junction {junction.id_} is used by the other road element."
                )

        self.junctions[junction.id_] = junction
        self.ids[junction.id_] = MapElement.JUNCTION

    def add_lane(self, lane: Lane):
        """This function adds a lane to the map.

        Args:
            lane (Lane): The lane to be added to the map.

        Raises:
            KeyError: If the id of the lane is used by any other road element.
        """
        if lane.id_ in self.ids:
            if lane.id_ in self.lanes:
                warnings.warn(f"Lane {lane.id_} already exists! Replacing the lane with new data.")
            else:
                raise KeyError(f"The id of Lane {lane.id_} is used by the other road element.")

        self.lanes[lane.id_] = lane
        self.ids[lane.id_] = MapElement.LANE

    def add_area(self, area: Area):
        """This function adds an area to the map.

        Args:
            area (Area): The area to be added to the map.

        Raises:
            KeyError: If the id of the area is used by any other road element.
        """
        if area.id_ in self.ids:
            if area.id_ in self.areas:
                warnings.warn(f"Area {area.id_} already exists! Replacing the area with new data.")
            else:
                raise KeyError(f"The id of Area {area.id_} is used by the other road element.")

        self.areas[area.id_] = area
        self.ids[area.id_] = MapElement.AREA

    def add_regulatory(self, regulatory: Regulatory):
        """This function adds a traffic regulation to the map.

        Args:
            regulatory (Regulatory): The regulatory to be added to the map.

        Raises:
            KeyError: If the id of the regulatory is used by any other road element.
        """
        if regulatory.id_ in self.ids:
            if regulatory.id_ in self.regulations:
                warnings.warn(
                    f"Regulatory {regulatory.id_} already exists! Replacing the regulatory with new data."
                )
            else:
                raise KeyError(
                    f"The id of Regulatory {regulatory.id_} is used by the other road element."
                )

        self.regulations[regulatory.id_] = regulatory
        self.ids[regulatory.id_] = MapElement.REGULATORY

    def set_boundary(self, boundary: tuple):
        """This function sets the boundary of the map.

        Args:
            boundary (tuple): The boundary of the map expressed in the form of (min_x, max_x, min_y, max_y).
        """
        self._boundary = boundary

    def get_by_id(self, id_: str):
        """This function returns the road element with the given id.

        Args:
            id_ (str): The id of the road element.
        """
        if not id_ in self.ids:
            warnings.warn(f"Cannot find element with id {id_}.")
            return None

        if self.ids[id_] == MapElement.NODE:
            return self.nodes[id_]
        elif self.ids[id_] == MapElement.LANE:
            return self.lanes[id_]
        elif self.ids[id_] == MapElement.AREA:
            return self.areas[id_]
        elif self.ids[id_] == MapElement.ROADLINE:
            return self.roadlines[id_]
        elif self.ids[id_] == MapElement.REGULATORY:
            return self.regulations[id_]
        elif self.ids[id_] == MapElement.CUSTOM:
            return self.customs[id_]

    def reset(self):
        """This function resets the map by clearing all the road elements."""
        self.ids.clear()
        self.nodes.clear()
        self.lanes.clear()
        self.areas.clear()
        self.roadlines.clear()
        self.regulations.clear()
        self.customs.clear()
        self._boundary = None
