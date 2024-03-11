##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_xodr.py
# @Description: This file defines a class for parsing the OpenDRIVE map format.
# @Author: Yueyuan Li
# @Version: 1.0.0

import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from tactics2d.map.element import Area, Lane, Map, Node, Regulatory, RoadLine
from tactics2d.math.geometry import Circle
from tactics2d.math.interpolate import Spiral


class XODRParser:
    """This class implements a parser for the OpenDRIVE format map.

    !!! quote "Reference
        [ASAM OpenDRIVE BS 1.8.0 Specification, 2023-11-22](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html)

    The general structure of the OpenDRIVE map is as follows:
    ```xml
    / header
        // geoReference
    / road []
        // link
        // type
            /// speed
        // planView
            /// geometry []
        // elevationProfile
        // lateralProfile
        // lanes
            /// laneOffset
            /// laneSection
        // objects
        // signals
            /// signal []
        // surface
    / controller
        // control []
    / junction
        // connection []
    ```
    """

    def _get_line(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])

        x_end = x_start + length * np.cos(heading)
        y_end = y_start + length * np.sin(heading)

        return [(x_start, y_start), (x_end, y_end)]

    def _get_spiral(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])
        curv_start = float(xml_node.find("spiral").attrib["curvStart"])
        curv_end = float(xml_node.find("spiral").attrib["curvEnd"])

        n_interpolate = int(length / 0.1)
        s_interpolate = np.linspace(0, length, n_interpolate+1)
        if length < 0.1: # TODO: check the threshold/precision 0.1
            points = [(x_start, y_start)]
        else:
            gamma = (curv_end - curv_start) / length
            points = Spiral.get_spiral(s_interpolate, [x_start, y_start], heading, curv_start, gamma)
            points = [(x, y) for x, y in points]

        return points

    def _get_arc(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])
        curvature = float(xml_node.find("arc").attrib["curvature"])

        center, radius = Circle.get_circle_by_tangent_vector(
            [x_start, y_start], heading, 1 / curvature
        )
        n_interpolate = int(length / 0.1)
        points = [(x_start, y_start)]
        theta = np.arctan2(y_start - center[1], x_start - center[0])
        d_theta = length / radius

        for i in range(n_interpolate):
            x = center[0] + radius * np.cos(theta + d_theta * i) * np.sign(curvature)
            y = center[1] + radius * np.sin(theta + d_theta * i) * np.sign(curvature)
            points.append((x, y))

        return points

    def _get_poly3(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])
        a = float(xml_node.find("poly3").attrib["a"])
        b = float(xml_node.find("poly3").attrib["b"])
        c = float(xml_node.find("poly3").attrib["c"])
        d = float(xml_node.find("poly3").attrib["d"])
        
        n_interpolate = int(length / 0.1)
        u_interpolate = np.linspace(0, length, n_interpolate+1)
        v_interpolate = a * u_interpolate ** 3 + b * u_interpolate ** 2 + c * u_interpolate + d
        coords_uv = np.array([u_interpolate, v_interpolate]).T
        transform = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]) # TODO: check the rotation direction
        coords_xy = np.dot(coords_uv, transform.T) + np.array([x_start, y_start])
        points = [(x, y) for x, y in coords_xy]

        return points

    def _get_param_poly3(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])
        if xml_node.find("paramPoly3").attrib["pRange"]:
            p_range = float(xml_node.find("paramPoly3").attrib["pRange"])
        else:
            p_range = 1
        aU = float(xml_node.find("paramPoly3").attrib["aU"])
        bU = float(xml_node.find("paramPoly3").attrib["bU"])
        cU = float(xml_node.find("paramPoly3").attrib["cU"])
        dU = float(xml_node.find("paramPoly3").attrib["dU"])
        aV = float(xml_node.find("paramPoly3").attrib["aV"])
        bV = float(xml_node.find("paramPoly3").attrib["bV"])
        cV = float(xml_node.find("paramPoly3").attrib["cV"])
        dV = float(xml_node.find("paramPoly3").attrib["dV"])
        
        n_interpolate = int(length / 0.1)
        p_interpolate = np.linspace(0, p_range, n_interpolate+1)
        u_interpolate = aU * p_interpolate ** 3 + bU * p_interpolate ** 2 + cU * p_interpolate + dU
        v_interpolate = aV * p_interpolate ** 3 + bV * p_interpolate ** 2 + cV * p_interpolate + dV
        coords_uv = np.array([u_interpolate, v_interpolate]).T
        transform = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]) # TODO: check the rotation direction
        coords_xy = np.dot(coords_uv, transform.T) + np.array([x_start, y_start])
        points = [(x, y) for x, y in coords_xy]
        
        return points

    def _get_geometry(self, xml_node: ET.Element) -> list:
        geometry = []

        if xml_node.find("line"):
            geometry = self._get_line(xml_node)
        elif xml_node.find("spiral"):
            geometry = self._get_spiral(xml_node)
        elif xml_node.find("arc"):
            geometry = self._get_arc(xml_node)
        elif xml_node.find("poly3"):
            geometry = self._get_poly3(xml_node)
        elif xml_node.find("paramPoly3"):
            geometry = self._get_param_poly3(xml_node)

        return geometry

    def _load_lane(self, xml_node: ET.Element):
        return

    def _load_lane_section(self, xml_node: ET.Element):
        if xml_node is None:
            return

        lanes = []
        if xml_node.get("left"):
            lanes.append(self._load_lane(xml_node.get("left")))
        if xml_node.get("center"):
            lanes.append(self._load_lane(xml_node.get("center")))
        if xml_node.get("right"):
            lanes.append(self._load_lane(xml_node.get("right")))
        return

    def load_header(self, xml_node: ET.Element):
        """This function loads the header of the OpenDRIVE map.

        Args:
            xml_node (ET.Element): The XML node of the header.
        """
        header_info = {
            "revMajor": xml_node.get("revMajor"),
            "revMinor": xml_node.get("revMinor"),
            "name": xml_node.get("name"),
            "version": xml_node.get("version"),
            "date": xml_node.get("date"),
            "north": xml_node.get("north"),
            "south": xml_node.get("south"),
            "east": xml_node.get("east"),
            "west": xml_node.get("west"),
            "vendor": xml_node.get("vendor"),
        }

        project_node = xml_node.find("geoReference")
        projector = None
        if not project_node is None:
            project_rule = project_node.text
            projector = CRS.from_proj4(project_rule)

        return header_info, projector

    def load_road(self, xml_node: ET.Element):
        link_node = xml_node.find("link")

        # type
        type_node = xml_node.find("type")
        if type_node.find("speed"):
            speed_limit = type_node.find("speed").attrib["max"]
            speed_unit = type_node.find("speed").attrib["unit"]
            print(speed_limit, speed_unit)

        # plan view
        points = []
        for geometry_node in xml_node.find("planView").findall("geometry"):
            new_points = self._get_geometry(geometry_node)
            points.extend(new_points)

        # elevation profile
        elevation_profile_node = xml_node.find("elevationProfile")

        # lateral profile
        lateral_profile_node = xml_node.find("lateralProfile")

        # lanes
        lanes_node = xml_node.find("lanes")
        lane_offset_node = lanes_node.find("laneOffset")
        self._load_lane_section(lanes_node.find("laneSection"))

        if lanes_node is None:
            raise ValueError("Road must have lanes element.")

        objects_node = xml_node.find("objects")

        return

    def load_junction(self, xml_node: ET.Element):
        return

    def parse(self, xml_root: ET.Element):
        header_node = xml_root.find("header")
        if header_node is not None:
            header_info, projector = self.load_header(header_node)

        map_ = Map(header_info["name"] if header_info["name"] != "" else None)
        if not projector is None:
            to_project = True

        for road in xml_root.findall("road"):
            self.load_road(road)

        # for junction in xml_root.findall("junction"):
        #     self.load_junction(junction)
