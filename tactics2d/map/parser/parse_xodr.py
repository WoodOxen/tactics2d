##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_xodr.py
# @Description: This file defines a class for parsing the OpenDRIVE map format.
# @Author: Tactics2D Team
# @Version: 1.0.0


import logging
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
from pyproj import CRS
from shapely.affinity import affine_transform, rotate
from shapely.geometry import LineString, Point, Polygon

from tactics2d.geometry import Circle
from tactics2d.interpolator import Spiral
from tactics2d.map.element import Area, Connection, Junction, Lane, Map, Node, Regulatory, RoadLine


class XODRParser:
    """This class implements a parser for the OpenDRIVE format map.

    !!! quote "Reference"
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

    _roadmark_dict = {
        "botts dots": "",
        "broken broken": "dashed_dashed",
        "broken solid": "dashed_solid",
        "broken": "dashed",
        "solid broken": "solid_dashed",
        "solid solid": "solid_solid",
        "solid": "solid",
    }

    def __init__(self):
        self.id_counter = 0

    def get_headings(self, points):
        diff = np.diff(np.array(points), axis=0)
        headings = np.arctan2(diff[:, 1], diff[:, 0]).tolist()
        headings.append(headings[-1])
        return headings

    def _get_line(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])

        points = [(x_start, y_start)]

        if length > 0.1:
            n_interpolate = int(length / 0.1)

            for i in np.linspace(0.1, length, n_interpolate - 1):
                x_end = x_start + i * np.cos(heading)
                y_end = y_start + i * np.sin(heading)
                points.append((x_end, y_end))

        points.append((x_start + length * np.cos(heading), y_start + length * np.sin(heading)))
        return points

    def _get_spiral(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])
        curv_start = float(xml_node.find("spiral").attrib["curvStart"])
        curv_end = float(xml_node.find("spiral").attrib["curvEnd"])

        if length < 0.1:  # TODO: check the threshold/precision 0.1
            points = [(x_start, y_start)]
        else:
            gamma = (curv_end - curv_start) / length
            points = Spiral.get_curve(length, [x_start, y_start], heading, curv_start, gamma)
            points = [(x, y) for x, y in points]
        return points

    def _get_arc(self, xml_node: ET.Element) -> list:
        x_start = float(xml_node.attrib["x"])
        y_start = float(xml_node.attrib["y"])
        heading = float(xml_node.attrib["hdg"])
        length = float(xml_node.attrib["length"])
        curvature = float(xml_node.find("arc").attrib["curvature"])

        center, radius = Circle.get_circle(
            tangent_point=[x_start, y_start],
            tangent_heading=heading,
            radius=abs(1 / curvature),
            side="L" if curvature > 0 else "R",
        )

        n_interpolate = int(length / 0.1)
        points = []

        arc_angle = length / radius * np.sign(curvature)
        start_heading = heading - np.pi / 2 * np.sign(curvature)

        for i in np.linspace(start_heading, start_heading + arc_angle, n_interpolate):
            x = center[0] + radius * np.cos(i)
            y = center[1] + radius * np.sin(i)
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
        u_interpolate = np.linspace(0, length, n_interpolate + 1)
        v_interpolate = a * u_interpolate**3 + b * u_interpolate**2 + c * u_interpolate + d
        coords_uv = np.array([u_interpolate, v_interpolate]).T
        transform = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )  # TODO: check the rotation direction
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
        p_interpolate = np.linspace(0, p_range, n_interpolate + 1)
        u_interpolate = aU * p_interpolate**3 + bU * p_interpolate**2 + cU * p_interpolate + dU
        v_interpolate = aV * p_interpolate**3 + bV * p_interpolate**2 + cV * p_interpolate + dV
        coords_uv = np.array([u_interpolate, v_interpolate]).T
        transform = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )  # TODO: check the rotation direction
        coords_xy = np.dot(coords_uv, transform.T) + np.array([x_start, y_start])
        points = [(x, y) for x, y in coords_xy]

        return points

    def _get_geometry(self, xml_node: ET.Element) -> list:
        """
        Road Reference line
        """
        geometry = []

        if not xml_node.find("line") is None:
            geometry = self._get_line(xml_node)
        elif not xml_node.find("spiral") is None:
            geometry = self._get_spiral(xml_node)
        elif not xml_node.find("arc") is None:
            geometry = self._get_arc(xml_node)
            # geometry = []
        elif not xml_node.find("poly3") is None:
            geometry = self._get_poly3(xml_node)
        elif not xml_node.find("paramPoly3") is None:
            geometry = self._get_param_poly3(xml_node)

        if len(geometry) >= 2:
            if geometry[-2] == geometry[-1]:
                geometry.pop(-1)
        return geometry

    def _check_continuity(self, new_points, points):
        if len(points) == 0:
            return True

        if len(new_points) == 0:
            return True

        return np.linalg.norm(np.array(new_points[0]) - np.array(points[-1])) < 0.1

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

    def load_roadmark(self, points: Union[list, LineString], xml_node: ET.Element):
        type_ = "virtual"
        subtype = None

        # TODO: handel "botts dots",
        if xml_node is None or xml_node.attrib["type"] == "none":
            pass
        elif xml_node.attrib["type"] == "curb":
            type_ = "curbstone"
        elif xml_node.attrib["type"] == "edge":
            type_ = "road_border"
        elif xml_node.attrib["type"] == "grass":
            type_ = "grass"
        else:
            type_ = "line_thin" if float(xml_node.attrib["width"]) <= 0.15 else "line_thick"
            subtype = self._roadmark_dict[xml_node.attrib["type"]]

        if xml_node is None or not hasattr(xml_node, "color"):
            color = None
        elif xml_node.attrib["color"] == "standard":
            color = "white"
        elif xml_node.attrib["color"] == "violet":
            color = "purple"
        else:
            color = xml_node.attrib["color"]

        lane_change = (False, False)
        if xml_node is None:
            lane_change = (True, True)
        elif xml_node.attrib["laneChange"] == "none":
            lane_change = None
        elif xml_node.attrib["laneChange"] == "both":
            lane_change = (True, True)
        elif xml_node.attrib["laneChange"] == "decrease":
            lane_change = (False, True)
        elif xml_node.attrib["laneChange"] == "increase":
            lane_change = (True, False)

        custom_tags = {
            "weight": (
                None
                if xml_node is None or not hasattr(xml_node, "weight")
                else xml_node.attrib["weight"]
            )
        }

        roadline = RoadLine(
            id_=self.id_counter,
            geometry=LineString(points),
            type_=type_,
            subtype=subtype,
            color=color,
            width=(
                None
                if xml_node is None or not hasattr(xml_node, "width")
                else xml_node.attrib["width"]
            ),
            height=(
                None
                if xml_node is None or not hasattr(xml_node, "height")
                else xml_node.attrib["height"]
            ),
            lane_change=lane_change,
            temporary=False,
            custom_tags=custom_tags,
        )
        self.id_counter += 1

        return roadline

    def load_lane(self, ref_line, xml_node: ET.Element, type_node: ET.Element):
        # ref_value should always be a positive value
        sign = np.sign(int(xml_node.attrib["id"]))

        if sign == 0:
            raise ValueError("Lane id of left/right lanes should not be 0.")

        location = type_node.attrib["type"] if hasattr(type_node, "type") else None

        default_speed_limit = (
            None
            if type_node.get("speed") is None or "max" not in type_node.attrib
            else float(type_node.attrib["max"])
        )
        default_speed_limit_unit = (
            "km/h"
            if type_node.get("speed") is None or "unit" not in type_node.attrib
            else type_node.attrib["unit"]
        )

        if sign > 0:
            right_side = ref_line.geometry
            left_side = ref_line.geometry.offset_curve(float(xml_node.find("width").attrib["a"]))
            new_ref_line = left_side
            line_ids = {"left": [self.id_counter], "right": [ref_line.id_]}
        else:
            left_side = ref_line.geometry
            right_side = ref_line.geometry.offset_curve(-float(xml_node.find("width").attrib["a"]))
            new_ref_line = right_side
            line_ids = {"left": [ref_line.id_], "right": [self.id_counter]}

        roadline = self.load_roadmark(new_ref_line, xml_node.find("roadMark"))

        lane = Lane(
            id_=self.id_counter,
            left_side=left_side,
            right_side=right_side,
            subtype=xml_node.attrib["type"],
            line_ids=line_ids,
            speed_limit=(
                default_speed_limit
                if xml_node.find("speed") is None or "max" not in xml_node.find("speed").attrib
                else float(xml_node.find("speed").attrib["max"])
            ),
            speed_limit_unit=(
                default_speed_limit_unit
                if xml_node.find("speed") is None or "unit" not in xml_node.find("speed").attrib
                else xml_node.find("speed").attrib["unit"]
            ),
            location=location,
        )
        self.id_counter += 1

        return lane, roadline

    def load_object(self, points, s_points, headings, xml_node: ET.Element):
        s = float(xml_node.attrib["s"])
        t = float(xml_node.attrib["t"])

        # find the closest point on the reference line
        idx = np.argmin(np.abs(s_points - s))
        ref_heading = headings[idx]
        x, y = points[idx]

        zOffset = float(xml_node.attrib["zOffset"])
        relative_heading = float(xml_node.attrib["hdg"]) if "hdg" in xml_node.attrib else 0

        width = float(xml_node.attrib["width"]) if "width" in xml_node.attrib else None
        length = float(xml_node.attrib["length"]) if "length" in xml_node.attrib else None
        height = float(xml_node.attrib["height"]) if "height" in xml_node.attrib else None
        radius = float(xml_node.attrib["radius"]) if "radius" in xml_node.attrib else None

        # convert local coordinate to global coordinate
        heading = ref_heading
        x_origin = x - t * np.sin(heading)
        y_origin = y + t * np.cos(heading)
        # object_heading = heading + relative_heading * orientation

        if not None in [width, length]:
            shape = Polygon(
                [
                    [0.5 * width, -0.5 * length],
                    [0.5 * width, 0.5 * length],
                    [-0.5 * width, 0.5 * length],
                    [-0.5 * width, -0.5 * length],
                ]
            )

        elif not radius is None:
            shape = np.array(
                [
                    (np.cos(theta) * radius, np.sin(theta) * radius)
                    for theta in np.linspace(0, 2 * np.pi, 100)
                ]
            )
        else:
            shape = Point(s, t)

        shape = rotate(shape, (relative_heading / np.pi) * 180)
        shape = affine_transform(
            shape,
            [
                np.cos(heading - np.pi / 2),
                -np.sin(heading - np.pi / 2),
                np.sin(heading - np.pi / 2),
                np.cos(heading - np.pi / 2),
                x_origin,
                y_origin,
            ],
        )

        # TODO: handle traffic participants and road areas separately
        area = Area(
            id_=self.id_counter,
            geometry=shape,
            subtype=xml_node.attrib["type"] if "type" in xml_node.attrib else None,
        )
        self.id_counter += 1

        return area

    def load_road(self, xml_node: ET.Element):
        objects = []
        lanes = []
        roadlines = []

        # refline points
        points = []

        link_node = xml_node.find("link")

        # type
        type_node = xml_node.find("type")

        # plan view
        first_geometry = xml_node.find("planView").find("geometry")
        x = float(first_geometry.attrib["x"])
        y = float(first_geometry.attrib["y"])
        heading = float(first_geometry.attrib["hdg"])
        for geometry_node in xml_node.find("planView").findall("geometry"):
            new_points = self._get_geometry(geometry_node)
            if self._check_continuity(new_points, points):
                points.extend(new_points)
            else:
                logging.warning("The geometry is not continuous.")
                points.extend(new_points)

        s_points = np.sqrt(
            np.sum((np.array(points) - np.array([points[0]] + points)[:-1, :]) ** 2, axis=1)
        )
        # I suggest you to put forward an issue to numpy to change this function name
        s_points = np.cumsum(s_points)

        # elevation profile
        elevation_profile_node = xml_node.find("elevationProfile")

        # lateral profile
        lateral_profile_node = xml_node.find("lateralProfile")

        # lanes
        lanes_node = xml_node.find("lanes")
        lane_sections_offset = [
            float(lane_section.attrib["s"]) for lane_section in lanes_node.findall("laneSection")
        ] + [s_points[-1]]
        for ls_idx, lane_section_node in enumerate(lanes_node.findall("laneSection")):
            # Load center line for lanesection
            ls_start_offset, ls_end_offset = (
                lane_sections_offset[ls_idx],
                lane_sections_offset[ls_idx + 1],
            )
            center_points = np.array(points)[
                (s_points >= ls_start_offset - 0.1) & (s_points <= ls_end_offset + 0.1)
            ].tolist()
            if len(center_points) == 1:
                center_points.append(center_points[0])

            center_line = RoadLine(id_=self.id_counter, geometry=LineString(center_points))
            self.id_counter += 1

            # Load road marks for lanesection
            road_marks = lane_section_node.find("center").find("lane").findall("roadMark")
            road_mark_offsets = [float(road_mark.attrib["sOffset"]) for road_mark in road_marks] + [
                s_points[-1]
            ]
            for road_mark_idx, road_mark in enumerate(road_marks):
                s_offset, e_offset = (
                    road_mark_offsets[road_mark_idx],
                    road_mark_offsets[road_mark_idx + 1],
                )
                part_refline = np.array(points)[
                    (s_points >= s_offset - 0.1) & (s_points <= e_offset + 0.1)
                ].tolist()

                if len(part_refline) == 1:
                    part_refline.append(part_refline[0])

                part_refline = self.load_roadmark(part_refline, road_mark)  # RoadLine
                if part_refline is None:
                    raise ValueError("Center line must be defined.")
                roadlines.append(part_refline)

            if not lane_section_node.find("left") is None:
                lane_nodes = sorted(
                    lane_section_node.find("left").findall("lane"), key=lambda x: x.attrib["id"]
                )

                ref_line = center_line
                for lane_node in lane_nodes:
                    if type_node is None:
                        continue
                    lane, ref_line = self.load_lane(ref_line, lane_node, type_node)
                    lanes.append(lane)
                    roadlines.append(ref_line)

            if not lane_section_node.find("right") is None:
                lane_nodes = sorted(
                    lane_section_node.find("right").findall("lane"), key=lambda x: x.attrib["id"]
                )
                ref_line = center_line
                for lane_node in lane_nodes:
                    if type_node is None:
                        continue
                    lane, ref_line = self.load_lane(ref_line, lane_node, type_node)
                    lanes.append(lane)
                    roadlines.append(ref_line)

        if lanes_node is None:
            raise ValueError("Road must have lanes element.")

        objects_node = xml_node.find("objects")
        if not objects_node is None:
            headings = self.get_headings(points)
            for object_node in objects_node.findall("object"):
                area = self.load_object(points, s_points, headings, object_node)
                objects.append(area)

        return lanes, roadlines, objects

    def load_junction(self, xml_node: ET.Element):
        junction = Junction(id_=self.id_counter)
        self.id_counter += 1

        for connection_node in xml_node.findall("connection"):
            connection = Connection(
                id_=self.id_counter,
                incoming_road=connection_node.attrib["incomingRoad"],
                connecting_road=connection_node.attrib["connectingRoad"],
                contact_point=connection_node.attrib["contactPoint"],
            )
            self.id_counter += 1

            for lane_link in xml_node.findall("laneLink"):
                connection.add_lane_link((lane_link.attrib["from"], lane_link.attrib["to"]))

            junction.add_connection(connection)

        return junction

    def parse(self, file_path: str):
        """This function parses the OpenDRIVE format map. To ensure that all road elements have an unique id, the function automatically reassign the id of the road elements.

        Args:
            file_path (str): The absolute path of the `.xodr` file.

        Returns:
            map_ (Map): The parsed map.
        """
        xml_root = ET.parse(file_path).getroot()
        header_node = xml_root.find("header")
        if header_node is not None:
            header_info, projector = self.load_header(header_node)

        map_ = Map(header_info["name"] if header_info["name"] != "" else None)
        if not projector is None:
            to_project = True

        for road in xml_root.findall("road"):
            lanes, roadlines, objects = self.load_road(road)
            for lane in lanes:
                map_.add_lane(lane)
            for roadline in roadlines:
                map_.add_roadline(roadline)
            for obj in objects:
                map_.add_area(obj)

        for junction in xml_root.findall("junction"):
            map_.add_junction(self.load_junction(junction))

        # reset the id counter
        self.id_counter = 0
        return map_
