##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_osm.py
# @Description: This file defines a parser for lanelet2 format map.
# @Author: Yueyuan Li
# @Version: 1.0.0


import logging
import xml.etree.ElementTree as ET
from typing import Tuple

from pyproj import Proj
from shapely.geometry import LineString, Polygon

from tactics2d.map.element import Area, Lane, Map, Node, Regulatory, RoadLine


class OSMParser:
    """This class implements a parser for the original OpenStreetMap format map.

    The parser is tested with data generated by [the official website](https://www.openstreetmap.org/) of OpenStreetMap and the software application [JOSM](https://josm.openstreetmap.de/).
    """

    def __init__(self, lanelet2: bool = False):
        """Initialize the parser.

        Args:
            lanelet2 (bool, optional): Whether the `.osm` file is annotated in Lanelet2 format. Defaults to False.
        """
        self.lanelet2 = lanelet2

    def _append_point_list(self, point_list, new_points, component_id):
        if point_list[-1] == new_points[0]:
            pass
        elif point_list[0] == new_points[0]:
            point_list.reverse()
        elif point_list[0] == new_points[-1]:
            point_list.reverse()
            new_points.reverse()
        elif point_list[-1] == new_points[-1]:
            new_points.reverse()
        else:
            raise SyntaxError(f"Points on the side of relation {component_id} is not continuous.")

        point_list += new_points[1:]

    def _get_tags(self, xml_node: ET.Element) -> dict:
        tags = dict()
        bool_tags = {"area", "oneway"}

        for tag in xml_node.findall("tag"):
            if tag.attrib["k"] in bool_tags:
                tags[tag.attrib["k"]] = tag.attrib["v"] == "yes"
            else:
                tags[tag.attrib["k"]] = tag.attrib["v"]

        return tags

    def _get_lanelet2_tags(self, xml_node: ET.Element) -> dict:
        tags = dict()
        tags["custom_tags"] = dict()

        directly_save = {
            "subtype",
            "color",
            "width",
            "location",
            "inferred_participants",
            "speed_limit",
        }
        bool_tags = {"temporary", "speed_limit_mandatory", "dynamic", "fallback", "oneway"}

        for tag in xml_node.findall("tag"):
            if tag.attrib["k"] == "type":
                tags["type_"] = tag.attrib["v"]

            elif tag.attrib["k"] in directly_save:
                tags[tag.attrib["k"]] = tag.attrib["v"]

            elif tag.attrib["k"] in bool_tags:
                tags[tag.attrib["k"]] = tag.attrib["v"] == "yes"

            elif "lane_change" in tag.attrib["k"]:
                if "lane_change" in tags:
                    raise SyntaxError("Conflict tags on lane changing property.")
                else:
                    if tag.attrib["k"] == "lane_change":
                        tags["lane_change"] = (
                            (True, True) if tag.attrib["v"] == "yes" else (False, False)
                        )
                    elif tag.attrib["k"] == "lane_change:left":
                        tags["lane_change"] = (
                            (True, False) if tag.attrib["v"] == "yes" else (False, False)
                        )
                    elif tag.attrib["k"] == "lane_change:right":
                        tags["lane_change"] = (
                            (False, True) if tag.attrib["v"] == "yes" else (False, False)
                        )

            else:
                tags["custom_tags"][tag.attrib["k"]] = tag.attrib["v"]

        return tags

    def _load_area(self, xml_node: ET.Element, map_: Map) -> Area:
        area_id = int(xml_node.attrib["id"])
        line_ids = dict(inner=[], outer=[])
        regulatory_ids = []

        for member in xml_node.findall("member"):
            member_id = int(member.attrib["ref"])
            if member.attrib["role"] == "outer":
                line_ids["outer"].append(member_id)
            elif member.attrib["role"] == "inner":
                line_ids["inner"].append(member_id)
            elif member.attrib["role"] == "regulatory_element":
                regulatory_ids.append(member_id)

        outer_point_list = []
        for line_id in line_ids["outer"]:
            if len(outer_point_list) == 0:
                if map_.roadlines.get(line_id):
                    outer_point_list = list(map_.roadlines[line_id].geometry.coords)
            else:
                if map_.roadlines.get(line_id):
                    new_points = list(map_.roadlines[line_id].geometry.coords)
                    try:
                        self._append_point_list(outer_point_list, new_points, area_id)
                    except SyntaxError as err:
                        logging.error(err)
                        return None

        if len(outer_point_list) == 0:
            return None

        if outer_point_list[0] != outer_point_list[-1]:
            logging.warning(f"The outer boundary of area {area_id} is not closed.")

        inner_point_list = [[]]
        inner_idx = 0
        for line_id in line_ids["inner"]:
            if len(inner_point_list[inner_idx]) == 0:
                if map_.roadlines.get(line_id):
                    inner_point_list[inner_idx] = list(map_.roadlines[line_id].geometry.coords)
            else:
                if map_.roadlines.get(line_id):
                    new_points = list(map_.roadlines[line_id].geometry.coords)
                    try:
                        self._append_point_list(inner_point_list[inner_idx], new_points, area_id)
                    except SyntaxError as err:
                        logging.error(err)
                        return None

            if inner_point_list[inner_idx][0] == inner_point_list[inner_idx][-1]:
                inner_point_list.append([])
                inner_idx += 1
        if len(inner_point_list[-1]) == 0:
            del inner_point_list[-1]
        elif inner_point_list[-1][0] != inner_point_list[-1][-1]:
            logging.warning(f"The inner boundary of area {area_id} is not closed.")
        polygon = Polygon(outer_point_list, inner_point_list)

        area_tags = self._get_lanelet2_tags(xml_node)

        return Area(area_id, polygon, line_ids, set(regulatory_ids), **area_tags)

    def _load_bounds_no_proj(self, xml_node: ET.Element) -> tuple:
        """This function loads the boundary of the map from the XML node. The coordinates will not be projected.

        Args:
            xml_node (ET.Element): The XML node of the boundary.

        Returns:
            The boundary of the map expressed as (min_lon, max_lon, min_lat, max_lat).
        """
        min_lon = float(xml_node.get("minlon"))
        max_lon = float(xml_node.get("maxlon"))
        min_lat = float(xml_node.get("minlat"))
        max_lat = float(xml_node.get("maxlat"))

        if not None in [min_lon, max_lon, min_lat, max_lat]:
            return (min_lon, max_lon, min_lat, max_lat)

        return None

    def _load_bounds(self, xml_node: ET.Element, projector: Proj, origin: tuple) -> tuple:
        """This function loads the boundary of the map from the XML node. The coordinates will be projected.

        Args:
            xml_node (ET.Element): The XML node of the boundary.
            projector (Proj): The projection rule of the map.
            origin (tuple): The origin of the GPS coordinates.

        Returns:
            The boundary of the map expressed as (min_x, max_x, min_y, max_y).
        """
        min_lon = float(xml_node.get("minlon"))
        max_lon = float(xml_node.get("maxlon"))
        min_lat = float(xml_node.get("minlat"))
        max_lat = float(xml_node.get("maxlat"))

        if not None in [min_lat, max_lat, min_lon, max_lon]:
            min_x, min_y = projector(min_lon, min_lat)
            max_x, max_y = projector(max_lon, max_lat)
            return (min_x - origin[0], max_x - origin[0], min_y - origin[1], max_y - origin[1])

        return None

    def _load_nodes_no_proj(self, xml_node: ET.Element) -> Node:
        """This function loads the nodes from the XML node. The coordinates will not be projected.

        Args:
            xml_node (ET.Element): The XML node of the nodes.

        Returns:
            A node under the GPS coordinates.
        """
        node_id = int(xml_node.attrib["id"])
        lon = float(xml_node.attrib["lon"])
        lat = float(xml_node.attrib["lat"])

        return Node(id_=node_id, x=lon, y=lat)

    def _load_nodes(self, xml_node: ET.Element, projector: Proj, origin: tuple) -> Node:
        """This function loads the nodes from the XML node. The coordinates will be projected.

        Args:
            xml_node (ET.Element): The XML node of the nodes.
            projector (Proj): The projection rule of the map.
            origin (tuple): The origin of the GPS coordinates.

        Returns:
            A node under the x-y coordinates.
        """
        node_id = int(xml_node.attrib["id"])
        x, y = projector(xml_node.attrib["lon"], xml_node.attrib["lat"])

        return Node(id_=node_id, x=x - origin[0], y=y - origin[1])

    def _load_way(self, xml_node: ET.Element, map_: Map) -> Tuple[Area, RoadLine]:
        """This function loads an OSM road elements from the XML node.

        Args:
            xml_node (ET.Element): The XML node of the road element.
            map_ (Map): The map that the road element belongs to.

        Returns:
            A road element.
        """
        id_ = int(xml_node.attrib["id"])
        point_list = []
        point_ids = []

        for node in xml_node.findall("nd"):
            node_id = int(node.attrib["ref"])
            point_list.append(map_.nodes[node_id].location)
            point_ids.append(node_id)

        tags = self._get_tags(xml_node)
        is_area = tags.pop("area", False)

        if is_area or point_ids[0] == point_ids[-1]:
            road_element = Area(id_, Polygon(point_list), custom_tags=tags)
        else:
            road_element = RoadLine(id_, LineString(point_list), custom_tags=tags)

        return road_element

    def _load_relation(self, xml_node: ET.Element, map_: Map) -> Tuple[Area, RoadLine, Regulatory]:
        """This function loads an OSM road elements from the XML node.

        Args:
            xml_node (ET.Element): The XML node of the road element.
            map_ (Map): The map that the road element belongs to.

        Returns:
            A road element.
        """
        id_ = int(xml_node.attrib["id"])
        tags = self._get_tags(xml_node)
        type_ = tags.pop("type")
        road_element = None

        if type_ == "multipolygon":
            road_element = self._load_area(xml_node, map_)

        elif type_ == "route":
            point_list = []
            line_ids = []
            for member in xml_node.findall("member"):
                if member.attrib["type"] == "way":
                    line_ids.append(int(member.attrib["ref"]))
            for line_id in line_ids:
                if len(point_list) == 0:
                    if map_.roadlines.get(line_id):
                        point_list = list(map_.roadlines[line_id].geometry.coords)
                if map_.roadlines.get(line_id):
                    new_points = list(map_.roadlines[line_id].geometry.coords)
                    try:
                        self._append_point_list(point_list, new_points, id_)
                    except SyntaxError as err:
                        logging.error(err)
                        return None
            road_element = RoadLine(id_, LineString(point_list), type_="route", custom_tags=tags)

        elif type_ == "restriction":
            subtype = tags.pop("restriction")
            froms = dict()
            tos = dict()
            vias = dict()
            for member in xml_node.findall("member"):
                member_id = int(member.attrib["ref"])
                if member.attrib["role"] == "from":
                    froms[member_id] = member.attrib["type"]
                elif member.attrib["role"] == "to":
                    tos[member_id] = member.attrib["type"]
                elif member.attrib["role"] == "via":
                    vias[member_id] = member.attrib["type"]

            tags["froms"] = froms
            tags["tos"] = tos
            tags["vias"] = vias
            road_element = Regulatory(id_, type_="restriction", subtype=subtype, custom_tags=tags)

        return road_element

    def _load_roadline_lanelet2(self, xml_node: ET.Element, map_: Map) -> RoadLine:
        """This function loads a Lanelet 2 roadline from the XML node.

        Args:
            xml_node (ET.Element): The XML node of the roadline.
            map_ (Map): The map that the roadline belongs to.

        Returns:
            A roadline labeled with Lanelet 2 tags.
        """
        line_id = int(xml_node.attrib["id"])
        point_list = []

        for node in xml_node.findall("nd"):
            point_list.append(map_.nodes[int(node.attrib["ref"])].location)
        linestring = LineString(point_list)

        tags = self._get_lanelet2_tags(xml_node)

        return RoadLine(id_=line_id, geometry=linestring, **tags)

    def _load_lane_lanelet2(self, xml_node: ET.Element, map_: Map) -> Lane:
        lane_id = int(xml_node.attrib["id"])
        line_ids = dict(left=[], right=[])
        regulatory_ids = []

        for member in xml_node.findall("member"):
            member_id = int(member.attrib["ref"])
            if member.attrib["role"] == "left":
                line_ids["left"].append(member_id)
            elif member.attrib["role"] == "right":
                line_ids["right"].append(member_id)
            elif member.attrib["role"] == "regulatory_element":
                regulatory_ids.append(member_id)

        point_list = dict()
        for side in ["left", "right"]:
            point_list[side] = list(map_.roadlines[line_ids[side][0]].geometry.coords)
            for line_id in line_ids[side][1:]:
                new_nodes = list(map_.roadlines[line_id].geometry.coords)
                self._append_point_list(point_list[side], new_nodes, lane_id)

        left_side = LineString(point_list["left"])
        right_side = LineString(point_list["right"])

        left_parallel = left_side.parallel_offset(0.1, "left")
        if left_side.hausdorff_distance(right_side) > left_parallel.hausdorff_distance(right_side):
            point_list["left"].reverse()
            left_side = LineString(point_list["left"])
        right_parallel = right_side.parallel_offset(0.1, "right")
        if right_side.hausdorff_distance(left_side) > right_parallel.hausdorff_distance(left_side):
            point_list["right"].reverse()
            right_side = LineString(point_list["right"])

        if left_side.crosses(right_side):
            logging.warning(f"The sides of lane {lane_id} is intersected.")

        lane_tags = self._get_lanelet2_tags(xml_node)

        return Lane(lane_id, left_side, right_side, line_ids, set(regulatory_ids), **lane_tags)

    def _load_area_lanelet2(self, xml_node: ET.Element, map_: Map) -> Area:
        """This function loads a Lanelet 2 area from the XML node.

        Args:
            xml_node (ET.Element): The XML node of the area.
            map_ (Map): The map that the area belongs to.

        Returns:
            An area labeled with Lanelet 2 tags.
        """
        area_id = int(xml_node.attrib["id"])
        line_ids = dict(inner=[], outer=[])
        regulatory_ids = []

        for member in xml_node.findall("member"):
            member_id = int(member.attrib["ref"])
            if member.attrib["role"] == "outer":
                line_ids["outer"].append(member_id)
            elif member.attrib["role"] == "inner":
                line_ids["inner"].append(member_id)
            elif member.attrib["role"] == "regulatory_element":
                regulatory_ids.append(member_id)

        outer_point_list = list(map_.roadlines[line_ids["outer"][0]].geometry.coords)
        for line_id in line_ids["outer"][1:]:
            new_points = list(map_.roadlines[line_id].geometry.coords)
            self._append_point_list(outer_point_list, new_points, area_id)
        if outer_point_list[0] != outer_point_list[-1]:
            logging.warning(f"The outer boundary of area {area_id} is not closed.")

        inner_point_list = [[]]
        inner_idx = 0
        for line_id in line_ids["inner"]:
            if len(inner_point_list[inner_idx]) == 0:
                inner_point_list[inner_idx] = list(map_.roadlines[line_id].geometry.coords)
            else:
                new_points = list(map_.roadlines[line_id].geometry.coords)
                self._append_point_list(inner_point_list[inner_idx], new_points, area_id)
            if inner_point_list[inner_idx][0] == inner_point_list[inner_idx][-1]:
                inner_point_list.append([])
                inner_idx += 1
        if len(inner_point_list[-1]) == 0:
            del inner_point_list[-1]
        elif inner_point_list[-1][0] != inner_point_list[-1][-1]:
            logging.warning(f"The inner boundary of area {area_id} is not closed.")
        polygon = Polygon(outer_point_list, inner_point_list)

        area_tags = self._get_lanelet2_tags(xml_node)

        return Area(area_id, polygon, line_ids, set(regulatory_ids), **area_tags)

    def _load_regulatory_lanelet2(self, xml_node: ET.Element) -> Regulatory:
        regulatory_id = int(xml_node.attrib["id"])
        relations = dict()
        ways = dict()
        for member in xml_node.findall("member"):
            if member.attrib["type"] == "relation":
                relations[int(member.attrib["ref"])] = member.attrib["role"]
            elif member.attrib["type"] == "way":
                ways[int(member.attrib["ref"])] = member.attrib["role"]

        regulatory_tags = self._get_lanelet2_tags(xml_node)
        return Regulatory(regulatory_id, relations, ways, **regulatory_tags)

    def parse(self, file_path: str, configs: dict = None) -> Map:
        """This function parses the OpenStreetMap format map.

        Args:
            file_path (str): The absolute path of the `.osm` file.
            configs (dict): The configurations of the map.
        """
        xml_root = ET.parse(file_path).getroot()

        if configs is not None:
            project_rule = configs.get("project_rule", None)
            gps_origin = configs.get("gps_origin", None)
        else:
            project_rule = None
            gps_origin = None

        projector = Proj(**project_rule) if project_rule else None

        to_project = not None in [projector, gps_origin]

        if to_project:
            origin = projector(*gps_origin)

        if configs is None:
            map_ = Map()
        else:
            map_ = Map(
                name=configs.get("name"),
                scenario_type=configs.get("scenario_type"),
                country=configs.get("country"),
            )

        node_boundary = xml_root.find("bounds")
        if node_boundary is not None:
            map_.set_boundary(
                self._load_bounds(node_boundary, projector, origin)
                if to_project
                else self._load_bounds_no_proj(node_boundary)
            )

        if to_project:
            for xml_node in xml_root.findall("node"):
                if xml_node.get("action") == "delete":
                    continue
                map_.add_node(self._load_nodes(xml_node, projector, origin))
        else:
            for xml_node in xml_root.findall("node"):
                if xml_node.get("action") == "delete":
                    continue
                map_.add_node(self._load_nodes_no_proj(xml_node))

        if self.lanelet2:
            for xml_node in xml_root.findall("way"):
                if xml_node.get("action") == "delete":
                    continue
                map_.add_roadline(self._load_roadline_lanelet2(xml_node, map_))

            for xml_node in xml_root.findall("relation"):
                if xml_node.get("action") == "delete":
                    continue
                for tag in xml_node.findall("tag"):
                    if tag.attrib["v"] == "lanelet":
                        map_.add_lane(self._load_lane_lanelet2(xml_node, map_))
                    elif tag.attrib["v"] in ["multipolygon", "area"]:
                        map_.add_area(self._load_area_lanelet2(xml_node, map_))

            for xml_node in xml_root.findall("relation"):
                if xml_node.get("action") == "delete":
                    continue
                if self.lanelet2:
                    for tag in xml_node.findall("tag"):
                        if tag.attrib["v"] == "regulatory_element":
                            map_.add_regulatory(self._load_regulatory_lanelet2(xml_node))

        else:
            for xml_node in xml_root.findall("way"):
                if xml_node.get("action") == "delete":
                    continue
                road_element = self._load_way(xml_node, map_)

                if isinstance(road_element, RoadLine):
                    map_.add_roadline(road_element)
                elif isinstance(road_element, Area):
                    map_.add_area(road_element)

            for xml_node in xml_root.findall("relation"):
                if xml_node.get("action") == "delete":
                    continue
                road_element = self._load_relation(xml_node, map_)

                if isinstance(road_element, RoadLine):
                    map_.add_roadline(road_element)
                elif isinstance(road_element, Area):
                    map_.add_area(road_element)
                elif isinstance(road_element, Regulatory):
                    map_.add_regulatory(road_element)

        return map_
