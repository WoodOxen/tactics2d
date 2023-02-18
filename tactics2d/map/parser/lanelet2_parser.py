import xml.etree.ElementTree as ET
import warnings

from pyproj import Proj
from shapely.geometry import Point, LineString, Polygon

from tactics2d.map.element.node import Node
from tactics2d.map.element.lane import Lane
from tactics2d.map.element.area import Area
from tactics2d.map.element.roadline import RoadLine
from tactics2d.map.element.regulatory import RegulatoryElement
from tactics2d.map.element.map import Map


LANE_CHANGE_MAPPING = {
    "line_thin": {
        "solid": (False, False),
        "solid_solid": (False, False),
        "dashed": (True, True),
        "dashed_solid": (True, False), # left->right: yes
        "solid_dashed": (False, True), # right->left: yes
    },
    "line_thick": {
        "solid": (False, False),
        "solid_solid": (False, False),
        "dashed": (True, True),
        "dashed_solid": (True, False), # left->right: yes
        "solid_dashed": (False, True), # right->left: yes
    },
    "curbstone": {
        "high": (False, False),
        "low":  (False, False),
    }
}


class MapParseWarning(Warning): ...


class MapParseError(Exception): ...


def _load_node(xml_node: ET.Element, projector: Proj, origin: Point) -> Node:
    node_id = xml_node.get("id")
    x, y = projector(xml_node.get("lon"), xml_node.get("lat"))
    proj_x = x - origin[0]
    proj_y = y - origin[1]
    return Node(node_id, proj_x, proj_y)


def _get_tags(xml_node: ET.Element) -> dict:
    tags = dict()
    tags["custom_tags"] = dict()
    for tag in xml_node.findall("tag"):
        if tag.get("k") == "type":
            tags["type_"] = tag.get("v")
        elif tag.get("k") == "subtype":
            tags["subtype"] = tag.get("v")
        elif tag.get("k") == "width": # only for roadline
            tags["width"] = tag.get("v")
        elif tag.get("k") == "height": # only for roadline
            tags["height"] = tag.get("v")
        elif tag.get("k") == "temporary": # only for roadline
            tags["temporary"] = (tag.get("v")=="yes")
        elif tag.get("k") == "lane_change": # only for roadline
            if "lane_change" not in tags:
                if tags.get("v") == "yes":
                    tags["lane_change"] = (True, True)
                else:
                    tags["lane_change"] = (False, False)
            else:
                raise MapParseError("Conflict tags on lane changing property.")
        elif tag.get("k") == "lane_change:left": # only for roadline
            if "lane_change" not in tags:
                if tags.get("v") == "yes":
                    tags["lane_change"] = (True, False)
                else:
                    tags["lane_change"] = (False, False)
            else:
                raise MapParseError("Conflict tags on lane changing property.")
        elif tag.get("k") == "lane_change:right": # only for roadline
            if "lane_change" not in tags:
                if tags.get("v") == "yes":
                    tags["lane_change"] = (False, True)
                else:
                    tags["lane_change"] = (False, False)
            else:
                raise MapParseError("Conflict tags on lane changing property.")
        elif tag.get("k") == "location": # lane or area
            tags["location"] = tag.get("v")
        elif tag.get("k") == "inferred_participants": # lane or area
            tags["inferred_participants"] = tag.get("v")
        elif tag.get("k") == "speed_limit": # lane or area
            tags["speed_limit"] = tag.get("v")
        elif tag.get("k") == "speed_limit_mandatory": # lane or area
            tags["speed_limit_mandatory"] = (tag.get("v")=="yes")
        elif tag.get("k") == "color":
            tags["color"] = tag.get("v")
        elif tag.get("k") == "dynamic": # only for regulatory
            tags["dynamic"] = tag.get("v")=="yes"
        elif tag.get("k") == "fallback": # only for regulatory
            tags["fallback"] = tag.get("v")=="yes"
        else:
            tags["custom_tags"][tag.get("k")] = tag.get("v")
    return tags


def _load_roadline(xml_node: ET.Element, map_: Map) -> Lane:
    line_id = xml_node.get("id")
    point_list = []
    for node in xml_node.findall("nd"):
        point_list.append(map_.nodes[node.get("ref")].location)
    linestring = LineString(point_list)

    line_tags = _get_tags(xml_node)

    if "lane_change" not in line_tags:
        if line_tags["type_"] in LANE_CHANGE_MAPPING:
            if ("subtype" in line_tags) \
                and (line_tags["subtype"] in LANE_CHANGE_MAPPING[line_tags["type_"]]):
                line_tags["lane_change"] = \
                    LANE_CHANGE_MAPPING[line_tags["type_"]][line_tags["subtype"]]

    return RoadLine(line_id, linestring, **line_tags)


def _append_point_list(point_list, new_points, component_id):
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
        raise MapParseError("Points on the side of relation %s is not continuous." % component_id)

    point_list += new_points[1:]


def _load_lane(xml_node: ET.Element, map_: Map) -> Lane:
    lane_id = xml_node.get("id")
    line_ids = dict(left=[], right=[])
    regulatory_id_list = []
    for member in xml_node.findall("member"):
        if member.get("role") == "left":
            line_ids["left"].append(member.get("ref"))
        elif member.get("role") == "right":
            line_ids["right"].append(member.get("ref"))
        elif member.get("role") == "regulatory_element":
            regulatory_id_list.append(member.get("ref"))

    point_list = dict()
    for side in ["left", "right"]:
        point_list[side] = list(map_.roadlines[line_ids[side][0]].linestring.coords)
        for line_id in line_ids[side][1:]:
            new_points = list(map_.roadlines[line_id].linestring.coords)
            _append_point_list(point_list[side], new_points, lane_id)

    left_side = LineString(point_list["left"])
    right_side = LineString(point_list["right"])

    left_parallel = left_side.parallel_offset(0.1, 'left')
    if left_side.hausdorff_distance(right_side) > left_parallel.hausdorff_distance(right_side):
        point_list["left"].reverse()
        left_side = LineString(point_list["left"])
    right_parallel = right_side.parallel_offset(0.1, 'right')
    if right_side.hausdorff_distance(left_side) > right_parallel.hausdorff_distance(left_side):
        point_list["right"].reverse()
        right_side = LineString(point_list["right"])

    if left_side.crosses(right_side):
        warnings.warn("The sides of lane %s is intersected." % lane_id, MapParseWarning)

    lane_tags = _get_tags(xml_node)

    return Lane(lane_id, left_side, right_side, line_ids, **lane_tags)


def _load_area(xml_node: ET.Element, map_: Map) -> Area:
    area_id = xml_node.get("id")
    line_ids = dict(inner=[],outer=[])
    regulatory_id_list = []
    for member in xml_node.findall("member"):
        if member.get("role") == "outer":
            line_ids["outer"].append(member.get("ref"))
        elif member.get("role") == "inner":
            line_ids["inner"].append(member.get("ref"))
        elif member.get("role") == "regulatory_element":
            regulatory_id_list.append(member.get("ref"))

    outer_point_list = list(map_.roadlines[line_ids["outer"][0]].linestring.coords)
    for line_id in line_ids["outer"][1:]:
        new_points = list(map_.roadlines[line_id].linestring.coords)
        _append_point_list(outer_point_list, new_points, area_id)
    if outer_point_list[0] != outer_point_list[-1]:
        warnings.warn("The outer boundary of area %s is not closed." % area_id, MapParseWarning)

    inner_point_list = [[]]
    inner_idx = 0
    for line_id in line_ids["inner"]:
        if len(inner_point_list[inner_idx]) == 0:
            inner_point_list[inner_idx] = list(map_.roadlines[line_id].linestring.coords)
        else:
            new_points = list(map_.roadlines[line_id].linestring.coords)
            _append_point_list(inner_point_list[inner_idx], new_points, area_id)
        if inner_point_list[inner_idx][0] == inner_point_list[inner_idx][-1]:
            inner_point_list.append([])
            inner_idx += 1
    if len(inner_point_list[-1]) == 0:
        del(inner_point_list[-1])
    else:
        if inner_point_list[-1][0] != inner_point_list[-1][-1]:
            warnings.warn("The inner boundary of area %s is not closed." % area_id, MapParseWarning)
    polygon = Polygon(outer_point_list, inner_point_list)

    area_tags = _get_tags(xml_node)
    
    return Area(area_id, polygon, line_ids, **area_tags)


def _load_regulatory(xml_node: ET.Element) -> RegulatoryElement:
    regulatory_id = xml_node.get("id")
    relation_list = []
    lane_list = []
    for member in xml_node.findall("member"):
        if member.get("type") == "relation":
            relation_list.append((member.get("ref"), member.get("role")))
        elif member.get("type") == "way":
            lane_list.append((member.get("ref"), member.get("role")))

    regulatory_tags = _get_tags(xml_node)
    return RegulatoryElement(regulatory_id, relation_list, lane_list, **regulatory_tags)


def _get_map_bounds(nodes: dict):
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for node in nodes.values():
        x_min = min(x_min, node.x)
        x_max = max(x_max, node.x)
        y_min = min(y_min, node.y)
        y_max = max(y_max, node.y)
    
    return x_min, x_max, y_min, y_max


class Lanelet2Parser(object):

    @staticmethod
    def parse(xml_root: ET.ElementTree, map_config: dict) -> Map:

        map_name = map_config["map_name"] # update map name
        map_info = map_name.split("_")
        scenario_type = map_info[0]
        country = map_info[3]

        map_ = Map(map_name, scenario_type, country)

        projector = Proj(**map_config["project_rule"])
        origin = projector(
        map_config["gps_origin"][0], map_config["gps_origin"][1])

        for xml_node in xml_root.findall("node"):
            if xml_node.get("action") == "delete":
                continue
            node = _load_node(xml_node, projector, origin)
            map_.add_node(node)
        
        for xml_node in xml_root.findall("way"):
            if xml_node.get("action") == "delete":
                continue
            roadline = _load_roadline(xml_node, map_)
            map_.add_roadline(roadline)
        
        for xml_node in xml_root.findall("relation"):
            if xml_node.get("action") == "delete":
                continue
            for tag in xml_node.findall("tag"):
                if tag.get("v") == "lanelet":
                    lane = _load_lane(xml_node, map_)
                    map_.add_lane(lane)
                elif tag.get("v") in ["multipolygon", "area"]:
                    area = _load_area(xml_node, map_)
                    map_.add_area(area)

        for xml_node in xml_root.findall("relation"):
            if xml_node.get("action") == "delete":
                continue
            for tag in xml_node.findall("tag"):
                if tag.get("v") == "regulatory_element":
                    regulatory = _load_regulatory(xml_node)
                    map_.add_regulatory(regulatory)

        x_min, x_max, y_min, y_max = _get_map_bounds(map_.nodes)
        map_.boundary = [x_min-10, x_max+10, y_min-10, y_max+10]

        return map_