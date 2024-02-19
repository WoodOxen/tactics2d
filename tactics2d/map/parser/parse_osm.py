import xml.etree.ElementTree as ET
import warnings

from pyproj import Proj
from shapely.geometry import Point, LineString, Polygon

from tactics2d.map.element import Node, Lane, Area, RoadLine, Regulatory, Map

LANE_CHANGE_MAPPING = {
    "line_thin": {
        "solid": (False, False),
        "solid_solid": (False, False),
        "dashed": (True, True),
        "dashed_solid": (True, False),  # left->right: yes
        "solid_dashed": (False, True),  # right->left: yes
    },
    "line_thick": {
        "solid": (False, False),
        "solid_solid": (False, False),
        "dashed": (True, True),
        "dashed_solid": (True, False),  # left->right: yes
        "solid_dashed": (False, True),  # right->left: yes
    },
    "curbstone": {"high": (False, False), "low": (False, False)},
}


def _load_node(xml_node: ET.Element, projector: Proj, origin: Point) -> Node:
    node_id = xml_node.attrib["id"]
    x, y = projector(xml_node.attrib["lon"], xml_node.attrib["lat"])
    proj_x = x - origin[0]
    proj_y = y - origin[1]
    return Node(node_id, proj_x, proj_y)


def _get_tags(xml_node: ET.Element) -> dict:
    tags = dict()
    tags["custom_tags"] = dict()
    for tag in xml_node.findall("tag"):
        if tag.attrib["k"] == "type":
            tags["type_"] = tag.attrib["v"]
        elif tag.attrib["k"] == "subtype":
            tags["subtype"] = tag.attrib["v"]
        elif tag.attrib["k"] == "width":  # only for roadline
            tags["width"] = tag.attrib["v"]
        elif tag.attrib["k"] == "height":  # only for roadline
            tags["height"] = tag.attrib["v"]
        elif tag.attrib["k"] == "temporary":  # only for roadline
            tags["temporary"] = tag.attrib["v"] == "yes"
        elif "lane_change" in tag.attrib["k"]:  # only for roadline
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
        elif tag.attrib["k"] == "location":  # lane or area
            tags["location"] = tag.attrib["v"]
        elif tag.attrib["k"] == "inferred_participants":  # lane or area
            tags["inferred_participants"] = tag.attrib["v"]
        elif tag.attrib["k"] == "speed_limit":  # lane or area
            tags["speed_limit"] = tag.attrib["v"]
        elif tag.attrib["k"] == "speed_limit_mandatory":  # lane or area
            tags["speed_limit_mandatory"] = tag.attrib["v"] == "yes"
        elif tag.attrib["k"] == "color":
            tags["color"] = tag.attrib["v"]
        elif tag.attrib["k"] == "dynamic":  # only for regulatory
            tags["dynamic"] = tag.attrib["v"] == "yes"
        elif tag.attrib["k"] == "fallback":  # only for regulatory
            tags["fallback"] = tag.attrib["v"] == "yes"
        else:
            tags["custom_tags"][tag.attrib["k"]] = tag.attrib["v"]

    return tags


def _load_roadline(xml_node: ET.Element, map_: Map) -> RoadLine:
    line_id = xml_node.attrib["id"]
    point_list = []
    for node in xml_node.findall("nd"):
        point_list.append(map_.nodes[node.attrib["ref"]].location)
    linestring = LineString(point_list)

    line_tags = _get_tags(xml_node)

    if "lane_change" not in line_tags:
        if line_tags["type_"] in LANE_CHANGE_MAPPING:
            if ("subtype" in line_tags) and (
                line_tags["subtype"] in LANE_CHANGE_MAPPING[line_tags["type_"]]
            ):
                line_tags["lane_change"] = LANE_CHANGE_MAPPING[line_tags["type_"]][
                    line_tags["subtype"]
                ]

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
        raise SyntaxError(f"Points on the side of relation {component_id} is not continuous.")

    point_list += new_points[1:]


def _load_lane(xml_node: ET.Element, map_: Map) -> Lane:
    lane_id = xml_node.attrib["id"]
    line_ids = dict(left=[], right=[])
    regulatory_id_list = []
    for member in xml_node.findall("member"):
        if member.attrib["role"] == "left":
            line_ids["left"].append(member.attrib["ref"])
        elif member.attrib["role"] == "right":
            line_ids["right"].append(member.attrib["ref"])
        elif member.attrib["role"] == "regulatory_element":
            regulatory_id_list.append(member.attrib["ref"])

    point_list = dict()
    for side in ["left", "right"]:
        point_list[side] = list(map_.roadlines[line_ids[side][0]].linestring.coords)
        for line_id in line_ids[side][1:]:
            new_points = list(map_.roadlines[line_id].linestring.coords)
            _append_point_list(point_list[side], new_points, lane_id)

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
        warnings.warn(f"The sides of lane {lane_id} is intersected.", SyntaxWarning)

    lane_tags = _get_tags(xml_node)

    return Lane(lane_id, left_side, right_side, line_ids, **lane_tags)


def _load_area(xml_node: ET.Element, map_: Map) -> Area:
    area_id = xml_node.attrib["id"]
    line_ids = dict(inner=[], outer=[])
    regulatory_id_list = []
    for member in xml_node.findall("member"):
        if member.attrib["role"] == "outer":
            line_ids["outer"].append(member.attrib["ref"])
        elif member.attrib["role"] == "inner":
            line_ids["inner"].append(member.attrib["ref"])
        elif member.attrib["role"] == "regulatory_element":
            regulatory_id_list.append(member.attrib["ref"])

    outer_point_list = list(map_.roadlines[line_ids["outer"][0]].linestring.coords)
    for line_id in line_ids["outer"][1:]:
        new_points = list(map_.roadlines[line_id].linestring.coords)
        _append_point_list(outer_point_list, new_points, area_id)
    if outer_point_list[0] != outer_point_list[-1]:
        warnings.warn(f"The outer boundary of area {area_id} is not closed.", SyntaxWarning)

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
        del inner_point_list[-1]
    else:
        if inner_point_list[-1][0] != inner_point_list[-1][-1]:
            warnings.warn(f"The inner boundary of area {area_id} is not closed.", SyntaxWarning)
    polygon = Polygon(outer_point_list, inner_point_list)

    area_tags = _get_tags(xml_node)

    return Area(area_id, polygon, line_ids, **area_tags)


def _load_regulatory(xml_node: ET.Element) -> Regulatory:
    regulatory_id = xml_node.attrib["id"]
    relation_list = []
    lane_list = []
    for member in xml_node.findall("member"):
        if member.attrib["type"] == "relation":
            relation_list.append((member.attrib["ref"], member.attrib["role"]))
        elif member.attrib["type"] == "way":
            lane_list.append((member.attrib["ref"], member.attrib["role"]))

    regulatory_tags = _get_tags(xml_node)
    return Regulatory(regulatory_id, relation_list, lane_list, **regulatory_tags)


class Lanelet2Parser(object):
    """This class provides a parser for lanelet2 format map."""

    @staticmethod
    def parse(xml_root: ET.ElementTree, map_config: dict) -> Map:
        """Parse the map from lanelet2 format.

        Args:
            xml_root (ET.ElementTree): A xml tree of the map. The xml tree should be
                denoted in lanelet2 format.
            map_config (dict): The configuration of the map. The configuration should include
                the following keys: name, scenario_type, country, gps_origin, project_rule.

        Returns:
            Map: A map instance.
        """
        name = map_config["name"]
        scenario_type = map_config["scenario_type"]
        country = map_config["country"]

        map_ = Map(name, scenario_type, country)

        projector = Proj(**map_config["project_rule"])
        origin = projector(map_config["gps_origin"][0], map_config["gps_origin"][1])

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
                if tag.attrib["v"] == "lanelet":
                    lane = _load_lane(xml_node, map_)
                    map_.add_lane(lane)
                elif tag.attrib["v"] in ["multipolygon", "area"]:
                    area = _load_area(xml_node, map_)
                    map_.add_area(area)

        for xml_node in xml_root.findall("relation"):
            if xml_node.get("action") == "delete":
                continue
            for tag in xml_node.findall("tag"):
                if tag.attrib["v"] == "regulatory_element":
                    regulatory = _load_regulatory(xml_node)
                    map_.add_regulatory(regulatory)

        return map_
