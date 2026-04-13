# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fix osm implementation."""

from __future__ import annotations

import defusedxml.ElementTree as ET

# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.


def re_id_elements(file, destination_file=None):
    """This function fix the OSM file to avoid id conflict between road elements.

    Args:
        file (str): path to the osm file
    """
    xml_root = ET.parse(file).getroot()
    id_cnt = 10000

    # Build ID mappings to avoid O(n³) complexity
    node_id_map = {}  # original_id -> new_id
    way_id_map = {}  # original_id -> new_id
    relation_id_map = {}  # original_id -> new_id

    # Collect all elements and build mappings
    nodes = list(xml_root.findall("node"))
    ways = list(xml_root.findall("way"))
    relations = list(xml_root.findall("relation"))

    # Process nodes first
    for xml_node in nodes:
        original_id = xml_node.attrib["id"]
        new_id = str(id_cnt)
        node_id_map[original_id] = new_id
        xml_node.attrib["id"] = new_id
        id_cnt += 1

    # Adjust id_cnt to next expected start (20000) if needed
    expected_start = 20000
    if id_cnt < expected_start:
        id_cnt = expected_start
    else:
        # Find next multiple of 1000 >= id_cnt
        while id_cnt >= expected_start:
            expected_start += 1000
        id_cnt = expected_start

    # Process ways with optimized reference updates
    # First build mapping for all ways
    for xml_way in ways:
        original_id = xml_way.attrib["id"]
        new_id = str(id_cnt)
        way_id_map[original_id] = new_id
        xml_way.attrib["id"] = new_id
        id_cnt += 1

    # Adjust id_cnt to next expected start (30000) if needed
    expected_start = 30000
    if id_cnt < expected_start:
        id_cnt = expected_start
    else:
        # Find next multiple of 1000 >= id_cnt
        while id_cnt >= expected_start:
            expected_start += 1000
        id_cnt = expected_start

    # Process relations
    for xml_relation in relations:
        original_id = xml_relation.attrib["id"]
        new_id = str(id_cnt)
        relation_id_map[original_id] = new_id
        xml_relation.attrib["id"] = new_id
        id_cnt += 1

    # Now update all references using the mappings (optimized O(n) operations)

    # Update node references in ways and relations
    # Build a list of all elements that might reference nodes
    for xml_way in ways:
        for xml_nd in xml_way.findall("nd"):
            original_ref = xml_nd.attrib["ref"]
            if original_ref in node_id_map:
                xml_nd.attrib["ref"] = node_id_map[original_ref]

    for xml_relation in relations:
        for xml_member in xml_relation.findall("member"):
            if xml_member.attrib["type"] == "node":
                original_ref = xml_member.attrib["ref"]
                if original_ref in node_id_map:
                    xml_member.attrib["ref"] = node_id_map[original_ref]

    # Update way references in relations
    for xml_relation in relations:
        for xml_member in xml_relation.findall("member"):
            if xml_member.attrib["type"] == "way":
                original_ref = xml_member.attrib["ref"]
                if original_ref in way_id_map:
                    xml_member.attrib["ref"] = way_id_map[original_ref]

    if destination_file is None:
        destination_file = file

    # xml_root.write(destination_file)
    xml_string = ET.tostring(xml_root, encoding="unicode")
    with open(destination_file, "w") as f:
        f.write(xml_string)
