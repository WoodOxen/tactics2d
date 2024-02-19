##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: fix_osm.py
# @Description:
# @Author: Yueyuan Li
# @Version: 1.0.0

import xml.etree.ElementTree as ET


def re_id_elements(file, destination_file=None):
    """This function fix the osm file to avoid id violation between road elements.

    Args:
        file (str): path to the osm file
    """
    xml_root = ET.parse(file).getroot()
    id_cnt = 10000

    for xml_node in xml_root.findall("node"):
        original_id = xml_node.attrib["id"]
        xml_node.attrib["id"] = str(id_cnt)
        for xml_way in xml_root.findall("way"):
            for xml_nd in xml_way.findall("nd"):
                if xml_nd.attrib["ref"] == original_id:
                    xml_nd.attrib["ref"] = str(id_cnt)

        for xml_relation in xml_root.findall("relation"):
            for xml_member in xml_relation.findall("member"):
                if xml_member.attrib["ref"] == original_id and xml_member.attrib["type"] == "node":
                    xml_member.attrib["ref"] = str(id_cnt)

        id_cnt += 1

    expected_start = 20000
    while True:
        if id_cnt < expected_start:
            id_cnt = expected_start
            break
        else:
            expected_start += 1000

    for xml_way in xml_root.findall("way"):
        original_id = xml_way.attrib["id"]
        xml_way.attrib["id"] = str(id_cnt)
        for xml_relation in xml_root.findall("relation"):
            for xml_member in xml_relation.findall("member"):
                if xml_member.attrib["ref"] == original_id and xml_member.attrib["type"] == "way":
                    xml_member.attrib["ref"] = str(id_cnt)

        id_cnt += 1

    expected_start = 30000
    while True:
        if id_cnt < expected_start:
            id_cnt = expected_start
            break
        else:
            expected_start += 1000

    for xml_relation in xml_root.findall("relation"):
        xml_relation.attrib["id"] = str(id_cnt)
        id_cnt += 1

    if destination_file is None:
        destination_file = file

    # xml_root.write(destination_file)
    xml_string = ET.tostring(xml_root, encoding="unicode")
    with open(destination_file, "w") as f:
        f.write(xml_string)
