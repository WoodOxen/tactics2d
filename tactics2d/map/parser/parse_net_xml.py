#! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_net_xml.py
# @Description: This file defines a class for parsing the SUMO network map format.
# @Author: Tactics2D Team
# @Version: 1.0.0

from __future__ import annotations

import logging
import os

import defusedxml.ElementTree as ET
from shapely.geometry import LineString, MultiPoint

# Connection is a nested class of Junction; it is not exported separately.
from tactics2d.map.element import Junction, Lane, Map, RoadLine


class NetXMLParser:
    """Parser for the SUMO network format (.net.xml) map.

    Reads a SUMO road-network file and converts it into a Tactics2D Map object
    containing lanes, road-mark lines, junctions, and connections.  The parser
    operates directly on the XML without any external SUMO dependency.

    Edges and their child lanes are mapped to :class:`~tactics2d.map.element.Lane`
    and :class:`~tactics2d.map.element.RoadLine` objects.  Junctions are mapped to
    :class:`~tactics2d.map.element.Junction` objects and SUMO ``<connection>``
    elements are attached to the junction at the receiving end of each edge.
    Internal edges (``function="internal"``) are skipped.

    !!! quote "Reference"
        [SUMO Road Networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)

    Example:
        >>> parser = NetXMLParser()
        >>> map_ = parser.parse("/path/to/map.net.xml")
        >>> print(len(map_.lanes))
    """

    _LANE_TYPE_DICT: dict = {
        "highway.motorway":    "highway",
        "highway.trunk":       "highway",
        "highway.primary":     "road",
        "highway.secondary":   "road",
        "highway.tertiary":    "road",
        "highway.residential": "road",
        "highway.service":     "road",
        "highway.pedestrian":  "walkway",
        "highway.footway":     "walkway",
        "highway.cycleway":    "bicycle_lane",
        "railway.rail":        "rail",
        "railway.tram":        "tram",
    }

    _DEFAULT_LANE_WIDTH: float = 3.2

    def __init__(self) -> None:
        self._id_counter = 0

    def _next_id(self) -> int:
        uid = self._id_counter
        self._id_counter += 1
        return uid

    def _parse_shape(self, shape_str: str) -> list:
        """Parse a SUMO shape string into a list of (x, y) coordinate tuples.

        Args:
            shape_str (str): Space-separated coordinate pairs,
                e.g. ``'0.00,1.00 2.00,3.00'``.

        Returns:
            list: List of (x, y) tuples parsed from the shape string.
        """
        points = []
        for pair in shape_str.strip().split():
            x, y = pair.split(",")
            points.append((float(x), float(y)))
        return points

    def _get_lane_subtype(self, edge_type: str) -> str:
        """Map a SUMO edge type string to a Tactics2D lane subtype.

        Args:
            edge_type (str): The SUMO edge ``type`` attribute string.

        Returns:
            str: The corresponding Tactics2D lane subtype. Defaults to ``'road'``.
        """
        return self._LANE_TYPE_DICT.get(edge_type, "road")

    def _load_lane(
        self,
        lane_node:  ET.Element,
        edge_type:  str,
        lane_width: float = None,
    ) -> tuple:
        """Parse one SUMO ``<lane>`` element into a Tactics2D Lane and its boundary RoadLines.

        The lane centre-line is taken directly from the ``shape`` attribute.
        Left and right boundaries are derived by offsetting the centre-line by
        half the lane width using Shapely's ``offset_curve``.

        Args:
            lane_node (ET.Element): The ``<lane>`` XML element to parse.
            edge_type (str): The ``type`` string of the parent edge, used to
                derive the lane subtype.
            lane_width (float, optional): Override lane width in metres.
                When ``None``, :attr:`_DEFAULT_LANE_WIDTH` is used.

        Returns:
            tuple:
                lane (Lane or None): The constructed Lane, or ``None`` if the
                    shape is missing or degenerate.
                left_line (RoadLine or None): RoadLine for the left boundary.
                right_line (RoadLine or None): RoadLine for the right boundary.

        Example:
            >>> parser = NetXMLParser()
            >>> import xml.etree.ElementTree as ET
            >>> lane_el = ET.fromstring(
            ...     '<lane id="E0_0" speed="13.89" shape="0.00,0.00 10.00,0.00"/>'
            ... )
            >>> lane, left, right = parser._load_lane(lane_el, "highway.primary")
        """
        shape_str = lane_node.attrib.get("shape", "")
        if not shape_str:
            return None, None, None

        shape = self._parse_shape(shape_str)
        if len(shape) < 2:
            return None, None, None

        speed_ms   = float(lane_node.attrib.get("speed", "13.89"))
        half_width = (lane_width if lane_width is not None else self._DEFAULT_LANE_WIDTH) / 2.0

        center     = LineString(shape)
        left_geom  = center.offset_curve( half_width)
        right_geom = center.offset_curve(-half_width)

        if left_geom is None or right_geom is None:
            return None, None, None

        left_id      = self._next_id()
        right_id     = self._next_id()
        lane_id      = self._next_id()
        lane_sumo_id = lane_node.attrib.get("id", "")

        left_line = RoadLine(
            id_=left_id,
            geometry=left_geom,
            type_="line_thin",
            subtype="dashed",
        )
        right_line = RoadLine(
            id_=right_id,
            geometry=right_geom,
            type_="line_thin",
            subtype="dashed",
        )
        lane = Lane(
            id_=lane_id,
            left_side=left_geom,
            right_side=right_geom,
            subtype=self._get_lane_subtype(edge_type),
            line_ids={"left": [left_id], "right": [right_id]},
            speed_limit=round(speed_ms * 3.6, 3),
            speed_limit_unit="km/h",
            custom_tags={"sumo_id": lane_sumo_id},
        )

        return lane, left_line, right_line

    def _load_junction(self, junction_node: ET.Element) -> Junction:
        """Parse a SUMO ``<junction>`` element into a Tactics2D Junction.

        Junction position, type, and shape polygon are stored in
        ``custom_tags`` for use by visualisers and format converters.

        Args:
            junction_node (ET.Element): The ``<junction>`` XML element to parse.

        Returns:
            Junction: A Tactics2D Junction object with SUMO metadata in
                ``custom_tags``: keys ``sumo_id``, ``x``, ``y``, ``type``,
                and ``shape`` (list of (x, y) tuples).

        Example:
            >>> parser = NetXMLParser()
            >>> import xml.etree.ElementTree as ET
            >>> junc_el = ET.fromstring(
            ...     '<junction id="J0" x="10.0" y="0.0" type="priority" shape=""/>'
            ... )
            >>> junction = parser._load_junction(junc_el)
        """
        shape_str = junction_node.attrib.get("shape", "")
        shape_pts = self._parse_shape(shape_str) if shape_str else []

        return Junction(
            id_=self._next_id(),
            custom_tags={
                "sumo_id": junction_node.attrib.get("id", ""),
                "x":       junction_node.attrib.get("x", ""),
                "y":       junction_node.attrib.get("y", ""),
                "type":    junction_node.attrib.get("type", ""),
                "shape":   shape_pts,
            },
        )

    def _load_connection(self, conn_node: ET.Element) -> Junction:
        """Parse a SUMO ``<connection>`` element into a Tactics2D Junction.

        All SUMO-specific routing attributes are stored in ``custom_tags`` so
        that the Tactics2D core data model remains format-agnostic.  The
        returned object is a ``Junction`` instance whose connection-level
        attributes (``incoming_road``, ``connecting_road``, etc.) are left at
        their defaults; SUMO routing data is stored in ``custom_tags`` instead.

        Args:
            conn_node (ET.Element): The ``<connection>`` XML element to parse.

        Returns:
            Junction: A Junction instance whose ``custom_tags`` contain keys
                ``from_edge``, ``to_edge``, ``from_lane``, ``to_lane``,
                ``via``, ``dir``, and ``state``.

        Example:
            >>> parser = NetXMLParser()
            >>> import xml.etree.ElementTree as ET
            >>> conn_el = ET.fromstring(
            ...     '<connection from="E0" to="E1" fromLane="0" toLane="0"'
            ...     ' via="" dir="s" state="M"/>'
            ... )
            >>> conn = parser._load_connection(conn_el)
            >>> conn.custom_tags["dir"]
            's'
        """
        return Junction(
            id_=self._next_id(),
            custom_tags={
                "from_edge": conn_node.attrib.get("from", ""),
                "to_edge":   conn_node.attrib.get("to", ""),
                "from_lane": conn_node.attrib.get("fromLane", ""),
                "to_lane":   conn_node.attrib.get("toLane", ""),
                "via":       conn_node.attrib.get("via", ""),
                "dir":       conn_node.attrib.get("dir", ""),
                "state":     conn_node.attrib.get("state", ""),
            },
        )

    def parse(self, file_path: str) -> Map:
        """Parse a SUMO network file (.net.xml) into a Tactics2D Map.

        The parser makes two linear passes over the XML: the first pass
        builds all lanes and junctions; the second pass attaches each
        ``<connection>`` to the junction at the receiving end of the
        originating edge.  Internal edges (``function="internal"``) are
        skipped throughout.

        Args:
            file_path (str): Absolute or relative path to the ``.net.xml`` file.

        Returns:
            Map: A Tactics2D Map populated with all lanes, roadlines, junctions,
                and connections parsed from the SUMO network file.

        Example:
            >>> parser = NetXMLParser()
            >>> map_ = parser.parse("/path/to/map.net.xml")
            >>> print(len(map_.lanes))
            >>> print(len(map_.junctions))
        """
        self._id_counter = 0

        xml_root = ET.parse(file_path).getroot()
        map_name = os.path.splitext(os.path.basename(file_path))[0]
        map_     = Map(name=map_name)

        location_node = xml_root.find("location")
        if location_node is not None:
            boundary_str = location_node.attrib.get("convBoundary", "")
            if boundary_str:
                x_min, y_min, x_max, y_max = map(float, boundary_str.split(","))
                map_.set_boundary((x_min, x_max, y_min, y_max))

        edge_to_junction: dict[str, str] = {}
        for edge_node in xml_root.findall("edge"):
            if edge_node.attrib.get("function") == "internal":
                continue
            edge_id = edge_node.attrib.get("id", "")
            to_node = edge_node.attrib.get("to", "")
            if edge_id and to_node:
                edge_to_junction[edge_id] = to_node

        for edge_node in xml_root.findall("edge"):
            if edge_node.attrib.get("function") == "internal":
                continue

            edge_type  = edge_node.attrib.get("type", "")
            lane_nodes = edge_node.findall("lane")

            lane_width = self._DEFAULT_LANE_WIDTH
            if len(lane_nodes) >= 2:
                try:
                    shape0 = self._parse_shape(lane_nodes[0].attrib.get("shape", ""))
                    shape1 = self._parse_shape(lane_nodes[1].attrib.get("shape", ""))
                    if shape0 and shape1:
                        dx       = shape1[0][0] - shape0[0][0]
                        dy       = shape1[0][1] - shape0[0][1]
                        computed = (dx**2 + dy**2) ** 0.5
                        if 1.5 < computed < 6.0:
                            lane_width = computed
                except Exception:
                    pass

            for lane_node in lane_nodes:
                try:
                    lane, left_line, right_line = self._load_lane(
                        lane_node, edge_type, lane_width
                    )
                    if lane is None:
                        continue
                    map_.add_lane(lane)
                    map_.add_roadline(left_line)
                    map_.add_roadline(right_line)
                except Exception as exc:
                    logging.warning(
                        "Failed to parse lane %s: %s",
                        lane_node.attrib.get("id", "unknown"), exc,
                    )

        sumo_to_tactics_junction: dict[str, int] = {}

        for junction_node in xml_root.findall("junction"):
            try:
                junction = self._load_junction(junction_node)
                map_.add_junction(junction)
                sumo_id = junction_node.attrib.get("id", "")
                if sumo_id:
                    sumo_to_tactics_junction[sumo_id] = junction.id_
            except Exception as exc:
                logging.warning(
                    "Failed to parse junction %s: %s",
                    junction_node.attrib.get("id", "unknown"), exc,
                )

        for conn_node in xml_root.findall("connection"):
            try:
                connection   = self._load_connection(conn_node)
                from_edge    = connection.custom_tags.get("from_edge", "")
                sumo_junc_id = edge_to_junction.get(from_edge, "")
                tactics_id   = sumo_to_tactics_junction.get(sumo_junc_id)

                if tactics_id is not None and tactics_id in map_.junctions:
                    map_.junctions[tactics_id].add_connection(connection)
                else:
                    logging.debug(
                        "Cannot find junction for connection from_edge=%s; skipping.",
                        from_edge,
                    )
            except Exception as exc:
                logging.warning("Failed to parse connection: %s", exc)

        junction_endpoints: dict[str, list] = {
            sumo_id: [] for sumo_id in sumo_to_tactics_junction
        }

        for lane in map_.lanes.values():
            edge_id = lane.custom_tags.get("sumo_id", "").rsplit("_", 1)[0]
            to_sumo = edge_to_junction.get(edge_id)
            if to_sumo and to_sumo in junction_endpoints:
                try:
                    for side in (lane.left_side, lane.right_side):
                        coords = list(side.coords)
                        junction_endpoints[to_sumo].append(coords[0])
                        junction_endpoints[to_sumo].append(coords[-1])
                except Exception:
                    pass

        for sumo_id, tactics_id in sumo_to_tactics_junction.items():
            junction = map_.junctions.get(tactics_id)
            if junction is None or junction.custom_tags.get("shape"):
                continue

            pts = junction_endpoints.get(sumo_id, [])
            if len(pts) < 3:
                continue

            try:
                hull = MultiPoint(pts).convex_hull
                if hull.geom_type == "Polygon" and not hull.is_empty:
                    junction.custom_tags["shape"] = list(hull.exterior.coords)
            except Exception as exc:
                logging.warning(
                    "Failed to compute convex hull for junction %s: %s", sumo_id, exc
                )

        self._id_counter = 0
        return map_