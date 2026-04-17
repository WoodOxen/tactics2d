# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml map parser implementation."""

from __future__ import annotations

import logging
import os

import sumolib
from shapely.geometry import LineString

from tactics2d.map.element import Junction, Lane, Map, RoadLine


class NetXMLParser:
    """This class implements a parser for the SUMO network format map (.net.xml).

    The parser reads SUMO road network files and converts them into the tactics2d
    internal map representation. It uses the ``sumolib`` library to read the network
    and maps edges/lanes to tactics2d ``Lane``, ``RoadLine``, and ``Junction`` objects.

    !!! quote "Reference"
        [SUMO Road Networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)

    Example:
        ```python
        from tactics2d.map.parser import NetXMLParser

        parser = NetXMLParser()
        map_ = parser.parse("/path/to/map.net.xml")
        print(f"Loaded {len(map_.lanes)} lanes")
        ```
    """

    _LANE_TYPE_DICT = {
        "highway.motorway":       "highway",
        "highway.trunk":          "highway",
        "highway.primary":        "road",
        "highway.secondary":      "road",
        "highway.tertiary":       "road",
        "highway.residential":    "road",
        "highway.living_street":  "road",
        "highway.service":        "road",
        "highway.pedestrian":     "walkway",
        "highway.footway":        "walkway",
        "highway.cycleway":       "bicycle_lane",
        "highway.path":           "walkway",
        "highway.unclassified":   "road",
        "highway.motorway_link":  "highway",
        "highway.trunk_link":     "highway",
        "highway.primary_link":   "road",
        "highway.secondary_link": "road",
        "highway.tertiary_link":  "road",
        "railway.rail":           "rail",
        "railway.subway":         "rail",
        "railway.tram":           "tram",
    }

    def __init__(self):
        self._id_counter = 0

    def _next_id(self) -> str:
        uid = str(self._id_counter)
        self._id_counter += 1
        return uid

    def _get_lane_subtype(self, edge_type: str) -> str:
        """Map a SUMO edge type string to a tactics2d lane subtype.

        Args:
            edge_type (str): The SUMO edge type string (e.g. ``'highway.residential'``).

        Returns:
            str: The corresponding tactics2d lane subtype. Defaults to ``'road'``.
        """
        return self._LANE_TYPE_DICT.get(edge_type, "road")

    def _load_lane(
        self,
        sumo_lane: sumolib.net.lane.Lane,
        edge_type: str,
    ) -> tuple[Lane, RoadLine, RoadLine]:
        """Parse one SUMO lane into a tactics2d Lane and its two boundary RoadLines.

        SUMO provides a centre-line shape for each lane together with a uniform
        width. The left and right boundaries are computed by offsetting the
        centre-line by +width/2 and -width/2 respectively.

        Args:
            sumo_lane: A sumolib ``Lane`` object.
            edge_type (str): The type string of the parent SUMO edge, used to
                determine the tactics2d lane subtype.

        Returns:
            tuple: A ``(Lane, left RoadLine, right RoadLine)`` triple.
        """
        shape = sumo_lane.getShape()      # list of (x, y)
        width = sumo_lane.getWidth()      # metres
        speed_mps = sumo_lane.getSpeed()  # m/s
        subtype = self._get_lane_subtype(edge_type)

        center = LineString(shape)
        left_geom  = center.offset_curve( width / 2.0)
        right_geom = center.offset_curve(-width / 2.0)

        left_id  = self._next_id()
        right_id = self._next_id()
        lane_id  = self._next_id()

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

        # Lane.__init__ converts speed_limit from km/h -> m/s internally
        lane = Lane(
            id_=lane_id,
            left_side=left_geom,
            right_side=right_geom,
            subtype=subtype,
            line_ids={"left": [left_id], "right": [right_id]},
            speed_limit=round(speed_mps * 3.6, 3),
            speed_limit_unit="km/h",
        )

        return lane, left_line, right_line

    def _load_junction(self, sumo_node: sumolib.net.node.Node) -> Junction:
        """Parse a SUMO junction node into a tactics2d Junction.

        Args:
            sumo_node: A sumolib ``Node`` object.

        Returns:
            Junction: A tactics2d Junction with an auto-assigned id.
        """
        return Junction(id_=self._next_id())

    def parse(self, file_path: str) -> Map:
        """Parse a SUMO network file (.net.xml) into a tactics2d Map.

        This function reads a SUMO ``.net.xml`` file using ``sumolib``, iterates
        over all edges and their lanes, and converts them into tactics2d ``Lane``,
        ``RoadLine``, and ``Junction`` objects assembled into a ``Map``.

        Args:
            file_path (str): The absolute path to the ``.net.xml`` file.

        Returns:
            Map: The parsed map containing all lanes, roadlines, and junctions.

        Example:
            ```python
            from tactics2d.map.parser import NetXMLParser

            parser = NetXMLParser()
            map_ = parser.parse("/path/to/cologne.net.xml")

            print(f"Loaded {len(map_.lanes)} lanes")
            print(f"Loaded {len(map_.junctions)} junctions")
            ```
        """
        net = sumolib.net.readNet(file_path, withInternal=False)

        map_name = os.path.splitext(os.path.basename(file_path))[0]
        map_ = Map(name=map_name)

        try:
            x_min, y_min, x_max, y_max = net.getBoundary()
            map_.set_boundary((x_min, x_max, y_min, y_max))
        except Exception as exc:
            logging.warning("Could not set map boundary: %s", exc)

        for edge in net.getEdges():
            edge_type = edge.getType() or ""
            for sumo_lane in edge.getLanes():
                try:
                    shape = sumo_lane.getShape()
                    if len(shape) < 2:
                        logging.warning(
                            "Lane %s has fewer than 2 points, skipping.",
                            sumo_lane.getID(),
                        )
                        continue
                    lane, left_line, right_line = self._load_lane(sumo_lane, edge_type)
                    map_.add_lane(lane)
                    map_.add_roadline(left_line)
                    map_.add_roadline(right_line)
                except Exception as exc:
                    logging.warning("Failed to parse lane %s: %s", sumo_lane.getID(), exc)

        for node in net.getNodes():
            try:
                junction = self._load_junction(node)
                map_.add_junction(junction)
            except Exception as exc:
                logging.warning("Failed to parse junction %s: %s", node.getID(), exc)

        self._id_counter = 0
        return map_