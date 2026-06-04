# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OSM Lanelet2 XML writer.

Writes a Tactics2D ``Map`` object to a Lanelet2-annotated OSM (.osm) XML tree.
Each ``Lane`` becomes a lanelet relation; boundaries become ``<way>`` elements;
speed limits become regulatory relations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


class OsmWriter:
    """Writes a Tactics2D Map to a Lanelet2 OSM XML tree.

    All OSM element IDs are generated from a single decrementing counter
    starting at -1, so they never collide with existing positive IDs in the
    Map. Boundary nodes, ways, lanelet relations, and speed regulatory
    relations are each written by dedicated public methods that can also be
    called individually.

    Boundary nodes are deduplicated within a 1 cm tolerance so that adjacent
    lanelets sharing an endpoint reference the same OSM node and appear
    topologically connected in any Lanelet2 viewer.

    Example:
    ```python
    from tactics2d.map.parser import XODRParser
    from tactics2d.map.writer import OsmWriter

    map_ = XODRParser().parse("map.xodr")
    root = OsmWriter().build(map_)
    ```
    """

    _M_PER_DEG_LON: float = 111320.0
    _M_PER_DEG_LAT: float = 110540.0

    _ROAD_TYPE_TO_LOCATION: dict[str, str] = {
        "motorway": "nonurban",
        "rural": "nonurban",
        "town": "urban",
        "pedestrian": "urban",
        "bicycle": "urban",
    }

    _LANE_TYPE_TO_SUBTYPE: dict[str, str] = {
        "driving": "road",
        "parking": "parking",
        "sidewalk": "walkway",
        "shoulder": "shoulder",
        "border": "border",
        "restricted": "restricted",
        "stop": "stop",
        "none": "none",
        "crosswalk": "crosswalk",
    }

    def __init__(self) -> None:
        """Initialise the writer with a fresh ID counter and an empty node cache."""
        self._next_id: int = -1
        self._node_cache: dict[tuple[int, int], int] = {}

    def build(self, map_) -> ET.Element:
        """Build a complete OSM XML tree from a Tactics2D ``Map``.

        Resets the ID counter and node cache before processing so that
        calling ``build`` multiple times on the same instance is safe.
        Iterates every lane in *map_*, extracts boundary geometry and
        metadata, and assembles ``<node>``, ``<way>``, and ``<relation>``
        elements with Lanelet2 tags.

        Args:
            map_: Tactics2D Map object containing ``.lanes`` and
                ``.roadlines`` registries.

        Returns:
            An ``<osm>`` ``ET.Element`` ready for serialisation.
        """
        self._next_id = -1
        self._node_cache = {}

        root = ET.Element("osm", {"version": "0.6", "generator": "Tactics2D xodr2osm"})
        speed_relations: list[tuple[int, float, list[int]]] = []

        for lane in map_.lanes.values():
            left_coords, right_coords = self._extract_boundaries(lane)
            if len(left_coords) < 2 or len(right_coords) < 2:
                continue

            left_mark = self._get_roadmark(lane, map_, "left")
            right_mark = self._get_roadmark(lane, map_, "right")
            lane_subtype = getattr(lane, "subtype", None) or "driving"
            location = getattr(lane, "location", None) or "urban"

            left_wid, right_wid = self.write_boundary_ways(
                root, left_coords, right_coords, left_mark, right_mark
            )
            rel_id = self.write_lanelet_relation(root, lane_subtype, location, left_wid, right_wid)

            speed_ms = getattr(lane, "speed_limit", None)
            if speed_ms is not None and speed_ms > 0:
                speed_kmh = round(speed_ms * 3.6, 3)
                speed_relations.append((self._alloc(), speed_kmh, [rel_id]))

        for reg_id, kmh, lane_ids in speed_relations:
            self.write_speed_regulatory(root, reg_id, kmh, lane_ids)

        return root

    def write_nodes(self, root: ET.Element, coords: np.ndarray) -> list[int]:
        """Append ``<node>`` elements for a sequence of Cartesian coordinates.

        Coordinates are deduplicated within a 1 cm tolerance using an internal
        cache keyed by ``(round(x * 100), round(y * 100))``. If a coordinate
        has already been written, the existing node ID is returned without
        creating a duplicate element, ensuring adjacent lanelets that share an
        endpoint reference the same OSM node and remain topologically connected.

        Args:
            root: Parent element to append ``<node>`` children to.
            coords: (N, 2) array of world coordinates ``(x, y)`` in metres.

        Returns:
            List of assigned node ID integers, in the same order as *coords*.
        """
        node_ids: list[int] = []
        for x, y in coords:
            key = (round(float(x) * 100), round(float(y) * 100))
            if key in self._node_cache:
                node_ids.append(self._node_cache[key])
                continue
            nid = self._alloc()
            ET.SubElement(
                root,
                "node",
                {
                    "id": str(nid),
                    "action": "modify",
                    "visible": "true",
                    "lat": f"{float(y) / self._M_PER_DEG_LAT:.7f}",
                    "lon": f"{float(x) / self._M_PER_DEG_LON:.7f}",
                },
            )
            self._node_cache[key] = nid
            node_ids.append(nid)
        return node_ids

    def write_way(
        self, root: ET.Element, way_id: int, node_ids: list[int], type_: str, subtype: str
    ) -> None:
        """Append a ``<way>`` element referencing a list of nodes.

        Args:
            root: Parent element to append the ``<way>`` to.
            way_id: Unique OSM way ID.
            node_ids: Ordered list of ``<node>`` IDs referenced by this way.
            type_: Lanelet2 ``type`` tag value (e.g. ``"line_thin"``).
            subtype: Lanelet2 ``subtype`` tag value (e.g. ``"solid"``).
        """
        way = ET.SubElement(
            root, "way", {"id": str(way_id), "action": "modify", "visible": "true"}
        )
        for nid in node_ids:
            ET.SubElement(way, "nd", {"ref": str(nid)})
        ET.SubElement(way, "tag", {"k": "type", "v": type_})
        ET.SubElement(way, "tag", {"k": "subtype", "v": subtype})

    def write_boundary_ways(
        self,
        root: ET.Element,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        left_mark: str,
        right_mark: str,
    ) -> tuple[int, int]:
        """Create ``<way>`` elements for the left and right boundaries of a lane.

        Both ways are tagged ``type="line_thin"``.

        Args:
            root: Parent ``<osm>`` element.
            left_coords: (N, 2) array of left-boundary points in metres.
            right_coords: (M, 2) array of right-boundary points in metres.
            left_mark: Lanelet2 ``subtype`` for the left boundary.
            right_mark: Lanelet2 ``subtype`` for the right boundary.

        Returns:
            A ``(left_way_id, right_way_id)`` tuple.
        """
        left_nids = self.write_nodes(root, left_coords)
        right_nids = self.write_nodes(root, right_coords)
        left_wid = self._alloc()
        right_wid = self._alloc()
        self.write_way(root, left_wid, left_nids, "line_thin", left_mark)
        self.write_way(root, right_wid, right_nids, "line_thin", right_mark)
        return left_wid, right_wid

    def write_lanelet_relation(
        self, root: ET.Element, subtype: str, location: str, left_wid: int, right_wid: int
    ) -> int:
        """Append a ``<relation type="lanelet">`` that binds two boundary ways.

        Args:
            root: Parent ``<osm>`` element.
            subtype: Tactics2D lane subtype (e.g. ``"driving"``).
            location: Tactics2D location string (e.g. ``"urban"``).
            left_wid: OSM way ID for the left boundary.
            right_wid: OSM way ID for the right boundary.

        Returns:
            The newly assigned OSM relation ID.
        """
        rel_id = self._alloc()
        rel = ET.SubElement(
            root, "relation", {"id": str(rel_id), "action": "modify", "visible": "true"}
        )
        ET.SubElement(rel, "member", {"type": "way", "ref": str(left_wid), "role": "left"})
        ET.SubElement(rel, "member", {"type": "way", "ref": str(right_wid), "role": "right"})
        ET.SubElement(rel, "tag", {"k": "type", "v": "lanelet"})
        ET.SubElement(
            rel, "tag", {"k": "subtype", "v": self._LANE_TYPE_TO_SUBTYPE.get(subtype, "road")}
        )
        ET.SubElement(
            rel,
            "tag",
            {"k": "location", "v": self._ROAD_TYPE_TO_LOCATION.get(location, "urban")},
        )
        return rel_id

    def write_speed_regulatory(
        self, root: ET.Element, reg_id: int, speed_kmh: float, lane_rel_ids: list[int]
    ) -> None:
        """Append a ``<relation type="regulatory_element">`` for a speed limit.

        Args:
            root: Parent ``<osm>`` element.
            reg_id: Unique OSM relation ID for this regulatory element.
            speed_kmh: Speed limit in kilometres per hour.
            lane_rel_ids: List of lanelet relation IDs this limit applies to.
        """
        reg = ET.SubElement(
            root, "relation", {"id": str(reg_id), "action": "modify", "visible": "true"}
        )
        for lid in lane_rel_ids:
            ET.SubElement(reg, "member", {"type": "relation", "ref": str(lid), "role": "refers"})
        ET.SubElement(reg, "tag", {"k": "type", "v": "regulatory_element"})
        ET.SubElement(reg, "tag", {"k": "subtype", "v": "speed_limit"})
        ET.SubElement(reg, "tag", {"k": "speed_limit", "v": f"{speed_kmh:.1f}"})
        ET.SubElement(reg, "tag", {"k": "speed_limit_mandatory", "v": "yes"})

    def _alloc(self) -> int:
        """Return a unique decrementing OSM element ID."""
        nid = self._next_id
        self._next_id -= 1
        return nid

    @staticmethod
    def _extract_boundaries(lane) -> tuple[np.ndarray, np.ndarray]:
        """Extract left and right boundary coordinates from a ``Lane``.

        Args:
            lane: Tactics2D ``Lane`` object.

        Returns:
            ``(left_coords, right_coords)`` as ``np.ndarray`` of shape
            ``(N, 2)`` each. Returns empty arrays when a side is ``None``.
        """
        left = lane.left_side
        right = lane.right_side
        left_pts = np.array(left.coords) if left else np.empty((0, 2))
        right_pts = np.array(right.coords) if right else np.empty((0, 2))
        return left_pts, right_pts

    @staticmethod
    def _get_roadmark(lane, map_, side: str) -> str:
        """Derive the Lanelet2 boundary subtype for a given side of a lane.

        Args:
            lane: Tactics2D ``Lane`` object.
            map_: Tactics2D ``Map`` object whose ``.roadlines`` dict is queried.
            side: ``"left"`` or ``"right"``.

        Returns:
            Lanelet2 boundary subtype string (``"solid"``, ``"dashed"``,
            ``"solid_solid"``, or ``"solid"`` as default).
        """
        line_ids = getattr(lane, "line_ids", {})
        for lid in line_ids.get(side, []):
            rl = map_.roadlines.get(lid)
            if rl is None:
                continue
            subtype = getattr(rl, "subtype", None)
            if subtype in ("solid", "dashed", "solid_solid"):
                return subtype
        return "solid"