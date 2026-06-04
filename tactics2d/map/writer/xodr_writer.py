# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE XODR XML writer.

Writes a Tactics2D ``Map`` object to an OpenDRIVE (.xodr) XML tree.
Each ``Lane`` becomes one ``<road>`` element with a single driving lane.
Centre-lines are derived from left and right boundaries, fitted to
``paramPoly3`` geometries per segment.  Lane width is fitted to a cubic
polynomial.  Predecessor / successor links and junction groups are inferred
from lane endpoint proximity.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString

from tactics2d.interpolator.param_poly3 import ParamPoly3, split_polyline


class _TopologyBuilder:
    """Infers predecessor / successor relationships and junction groups
    from lane endpoint proximity.

    A junction is flagged when three or more lane endpoints fall within
    ``_ENDPOINT_TOL`` of each other.
    """

    _ENDPOINT_TOL: float = 0.5

    def __init__(self, map_: Map) -> None:
        self._map = map_
        self.predecessors: dict[int, list[int]] = {}
        self.successors: dict[int, list[int]] = {}
        self.junction_groups: list[set[int]] = []
        self._build()

    def junction_id_for(self, lane_id: int) -> int | None:
        """Return the junction index containing *lane_id*, or ``None``."""
        for i, g in enumerate(self.junction_groups):
            if lane_id in g:
                return i
        return None

    def _endpoint(self, lane, which: str) -> tuple:
        coords = list(lane.left_side.coords)
        return coords[0] if which == "start" else coords[-1]

    @staticmethod
    def _dist(a: tuple, b: tuple) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _build(self) -> None:
        lanes = self._map.lanes
        ids = list(lanes.keys())

        for lid in ids:
            self.predecessors[lid] = []
            self.successors[lid] = []

        starts = {lid: self._endpoint(lanes[lid], "start") for lid in ids}
        ends = {lid: self._endpoint(lanes[lid], "end") for lid in ids}

        for a in ids:
            for b in ids:
                if a == b:
                    continue
                if self._dist(ends[a], starts[b]) < self._ENDPOINT_TOL:
                    if b not in self.successors[a]:
                        self.successors[a].append(b)
                    if a not in self.predecessors[b]:
                        self.predecessors[b].append(a)

        node_map: dict[tuple, list[tuple[int, str]]] = {}
        for lid in ids:
            for which, pt in [("start", starts[lid]), ("end", ends[lid])]:
                key = (round(pt[0] / self._ENDPOINT_TOL), round(pt[1] / self._ENDPOINT_TOL))
                node_map.setdefault(key, []).append((lid, which))

        for owners in node_map.values():
            if len(owners) < 3:
                continue
            merged = {lid for lid, _ in owners}
            remaining: list[set[int]] = []
            for g in self.junction_groups:
                if g & merged:
                    merged |= g
                else:
                    remaining.append(g)
            remaining.append(merged)
            self.junction_groups = remaining


class XodrWriter:
    """Writes a Tactics2D Map to an OpenDRIVE XODR XML tree.

    Each Tactics2D ``Lane`` becomes one ``<road>`` with a single
    ``<lane>`` on the right side of its centreline.  The centreline is
    interpolated from ``Lane.left_side`` and ``Lane.right_side``, and
    lane width is fitted to a cubic polynomial in arc-length.
    Predecessor / successor links and junction groups are inferred from
    lane endpoint proximity.

    Examples:
    ```python
    from tactics2d.map.parser import OSMParser
    from tactics2d.map.writer import XodrWriter

    map_ = OSMParser(lanelet2=True).parse("map.osm")
    root = XodrWriter().build(map_)
    ```
    """

    _MAX_SEG_LENGTH: float = 10.0

    _SUBTYPE_TO_ROADMARK: dict[str, str] = {
        "dashed": "broken",
        "solid": "solid",
        "solid_solid": "solid solid",
    }

    _LOCATION_TO_ROAD_TYPE: dict[str, str] = {"urban": "town", "nonurban": "motorway"}

    def __init__(self) -> None:
        """Initialise the writer."""

    def build(self, map_: Map) -> ET.Element:
        """Build a complete OpenDRIVE XML tree from a Tactics2D ``Map``.

        Iterates every lane in *map_*, computes centreline geometry and
        width polynomials, and assembles ``<road>`` elements.  Topology
        and junction groups are inferred from lane endpoint proximity.

        Args:
            map_: Tactics2D Map object containing ``.lanes`` and
                ``.roadlines`` registries.

        Returns:
            An ``<OpenDRIVE>`` ``ET.Element`` ready for serialisation.
        """
        topology = _TopologyBuilder(map_)

        root = ET.Element("OpenDRIVE")
        self.write_header(root)

        for lane_id, lane in map_.lanes.items():
            pts = self._get_centerline(lane)
            if len(pts) < 2:
                continue

            jid = topology.junction_id_for(lane_id)
            preds = topology.predecessors.get(lane_id, [])
            succs = topology.successors.get(lane_id, [])

            self.write_road(root, lane, map_, pts, preds, succs, jid, topology)

        self.write_junctions(root, topology, map_)
        return root

    def write_header(self, root: ET.Element) -> None:
        """Write the ``<header>`` element with default metadata.

        Args:
            root: The ``<OpenDRIVE>`` root element.
        """
        ET.SubElement(
            root,
            "header",
            {
                "revMajor": "1",
                "revMinor": "6",
                "name": "",
                "version": "1.00",
                "date": "",
                "north": "0",
                "south": "0",
                "east": "0",
                "west": "0",
            },
        )

    def write_road(
        self,
        root: ET.Element,
        lane,
        map_: Map,
        centerline_pts: list,
        predecessors: list,
        successors: list,
        junction_id: int | None,
        topology: _TopologyBuilder,
    ) -> ET.Element:
        """Write a complete ``<road>`` element for a single ``Lane``.

        Args:
            root: Parent ``<OpenDRIVE>`` element.
            lane: Tactics2D ``Lane`` object.
            map_: Tactics2D ``Map`` for roadline lookups.
            centerline_pts: (x, y) tuples for the lane centreline.
            predecessors: Lane IDs that precede this lane.
            successors: Lane IDs that follow this lane.
            junction_id: Junction index if this lane is part of a
                junction, ``None`` otherwise.
            topology: Topology builder for junction ID lookups.

        Returns:
            The newly created ``<road>`` element.
        """
        total_length = float(LineString(centerline_pts).length)

        road = ET.SubElement(
            root,
            "road",
            {
                "name": str(lane.id_),
                "length": f"{total_length:.4f}",
                "id": str(lane.id_),
                "junction": str(junction_id) if junction_id is not None else "-1",
            },
        )

        self.write_link(road, predecessors, successors, topology)

        location = getattr(lane, "location", None) or "urban"
        road_type = self._LOCATION_TO_ROAD_TYPE.get(location, "town")
        ET.SubElement(road, "type", {"s": "0.0", "type": road_type})

        self.write_plan_view(road, centerline_pts)
        ET.SubElement(road, "elevationProfile")
        ET.SubElement(road, "lateralProfile")
        self.write_lanes(road, lane, map_)

        return road

    def write_plan_view(self, road_elem: ET.Element, pts: list) -> None:
        """Write the ``<planView>`` element using ``paramPoly3`` geometry.

        The centreline is split into segments no longer than
        ``_MAX_SEG_LENGTH``, and each segment is fitted to a parametric
        cubic polynomial.

        Args:
            road_elem: The ``<road>`` element to append to.
            pts: List of (x, y) tuples for the road centreline.
        """
        plan_view = ET.SubElement(road_elem, "planView")
        s_offset = 0.0

        for seg in split_polyline(pts, self._MAX_SEG_LENGTH):
            fit = ParamPoly3.fit(seg)
            if fit is None:
                continue
            x, y, hdg, length, aU, bU, cU, dU, aV, bV, cV, dV = fit

            geom = ET.SubElement(
                plan_view,
                "geometry",
                {
                    "s": f"{s_offset:.4f}",
                    "x": f"{x:.4f}",
                    "y": f"{y:.4f}",
                    "hdg": f"{hdg:.6f}",
                    "length": f"{length:.4f}",
                },
            )
            pp3 = ET.SubElement(geom, "paramPoly3", {"pRange": "normalized"})
            for k, v in zip(
                ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV"), (aU, bU, cU, dU, aV, bV, cV, dV)
            ):
                pp3.set(k, f"{v:.6f}")
            s_offset += length

    def write_lanes(self, road_elem: ET.Element, lane, map_: Map) -> None:
        """Write the ``<lanes>`` element with a single right-side lane.

        Lane width is fitted to a cubic polynomial.  RoadMark types are
        reverse-mapped from ``RoadLine.subtype``.  Speed limits are
        written as ``<speed>`` children in km/h.

        Args:
            road_elem: The ``<road>`` element to append to.
            lane: Tactics2D ``Lane`` object.
            map_: Tactics2D ``Map`` for roadline lookups.
        """
        a, b, c, d = self._fit_width(lane)
        left_mark = self._get_xodr_roadmark(lane, map_, "left")
        right_mark = self._get_xodr_roadmark(lane, map_, "right")

        subtype = getattr(lane, "subtype", None) or "driving"
        lane_type = (
            subtype
            if subtype
            in {
                "driving",
                "parking",
                "sidewalk",
                "shoulder",
                "border",
                "restricted",
                "stop",
                "none",
                "crosswalk",
            }
            else "driving"
        )

        lanes_elem = ET.SubElement(road_elem, "lanes")
        lane_section = ET.SubElement(lanes_elem, "laneSection", {"s": "0.0"})
        ET.SubElement(lane_section, "left")

        center = ET.SubElement(lane_section, "center")
        center_lane = ET.SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})
        ET.SubElement(
            center_lane,
            "roadMark",
            {
                "sOffset": "0",
                "type": left_mark,
                "weight": "standard",
                "color": "standard",
                "width": "0.13",
            },
        )

        right_elem = ET.SubElement(lane_section, "right")
        right_lane = ET.SubElement(
            right_elem, "lane", {"id": "-1", "type": lane_type, "level": "false"}
        )
        ET.SubElement(
            right_lane,
            "width",
            {"sOffset": "0", "a": f"{a:.4f}", "b": f"{b:.6f}", "c": f"{c:.6f}", "d": f"{d:.6f}"},
        )
        ET.SubElement(
            right_lane,
            "roadMark",
            {
                "sOffset": "0",
                "type": right_mark,
                "weight": "standard",
                "color": "standard",
                "width": "0.13",
            },
        )

        speed_ms = getattr(lane, "speed_limit", None)
        if speed_ms is not None and speed_ms > 0:
            ET.SubElement(
                right_lane,
                "speed",
                {"sOffset": "0", "max": f"{speed_ms * 3.6:.3f}", "unit": "km/h"},
            )

    def write_link(
        self,
        road_elem: ET.Element,
        predecessors: list,
        successors: list,
        topology: _TopologyBuilder,
    ) -> None:
        """Write the ``<link>`` element with predecessor / successor refs.

        Args:
            road_elem: The ``<road>`` element to append to.
            predecessors: Lane IDs preceding this road.
            successors: Lane IDs following this road.
            topology: Topology builder for junction lookups.
        """
        if not predecessors and not successors:
            return

        link = ET.SubElement(road_elem, "link")
        for pred_id in predecessors:
            jid = topology.junction_id_for(pred_id)
            if jid is not None:
                ET.SubElement(
                    link, "predecessor", {"elementType": "junction", "elementId": str(jid)}
                )
            else:
                ET.SubElement(
                    link,
                    "predecessor",
                    {"elementType": "road", "elementId": str(pred_id), "contactPoint": "end"},
                )
        for succ_id in successors:
            jid = topology.junction_id_for(succ_id)
            if jid is not None:
                ET.SubElement(link, "successor", {"elementType": "junction", "elementId": str(jid)})
            else:
                ET.SubElement(
                    link,
                    "successor",
                    {"elementType": "road", "elementId": str(succ_id), "contactPoint": "start"},
                )

    def write_junctions(self, root: ET.Element, topology: _TopologyBuilder, map_: Map) -> None:
        """Write ``<junction>`` elements for inferred junction groups.

        Args:
            root: Parent ``<OpenDRIVE>`` element.
            topology: Topology builder with ``.junction_groups``.
            map_: Tactics2D ``Map`` for lane existence checks.
        """
        for jid, group in enumerate(topology.junction_groups):
            junction = ET.SubElement(root, "junction", {"name": f"junction_{jid}", "id": str(jid)})
            conn_id = 0
            for lane_id in group:
                if map_.lanes.get(lane_id) is None:
                    continue
                for incoming in topology.predecessors.get(lane_id, []):
                    for outgoing in topology.successors.get(lane_id, []):
                        conn = ET.SubElement(
                            junction,
                            "connection",
                            {
                                "id": str(conn_id),
                                "incomingRoad": str(incoming),
                                "connectingRoad": str(lane_id),
                                "contactPoint": "start",
                            },
                        )
                        ET.SubElement(conn, "laneLink", {"from": "-1", "to": "-1"})
                        conn_id += 1

    @staticmethod
    def _get_centerline(lane) -> list:
        """Get the reference line for an OpenDRIVE road.

        In OpenDRIVE, the reference line is the left boundary of the
        rightmost lane section. Since we write a single lane with id=-1
        (right side), the reference line should be the lane's left_side
        boundary. This ensures adjacent lanes share boundaries without gaps.

        Args:
            lane: Tactics2D Lane object.

        Returns:
            List of (x, y) tuples for the reference line.
        """
        left = lane.left_side
        if left is None:
            return []
        if left.length < 1e-6:
            return []
        n = max(2, int(left.length / 0.1))
        return [
            (
                left.interpolate(left.length * i / (n - 1)).x,
                left.interpolate(left.length * i / (n - 1)).y,
            )
            for i in range(n)
        ]

    @staticmethod
    def _sample_widths(lane, n: int = 20) -> np.ndarray:
        """Sample lane width at *n* evenly spaced normalised positions.

        Args:
            lane: Tactics2D ``Lane`` object.
            n: Number of samples. Defaults to 20.

        Returns:
            1-D array of width values in metres.
        """
        left, right = lane.left_side, lane.right_side
        if left is None or right is None:
            return np.array([3.5])
        s_vals = np.linspace(0, 1, n)
        return np.array(
            [
                left.interpolate(s, normalized=True).distance(right.interpolate(s, normalized=True))
                for s in s_vals
            ]
        )

    @staticmethod
    def _fit_width(lane) -> tuple[float, float, float, float]:
        widths = XodrWriter._sample_widths(lane)
        n = len(widths)
        if n < 2:
            return (float(widths[0]), 0.0, 0.0, 0.0)
        length = min(lane.left_side.length, lane.right_side.length)
        s = np.linspace(0, length, n)
        coeffs = np.polyfit(s, widths, min(3, n - 1))
        coeffs = np.concatenate([np.zeros(4 - len(coeffs)), coeffs])
        d, c, b, a = coeffs
        return (float(a), float(b), float(c), float(d))

    @staticmethod
    def _get_xodr_roadmark(lane, map_: Map, side: str) -> str:
        """Reverse-map ``RoadLine.subtype`` to a xodr ``roadMark`` type.

        Args:
            lane: Tactics2D ``Lane`` object.
            map_: Tactics2D ``Map`` with ``.roadlines`` registry.
            side: ``"left"`` or ``"right"``.

        Returns:
            xodr roadMark type (``"broken"``, ``"solid"``,
            ``"solid solid"``), defaulting to ``"solid"``.
        """
        line_ids = getattr(lane, "line_ids", {})
        for lid in line_ids.get(side, []):
            rl = map_.roadlines.get(lid)
            if rl is None:
                continue
            subtype = getattr(rl, "subtype", None)
            if subtype in XodrWriter._SUBTYPE_TO_ROADMARK:
                return XodrWriter._SUBTYPE_TO_ROADMARK[subtype]
        return "solid"
