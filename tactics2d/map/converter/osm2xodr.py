# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenStreetMap Lanelet2 to OpenDRIVE converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.parser import OSMParser


class TopologyBuilder:
    """Builds predecessor/successor relationships and identifies junctions
    from a Tactics2D Map parsed from Lanelet2 format.

    The algorithm matches lane endpoints within a spatial tolerance. If a
    lane end-point is shared by more than two lanes, the group is flagged
    as a junction.
    """

    ENDPOINT_TOL = 0.5

    def __init__(self, map_):
        self._map = map_
        self.successors: dict[int, list[int]] = {}
        self.predecessors: dict[int, list[int]] = {}
        self.junction_groups: list[set[int]] = []
        self._build()

    def is_junction_lane(self, lane_id: int) -> bool:
        return any(lane_id in g for g in self.junction_groups)

    def junction_id_for(self, lane_id: int) -> int | None:
        for i, g in enumerate(self.junction_groups):
            if lane_id in g:
                return i
        return None

    def _endpoint(self, lane, which: str) -> tuple:
        coords = list(lane.left_side.coords)
        return coords[0] if which == "start" else coords[-1]

    def _dist(self, a: tuple, b: tuple) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _build(self):
        lanes = self._map.lanes
        ids = list(lanes.keys())

        for lid in ids:
            self.successors[lid] = []
            self.predecessors[lid] = []

        starts = {lid: self._endpoint(lanes[lid], "start") for lid in ids}
        ends = {lid: self._endpoint(lanes[lid], "end") for lid in ids}

        for a in ids:
            for b in ids:
                if a == b:
                    continue
                if self._dist(ends[a], starts[b]) < self.ENDPOINT_TOL:
                    if b not in self.successors[a]:
                        self.successors[a].append(b)
                    if a not in self.predecessors[b]:
                        self.predecessors[b].append(a)

        node_map: dict[tuple, list[tuple[int, str]]] = {}
        for lid in ids:
            for which, pt in [("start", starts[lid]), ("end", ends[lid])]:
                key = (round(pt[0] / self.ENDPOINT_TOL), round(pt[1] / self.ENDPOINT_TOL))
                node_map.setdefault(key, []).append((lid, which))

        for key, owners in node_map.items():
            if len(owners) < 3:
                continue
            merged = {lid for lid, _ in owners}
            remaining = []
            for g in self.junction_groups:
                if g & merged:
                    merged = merged | g
                else:
                    remaining.append(g)
            remaining.append(merged)
            self.junction_groups = remaining


class Osm2XodrConverter:
    """Converts a Lanelet2-annotated OSM file to OpenDRIVE (.xodr) format.

    Each Tactics2D Lane becomes an OpenDRIVE road. Predecessor/successor
    topology and junction areas are derived from lane endpoint proximity.
    Lane boundary subtypes are mapped to xodr roadMark types. Speed limits
    are converted from km/h to m/s.

    Example:
        >>> converter = Osm2XodrConverter()
        >>> converter.convert("map.osm", "map.xodr")
    """

    _MAX_SEG_LENGTH = 20.0

    def _get_centerline(self, lane) -> list:
        left, right = lane.left_side, lane.right_side
        if left is None or right is None:
            return []
        length = min(left.length, right.length)
        if length < 1e-6:
            return []
        n = max(2, int(length / 0.5))
        s_vals = np.linspace(0, 1, n)
        pts = []
        for s in s_vals:
            lp = left.interpolate(s, normalized=True)
            rp = right.interpolate(s, normalized=True)
            pts.append(((lp.x + rp.x) / 2, (lp.y + rp.y) / 2))
        return pts

    def _sample_widths(self, lane, n: int = 20) -> np.ndarray:
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

    def _width_poly(self, lane) -> tuple:
        """Fit a cubic polynomial to sampled lane widths.

        Returns:
            tuple: Coefficients (a, b, c, d) for a + b*s + c*s^2 + d*s^3.
        """
        widths = self._sample_widths(lane)
        n = len(widths)
        if n < 2:
            return (float(widths[0]), 0.0, 0.0, 0.0)
        s = np.linspace(0, 1, n)
        coeffs = np.polyfit(s, widths, min(3, n - 1))
        coeffs = np.concatenate([np.zeros(4 - len(coeffs)), coeffs])
        d, c, b, a = coeffs
        return (float(a), float(b), float(c), float(d))

    def _fit_param_poly3(self, pts: np.ndarray) -> dict | None:
        """Fit a paramPoly3 geometry to a polyline segment in local coordinates.

        Args:
            pts (np.ndarray): Polyline points in world coordinates, shape (N, 2).

        Returns:
            dict: Geometry parameters, or None if the segment is degenerate.
        """
        x0, y0 = pts[0]
        dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        hdg = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
        cos_h, sin_h = np.cos(hdg), np.sin(hdg)

        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total = float(seg_lengths.sum())
        if total < 1e-6:
            return None

        s_cum = np.concatenate([[0], np.cumsum(seg_lengths)])
        p = s_cum / total
        local = pts - pts[0]
        u = local[:, 0] * cos_h + local[:, 1] * sin_h
        v = -local[:, 0] * sin_h + local[:, 1] * cos_h

        coeffs_u = np.polyfit(p, u, 3)[::-1]
        coeffs_v = np.polyfit(p, v, 3)[::-1]

        return {
            "x": x0,
            "y": y0,
            "hdg": hdg,
            "length": total,
            "aU": coeffs_u[0],
            "bU": coeffs_u[1],
            "cU": coeffs_u[2],
            "dU": coeffs_u[3],
            "aV": coeffs_v[0],
            "bV": coeffs_v[1],
            "cV": coeffs_v[2],
            "dV": coeffs_v[3],
        }

    def _split_segments(self, pts: list) -> list[np.ndarray]:
        arr = np.array(pts)
        diffs = np.diff(arr, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0], np.cumsum(lens)])
        total = cum[-1]
        if total <= self._MAX_SEG_LENGTH:
            return [arr]
        n_segs = max(1, int(np.ceil(total / self._MAX_SEG_LENGTH)))
        breaks = np.linspace(0, total, n_segs + 1)
        segments = []
        for i in range(n_segs):
            s0, s1 = breaks[i], breaks[i + 1]
            mask = (cum >= s0 - 1e-9) & (cum <= s1 + 1e-9)
            seg = arr[mask]
            if len(seg) < 2:
                idx0 = np.searchsorted(cum, s0)
                idx1 = np.searchsorted(cum, s1)
                seg = arr[max(0, idx0) : min(len(arr), idx1 + 1)]
            if len(seg) >= 2:
                segments.append(seg)
        return segments if segments else [arr]

    def _roadmark_type(self, lane, map_, side: str) -> str:
        """Derive xodr roadMark type from the RoadLine subtype on the given side.

        Args:
            lane: Tactics2D Lane.
            map_: Tactics2D Map.
            side (str): 'left' or 'right'.

        Returns:
            str: xodr roadMark type string.
        """
        line_ids = getattr(lane, "line_ids", {})
        for lid in line_ids.get(side, []):
            rl = map_.roadlines.get(lid)
            if rl is None:
                continue
            subtype = getattr(rl, "subtype", None)
            if subtype == "dashed":
                return "broken"
            if subtype == "solid":
                return "solid"
            if subtype == "solid_solid":
                return "solid solid"
        return "solid"

    def _write_plan_view(self, road_elem: ET.Element, pts: list) -> None:
        plan_view = ET.SubElement(road_elem, "planView")
        s_offset = 0.0
        for seg in self._split_segments(pts):
            fit = self._fit_param_poly3(seg)
            if fit is None:
                continue
            geom = ET.SubElement(
                plan_view,
                "geometry",
                {
                    "s": f"{s_offset:.4f}",
                    "x": f"{fit['x']:.4f}",
                    "y": f"{fit['y']:.4f}",
                    "hdg": f"{fit['hdg']:.6f}",
                    "length": f"{fit['length']:.4f}",
                },
            )
            pp3 = ET.SubElement(geom, "paramPoly3", {"pRange": "normalized"})
            for k in ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV"):
                pp3.set(k, f"{fit[k]:.6f}")
            s_offset += fit["length"]

    def _write_lanes(self, road_elem: ET.Element, lane, map_) -> None:
        a, b, c, d = self._width_poly(lane)
        left_mark = self._roadmark_type(lane, map_, "left")
        right_mark = self._roadmark_type(lane, map_, "right")

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
        if speed_ms and speed_ms > 0:
            ET.SubElement(
                right_lane,
                "speed",
                {"sOffset": "0", "max": f"{speed_ms * 3.6:.3f}", "unit": "km/h"},
            )

    def _write_link(self, road_elem: ET.Element, topo: TopologyBuilder, lane_id: int) -> None:
        preds = topo.predecessors.get(lane_id, [])
        succs = topo.successors.get(lane_id, [])
        if not preds and not succs:
            return

        link = ET.SubElement(road_elem, "link")
        for pred_id in preds:
            jid = topo.junction_id_for(pred_id)
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
        for succ_id in succs:
            jid = topo.junction_id_for(succ_id)
            if jid is not None:
                ET.SubElement(link, "successor", {"elementType": "junction", "elementId": str(jid)})
            else:
                ET.SubElement(
                    link,
                    "successor",
                    {"elementType": "road", "elementId": str(succ_id), "contactPoint": "start"},
                )

    def _write_junctions(self, root: ET.Element, topo: TopologyBuilder, map_) -> None:
        for jid, group in enumerate(topo.junction_groups):
            junction = ET.SubElement(root, "junction", {"name": f"junction_{jid}", "id": str(jid)})
            conn_id = 0
            for lane_id in group:
                if map_.lanes.get(lane_id) is None:
                    continue
                for incoming in topo.predecessors.get(lane_id, []):
                    for outgoing in topo.successors.get(lane_id, []):
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

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a Lanelet2 OSM file to an OpenDRIVE xodr file.

        Args:
            input_path (str): Path to the input Lanelet2 .osm file.
            output_path (str): Path to the output .xodr file.

        Returns:
            str: The output file path.
        """
        map_ = OSMParser(lanelet2=True).parse(input_path)
        topo = TopologyBuilder(map_)

        root = ET.Element("OpenDRIVE")
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

        for lane_id, lane in map_.lanes.items():
            pts = self._get_centerline(lane)
            if len(pts) < 2:
                continue

            total_length = float(LineString(pts).length)
            jid = topo.junction_id_for(lane_id)

            road = ET.SubElement(
                root,
                "road",
                {
                    "name": str(lane_id),
                    "length": f"{total_length:.4f}",
                    "id": str(lane_id),
                    "junction": str(jid) if jid is not None else "-1",
                },
            )

            self._write_link(road, topo, lane_id)

            location = getattr(lane, "location", "urban")
            road_type = "motorway" if location == "nonurban" else "town"
            ET.SubElement(road, "type", {"s": "0.0", "type": road_type})

            self._write_plan_view(road, pts)
            ET.SubElement(road, "elevationProfile")
            ET.SubElement(road, "lateralProfile")
            self._write_lanes(road, lane, map_)

        self._write_junctions(root, topo, map_)

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
