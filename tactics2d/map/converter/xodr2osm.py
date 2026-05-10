# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenDRIVE to OpenStreetMap Lanelet2 converter implementation."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from tactics2d.interpolator.param_poly3 import ParamPoly3


def _unit_left_normals(pts: np.ndarray) -> np.ndarray:
    """Return unit left-pointing normals for every sample of a polyline."""
    tangents = np.empty_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.where(norms < 1e-12, 1.0, norms)
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _eval_cubic_poly(records: list, s: float, s_key: str = "s") -> float:
    """Evaluate a piecewise cubic polynomial at arc-length s."""
    if not records:
        return 0.0
    active = records[0]
    for rec in records:
        if s >= rec[s_key]:
            active = rec
        else:
            break
    ds = s - active[s_key]
    return active["a"] + active["b"] * ds + active["c"] * ds**2 + active["d"] * ds**3


def _offset_polyline(ref_pts: np.ndarray, normals: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Offset ref_pts laterally by signed distances t along normals."""
    return ref_pts + t[:, np.newaxis] * normals


class _LaneGeom:
    """Geometry and metadata for a single xodr lane, ready for OSM output."""

    __slots__ = (
        "left_boundary",
        "right_boundary",
        "left_mark",
        "right_mark",
        "lane_type",
        "road_type",
        "speed_kmh",
    )

    def __init__(self):
        self.left_boundary = np.empty((0, 2))
        self.right_boundary = np.empty((0, 2))
        self.left_mark = "solid"
        self.right_mark = "solid"
        self.lane_type = "driving"
        self.road_type = "town"
        self.speed_kmh = None


class _XodrReader:
    """Parses an OpenDRIVE XML tree into a list of :class:`_LaneGeom` objects.

    Uses the same cumulative lateral-offset approach as XODRParser: all
    boundaries are computed from the road reference line so width polynomials
    are evaluated at the correct arc-length s. Every individual xodr lane
    produces one _LaneGeom (one Lanelet2 lanelet).
    """

    _SAMPLES_PER_METRE = 5

    _MARK_TO_SUBTYPE: dict[str, str] = {
        "solid": "solid",
        "solid solid": "solid_solid",
        "broken": "dashed",
        "none": "virtual",
        "curb": "solid",
        "edge": "solid",
    }

    def read(self, root: ET.Element) -> list[_LaneGeom]:
        lanes = []
        for road_elem in root.findall("road"):
            lanes.extend(self._parse_road(road_elem))
        return lanes

    def _parse_road(self, road_elem: ET.Element) -> list[_LaneGeom]:
        road_type = "town"
        type_elem = road_elem.find("type")
        if type_elem is not None:
            road_type = type_elem.get("type", "town")

        ref_pts, s_arr = self._sample_ref_line(road_elem.find("planView"))
        if ref_pts is None or len(ref_pts) < 2:
            logging.warning("Road %s has no usable planView; skipped.", road_elem.get("id", "?"))
            return []

        ref_normals = _unit_left_normals(ref_pts)

        lanes_elem = road_elem.find("lanes")
        if lanes_elem is None:
            return []

        lo_records = self._parse_poly_records(lanes_elem, "laneOffset", s_key="s")
        lo_t = np.array([_eval_cubic_poly(lo_records, float(s), s_key="s") for s in s_arr])

        results: list[_LaneGeom] = []
        for ls_elem in lanes_elem.findall("laneSection"):
            results.extend(
                self._parse_lane_section(ls_elem, road_type, ref_pts, s_arr, ref_normals, lo_t)
            )
        return results

    def _parse_lane_section(
        self,
        ls_elem: ET.Element,
        road_type: str,
        ref_pts: np.ndarray,
        s_arr: np.ndarray,
        ref_normals: np.ndarray,
        lo_t: np.ndarray,
    ) -> list[_LaneGeom]:
        ls_s0 = float(ls_elem.get("s", 0.0))
        mask = s_arr >= ls_s0 - 1e-6
        if not np.any(mask):
            return []

        seg_pts = ref_pts[mask]
        seg_s = s_arr[mask]
        seg_n = ref_normals[mask]
        seg_lo = lo_t[mask]

        center_mark = "virtual"
        center_elem = ls_elem.find("center")
        if center_elem is not None:
            lane0 = center_elem.find("lane")
            if lane0 is not None:
                rm = lane0.find("roadMark")
                if rm is not None:
                    center_mark = self._MARK_TO_SUBTYPE.get(rm.get("type", "none"), "virtual")

        results: list[_LaneGeom] = []
        for side_tag, sign in (("left", 1), ("right", -1)):
            side_elem = ls_elem.find(side_tag)
            if side_elem is None:
                continue

            lane_nodes = sorted(side_elem.findall("lane"), key=lambda n: abs(int(n.attrib["id"])))
            cumulative_t = seg_lo.copy()
            inner_mark = center_mark

            for ln in lane_nodes:
                geom, outer_mark, cumulative_t = self._build_lane_geom(
                    ln, road_type, sign, seg_pts, seg_s, seg_n, cumulative_t, inner_mark
                )
                if geom is not None:
                    results.append(geom)
                inner_mark = outer_mark

        return results

    def _build_lane_geom(
        self,
        ln: ET.Element,
        road_type: str,
        sign: int,
        ref_pts: np.ndarray,
        ref_s: np.ndarray,
        ref_normals: np.ndarray,
        inner_t: np.ndarray,
        inner_mark: str,
    ) -> tuple[_LaneGeom | None, str, np.ndarray]:
        width_records = self._parse_poly_records(ln, "width", s_key="sOffset")
        ls_s0 = float(ref_s[0])
        width_at_s = np.array(
            [_eval_cubic_poly(width_records, float(s) - ls_s0, s_key="sOffset") for s in ref_s]
        )
        outer_t = inner_t + sign * width_at_s

        rm = ln.find("roadMark")
        outer_mark = self._MARK_TO_SUBTYPE.get(
            rm.get("type", "solid") if rm is not None else "solid", "solid"
        )

        lane_type = ln.get("type", "driving")
        if not np.any(width_at_s > 0.01):
            return None, outer_mark, outer_t

        inner_pts = _offset_polyline(ref_pts, ref_normals, inner_t)
        outer_pts = _offset_polyline(ref_pts, ref_normals, outer_t)

        if sign > 0:
            left_pts, right_pts = outer_pts, inner_pts
            left_mark, right_mark = outer_mark, inner_mark
        else:
            left_pts, right_pts = inner_pts, outer_pts
            left_mark, right_mark = inner_mark, outer_mark

        speed_elem = ln.find("speed")
        speed_kmh = None
        if speed_elem is not None:
            val = float(speed_elem.get("max", 0))
            unit = speed_elem.get("unit", "km/h")
            speed_kmh = val if unit == "km/h" else val * 3.6

        geom = _LaneGeom()
        geom.left_boundary = left_pts
        geom.right_boundary = right_pts
        geom.left_mark = left_mark
        geom.right_mark = right_mark
        geom.lane_type = lane_type
        geom.road_type = road_type
        geom.speed_kmh = speed_kmh
        return geom, outer_mark, outer_t

    def _sample_ref_line(
        self, plan_view: ET.Element | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if plan_view is None:
            return None, None

        all_pts: list[np.ndarray] = []
        all_s: list[np.ndarray] = []
        for geom_elem in plan_view.findall("geometry"):
            s0 = float(geom_elem.get("s", 0))
            length = float(geom_elem.get("length", 0))
            pts = self._sample_geometry(geom_elem)
            if pts is None or len(pts) < 2:
                continue
            s_seg = np.linspace(s0, s0 + length, len(pts))
            if all_pts:
                pts = pts[1:]
                s_seg = s_seg[1:]
            all_pts.append(pts)
            all_s.append(s_seg)

        if not all_pts:
            return None, None
        return np.vstack(all_pts), np.concatenate(all_s)

    def _sample_geometry(self, geom_elem: ET.Element) -> np.ndarray | None:
        x = float(geom_elem.get("x", 0))
        y = float(geom_elem.get("y", 0))
        hdg = float(geom_elem.get("hdg", 0))
        length = float(geom_elem.get("length", 0))
        if length < 1e-6:
            return None
        n = max(2, int(length * self._SAMPLES_PER_METRE))

        pp3 = geom_elem.find("paramPoly3")
        if pp3 is not None:
            kwargs = {
                k: float(pp3.get(k, 0)) for k in ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV")
            }
            return ParamPoly3.get_curve(
                length=length,
                start_point=(x, y),
                heading=hdg,
                p_range_type=pp3.get("pRange", "normalized"),
                **kwargs,
            )

        if geom_elem.find("line") is not None:
            t = np.linspace(0, length, n)
            return np.column_stack([x + t * np.cos(hdg), y + t * np.sin(hdg)])

        arc = geom_elem.find("arc")
        if arc is not None:
            curvature = float(arc.get("curvature", 0))
            t = np.linspace(0, length, n)
            if abs(curvature) < 1e-9:
                return np.column_stack([x + t * np.cos(hdg), y + t * np.sin(hdg)])
            radius = 1.0 / curvature
            cx = x - radius * np.sin(hdg)
            cy = y + radius * np.cos(hdg)
            angles = hdg - np.pi / 2 + t * curvature
            return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])

        logging.warning("Unsupported geometry at (%.2f, %.2f); skipped.", x, y)
        return None

    @staticmethod
    def _parse_poly_records(parent: ET.Element, tag: str, s_key: str) -> list:
        records = []
        for elem in parent.findall(tag):
            try:
                records.append(
                    {
                        s_key: float(elem.get(s_key, 0.0)),
                        "a": float(elem.get("a", 0)),
                        "b": float(elem.get("b", 0)),
                        "c": float(elem.get("c", 0)),
                        "d": float(elem.get("d", 0)),
                    }
                )
            except (ValueError, TypeError):
                pass
        records.sort(key=lambda r: r[s_key])
        return records


class _OsmWriter:
    """Writes a list of :class:`_LaneGeom` objects to a Lanelet2 OSM XML tree.

    All element types share a single decrementing ID counter so they never
    collide in the Map.ids namespace.
    """

    # Inverse of OSMParser._load_nodes_no_proj with origin (lat0=0, lon0=0).
    _M_PER_DEG_LON = 111320.0
    _M_PER_DEG_LAT = 110540.0

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

    def __init__(self):
        self._next_id = -1

    def _alloc(self) -> int:
        nid = self._next_id
        self._next_id -= 1
        return nid

    def build(self, lanes: list[_LaneGeom]) -> ET.Element:
        root = ET.Element("osm", {"version": "0.6", "generator": "Tactics2D xodr2osm"})
        speed_relations: list[tuple[int, float, list[int]]] = []

        for lane in lanes:
            left_wid, right_wid = self._write_boundary_ways(
                root, lane.left_boundary, lane.right_boundary, lane.left_mark, lane.right_mark
            )
            rel_id = self._write_lanelet_relation(root, lane, left_wid, right_wid)
            if lane.speed_kmh is not None and lane.speed_kmh > 0:
                speed_relations.append((self._alloc(), lane.speed_kmh, [rel_id]))

        for reg_id, kmh, lane_ids in speed_relations:
            self._write_speed_regulatory(root, reg_id, kmh, lane_ids)

        return root

    def _write_nodes(self, root: ET.Element, coords: np.ndarray) -> list[int]:
        node_ids = []
        for x, y in coords:
            nid = self._alloc()
            ET.SubElement(
                root,
                "node",
                {
                    "id": str(nid),
                    "action": "modify",
                    "visible": "true",
                    "lat": f"{y / self._M_PER_DEG_LAT:.7f}",
                    "lon": f"{x / self._M_PER_DEG_LON:.7f}",
                },
            )
            node_ids.append(nid)
        return node_ids

    def _write_way(
        self, root: ET.Element, way_id: int, node_ids: list[int], type_: str, subtype: str
    ) -> None:
        way = ET.SubElement(root, "way", {"id": str(way_id), "action": "modify", "visible": "true"})
        for nid in node_ids:
            ET.SubElement(way, "nd", {"ref": str(nid)})
        ET.SubElement(way, "tag", {"k": "type", "v": type_})
        ET.SubElement(way, "tag", {"k": "subtype", "v": subtype})

    def _write_boundary_ways(
        self,
        root: ET.Element,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        left_mark: str,
        right_mark: str,
    ) -> tuple[int, int]:
        left_nids = self._write_nodes(root, left_coords)
        right_nids = self._write_nodes(root, right_coords)
        left_wid, right_wid = self._alloc(), self._alloc()
        self._write_way(root, left_wid, left_nids, "line_thin", left_mark)
        self._write_way(root, right_wid, right_nids, "line_thin", right_mark)
        return left_wid, right_wid

    def _write_lanelet_relation(
        self, root: ET.Element, lane: _LaneGeom, left_wid: int, right_wid: int
    ) -> int:
        rel_id = self._alloc()
        rel = ET.SubElement(
            root, "relation", {"id": str(rel_id), "action": "modify", "visible": "true"}
        )
        ET.SubElement(rel, "member", {"type": "way", "ref": str(left_wid), "role": "left"})
        ET.SubElement(rel, "member", {"type": "way", "ref": str(right_wid), "role": "right"})
        ET.SubElement(rel, "tag", {"k": "type", "v": "lanelet"})
        ET.SubElement(
            rel,
            "tag",
            {"k": "subtype", "v": self._LANE_TYPE_TO_SUBTYPE.get(lane.lane_type, "road")},
        )
        ET.SubElement(
            rel,
            "tag",
            {"k": "location", "v": self._ROAD_TYPE_TO_LOCATION.get(lane.road_type, "urban")},
        )
        return rel_id

    def _write_speed_regulatory(
        self, root: ET.Element, reg_id: int, speed_kmh: float, lane_rel_ids: list[int]
    ) -> None:
        reg = ET.SubElement(
            root, "relation", {"id": str(reg_id), "action": "modify", "visible": "true"}
        )
        for lid in lane_rel_ids:
            ET.SubElement(reg, "member", {"type": "relation", "ref": str(lid), "role": "refers"})
        ET.SubElement(reg, "tag", {"k": "type", "v": "regulatory_element"})
        ET.SubElement(reg, "tag", {"k": "subtype", "v": "speed_limit"})
        ET.SubElement(reg, "tag", {"k": "speed_limit", "v": f"{speed_kmh:.1f}"})
        ET.SubElement(reg, "tag", {"k": "speed_limit_mandatory", "v": "yes"})


class Xodr2OsmConverter:
    """Converts an OpenDRIVE (.xodr) file to a Lanelet2-annotated OSM (.osm) file.

    Each xodr lane becomes one Lanelet2 lanelet relation. Boundaries are built
    using the same cumulative lateral-offset method as XODRParser so geometry
    is consistent with the rest of Tactics2D.

    Example:
        >>> converter = Xodr2OsmConverter()
        >>> converter.convert("map.xodr", "map.osm")
    """

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert an OpenDRIVE xodr file to a Lanelet2 OSM file.

        Args:
            input_path (str): Path to the input .xodr file.
            output_path (str): Path to the output .osm file.

        Returns:
            str: The output file path.
        """
        xodr_root = ET.parse(input_path).getroot()

        lanes = _XodrReader().read(xodr_root)
        logging.info("Parsed %d lanes from %s.", len(lanes), input_path)

        osm_root = _OsmWriter().build(lanes)

        xml_str = minidom.parseString(ET.tostring(osm_root, encoding="unicode")).toprettyxml(
            indent="    "
        )
        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logging.info("Written %s.", output_path)
        return output_path
