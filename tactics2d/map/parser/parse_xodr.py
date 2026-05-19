# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import CRS
from shapely.affinity import affine_transform, rotate
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from tactics2d.geometry import Circle
from tactics2d.interpolator import ParamPoly3, Spiral
from tactics2d.map.element import Area, Junction, Lane, Map, RoadLine


def _unit_left_normals(points: np.ndarray) -> np.ndarray:
    """Return the unit left-pointing normal at every sample of a polyline.

    Uses central finite differences in the interior and one-sided differences
    at the endpoints, giving O(h^2) accuracy in the interior.

    Args:
        points (np.ndarray): Input polyline samples. Shape is (N, 2).

    Returns:
        np.ndarray: Unit left-pointing normals. Shape is (N, 2). Each row is a
            unit vector perpendicular to the polyline tangent, rotated 90 degrees
            counter-clockwise (pointing to the left of travel).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.empty((0, 2))
    if n == 1:
        return np.zeros((1, 2))

    tangents = np.empty_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents /= norms

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def _signed_curvature(points: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """Estimate the signed curvature kappa(s) of a polyline at every sample.

    Uses the cross-product formula with central finite differences. This is
    used as a fallback when analytic curvature is not available.

    Args:
        points (np.ndarray): Input polyline samples. Shape is (N, 2).
        s_vals (np.ndarray): Arc-length parameter values. Shape is (N,).

    Returns:
        np.ndarray: Signed curvature at each sample. Shape is (N,).
    """
    pts = np.asarray(points, dtype=np.float64)
    s = np.asarray(s_vals, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return np.zeros(n)

    ds2 = np.maximum(s[2:] - s[:-2], 1e-12)
    dx = np.empty(n)
    dy = np.empty(n)
    dx[1:-1] = (pts[2:, 0] - pts[:-2, 0]) / ds2
    dy[1:-1] = (pts[2:, 1] - pts[:-2, 1]) / ds2
    dx[0] = dx[1]
    dy[0] = dy[1]
    dx[-1] = dx[-2]
    dy[-1] = dy[-2]

    ds_vec = np.maximum(np.gradient(s), 1e-12)
    d2x = np.gradient(dx) / ds_vec
    d2y = np.gradient(dy) / ds_vec

    speed_sq = np.maximum(dx**2 + dy**2, 1e-12)
    return (dx * d2y - dy * d2x) / speed_sq**1.5


def _eval_cubic_poly(records: list, s: float, s_key: str = "s") -> float:
    """Evaluate a piecewise cubic polynomial at arc-length s.

    Args:
        records (list): List of dicts with keys s_key, a, b, c, d.
        s (float): Arc-length value at which to evaluate.
        s_key (str): Key for the segment start value. Defaults to "s".

    Returns:
        float: Evaluated polynomial value. Returns 0.0 when records is empty.
    """
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


def _build_offset_polyline(
    ref_pts: np.ndarray,
    ref_s: np.ndarray,
    ref_normals: np.ndarray,
    t_vals: np.ndarray,
    ref_kappa: np.ndarray = None,
) -> np.ndarray:
    """Build an offset polyline with curvature-aware clamping.

    The offset satisfies P_out = P_ref + t_effective * n_ref, where
    t_effective is clamped so that the arc-length scaling factor
    (1 - kappa * t) stays strictly positive, preventing swallowtail
    self-intersection on tight curves.

    The clamping boundary is derived analytically: setting
    1 - kappa * t = epsilon gives t_limit = (1 - epsilon) / kappa.
    A safety margin of epsilon = 0.01 is used (1% of the collapse
    boundary), which is tight enough to preserve geometry fidelity
    while guaranteeing no self-intersection.

    When ref_kappa is provided (analytic curvature computed directly
    from geometry coefficients), it is used in preference to the
    finite-difference estimate from _signed_curvature, eliminating
    numerical noise at segment boundaries.

    Args:
        ref_pts (np.ndarray): Reference-line sample points. Shape is (N, 2).
        ref_s (np.ndarray): Arc-length values along the reference line. Shape is (N,).
        ref_normals (np.ndarray): Unit left-pointing normals. Shape is (N, 2).
        t_vals (np.ndarray): Signed lateral offset at each sample. Shape is (N,).
            Positive values are left of the reference line.
        ref_kappa (np.ndarray, optional): Analytic signed curvature at each
            sample. Shape is (N,). When provided, skips numerical estimation.

    Returns:
        np.ndarray: Offset polyline points in world coordinates. Shape is (N, 2).
    """
    kappa = ref_kappa if ref_kappa is not None else _signed_curvature(ref_pts, ref_s)
    correction = 1.0 - kappa * t_vals
    collapsed = correction <= 0.0

    if np.any(collapsed):
        with np.errstate(divide="ignore", invalid="ignore"):
            t_limit = np.where(np.abs(kappa) > 1e-6, 0.99 / kappa, t_vals)
        t_effective = np.where(collapsed, t_limit, t_vals)
    else:
        t_effective = t_vals

    return ref_pts + t_effective[:, np.newaxis] * ref_normals


def _sanitise_linestring(pts: np.ndarray, dedup_tolerance: float = 0.02) -> np.ndarray:
    """Remove near-duplicate consecutive points and repair self-intersections.

    Args:
        pts (np.ndarray): Input polyline points. Shape is (N, 2).
        dedup_tolerance (float): Minimum chord length to retain a point.

    Returns:
        np.ndarray: Cleaned polyline points. Shape is (M, 2) where M <= N.
    """
    if len(pts) < 2:
        return pts

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], dists > dedup_tolerance])
    pts = pts[keep]

    if len(pts) < 2:
        return pts

    if len(pts) >= 3:
        vecs = np.diff(pts, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        vecs = vecs / norms
        dots = np.sum(vecs[:-1] * vecs[1:], axis=1)
        keep = np.concatenate([[True], dots > 0.0, [True]])
        pts = pts[keep]

    if len(pts) < 2:
        return pts

    ls = LineString(pts)
    if not ls.is_valid:
        fixed = make_valid(ls)
        if fixed.geom_type == "LineString":
            pts = np.array(fixed.coords, dtype=np.float64)
        elif fixed.geom_type == "MultiLineString":
            longest = max(fixed.geoms, key=lambda g: g.length)
            if longest.length >= 0.8 * ls.length:
                pts = np.array(longest.coords, dtype=np.float64)

    return pts


class XODRParser:
    """Parser for the OpenDRIVE (.xodr) map format.

    Converts an OpenDRIVE file into a Tactics2D Map object containing lanes,
    road-mark lines, junctions, and optional area objects (crosswalks, etc.).

    The parser follows the canonical offset-curve approach used by libopendrive
    and esmini:

        P_border(s, t) = P_ref(s) + t * n_ref(s)

    where s is the arc-length along the reference line, t is the signed lateral
    distance, and n_ref(s) is the unit left-pointing normal. Lane widths are
    accumulated outward from the centre lane so that all boundaries share the
    same s-sample array. Analytic curvature is computed directly from geometry
    coefficients (paramPoly3, arc, spiral, poly3) and passed to the offset
    builder, eliminating the numerical noise introduced by finite-difference
    curvature estimation on sampled polylines.

    Example:
    ```python
    from tactics2d.map.parser import XODRParser

    parser = XODRParser()
    map_ = parser.parse("path/to/map.xodr")
    ```
    """

    _ROADMARK_SUBTYPE: dict = {
        "botts dots": "",
        "broken broken": "dashed_dashed",
        "broken solid": "dashed_solid",
        "broken": "dashed",
        "solid broken": "solid_dashed",
        "solid solid": "solid_solid",
        "solid": "solid",
    }

    def __init__(self) -> None:
        self._id_counter = 0

    def _next_id(self) -> int:
        uid = self._id_counter
        self._id_counter += 1
        return uid

    def _sample_line(self, node: ET.Element) -> tuple[list, list]:
        """Sample a straight-line geometry at 0.1 m intervals.

        Args:
            node (ET.Element): A ``<geometry>`` element whose child is ``<line>``.

        Returns:
            tuple[list, list]: World-coordinate point list and curvature list
                (all zeros for a straight line).
        """
        x0 = float(node.attrib["x"])
        y0 = float(node.attrib["y"])
        hdg = float(node.attrib["hdg"])
        L = float(node.attrib["length"])

        if L <= 0.0:
            return [(x0, y0)], [0.0]

        n = max(2, int(L / 0.1) + 1)
        s_arr = np.linspace(0.0, L, n)
        pts = [(x0 + s * np.cos(hdg), y0 + s * np.sin(hdg)) for s in s_arr]
        kappas = [0.0] * len(pts)
        return pts, kappas

    def _sample_spiral(self, node: ET.Element) -> tuple[list, list]:
        """Sample a Euler spiral geometry using the Spiral integrator.

        Curvature varies linearly from curvStart to curvEnd along the arc.

        Args:
            node (ET.Element): A ``<geometry>`` element whose child is ``<spiral>``.

        Returns:
            tuple[list, list]: World-coordinate point list and analytic curvature list.
        """
        x0 = float(node.attrib["x"])
        y0 = float(node.attrib["y"])
        hdg = float(node.attrib["hdg"])
        L = float(node.attrib["length"])
        k0 = float(node.find("spiral").attrib["curvStart"])
        k1 = float(node.find("spiral").attrib["curvEnd"])

        if L < 1e-6:
            return [(x0, y0)], [k0]

        gamma = (k1 - k0) / L
        pts_arr = Spiral.get_curve(L, [x0, y0], hdg, k0, gamma)
        pts = [(x, y) for x, y in pts_arr]
        kappas = np.linspace(k0, k1, len(pts)).tolist()
        return pts, kappas

    def _sample_arc(self, node: ET.Element) -> tuple[list, list]:
        """Sample a constant-curvature arc geometry at 0.1 m intervals.

        Falls back to straight-line sampling when curvature is negligible.

        Args:
            node (ET.Element): A ``<geometry>`` element whose child is ``<arc>``.

        Returns:
            tuple[list, list]: World-coordinate point list and analytic curvature list
                (constant value equal to the arc curvature).
        """
        x0 = float(node.attrib["x"])
        y0 = float(node.attrib["y"])
        hdg = float(node.attrib["hdg"])
        L = float(node.attrib["length"])
        k = float(node.find("arc").attrib["curvature"])

        if abs(k) < 1e-9:
            return self._sample_line(node)

        r = abs(1.0 / k)
        center, radius = Circle.get_circle(
            tangent_point=np.array([x0, y0]),
            tangent_heading=hdg,
            radius=r,
            side="L" if k > 0 else "R",
        )
        arc_angle = L / radius
        start_angle = hdg - np.pi / 2.0 * np.sign(k)
        clockwise = k < 0

        pts_arr = Circle.get_arc(
            center_point=center,
            radius=radius,
            delta_angle=arc_angle,
            start_angle=start_angle,
            clockwise=clockwise,
            step_size=0.1,
        )
        pts = [(x, y) for x, y in pts_arr.tolist()]
        kappas = [k] * len(pts)
        return pts, kappas

    def _sample_poly3(self, node: ET.Element) -> tuple[list, list]:
        """Sample a cubic polynomial geometry (local u/v frame) at 0.1 m intervals.

        Analytic curvature is computed from the polynomial coefficients using
        the Frenet-Serret formula in the local frame where U is the arc-length
        parameter (U'=1, U''=0).

        Args:
            node (ET.Element): A ``<geometry>`` element whose child is ``<poly3>``.

        Returns:
            tuple[list, list]: World-coordinate point list and analytic curvature list.
        """
        x0 = float(node.attrib["x"])
        y0 = float(node.attrib["y"])
        hdg = float(node.attrib["hdg"])
        L = float(node.attrib["length"])
        p = node.find("poly3")
        a, b, c, d = (float(p.attrib[k]) for k in ("a", "b", "c", "d"))

        n = max(2, int(L / 0.1) + 1)
        u = np.linspace(0.0, L, n)
        v = a + b * u + c * u**2 + d * u**3

        cos_h, sin_h = np.cos(hdg), np.sin(hdg)
        xs = x0 + u * cos_h - v * sin_h
        ys = y0 + u * sin_h + v * cos_h
        pts = list(zip(xs.tolist(), ys.tolist()))

        # In local frame: U'=1, U''=0, V'=dv/du, V''=d²v/du²
        dv = b + 2 * c * u + 3 * d * u**2
        d2v = 2 * c + 6 * d * u
        speed_sq = np.maximum(1.0 + dv**2, 1e-12)
        kappas = (d2v / speed_sq**1.5).tolist()
        return pts, kappas

    def _sample_param_poly3(self, node: ET.Element) -> tuple[list, list]:
        """Sample a parametric cubic polynomial geometry.

        Analytic curvature is computed directly from the polynomial
        coefficients using the Frenet-Serret formula:

            kappa(p) = (U'V'' - V'U'') / (U'² + V'²)^(3/2)

        where U', V' are first derivatives and U'', V'' are second derivatives
        of the U and V polynomials with respect to the parameter p. This
        avoids the numerical noise introduced by finite-difference estimation
        on sampled polylines, especially at segment boundaries.

        Args:
            node (ET.Element): A ``<geometry>`` element whose child is
                ``<paramPoly3>``.

        Returns:
            tuple[list, list]: World-coordinate point list and analytic curvature list.
        """
        x0 = float(node.attrib["x"])
        y0 = float(node.attrib["y"])
        hdg = float(node.attrib["hdg"])
        L = float(node.attrib["length"])
        p = node.find("paramPoly3")
        p_range = p.attrib.get("pRange", "normalized")
        c = {k: float(p.attrib[k]) for k in ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV")}

        pts_arr = ParamPoly3.get_curve(
            L,
            (x0, y0),
            hdg,
            c["aU"],
            c["bU"],
            c["cU"],
            c["dU"],
            c["aV"],
            c["bV"],
            c["cV"],
            c["dV"],
            p_range,
        )
        pts = [(x, y) for x, y in pts_arr]

        p_max = L if p_range == "arcLength" else 1.0
        n = len(pts)
        p_vals = np.linspace(0.0, p_max, n)

        # Analytic first and second derivatives of U(p) and V(p)
        dU = c["bU"] + 2 * c["cU"] * p_vals + 3 * c["dU"] * p_vals**2
        dV = c["bV"] + 2 * c["cV"] * p_vals + 3 * c["dV"] * p_vals**2
        d2U = 2 * c["cU"] + 6 * c["dU"] * p_vals
        d2V = 2 * c["cV"] + 6 * c["dV"] * p_vals

        speed_sq = np.maximum(dU**2 + dV**2, 1e-12)
        # Curvature in local (U, V) frame; rotation to world frame preserves
        # curvature magnitude and sign.
        kappas = ((dU * d2V - dV * d2U) / speed_sq**1.5).tolist()
        return pts, kappas

    def _sample_geometry(self, node: ET.Element) -> tuple[list, list]:
        """Dispatch to the appropriate sampler and remove trailing duplicate.

        Args:
            node (ET.Element): A ``<geometry>`` element from ``<planView>``.

        Returns:
            tuple[list, list]: World-coordinate point list and curvature list.
                Both lists are empty for unknown geometry types.
        """
        if node.find("line") is not None:
            pts, kappas = self._sample_line(node)
        elif node.find("spiral") is not None:
            pts, kappas = self._sample_spiral(node)
        elif node.find("arc") is not None:
            pts, kappas = self._sample_arc(node)
        elif node.find("poly3") is not None:
            pts, kappas = self._sample_poly3(node)
        elif node.find("paramPoly3") is not None:
            pts, kappas = self._sample_param_poly3(node)
        else:
            logging.warning("Unknown geometry type in planView; skipping.")
            return [], []

        if len(pts) >= 2 and pts[-1] == pts[-2]:
            pts.pop()
            kappas.pop()
        return pts, kappas

    def load_header(self, node: ET.Element) -> tuple:
        """Parse the ``<header>`` element of an OpenDRIVE file.

        Args:
            node (ET.Element): The ``<header>`` XML element.

        Returns:
            tuple:
                info (dict): Header attributes including name, version, date,
                    and geographic bounding box.
                projector (CRS or None): A pyproj CRS object parsed from the
                    ``<geoReference>`` child element, or None if absent.
        """
        attrs = (
            "revMajor",
            "revMinor",
            "name",
            "version",
            "date",
            "north",
            "south",
            "east",
            "west",
            "vendor",
        )
        info = {a: node.get(a) for a in attrs}

        projector = None
        geo = node.find("geoReference")
        if geo is not None and geo.text:
            try:
                projector = CRS.from_proj4(geo.text)
            except Exception:
                logging.warning("Could not parse geoReference CRS.")

        return info, projector

    def _make_roadline(self, geometry: list | LineString, rm_node: ET.Element) -> RoadLine:
        """Construct a RoadLine from coordinate geometry and a ``<roadMark>`` node.

        Args:
            geometry (list or LineString): Coordinate sequence for the line.
            rm_node (ET.Element or None): The ``<roadMark>`` XML element.

        Returns:
            RoadLine: A fully initialised RoadLine object.
        """
        rm = rm_node

        rm_type = rm.attrib.get("type", "none") if rm is not None else "none"

        if rm_type == "none":
            line_type = "virtual"
            subtype = None
        elif rm_type == "curb":
            line_type = "curbstone"
            subtype = None
        elif rm_type == "edge":
            line_type = "road_border"
            subtype = None
        elif rm_type == "grass":
            line_type = "grass"
            subtype = None
        else:
            width_val = float(rm.attrib.get("width", "0.15")) if rm is not None else 0.15
            line_type = "line_thin" if width_val <= 0.15 else "line_thick"
            subtype = self._ROADMARK_SUBTYPE.get(rm_type)

        color_raw = rm.attrib.get("color") if rm is not None else None
        color_map = {"standard": "white", "violet": "purple"}
        color = color_map.get(color_raw, color_raw)

        lc_str = rm.attrib.get("laneChange", "both") if rm is not None else "both"
        lc_map = {
            "none": None,
            "both": (True, True),
            "increase": (True, False),
            "decrease": (False, True),
        }
        lane_change = lc_map.get(lc_str, (True, True))

        geom = geometry if isinstance(geometry, LineString) else LineString(geometry)

        return RoadLine(
            id_=self._next_id(),
            geometry=geom,
            type_=line_type,
            subtype=subtype,
            color=color,
            width=rm.attrib.get("width") if rm is not None else None,
            height=rm.attrib.get("height") if rm is not None else None,
            lane_change=lane_change,
            temporary=False,
            custom_tags={"weight": rm.attrib.get("weight") if rm is not None else None},
        )

    def _parse_width_records(self, lane_node: ET.Element) -> list:
        """Return width-polynomial records sorted by sOffset."""
        records = []
        for w in lane_node.findall("width"):
            try:
                records.append(
                    {
                        "sOffset": float(w.attrib.get("sOffset", 0.0)),
                        "a": float(w.attrib["a"]),
                        "b": float(w.attrib["b"]),
                        "c": float(w.attrib["c"]),
                        "d": float(w.attrib["d"]),
                    }
                )
            except KeyError:
                pass
        records.sort(key=lambda r: r["sOffset"])
        return records

    def _load_lane(
        self,
        lane_node: ET.Element,
        type_node: ET.Element,
        ref_pts: np.ndarray,
        ref_s: np.ndarray,
        ref_normals: np.ndarray,
        inner_t: np.ndarray,
        inner_line_id: int,
        road_id: str = "",
        ref_kappa: np.ndarray = None,
    ):
        """Build one lane and its outer boundary from the shared reference line.

        Args:
            lane_node (ET.Element): The ``<lane>`` XML element to parse.
            type_node (ET.Element): The ``<type>`` XML element of the parent road.
            ref_pts (np.ndarray): Reference-line sample points. Shape is (N, 2).
            ref_s (np.ndarray): Arc-length values along the reference line. Shape is (N,).
            ref_normals (np.ndarray): Unit left-pointing normals. Shape is (N, 2).
            inner_t (np.ndarray): Signed lateral offset of this lane's inner boundary. Shape is (N,).
            inner_line_id (int): ID of the RoadLine for the inner boundary.
            road_id (str): OpenDRIVE road id stored in custom_tags. Defaults to "".
            ref_kappa (np.ndarray, optional): Analytic curvature at each sample.
                Shape is (N,). When provided, used directly in offset computation.

        Returns:
            tuple:
                lane (Lane): The constructed Lane object.
                roadline (RoadLine): The RoadLine for the outer boundary.
                outer_t (np.ndarray): Cumulative signed offset for the next lane outward.
        """
        sign = np.sign(int(lane_node.attrib["id"]))
        if sign == 0:
            raise ValueError(
                "Lane id 0 is reserved for the centre lane and cannot appear "
                "in the left/right lane list."
            )

        location = type_node.attrib.get("type") if type_node is not None else None

        speed_node = lane_node.find("speed")
        speed_limit = (
            float(speed_node.attrib["max"])
            if speed_node is not None and "max" in speed_node.attrib
            else None
        )
        speed_limit_unit = (
            speed_node.attrib["unit"]
            if speed_node is not None and "unit" in speed_node.attrib
            else None
        )

        width_records = self._parse_width_records(lane_node)

        ls_s0 = float(ref_s[0])
        width_at_s = np.array(
            [_eval_cubic_poly(width_records, float(s) - ls_s0, s_key="sOffset") for s in ref_s]
        )

        outer_t = inner_t + sign * width_at_s

        inner_pts = _build_offset_polyline(ref_pts, ref_s, ref_normals, inner_t, ref_kappa)
        outer_pts = _build_offset_polyline(ref_pts, ref_s, ref_normals, outer_t, ref_kappa)

        inner_pts = _sanitise_linestring(inner_pts)
        outer_pts = _sanitise_linestring(outer_pts)
        if len(outer_pts) < 2:
            outer_pts = _build_offset_polyline(ref_pts, ref_s, ref_normals, outer_t, ref_kappa)

        inner_coords = inner_pts.tolist()
        outer_coords = outer_pts.tolist()

        if sign > 0:
            left_geom = LineString(outer_coords)
            right_geom = LineString(inner_coords)
        else:
            left_geom = LineString(inner_coords)
            right_geom = LineString(outer_coords)

        roadline = self._make_roadline(outer_coords, lane_node.find("roadMark"))

        line_ids = (
            {"left": [roadline.id_], "right": [inner_line_id]}
            if sign > 0
            else {"left": [inner_line_id], "right": [roadline.id_]}
        )

        lane = Lane(
            id_=self._next_id(),
            left_side=left_geom,
            right_side=right_geom,
            subtype=lane_node.attrib.get("type"),
            line_ids=line_ids,
            speed_limit=speed_limit,
            speed_limit_unit=speed_limit_unit,
            location=location,
            custom_tags={"xodr_road_id": road_id},
        )

        return lane, roadline, outer_t

    def _load_object(self, ref_pts: list, s_vals: np.ndarray, headings: list, obj_node: ET.Element):
        """Parse a <object> element and return an Area, or None if skipped."""
        obj_type = obj_node.attrib.get("type", "none").lower()
        allowed_types = {"crosswalk", "stopline", "parkingspace", "pedestriancrossing"}
        if obj_type not in allowed_types:
            return None

        s = float(obj_node.attrib["s"])
        t = float(obj_node.attrib["t"])

        idx = int(np.argmin(np.abs(s_vals - s)))
        heading = headings[idx]
        x, y = ref_pts[idx]
        x_origin = x - t * np.sin(heading)
        y_origin = y + t * np.cos(heading)
        rel_hdg = float(obj_node.attrib.get("hdg", 0.0))

        shape = None

        outline = obj_node.find("outline")
        if outline is not None:
            global_corners = outline.findall("cornerGlobal")
            local_corners = outline.findall("cornerLocal")

            if global_corners:
                poly_pts = [[float(c.attrib["x"]), float(c.attrib["y"])] for c in global_corners]
                if len(poly_pts) >= 3:
                    raw = Polygon(poly_pts)
                    shape = make_valid(raw) if not raw.is_valid else raw
                    if shape.geom_type == "MultiPolygon":
                        shape = max(shape.geoms, key=lambda g: g.area)
                    return Area(id_=self._next_id(), geometry=shape, subtype=obj_type)

            elif local_corners:
                poly_pts = [[float(c.attrib["u"]), float(c.attrib["v"])] for c in local_corners]
                if len(poly_pts) >= 3:
                    raw = Polygon(poly_pts)
                    shape = make_valid(raw) if not raw.is_valid else raw
                    if shape.geom_type == "MultiPolygon":
                        shape = max(shape.geoms, key=lambda g: g.area)

        if shape is None:
            w_str = obj_node.attrib.get("width")
            l_str = obj_node.attrib.get("length")
            if w_str is not None and l_str is not None:
                w, l = float(w_str), float(l_str)
                shape = Polygon(
                    [
                        [0.5 * l, -0.5 * w],
                        [0.5 * l, 0.5 * w],
                        [-0.5 * l, 0.5 * w],
                        [-0.5 * l, -0.5 * w],
                    ]
                )

        if shape is None:
            return None

        shape = rotate(shape, rel_hdg, use_radians=True, origin=(0, 0))
        shape = rotate(shape, heading, use_radians=True, origin=(0, 0))
        shape = affine_transform(shape, [1, 0, 0, 1, x_origin, y_origin])

        return Area(id_=self._next_id(), geometry=shape, subtype=obj_type)

    def load_road(self, road_node: ET.Element) -> tuple:
        """Parse a <road> element.

        Args:
            road_node (ET.Element): The ``<road>`` XML element to parse.

        Returns:
            tuple:
                lanes (list[Lane]): List of parsed Lane objects.
                roadlines (list[RoadLine]): List of parsed RoadLine objects.
                objects (list[Area]): List of parsed Area objects.
        """
        lanes, roadlines, objects = [], [], []
        type_node = road_node.find("type")
        road_id = road_node.attrib.get("id", "")

        raw_pts: list = []
        raw_s: list = []
        raw_kappas: list = []

        for geom_node in road_node.find("planView").findall("geometry"):
            new_pts, new_kappas = self._sample_geometry(geom_node)
            if not new_pts:
                continue

            s0 = float(geom_node.attrib["s"])
            sL = float(geom_node.attrib["length"])
            new_s = np.linspace(s0, s0 + sL, len(new_pts)).tolist()

            if raw_pts:
                gap = np.linalg.norm(np.array(new_pts[0]) - np.array(raw_pts[-1]))
                if gap > 0.1:
                    logging.warning("planView discontinuity of %.3f m at s=%.3f.", gap, s0)

            raw_pts.extend(new_pts)
            raw_s.extend(new_s)
            raw_kappas.extend(new_kappas)

        if not raw_pts:
            return lanes, roadlines, objects

        pts_arr = np.array(raw_pts, dtype=np.float64)
        s_arr = np.array(raw_s, dtype=np.float64)
        kappa_arr = np.array(raw_kappas, dtype=np.float64)

        dists = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
        keep = np.concatenate([[True], dists > 0.02])
        pts_arr = pts_arr[keep]
        s_arr = s_arr[keep]
        kappa_arr = kappa_arr[keep]

        if len(pts_arr) < 2:
            return lanes, roadlines, objects

        ref_normals = _unit_left_normals(pts_arr)

        lanes_node = road_node.find("lanes")
        if lanes_node is None:
            raise ValueError("<road> element has no <lanes> child.")

        lane_offset_records = sorted(
            [
                {
                    "s": float(lo.attrib["s"]),
                    "a": float(lo.attrib["a"]),
                    "b": float(lo.attrib["b"]),
                    "c": float(lo.attrib["c"]),
                    "d": float(lo.attrib["d"]),
                }
                for lo in lanes_node.findall("laneOffset")
            ],
            key=lambda r: r["s"],
        )

        lane_offset_t = np.array(
            [_eval_cubic_poly(lane_offset_records, float(s), s_key="s") for s in s_arr]
        )

        center_pts = pts_arr + lane_offset_t[:, np.newaxis] * ref_normals

        ls_nodes = lanes_node.findall("laneSection")
        ls_s_starts = [float(ls.attrib["s"]) for ls in ls_nodes]
        ls_s_ends = ls_s_starts[1:] + [float(s_arr[-1])]

        eps = 1e-6

        for ls_idx, ls_node in enumerate(ls_nodes):
            ls_s0 = ls_s_starts[ls_idx]
            ls_s1 = ls_s_ends[ls_idx]

            if ls_idx == 0:
                mask = (s_arr >= ls_s0) & (s_arr <= ls_s1 + eps)
            else:
                mask = (s_arr > ls_s0 - eps) & (s_arr <= ls_s1 + eps)

            if not np.any(mask):
                continue

            seg_ref_pts = pts_arr[mask]
            seg_ref_s = s_arr[mask]
            seg_ref_n = ref_normals[mask]
            seg_lo_t = lane_offset_t[mask]
            seg_center = center_pts[mask]
            seg_kappa = kappa_arr[mask]

            if len(seg_center) < 2:
                continue

            center_line = RoadLine(id_=self._next_id(), geometry=LineString(seg_center.tolist()))

            center_lane_node = ls_node.find("center").find("lane")
            rm_nodes = center_lane_node.findall("roadMark")
            rm_s_abs_starts = [ls_s0 + float(rm.attrib["sOffset"]) for rm in rm_nodes]
            rm_s_abs_ends = rm_s_abs_starts[1:] + [float(seg_ref_s[-1])]

            for rm_i, rm_node in enumerate(rm_nodes):
                rm_abs_s0 = rm_s_abs_starts[rm_i]
                rm_abs_s1 = rm_s_abs_ends[rm_i]
                rm_mask = (seg_ref_s >= rm_abs_s0 - eps) & (seg_ref_s <= rm_abs_s1 + eps)
                if not np.any(rm_mask):
                    continue
                rm_pts = seg_center[rm_mask].tolist()
                if len(rm_pts) < 2:
                    rm_pts = rm_pts + rm_pts
                roadlines.append(self._make_roadline(rm_pts, rm_node))

            left_node = ls_node.find("left")
            if left_node is not None:
                left_lane_nodes = sorted(
                    left_node.findall("lane"), key=lambda n: int(n.attrib["id"])
                )
                cumulative_t = seg_lo_t.copy()
                prev_id = center_line.id_

                for ln in left_lane_nodes:
                    lane, roadline, cumulative_t = self._load_lane(
                        ln,
                        type_node,
                        seg_ref_pts,
                        seg_ref_s,
                        seg_ref_n,
                        cumulative_t,
                        prev_id,
                        road_id=road_id,
                        ref_kappa=seg_kappa,
                    )
                    lanes.append(lane)
                    roadlines.append(roadline)
                    prev_id = roadline.id_

            right_node = ls_node.find("right")
            if right_node is not None:
                right_lane_nodes = sorted(
                    right_node.findall("lane"), key=lambda n: abs(int(n.attrib["id"]))
                )
                cumulative_t = seg_lo_t.copy()
                prev_id = center_line.id_

                for ln in right_lane_nodes:
                    lane, roadline, cumulative_t = self._load_lane(
                        ln,
                        type_node,
                        seg_ref_pts,
                        seg_ref_s,
                        seg_ref_n,
                        cumulative_t,
                        prev_id,
                        road_id=road_id,
                        ref_kappa=seg_kappa,
                    )
                    lanes.append(lane)
                    roadlines.append(roadline)
                    prev_id = roadline.id_

        objects_node = road_node.find("objects")
        if objects_node is not None:
            headings = self._compute_headings(pts_arr.tolist())
            for obj_node in objects_node.findall("object"):
                area = self._load_object(pts_arr.tolist(), s_arr, headings, obj_node)
                if area is not None:
                    objects.append(area)

        return lanes, roadlines, objects

    @staticmethod
    def _compute_headings(points: list) -> list:
        """Return the forward heading (radians) at each point of a polyline."""
        diff = np.diff(np.array(points), axis=0)
        hdgs = np.arctan2(diff[:, 1], diff[:, 0]).tolist()
        hdgs.append(hdgs[-1])
        return hdgs

    def load_junction(self, junction_node: ET.Element) -> Junction:
        """Parse a ``<junction>`` element and return a Junction object.

        Args:
            junction_node (ET.Element): The ``<junction>`` XML element to parse.

        Returns:
            Junction: A Junction populated with all ``<connection>`` child elements.

        Example:
        ```python
        import xml.etree.ElementTree as ET

        from tactics2d.map.parser import XODRParser

        parser = XODRParser()
        tree = ET.parse("path/to/map.xodr")
        junction_node = tree.getroot().find("junction")
        junction = parser.load_junction(junction_node)
        ```
        """
        junction = Junction(id_=self._next_id())

        for conn_node in junction_node.findall("connection"):
            connection = Junction(
                id_=self._next_id(),
                incoming_road=conn_node.attrib["incomingRoad"],
                connecting_road=conn_node.attrib["connectingRoad"],
                contact_point=conn_node.attrib["contactPoint"],
            )
            for ll in conn_node.findall("laneLink"):
                connection.add_lane_link((ll.attrib["from"], ll.attrib["to"]))
            junction.add_connection(connection)

        return junction

    def parse(self, file_path: str) -> Map:
        """Parse an OpenDRIVE file and return a Tactics2D Map object.

        Args:
            file_path (str): Absolute or relative path to the .xodr file.

        Returns:
            Map: A Tactics2D Map populated with lanes, roadlines, junctions,
                and area objects parsed from the OpenDRIVE file.
        """
        xml_root = ET.parse(file_path).getroot()

        header_info: dict = {}
        header_node = xml_root.find("header")
        if header_node is not None:
            header_info, _ = self.load_header(header_node)

        map_ = Map(header_info.get("name") or None)

        for road_node in xml_root.findall("road"):
            lanes, roadlines, objects = self.load_road(road_node)
            for lane in lanes:
                map_.add_lane(lane)
            for roadline in roadlines:
                map_.add_roadline(roadline)
            for obj in objects:
                map_.add_area(obj)

        for junction_node in xml_root.findall("junction"):
            map_.add_junction(self.load_junction(junction_node))

        self._id_counter = 0
        return map_
