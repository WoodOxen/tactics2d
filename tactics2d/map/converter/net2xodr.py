# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""SUMO net.xml to OpenDRIVE xodr converter implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.parser import NetXMLParser


class Net2XodrConverter:
    """This class implements a converter from SUMO (.net.xml) to OpenDRIVE (.xodr).

    The converter reads a SUMO net.xml file using NetXMLParser, then writes the
    parsed map into OpenDRIVE xodr format. Each Tactics2D Lane becomes an
    OpenDRIVE road. The lane centre-line is derived from left and right boundaries
    and fitted to a paramPoly3 geometry per segment for compact, accurate output.
    Lane width is estimated as the mean point-to-point distance between boundaries.

    Example:
        >>> from tactics2d.map.converter import Net2XodrConverter
        >>> converter = Net2XodrConverter()
        >>> converter.convert("path/to/map.net.xml", "path/to/output.xodr")
    """

    _MAX_SEG_LENGTH = 20.0

    def _get_centerline(self, lane, n: int = 50) -> list:
        """Derive centre-line from left and right lane boundaries.

        Args:
            lane: Tactics2D Lane object.
            n (int): Number of sample points. Defaults to 50.

        Returns:
            list: List of (x, y) tuples. Empty if boundaries are degenerate.
        """
        left = lane.left_side
        right = lane.right_side
        if left is None or right is None:
            return []
        length = min(left.length, right.length)
        if length < 1e-6:
            return []
        n = max(2, int(length / 0.5))
        s_vals = np.linspace(0, 1, n)
        return [
            (
                (left.interpolate(s, normalized=True).x + right.interpolate(s, normalized=True).x)
                / 2,
                (left.interpolate(s, normalized=True).y + right.interpolate(s, normalized=True).y)
                / 2,
            )
            for s in s_vals
        ]

    def _get_width(self, lane, n: int = 10) -> float:
        """Estimate lane width as mean point-to-point distance between boundaries.

        Args:
            lane: Tactics2D Lane object.
            n (int): Number of sample points. Defaults to 10.

        Returns:
            float: Estimated lane width in metres.
        """
        left = lane.left_side
        right = lane.right_side
        if left is None or right is None:
            return 3.2
        s_vals = np.linspace(0, 1, n)
        return float(
            np.mean(
                [
                    left.interpolate(s, normalized=True).distance(
                        right.interpolate(s, normalized=True)
                    )
                    for s in s_vals
                ]
            )
        )

    def _fit_param_poly3(self, pts: np.ndarray) -> tuple:
        """Fit a paramPoly3 geometry to a polyline segment in local coordinates.

        Transforms the segment to a local frame (origin at start, x-axis along
        initial heading), then fits cubic polynomials U(p) and V(p) where p is
        the normalised arc-length parameter in [0, 1].

        Args:
            pts (np.ndarray): Polyline points in world coordinates. Shape (N, 2).

        Returns:
            tuple: (x, y, hdg, length, coeffs_u, coeffs_v) or None if degenerate.
        """
        x0, y0 = pts[0]
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        hdg = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else 0.0

        cos_h, sin_h = np.cos(hdg), np.sin(hdg)

        # arc-length parameter
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total = float(seg_lengths.sum())
        if total < 1e-6:
            return None

        s_cum = np.concatenate([[0], np.cumsum(seg_lengths)])
        p = s_cum / total  # normalised [0,1]

        # local coordinates
        local = pts - pts[0]
        u = local[:, 0] * cos_h + local[:, 1] * sin_h
        v = -local[:, 0] * sin_h + local[:, 1] * cos_h

        # fit cubic polynomials with constraint at p=0: coeff[0]=0
        coeffs_u = np.polyfit(p, u, 3)[::-1]
        coeffs_v = np.polyfit(p, v, 3)[::-1]

        return x0, y0, hdg, total, coeffs_u, coeffs_v

    def _split_segments(self, pts: list) -> list[np.ndarray]:
        """Split a polyline into segments no longer than _MAX_SEG_LENGTH.

        Args:
            pts (list): List of (x, y) tuples.

        Returns:
            list: List of numpy arrays, each a segment.
        """
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

    def convert(self, input_path: str, output_path: str) -> str:
        """Convert a SUMO net.xml file to an OpenDRIVE xodr file.

        Reads the SUMO network file with NetXMLParser, maps the resulting
        Tactics2D Map to OpenDRIVE elements, and writes the output file.
        Each Tactics2D Lane becomes an OpenDRIVE road with a single driving
        lane. Centre-lines are fitted to paramPoly3 geometries for accuracy.
        Junctions and connections are preserved.

        Args:
            input_path (str): Path to the input .net.xml file.
            output_path (str): Path to the output .xodr file.

        Returns:
            str: The output file path.

        Example:
            >>> from tactics2d.map.converter import Net2XodrConverter
            >>> converter = Net2XodrConverter()
            >>> converter.convert("map.net.xml", "map.xodr")
        """
        map_ = NetXMLParser().parse(input_path)
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

            width = self._get_width(lane)
            speed = lane.speed_limit if lane.speed_limit else 50.0 / 3.6
            sumo_id = (lane.custom_tags or {}).get("sumo_id", str(lane_id))
            total_length = float(LineString(pts).length)

            road = ET.SubElement(
                root,
                "road",
                {
                    "name": sumo_id,
                    "length": f"{total_length:.4f}",
                    "id": str(lane_id),
                    "junction": "-1",
                },
            )

            ET.SubElement(road, "type", {"s": "0.0", "type": "town"})

            plan_view = ET.SubElement(road, "planView")
            segments = self._split_segments(pts)
            s_offset = 0.0

            for seg in segments:
                fit = self._fit_param_poly3(seg)
                if fit is None:
                    continue
                x, y, hdg, length, coeffs_u, coeffs_v = fit
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
                    ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV"), (*coeffs_u, *coeffs_v)
                ):
                    pp3.set(k, f"{v:.6f}")
                s_offset += length

            ET.SubElement(road, "elevationProfile")
            ET.SubElement(road, "lateralProfile")

            lanes_elem = ET.SubElement(road, "lanes")
            lane_section = ET.SubElement(lanes_elem, "laneSection", {"s": "0.0"})
            ET.SubElement(lane_section, "left")

            center = ET.SubElement(lane_section, "center")
            center_lane = ET.SubElement(
                center, "lane", {"id": "0", "type": "none", "level": "false"}
            )
            ET.SubElement(
                center_lane,
                "roadMark",
                {
                    "sOffset": "0",
                    "type": "solid",
                    "weight": "standard",
                    "color": "standard",
                    "width": "0.13",
                },
            )

            right_elem = ET.SubElement(lane_section, "right")
            right_lane = ET.SubElement(
                right_elem,
                "lane",
                {"id": "-1", "type": lane.subtype if lane.subtype else "driving", "level": "false"},
            )
            ET.SubElement(
                right_lane,
                "width",
                {"sOffset": "0", "a": f"{width:.4f}", "b": "0", "c": "0", "d": "0"},
            )
            ET.SubElement(
                right_lane,
                "roadMark",
                {
                    "sOffset": "0",
                    "type": "solid",
                    "weight": "standard",
                    "color": "standard",
                    "width": "0.13",
                },
            )
            ET.SubElement(
                right_lane, "speed", {"sOffset": "0", "max": f"{speed:.3f}", "unit": "m/s"}
            )

        for junc_id, junction in map_.junctions.items():
            junc_elem = ET.SubElement(root, "junction", {"name": "", "id": str(junc_id)})
            for conn_id, conn in junction.connections.items():
                tags = conn.custom_tags or {}
                from_edge = tags.get("from_edge", "")
                to_edge = tags.get("to_edge", "")
                from_lane = tags.get("from_lane", "0")
                to_lane = tags.get("to_lane", "0")
                if not from_edge or not to_edge:
                    continue
                conn_elem = ET.SubElement(
                    junc_elem,
                    "connection",
                    {
                        "id": str(conn_id),
                        "incomingRoad": from_edge,
                        "connectingRoad": to_edge,
                        "contactPoint": "start",
                    },
                )
                ET.SubElement(
                    conn_elem,
                    "laneLink",
                    {
                        "from": f"-{from_lane}" if from_lane != "0" else "-1",
                        "to": f"-{to_lane}" if to_lane != "0" else "-1",
                    },
                )

        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
            indent="    "
        )

        lines = [line for line in xml_str.splitlines() if line.strip()]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
