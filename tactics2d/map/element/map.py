# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Map implementation."""


import enum
import warnings

import numpy as np

try:
    from shapely.strtree import STRtree

    HAS_STRTREE = True
except ImportError:
    HAS_STRTREE = False
    STRtree = None

from shapely.geometry import Point

from .area import Area
from .junction import Junction
from .lane import Lane
from .node import Node
from .regulatory import Regulatory
from .roadline import RoadLine


class MapElement(enum.Enum):
    NODE = enum.auto()
    LANE = enum.auto()
    AREA = enum.auto()
    ROADLINE = enum.auto()
    REGULATORY = enum.auto()
    JUNCTION = enum.auto()
    CUSTOM = enum.auto()


class Map:
    """This class implements a map to manage the road elements. The elements in the map include nodes, lanes, areas, roadlines, and regulations. The map is used to store the road elements and provide the interface to access the elements. Every element in the map should have a unique id.

    Attributes:
        name (str): The name of the map. Defaults to None.
        scenario_type (str): The scenario type of the map. Defaults to None.
        country (str): The country that the map is located in. Defaults to None.
        ids (set): The identifier of the elements in the map. All elements in the map should have a unique id. A conflict in ids will raise a KeyError.
        nodes (dict): The nodes in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_node](#tactics2d.map.element.Map.add_node) method.
        lanes (dict): The lanes in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_lane](#tactics2d.map.element.Map.add_lane) method.
        areas (dict): The areas in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_area](#tactics2d.map.element.Map.add_area) method.
        roadlines (dict): The roadlines in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_roadline](#tactics2d.map.element.Map.add_roadline) method.
        regulations (dict): The regulations in the map. Defaults to an empty dictionary. This attribute needs to be set manually by trigger the [add_regulatory](#tactics2d.map.element.Map.add_regulatory) method.
        boundary (tuple): The boundary of the map expressed in the form of (min_x, max_x, min_y, max_y). This attribute is **read-only**.
    """

    def __init__(self, name: str = None, scenario_type: str = None, country: str = None):
        """Initialize the attributes in the class.

        Args:
            name (str, optional): The name of the map. Defaults to None.
            scenario_type (str, optional): The scenario type of the map. Defaults to None.
            country (str, optional): The country that the map is located in. Defaults to None.
        """
        self.name = name
        self.scenario_type = scenario_type
        self.country = country

        self.ids = dict()
        self.nodes = dict()
        self.lanes = dict()
        self.areas = dict()
        self.junctions = dict()
        self.roadlines = dict()
        self.regulations = dict()
        self.customs = dict()

        # Spatial indexing for fast queries
        self._spatial_index = None
        self._spatial_index_dirty = True
        self._element_geometries = {}  # element_id -> geometry
        self._spatial_geometries = []  # List of geometries for STRtree (parallel to _spatial_ids)
        self._spatial_ids = []  # List of element IDs corresponding to geometries
        self._geometry_to_id = {}  # geometry id -> element_id for fast lookup

        self._boundary = None
        # For incremental boundary updates
        self._min_x = None
        self._max_x = None
        self._min_y = None
        self._max_y = None

    @property
    def boundary(self):
        if self._boundary is not None:
            return self._boundary

        # If min/max values are available, use them to compute boundary
        if (
            self._min_x is not None
            and self._max_x is not None
            and self._min_y is not None
            and self._max_y is not None
        ):
            self._boundary = (
                np.floor(self._min_x),
                np.ceil(self._max_x),
                np.floor(self._min_y),
                np.ceil(self._max_y),
            )
            return self._boundary

        # Otherwise compute boundary from all elements and update min/max values
        # Collect all coordinates for vectorized min/max computation
        all_x = []
        all_y = []

        # Collect node coordinates
        for node in self.nodes.values():
            all_x.append(node.x)
            all_y.append(node.y)

        # Collect lane coordinates
        for lane in self.lanes.values():
            if not hasattr(lane, "geometry") or lane.geometry is None:
                continue
            coords = np.array(lane.geometry.coords)
            if len(coords) > 0:
                all_x.extend(coords[:, 0])
                all_y.extend(coords[:, 1])

        # Collect area coordinates (exterior only)
        for area in self.areas.values():
            if not hasattr(area, "geometry") or area.geometry is None:
                continue
            coords = np.array(area.geometry.exterior.coords)
            if len(coords) > 0:
                all_x.extend(coords[:, 0])
                all_y.extend(coords[:, 1])

        # Collect roadline coordinates
        for roadline in self.roadlines.values():
            if not hasattr(roadline, "geometry") or roadline.geometry is None:
                continue
            coords = np.array(roadline.geometry.coords)
            if len(coords) > 0:
                all_x.extend(coords[:, 0])
                all_y.extend(coords[:, 1])

        # Regulatory elements currently not included in boundary calculation
        # TODO: get shape of regulations

        if len(all_x) == 0:
            self._boundary = (0.0, 0.0, 0.0, 0.0)
            # Keep min/max as None for empty map to allow proper incremental updates
        else:
            x_min = np.min(all_x)
            x_max = np.max(all_x)
            y_min = np.min(all_y)
            y_max = np.max(all_y)
            self._boundary = (np.floor(x_min), np.ceil(x_max), np.floor(y_min), np.ceil(y_max))
            # Update min/max values for future incremental updates
            self._min_x = x_min
            self._max_x = x_max
            self._min_y = y_min
            self._max_y = y_max

        return self._boundary

    def _get_element_geometry(self, element):
        """Get geometry object for spatial indexing."""
        if hasattr(element, "geometry") and element.geometry is not None:
            return element.geometry
        elif hasattr(element, "x") and hasattr(element, "y"):
            # Node or similar point element
            return Point(element.x, element.y)
        return None

    def _update_boundary_with_coords(self, coords):
        """Update min/max coordinates with new coordinate array.

        Args:
            coords: numpy array of shape (N, 2) or list of (x, y) tuples
        """
        if coords is None or len(coords) == 0:
            return

        coords_array = np.array(coords)
        if len(coords_array.shape) == 1:
            # Single point (x, y)
            x, y = coords_array
            x_vals = np.array([x])
            y_vals = np.array([y])
        else:
            x_vals = coords_array[:, 0]
            y_vals = coords_array[:, 1]

        # Update min/max values
        if self._min_x is None:
            self._min_x = np.min(x_vals)
            self._max_x = np.max(x_vals)
            self._min_y = np.min(y_vals)
            self._max_y = np.max(y_vals)
        else:
            self._min_x = min(self._min_x, np.min(x_vals))
            self._max_x = max(self._max_x, np.max(x_vals))
            self._min_y = min(self._min_y, np.min(y_vals))
            self._max_y = max(self._max_y, np.max(y_vals))

    def _update_boundary_with_element(self, element):
        """Update boundary with coordinates from a map element."""
        if hasattr(element, "x") and hasattr(element, "y"):
            # Node-like element
            self._update_boundary_with_coords([(element.x, element.y)])
        elif hasattr(element, "geometry") and element.geometry is not None:
            # Geometry-based element
            geom = element.geometry
            # Handle Polygon first (has exterior attribute, coords raises NotImplementedError)
            try:
                # Try to access exterior for Polygon
                self._update_boundary_with_coords(list(geom.exterior.coords))
            except AttributeError:
                # Not a Polygon, try coords for LineString, LinearRing, etc.
                try:
                    self._update_boundary_with_coords(list(geom.coords))
                except NotImplementedError:
                    # Polygon's coords raises NotImplementedError, but we already tried exterior
                    # This shouldn't happen if exterior check worked
                    pass
            # Other geometry types could be added here

    def _add_element_to_spatial_index(self, element_id, element):
        """Add or update element geometry in spatial index data."""
        geometry = self._get_element_geometry(element)
        if geometry is not None:
            self._element_geometries[element_id] = geometry
            self._spatial_index_dirty = True
        elif element_id in self._element_geometries:
            # Remove element if it no longer has geometry
            del self._element_geometries[element_id]
            self._spatial_index_dirty = True

    def _rebuild_spatial_index(self):
        """Rebuild spatial index if dirty and STRtree is available."""
        if not HAS_STRTREE:
            return
        if self._spatial_index_dirty and self._element_geometries:
            # Build parallel lists of geometries and IDs
            self._spatial_geometries = []
            self._spatial_ids = []
            # Build mapping from geometry to ID for fast lookup
            self._geometry_to_id = {}
            for elem_id, geom in self._element_geometries.items():
                self._spatial_geometries.append(geom)
                self._spatial_ids.append(elem_id)
                # Use id(geom) as key since shapely geometries may not be hashable
                self._geometry_to_id[id(geom)] = elem_id

            self._spatial_index = STRtree(self._spatial_geometries)
            self._spatial_index_dirty = False
        elif not self._element_geometries:
            self._spatial_index = None
            self._spatial_geometries = []
            self._spatial_ids = []
            self._geometry_to_id = {}

    def _ensure_spatial_index(self):
        """Ensure spatial index is up to date."""
        if self._spatial_index_dirty:
            self._rebuild_spatial_index()

    def query_point(self, point, buffer=1.0):
        """Query elements near a point within given buffer distance.

        Args:
            point: Tuple (x, y) or shapely Point
            buffer: Buffer distance in meters

        Returns:
            List of element IDs near the point
        """
        if not HAS_STRTREE or not self._element_geometries:
            return []

        self._ensure_spatial_index()
        from shapely.geometry import Point as ShapelyPoint

        if not isinstance(point, ShapelyPoint):
            point = ShapelyPoint(point)

        query_geom = point.buffer(buffer)
        results = self._spatial_index.query(query_geom)

        # Convert geometry results back to element IDs using geometry to ID mapping
        element_ids = []
        for geom in results:
            # Use id(geom) to lookup element ID
            elem_id = self._geometry_to_id.get(id(geom))
            if elem_id is not None:
                element_ids.append(elem_id)
        return element_ids

    def query_bbox(self, bbox):
        """Query elements within a bounding box.

        Args:
            bbox: Tuple (min_x, max_x, min_y, max_y)

        Returns:
            List of element IDs within the bounding box
        """
        if not HAS_STRTREE or not self._element_geometries:
            return []

        self._ensure_spatial_index()
        from shapely.geometry import box

        query_box = box(bbox[0], bbox[2], bbox[1], bbox[3])
        results = self._spatial_index.query(query_box)

        # Convert geometry results back to element IDs using geometry to ID mapping
        element_ids = []
        for geom in results:
            # Use id(geom) to lookup element ID
            elem_id = self._geometry_to_id.get(id(geom))
            if elem_id is not None:
                element_ids.append(elem_id)
        return element_ids

    def add_node(self, node: Node):
        """This function adds a node to the map.

        Args:
            node (Node): The node to be added to the map.

        Raises:
            KeyError: If the id of the node is used by any other road element.
        """
        if node.id_ in self.ids:
            if node.id_ in self.nodes:
                warnings.warn(f"Node {node.id_} already exists! Replaced the node with new data.")
                # Reset boundary on replacement since old coordinates may affect min/max
                self._boundary = None
                self._min_x = self._max_x = self._min_y = self._max_y = None
            else:
                raise KeyError(f"The id of Node {node.id_} is used by the other road element.")
        self.nodes[node.id_] = node
        self.ids[node.id_] = MapElement.NODE
        self._add_element_to_spatial_index(node.id_, node)
        # Update boundary with new element
        self._update_boundary_with_element(node)
        self._boundary = None  # Invalidate cached boundary

    def add_roadline(self, roadline: RoadLine):
        """This function adds a roadline to the map.

        Args:
            roadline (RoadLine): The roadline to be added to the map.

        Raises:
            KeyError: If the id of the roadline is used by any other road element.
        """
        if roadline.id_ in self.ids:
            if roadline.id_ in self.roadlines:
                warnings.warn(
                    f"Roadline {roadline.id_} already exists! Replaced the roadline with new data."
                )
                # Reset boundary on replacement since old coordinates may affect min/max
                self._boundary = None
                self._min_x = self._max_x = self._min_y = self._max_y = None
            else:
                raise KeyError(
                    f"The id of Roadline {roadline.id_} is used by the other road element."
                )

        self.roadlines[roadline.id_] = roadline
        self.ids[roadline.id_] = MapElement.ROADLINE
        self._add_element_to_spatial_index(roadline.id_, roadline)
        # Update boundary with new element
        self._update_boundary_with_element(roadline)
        self._boundary = None  # Invalidate cached boundary

    def add_junction(self, junction: Junction):
        """This function adds a junction to the map.

        Args:
            junction (Junction): The junction to be added to the map.
        """
        if junction.id_ in self.ids:
            if junction.id_ in self.junctions:
                warnings.warn(
                    f"Junction {junction.id_} already exists! Replaced the junction with new data."
                )
                # Reset boundary on replacement since old coordinates may affect min/max
                self._boundary = None
                self._min_x = self._max_x = self._min_y = self._max_y = None
            else:
                raise KeyError(
                    f"The id of Junction {junction.id_} is used by the other road element."
                )

        self.junctions[junction.id_] = junction
        self.ids[junction.id_] = MapElement.JUNCTION
        self._add_element_to_spatial_index(junction.id_, junction)
        # Update boundary with new element (junction may not have geometry)
        self._update_boundary_with_element(junction)
        self._boundary = None  # Invalidate cached boundary

    def add_lane(self, lane: Lane):
        """This function adds a lane to the map.

        Args:
            lane (Lane): The lane to be added to the map.

        Raises:
            KeyError: If the id of the lane is used by any other road element.
        """
        if lane.id_ in self.ids:
            if lane.id_ in self.lanes:
                warnings.warn(f"Lane {lane.id_} already exists! Replacing the lane with new data.")
                # Reset boundary on replacement since old coordinates may affect min/max
                self._boundary = None
                self._min_x = self._max_x = self._min_y = self._max_y = None
            else:
                raise KeyError(f"The id of Lane {lane.id_} is used by the other road element.")

        self.lanes[lane.id_] = lane
        self.ids[lane.id_] = MapElement.LANE
        self._add_element_to_spatial_index(lane.id_, lane)
        # Update boundary with new element
        self._update_boundary_with_element(lane)
        self._boundary = None  # Invalidate cached boundary

    def add_area(self, area: Area):
        """This function adds an area to the map.

        Args:
            area (Area): The area to be added to the map.

        Raises:
            KeyError: If the id of the area is used by any other road element.
        """
        if area.id_ in self.ids:
            if area.id_ in self.areas:
                warnings.warn(f"Area {area.id_} already exists! Replacing the area with new data.")
                # Reset boundary on replacement since old coordinates may affect min/max
                self._boundary = None
                self._min_x = self._max_x = self._min_y = self._max_y = None
            else:
                raise KeyError(f"The id of Area {area.id_} is used by the other road element.")

        self.areas[area.id_] = area
        self.ids[area.id_] = MapElement.AREA
        self._add_element_to_spatial_index(area.id_, area)
        # Update boundary with new element
        self._update_boundary_with_element(area)
        self._boundary = None  # Invalidate cached boundary

    def add_regulatory(self, regulatory: Regulatory):
        """This function adds a traffic regulation to the map.

        Args:
            regulatory (Regulatory): The regulatory to be added to the map.

        Raises:
            KeyError: If the id of the regulatory is used by any other road element.
        """
        if regulatory.id_ in self.ids:
            if regulatory.id_ in self.regulations:
                warnings.warn(
                    f"Regulatory {regulatory.id_} already exists! Replacing the regulatory with new data."
                )
                # Reset boundary on replacement since old coordinates may affect min/max
                self._boundary = None
                self._min_x = self._max_x = self._min_y = self._max_y = None
            else:
                raise KeyError(
                    f"The id of Regulatory {regulatory.id_} is used by the other road element."
                )

        self.regulations[regulatory.id_] = regulatory
        self.ids[regulatory.id_] = MapElement.REGULATORY
        self._add_element_to_spatial_index(regulatory.id_, regulatory)
        # Update boundary with new element (regulatory may not have geometry)
        self._update_boundary_with_element(regulatory)
        self._boundary = None  # Invalidate cached boundary

    def set_boundary(self, boundary: tuple):
        """This function sets the boundary of the map.

        Args:
            boundary (tuple): The boundary of the map expressed in the form of (min_x, max_x, min_y, max_y).
        """
        self._boundary = boundary
        # Update min/max values from boundary
        if boundary is not None:
            self._min_x = boundary[0]
            self._max_x = boundary[1]
            self._min_y = boundary[2]
            self._max_y = boundary[3]

    def get_by_id(self, id_: str):
        """This function returns the road element with the given id.

        Args:
            id_ (str): The id of the road element.
        """
        if not id_ in self.ids:
            warnings.warn(f"Cannot find element with id {id_}.")
            return None

        if self.ids[id_] == MapElement.NODE:
            return self.nodes[id_]
        elif self.ids[id_] == MapElement.LANE:
            return self.lanes[id_]
        elif self.ids[id_] == MapElement.AREA:
            return self.areas[id_]
        elif self.ids[id_] == MapElement.ROADLINE:
            return self.roadlines[id_]
        elif self.ids[id_] == MapElement.REGULATORY:
            return self.regulations[id_]
        elif self.ids[id_] == MapElement.CUSTOM:
            return self.customs[id_]

    def reset(self):
        """This function resets the map by clearing all the road elements."""
        self.ids.clear()
        self.nodes.clear()
        self.lanes.clear()
        self.areas.clear()
        self.roadlines.clear()
        self.regulations.clear()
        self.customs.clear()
        # Clear spatial indexing data
        self._element_geometries.clear()
        self._spatial_geometries.clear()
        self._spatial_ids.clear()
        self._geometry_to_id.clear()
        self._spatial_index = None
        self._spatial_index_dirty = True
        self._boundary = None
        self._min_x = self._max_x = self._min_y = self._max_y = None
