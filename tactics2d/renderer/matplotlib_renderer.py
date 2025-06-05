###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: matplotlib_renderer.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9


import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon


class MatplotlibRenderer:
    def __init__(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        resolution: Tuple[float, float],
        dpi: int = 200,
    ):
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution
        self.dpi = dpi
        self.width = self.resolution[0] / self.dpi
        self.height = self.resolution[1] / self.dpi

        aspect_ratio = (self.ylim[1] - self.ylim[0]) / (self.xlim[1] - self.xlim[0])
        self.height = self.width * aspect_ratio + 0.5

        self.camera_position = None
        self.camera_yaw = None
        self.road_lines = dict()
        self.road_line_geometry = dict()
        self.road_polygons = dict()
        self.road_polygon_geometry = dict()
        self.participants = dict()

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(self.width, self.height)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        self.ax.set_aspect("equal")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_axis_off()

    def _create_polygon(self, element):
        return Polygon(
            xy=element["geometry"],
            closed=True,
            facecolor=element["color"],
            linewidth=element["line_width"],
            zorder=element["order"],
        )

    def _create_circle(self, element):
        return Circle(
            xy=(0, 0),
            radius=element["radius"],
            facecolor=element["color"],
            linewidth=element["line_width"],
            zorder=element["order"],
        )

    def _create_line(self, element):
        line_shape = np.array(element["geometry"])
        lines = []

        # TODO: Handle "solid_solid", "solid_dashed", "dashed_solid", and "dashed_dashed"
        if "dashed" in element["type"]:
            lines.append(
                Line2D(
                    line_shape[:, 0],
                    line_shape[:, 1],
                    linewidth=element["line_width"],
                    linestyle=(0, (5, 5)),
                    color=element["color"],
                    zorder=element["order"],
                )
            )

        elif "solid" in element["type"]:
            lines.append(
                Line2D(
                    line_shape[:, 0],
                    line_shape[:, 1],
                    linewidth=element["line_width"],
                    color=element["color"],
                    zorder=element["order"],
                )
            )

        return lines

    def _transform_to_camera_view(self, points):
        points = np.array(points)
        dx, dy = -self.camera_position
        cos_theta = np.cos(-self.camera_yaw)
        sin_theta = np.sin(-self.camera_yaw)

        translated = points + np.array([dx, dy])
        rotated = np.dot(translated, np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]))

        return rotated

    def _update_polygon(self, polygon, geometry, position=[0, 0], rotation=0):
        center = np.array(position)
        yaw = rotation
        shape = np.array(geometry)

        cos_theta = np.cos(yaw)
        sin_theta = np.sin(yaw)
        rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        transformed = (shape @ rotation.T) + center
        transformed = self._transform_to_camera_view(transformed)
        polygon.set_xy(transformed)

    def _update_circle(self, circle, position):
        transformed_center = self._transform_to_camera_view(position)
        circle.set_center(transformed_center)

    def _update_line(self, lines, geometry, position=[0, 0], rotation=0):
        for line in lines:
            # TODO: Handle "solid_solid", "solid_dashed", "dashed_solid", and "dashed_dashed"
            center = np.array(position)
            yaw = rotation
            shape = np.array(geometry)

            cos_theta = np.cos(yaw)
            sin_theta = np.sin(yaw)
            rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            transformed = (shape @ rotation.T) + center
            transformed = self._transform_to_camera_view(transformed)
            line.set_data(transformed[:, 0], transformed[:, 1])

    def update(
        self, geometry_data: dict, camera_position: Tuple[float, float], camera_yaw: float = 0
    ):
        self.camera_position = camera_position
        self.camera_yaw = camera_yaw

        road_id_to_remove = geometry_data["map_data"]["road_id_to_remove"]
        road_elements = geometry_data["map_data"]["road_elements"]
        participant_id_to_create = geometry_data["participant_data"]["participant_id_to_create"]
        participant_id_to_remove = geometry_data["participant_data"]["participant_id_to_remove"]
        participants = geometry_data["participant_data"]["participants"]

        # Add new road elements
        for element in road_elements:
            element_id = element["id"]
            if element_id in self.road_polygons or element_id in self.road_lines:
                continue  # Skip if already exists

            if element["type"] == "polygon":
                polygon = self._create_polygon(element)
                self.road_polygons[element_id] = polygon
                self.road_polygon_geometry[element_id] = element["geometry"]
                self.ax.add_patch(polygon)

            elif "line" in element["type"]:
                lines = self._create_line(element)
                self.road_lines[element_id] = lines
                self.road_line_geometry[element_id] = element["geometry"]
                for line in lines:
                    self.ax.add_line(line)

        # Update road elements
        for element_id, geometry in self.road_polygon_geometry.items():
            self._update_polygon(self.road_polygons[element_id], geometry)

        for element_id, geometry in self.road_line_geometry.items():
            self._update_line(self.road_lines[element_id], geometry)

        # Remove road elements
        for element_id in road_id_to_remove:
            if element_id in self.road_polygons:
                polygon = self.road_polygons.pop(element_id, None)
                if polygon is not None:
                    polygon.remove()
                self.road_polygon_geometry.pop(element_id, None)

            elif element_id in self.road_lines:
                lines = self.road_lines.pop(element_id, None)
                for line in lines:
                    line.remove()
                self.road_line_geometry.pop(element_id, None)

        # Remove dead participants
        for id_ in participant_id_to_remove:
            participant = self.participants.pop(id_, None)
            if participant is not None:
                participant.remove()

        for participant in participants:
            id_ = participant["id"]

            # Create
            if id_ in participant_id_to_create and id_ not in self.participants:
                if participant["type"] == "polygon":
                    patch = self._create_polygon(participant)
                elif participant["type"] == "circle":
                    patch = self._create_circle(participant)
                else:
                    continue
                self.participants[id_] = patch
                self.ax.add_patch(patch)

            # Update
            if id_ in self.participants:
                if participant["type"] == "polygon":
                    self._update_polygon(
                        self.participants[id_],
                        participant["geometry"],
                        participant["position"],
                        participant["rotation"],
                    )
                elif participant["type"] == "circle":
                    self._update_circle(self.participants[id_], participant["position"])

    def reset(self):
        self.camera_position = None
        self.camera_yaw = None

        # Remove road lines
        for lines in self.road_lines.values():
            for line in lines:
                line.remove()
        self.road_lines.clear()
        self.road_line_geometry.clear()

        # Remove road polygons
        for polygon in self.road_polygons.values():
            polygon.remove()
        self.road_polygons.clear()
        self.road_polygon_geometry.clear()

        # Remove participants
        for patch in self.participants.values():
            patch.remove()
        self.participants.clear()

        # Optional: remove other artists like annotations/text if you later add them
        for artist in reversed(self.ax.artists):
            artist.remove()

        # Do NOT use ax.clear(), instead reapply axis limits and settings
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
